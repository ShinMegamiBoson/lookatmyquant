"""Lossless compiler artifacts and exact search snapshots.

The packed format is artifact.json plus weights.bin. It retains fp32 scales and
matrix-specific tables rather than silently rounding them during export. File
size is derived from this format's real tensor layout and canonical manifest.
Optional table payload aliases preserve logical group identities and expose
immutable shared views; they are separate from tied model-state aliases.
Imports stay lightweight so the allocator can price shapes without torch/MLX.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil

FORMAT = "opgraph-packed-1"
STATE_BITS = 12
TABLE_ENTRIES = 1 << STATE_BITS
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "U32": 4, "I32": 4,
               "I64": 8, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}
NP_DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2", "U32": "<u4",
             "I32": "<i4", "I64": "<i8", "I16": "<i2", "U16": "<u2", "I8": "i1", "U8": "u1", "BOOL": "?"}


def row_group_payload_bytes(rows, columns, bits, *, scale_bytes=4, table_entries=TABLE_ENTRIES, include_rows=True):
    """Exact tensor payload for one group; shared input signs are priced once per matrix."""
    if rows < 0 or columns < 1 or not 1 <= bits <= 8 or scale_bytes not in (2, 4, 8):
        raise ValueError("invalid row-group shape or precision")
    return rows * ((columns + 31) // 32 * bits * 4 + scale_bytes + (4 if include_rows else 0)) + table_entries * 4


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_spec(dtype, shape):
    shape = [int(v) for v in shape]
    if dtype not in DTYPE_BYTES or any(v < 0 for v in shape):
        raise ValueError("invalid tensor specification")
    return {"dtype": dtype, "shape": shape, "nbytes": math.prod(shape) * DTYPE_BYTES[dtype]}


def table_tensor_names(semantic):
    """Logical codebook names referenced by groups, including embedding buckets."""
    names = set()

    def visit(matrix):
        if matrix.get('kind') == 'grouped':
            names.update(group['key'] + '.table' for group in matrix['groups'])
        elif matrix.get('kind') == 'table':
            for bucket in matrix['buckets']:
                if 'matrix' in bucket:
                    visit(bucket['matrix'])

    for matrix in semantic.get('matrices', {}).values():
        visit(matrix)
    return names


def validate_artifact_layout(meta):
    """Return logical name -> physical owner after validating the entire layout.

    Table aliases are direct links to an earlier, non-alias table. Every stored
    descriptor must agree, including its checksum. Other tensors cannot share
    payload, and unique payload owners cover the blob exactly once, in order.
    This helper has no NumPy/Torch/MLX dependency and also serves streaming readers.
    """
    if meta.get('format') != FORMAT or meta.get('blob') != 'weights.bin':
        raise ValueError('unsupported compiler artifact')
    if type(meta.get('blob_bytes')) is not int or meta['blob_bytes'] < 0:
        raise ValueError('invalid blob size')
    tensors = meta['tensors']
    tables, owners, offset = table_tensor_names(meta), {}, 0
    for name, spec in sorted(tensors.items()):
        expected = tensor_spec(spec['dtype'], spec['shape'])
        digest = spec.get('sha256')
        if (expected['nbytes'] != spec['nbytes'] or type(spec['offset']) is not int
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError(f'invalid tensor layout: {name}')
        if 'alias_of' in spec:
            owner = spec['alias_of']
            if (not isinstance(owner, str) or name not in tables or owner not in tables or owner not in owners
                    or owners[owner] != owner):
                raise ValueError(f'invalid table alias owner: {name}')
            anchor = tensors[owner]
            if any(spec[field] != anchor[field] for field in ('dtype', 'shape', 'nbytes', 'sha256', 'offset')):
                raise ValueError(f'table alias descriptor mismatch: {name}')
            owners[name] = owner
        else:
            if spec['offset'] != offset:
                raise ValueError(f'invalid tensor layout: {name}')
            owners[name] = name
            offset += spec['nbytes']
    if offset != meta['blob_bytes']:
        raise ValueError('unaccounted artifact payload')
    return owners


def artifact_manifest(specs, semantic, *, tensor_aliases=None):
    """Plan a real manifest without allocating tensors; hashes have a fixed encoded length."""
    if set(semantic) & {'format', 'blob', 'blob_bytes', 'tensors'}:
        raise ValueError('semantic metadata cannot override the storage contract')
    aliases = dict(tensor_aliases or {})
    if not set(aliases).issubset(specs):
        raise ValueError('unknown table alias name')
    if any(not isinstance(owner, str) for owner in aliases.values()):
        raise ValueError('invalid table alias owner')
    offset, tensors = 0, {}
    for name, spec in sorted(specs.items()):
        spec = tensor_spec(spec["dtype"], spec["shape"])
        owner = aliases.get(name)
        if owner is not None:
            if owner not in tensors:
                raise ValueError(f'table alias owner must precede its alias: {name}')
            tensors[name] = {**spec, "offset": tensors[owner]['offset'], "sha256": "0" * 64, 'alias_of': owner}
        else:
            tensors[name] = {**spec, "offset": offset, "sha256": "0" * 64}
            offset += spec["nbytes"]
    manifest = {"format": FORMAT, "blob": "weights.bin", "blob_bytes": offset, "tensors": tensors, **semantic}
    validate_artifact_layout(manifest)
    return manifest


def artifact_serialized_bytes(specs, semantic, *, tensor_aliases=None):
    """Exact bytes of artifact.json + weights.bin (tokenizer/config assets are separate)."""
    manifest = artifact_manifest(specs, semantic, tensor_aliases=tensor_aliases)
    return manifest["blob_bytes"] + len(canonical_json(manifest))


def grouped_matrix_specs(name, shape, groups, *, state_bits=STATE_BITS):
    """Return (tensor specs, matrix metadata) from rows/bits descriptors, no tensor allocation.

    Groups are dictionaries with rows=count, bits=k, and optionally kind='scalar'
    and table_entries. Each group has an explicit uint32 row map, fp32 scale/table.
    """
    rows, columns = map(int, shape)
    if sum(int(g["rows"]) for g in groups) != rows:
        raise ValueError("groups must cover the matrix")
    specs = {name + ".signs": tensor_spec("I8", [columns])}
    records = []
    for i, group in enumerate(groups):
        m, bits = int(group["rows"]), int(group["bits"])
        if m <= 0 or not 1 <= bits <= 8:
            raise ValueError("empty group or unsupported width")
        key, kind = f"{name}.g{i}", group.get("kind", "trellis")
        table_entries = int(group.get("table_entries", (1 << state_bits) if kind == "trellis" else (1 << bits)))
        specs.update({key + ".rows": tensor_spec("U32", [m]),
                      key + ".codes": tensor_spec("U32", [m, (columns + 31) // 32 * bits]),
                      key + ".scales": tensor_spec("F32", [m]),
                      key + ".table": tensor_spec("F32", [table_entries])})
        records.append({"key": key, "kind": kind, "bits": bits, "rows": m,
                        "columns": columns, "padded_columns": (columns + 31) // 32 * 32,
                        **({"state_bits": state_bits, "start_state": 0} if kind == "trellis" else {})})
    return specs, {"kind": "grouped", "shape": [rows, columns], "signs": name + ".signs", "groups": records}


def _np():
    import numpy as np
    return np


def _array(value):
    np = _np()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _typed_array(value):
    np = _np()
    if hasattr(value, "detach") and str(value.dtype) == "torch.bfloat16":
        import torch
        return value.detach().cpu().contiguous().view(torch.uint16).numpy(), "BF16"
    a = _array(value)
    names = {np.dtype(v): k for k, v in NP_DTYPES.items() if k != "BF16"}
    if a.dtype not in names:
        raise ValueError(f"unsupported storage dtype {a.dtype}")
    code = names[a.dtype]
    return np.ascontiguousarray(a.astype(NP_DTYPES[code], copy=False)), code


def _storage_identity(value):
    """Only identify exact aliases, never merely equal-valued independent weights."""
    if hasattr(value, 'untyped_storage'):
        return ('torch', str(value.device), str(value.dtype), value.untyped_storage().data_ptr(),
                value.storage_offset(), tuple(value.shape), tuple(value.stride()))
    a = _array(value)
    return ('numpy', a.dtype.str, a.__array_interface__['data'][0], tuple(a.shape), tuple(a.strides))


def state_aliases(dense_tensors, coded):
    """Alias->anchor, preferring a coded anchor so a tied head is not duplicated dense."""
    groups = {}
    for name, value in dense_tensors.items():
        # Empty tensors have ambiguous shared zero pointers and cost no payload.
        if math.prod(value.shape):
            groups.setdefault(_storage_identity(value), []).append(name)
    aliases = {}
    for names in groups.values():
        if len(names) < 2:
            continue
        compressed = [name for name in names if name in coded]
        if len(compressed) > 1:
            raise ValueError('multiple coded aliases require an explicit untied selected state')
        anchor = compressed[0] if compressed else sorted(names)[0]
        if compressed and 'buckets' not in coded[anchor]:
            raise ValueError('packed alias support is currently limited to tied embedding/output tables')
        aliases.update({name: anchor for name in names if name != anchor})
    return aliases


def pack_symbols(symbols, bits):
    """Forward scalar symbols, little-endian, each row padded to groups of 32."""
    np = _np()
    q = _array(symbols)
    if q.ndim != 2 or not 1 <= bits <= 8 or np.any(q < 0) or np.any(q >= 1 << bits):
        raise ValueError("invalid symbols or width")
    m, n = q.shape
    q = np.pad(q, ((0, 0), (0, (-n) % 32))).astype(np.uint64).reshape(m, -1, 32)
    result = np.zeros((m, q.shape[1], bits), dtype=np.uint32)
    for i in range(32):
        word, shift = divmod(i * bits, 32)
        value = q[:, :, i] << np.uint64(shift)
        result[:, :, word] |= (value & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        if shift + bits > 32:
            result[:, :, word + 1] |= (value >> np.uint64(32)).astype(np.uint32)
    return result.reshape(m, -1)


def unpack_symbols(codes, bits, columns):
    np = _np()
    codes = _array(codes).astype(np.uint32, copy=False)
    m = codes.shape[0]
    blocks = (columns + 31) // 32
    if codes.shape != (m, blocks * bits):
        raise ValueError("code dimensions disagree with width")
    words = codes.reshape(m, blocks, bits).astype(np.uint64)
    result = np.empty((m, blocks, 32), dtype=np.uint32)
    for i in range(32):
        word, shift = divmod(i * bits, 32)
        value = words[:, :, word] >> np.uint64(shift)
        if shift + bits > 32:
            value |= words[:, :, word + 1] << np.uint64(32 - shift)
        result[:, :, i] = (value & np.uint64((1 << bits) - 1)).astype(np.uint32)
    return result.reshape(m, -1)[:, :columns]


def valid_path(states, bits, state_bits=STATE_BITS):
    np = _np()
    states = _array(states).astype(np.int64, copy=False)
    return (states.ndim == 2 and states.shape[1] > 0 and bool(np.all((states >= 0) & (states < (1 << state_bits))))
            and bool(np.all(states[:, 0] < (1 << bits)))
            and bool(np.all(states[:, 1:] >> bits == states[:, :-1] & ((1 << (state_bits - bits)) - 1))))


def infer_path_bits(states, state_bits=STATE_BITS):
    choices = [bits for bits in range(1, min(8, state_bits - 1) + 1) if valid_path(states, bits, state_bits)]
    if len(choices) != 1:
        raise ValueError(f"trellis width is ambiguous or invalid ({choices}); save explicit row_groups metadata")
    return choices[0]


def pack_trellis(states, bits, state_bits=STATE_BITS):
    np = _np()
    states = _array(states).astype(np.int64, copy=False)
    if not valid_path(states, bits, state_bits):
        raise ValueError("states are not a row trellis from zero")
    symbols = states & ((1 << bits) - 1)
    # Pad the *forward* sequence, then reverse. Padding must not precede real states.
    symbols = np.pad(symbols, ((0, 0), (0, (-symbols.shape[1]) % 32)))
    return pack_symbols(symbols[:, ::-1], bits)


def unpack_trellis(codes, bits, columns, state_bits=STATE_BITS):
    np = _np()
    padded = (columns + 31) // 32 * 32
    symbols = unpack_symbols(codes, bits, padded)[:, ::-1]
    state = np.zeros(len(symbols), dtype=np.uint32)
    result = np.empty((len(symbols), columns), dtype=np.uint32)
    mask = (1 << state_bits) - 1
    for i in range(columns):
        state = ((state << bits) | symbols[:, i]) & mask
        result[:, i] = state
    return result


def _input_signs(columns, seed):
    # Preserve the quantizer's exact RNG definition, not a NumPy approximation.
    import torch
    return torch.where(torch.rand(columns, generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float64) < .5,
                       -1, 1).numpy().astype('int8')


def encode_entry(name, entry, *, signs=None):
    """Encode one saved Coded/TrellisCoded entry without changing a value or path.

    Explicit row_groups use original row ids, bits, table_index. Legacy mixed
    tables are recognized by per-row 4096-entry state offsets. Additional trained
    transforms/scales are rejected until explicitly represented by this format.
    """
    np = _np()
    unsupported = set(entry) & {"log_g", "group_scales", "log_c", "col_scales", "kron", "A", "B"}
    if unsupported:
        raise ValueError(f"unsupported trained transforms: {sorted(unsupported)}")
    state_based = "states" in entry
    encoded = _array(entry["states" if state_based else "codes"])
    if encoded.ndim != 2 or not np.issubdtype(encoded.dtype, np.integer):
        raise ValueError("saved entry must be a matrix")
    m, n = encoded.shape
    scales = _array(entry["scales"])
    if scales.shape != (m,) or scales.dtype != np.float32:
        raise ValueError("saved row scales must be fp32, one per row")
    if not np.isfinite(scales).all():
        raise ValueError("nonfinite row scales")
    table = _array(entry["table" if state_based else "levels"])
    if table.ndim != 1 or table.dtype != np.float32 or not np.isfinite(table).all():
        raise ValueError("saved table must be a finite fp32 vector")
    signs = _input_signs(n, entry["in_seed"]) if signs is None else _array(signs).astype(np.int8)
    if signs.shape != (n,) or not np.isin(signs, [-1, 1]).all():
        raise ValueError("invalid input signs")
    if state_based:
        if len(table) % TABLE_ENTRIES or np.any(encoded < 0) or np.any(encoded >= len(table)):
            raise ValueError("invalid saved trellis table or states")
        group_ids = encoded[:, 0] // TABLE_ENTRIES
        if not np.all(encoded // TABLE_ENTRIES == group_ids[:, None]):
            raise ValueError("a row crosses table identities")
        groups = entry.get("row_groups")
        if groups is None:
            groups = []
            for table_index in sorted(set(group_ids.tolist())):
                rows = np.flatnonzero(group_ids == table_index)
                states = encoded[rows] - table_index * TABLE_ENTRIES
                groups.append({"rows": rows, "bits": infer_path_bits(states), "table_index": table_index})
        prepared = []
        for group in groups:
            rows = _array(group["rows"]).astype(np.int64)
            bits, ti = int(group["bits"]), int(group["table_index"])
            if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= m) or not np.all(group_ids[rows] == ti):
                raise ValueError("invalid row group")
            states = encoded[rows] - ti * TABLE_ENTRIES
            prepared.append((rows, bits, "trellis", table[ti * TABLE_ENTRIES:(ti + 1) * TABLE_ENTRIES], pack_trellis(states, bits)))
    else:
        bits = max(1, (len(table) - 1).bit_length())
        if np.any(encoded >= len(table)):
            raise ValueError("scalar index outside table")
        prepared = [(np.arange(m), bits, "scalar", table, pack_symbols(encoded, bits))]
    if sorted(np.concatenate([p[0] for p in prepared]).tolist()) != list(range(m)):
        raise ValueError("row groups must cover every row exactly once")
    descriptors = [{"rows": len(rows), "bits": bits, "kind": kind, "table_entries": len(values)}
                   for rows, bits, kind, values, codes in prepared]
    specs, meta = grouped_matrix_specs(name, [m, n], descriptors)
    tensors = {name + ".signs": signs}
    for i, (rows, bits, kind, values, codes) in enumerate(prepared):
        key = f"{name}.g{i}"
        tensors.update({key + ".rows": rows.astype(np.uint32), key + ".codes": codes,
                        key + ".scales": scales[rows], key + ".table": values})
    assert all(_typed_array(value)[0].nbytes == specs[key]["nbytes"] for key, value in tensors.items())
    return tensors, meta


def export_saved_codes(saved, out, *, dense_tensors, model_dir=None):
    """Pack a complete model state using exact saved codes plus all uncoded tensors.

    dense_tensors must be the correctly named complete model state: a source
    model state_dict for a calibration-only build, or the selected state for a
    trained build. A codes file alone does not contain uncoded weights or dense
    embedding buckets. This function never loads a model or reconstructs codes.
    """
    np = _np()
    coded = saved['matrices']
    if not set(coded).issubset(dense_tensors):
        raise ValueError('complete dense state is missing named coded matrices')
    aliases = state_aliases(dense_tensors, coded)
    tensors, matrices = {}, {}
    for name, entry in coded.items():
        if 'buckets' not in entry:
            values, meta = encode_entry(name, entry)
            tensors.update(values)
            matrices[name] = meta
            continue
        dense = dense_tensors[name]
        shape = list(dense.shape)
        buckets, row_ids = [], []
        for i, bucket in enumerate(entry['buckets']):
            key = f'{name}.b{i}'
            rows = _array(bucket['rows']).astype(np.uint32)
            row_ids.extend(rows.tolist())
            tensors[key + '.rows'] = rows
            if 'states' in bucket or 'codes' in bucket:
                values, meta = encode_entry(key, bucket)
                tensors.update(values)
                buckets.append({'rows': key + '.rows', 'matrix': meta})
            else:
                # Preserve dtype and bits of kept rows; do not cast them to bf16.
                subset = dense[rows.astype(np.int64)]
                tensors[key + '.dense'] = subset
                buckets.append({'rows': key + '.rows', 'dense': key + '.dense'})
        if sorted(row_ids) != list(range(shape[0])):
            raise ValueError(f'embedding buckets do not exactly cover {name}')
        matrices[name] = {'kind': 'table', 'shape': shape, 'buckets': buckets}
    dense_names = []
    for name, value in dense_tensors.items():
        if name not in coded and name not in aliases:
            tensors[name] = value
            dense_names.append(name)
    assets = {}
    if model_dir is not None:
        for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'chat_template.jinja', 'tokenizer.model'):
            source = Path(model_dir) / name
            if source.exists():
                assets[name] = source
    semantic = {'matrices': matrices, 'dense': dense_names, 'aliases': aliases, 'self_contained_weights': True,
                'source_model': saved.get('model'), 'scale_precision': 'fp32 preserved',
                'assets': {name: {'bytes': source.stat().st_size, 'sha256': file_sha256(source)}
                           for name, source in assets.items()}}
    return write_artifact(out, tensors, semantic, assets=assets)


def decode_rotated(tensors, meta):
    """Reconstruct exact stored table*scale values before the input transform."""
    np = _np()
    result = np.empty(meta["shape"], dtype=np.float32)
    for group in meta["groups"]:
        key, bits, n = group["key"], group["bits"], group["columns"]
        indices = (unpack_trellis(tensors[key + ".codes"], bits, n, group["state_bits"])
                   if group["kind"] == "trellis" else unpack_symbols(tensors[key + ".codes"], bits, n))
        result[tensors[key + ".rows"]] = tensors[key + ".table"][indices] * tensors[key + ".scales"][:, None]
    return result


def _table_fingerprint(dtype, shape, raw):
    return dtype, tuple(shape), hashlib.sha256(raw).digest()


def _exact_table_aliases(values):
    """Hash only proposes a match; complete bytes, dtype and shape must agree."""
    pools, aliases = {}, {}
    for name, dtype, shape, raw in values:
        key = _table_fingerprint(dtype, shape, raw)
        pool = pools.setdefault(key, [])
        owner = next((n for n, d, s, b in pool if d == dtype and s == tuple(shape) and b == raw), None)
        if owner is None:
            pool.append((name, dtype, tuple(shape), raw))
        else:
            aliases[name] = owner
    return aliases


def find_table_aliases(tensors, semantic):
    """Find exact immutable codebook sharing; never aliases scales or dense state."""
    def values():
        for name in sorted(table_tensor_names(semantic)):
            value, dtype = _typed_array(tensors[name])
            yield name, dtype, value.shape, value.tobytes(order='C')

    return _exact_table_aliases(values())


def manifest_table_aliases(meta):
    """Content-hash sharing plan for pricing only; repacking verifies actual bytes.

    A source manifest does not prove that two byte strings with the same digest
    are identical. find_table_aliases/repack_artifact additionally compare bytes.
    """
    validate_artifact_layout(meta)
    pools, aliases = {}, {}
    for name in sorted(table_tensor_names(meta)):
        spec = meta['tensors'][name]
        key = spec['dtype'], tuple(spec['shape']), spec['sha256']
        if key in pools:
            aliases[name] = pools[key]
        else:
            pools[key] = name
    return aliases


def write_artifact(path, tensors, semantic, *, assets=None, tensor_aliases=None):
    """Write once, verify all bytes on reload, report actual storage. Never overwrite an artifact."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    typed = {name: _typed_array(value) for name, value in tensors.items()}
    specs = {name: tensor_spec(dtype, value.shape) for name, (value, dtype) in typed.items()}
    manifest = artifact_manifest(specs, semantic, tensor_aliases=tensor_aliases)
    owners = validate_artifact_layout(manifest)
    for name, owner in owners.items():
        if owner != name and typed[name][0].tobytes() != typed[owner][0].tobytes():
            raise ValueError(f'table alias bytes differ: {name}')
    path.mkdir(parents=True)
    with (path / "weights.bin").open("xb") as out:
        for name in sorted(typed):
            if owners[name] != name:
                manifest['tensors'][name]['sha256'] = manifest['tensors'][owners[name]]['sha256']
                continue
            raw = typed[name][0].tobytes(order='C')
            manifest["tensors"][name]["sha256"] = hashlib.sha256(raw).hexdigest()
            out.write(raw)
    (path / "artifact.json").write_bytes(canonical_json(manifest))
    for name, source in (assets or {}).items():
        if Path(name).name != name or name in ("weights.bin", "artifact.json"):
            raise ValueError("invalid asset name")
        shutil.copyfile(source, path / name)
    loaded, checked = read_artifact(path)
    assert checked == manifest
    for name, (value, dtype) in typed.items():
        if loaded[name].tobytes() != value.tobytes():
            raise RuntimeError(f"artifact round-trip failed for {name}")
    payload_bytes = (path / "weights.bin").stat().st_size
    manifest_bytes = (path / "artifact.json").stat().st_size
    assert payload_bytes + manifest_bytes == artifact_serialized_bytes(specs, semantic, tensor_aliases=tensor_aliases)
    return {"path": str(path), "payload_bytes": payload_bytes, "manifest_bytes": manifest_bytes,
            "asset_bytes": sum(p.stat().st_size for p in path.iterdir() if p.name not in ("weights.bin", "artifact.json")),
            "total_bytes": sum(p.stat().st_size for p in path.iterdir()),
            "manifest_sha256": hashlib.sha256((path / "artifact.json").read_bytes()).hexdigest(),
            "tensor_count": len(tensors), "physical_tensor_count": sum(n == owner for n, owner in owners.items()),
            "table_alias_count": sum(n != owner for n, owner in owners.items()), "all_tensor_bytes_verified": True}


def read_artifact(path, *, verify=True):
    np = _np()
    path = Path(path)
    meta = json.loads((path / 'artifact.json').read_text())
    owners = validate_artifact_layout(meta)
    blob = path / 'weights.bin'
    if blob.stat().st_size != meta['blob_bytes']:
        raise ValueError('artifact blob size mismatch')
    tensors = {}
    with blob.open('rb') as file:
        for name, spec in sorted(meta['tensors'].items()):
            if owners[name] != name:
                tensors[name] = tensors[owners[name]]
                continue
            file.seek(spec['offset'])
            raw = file.read(spec['nbytes'])
            if len(raw) != spec['nbytes'] or (verify and hashlib.sha256(raw).hexdigest() != spec['sha256']):
                raise ValueError(f'tensor checksum mismatch: {name}')
            tensors[name] = np.frombuffer(raw, dtype=NP_DTYPES[spec['dtype']]).reshape(spec['shape'])
    for name, spec in meta.get('assets', {}).items():
        if Path(name).name != name:
            raise ValueError('invalid asset path')
        asset = path / name
        if asset.stat().st_size != spec['bytes'] or (verify and file_sha256(asset) != spec['sha256']):
            raise ValueError(f'asset checksum mismatch: {name}')
    return tensors, meta


def _artifact_metadata(path):
    """Read metadata and account for every file; payload verification is separate."""
    path = Path(path)
    raw = (path / 'artifact.json').read_bytes()
    meta = json.loads(raw)
    owners = validate_artifact_layout(meta)
    if (path / 'weights.bin').stat().st_size != meta['blob_bytes']:
        raise ValueError('artifact blob size mismatch')
    assets = meta.get('assets', {})
    for name, spec in assets.items():
        if Path(name).name != name or name in ('.', '..', 'weights.bin', 'artifact.json'):
            raise ValueError('invalid asset path')
        if (path / name).stat().st_size != spec['bytes']:
            raise ValueError(f'asset size mismatch: {name}')
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    expected = {'artifact.json', 'weights.bin'} | set(assets)
    if actual != expected:
        raise ValueError(f'unaccounted artifact files: {sorted(actual ^ expected)}')
    return meta, owners, raw


def _stream_blob(path, meta, owners, *, output=None, kept=None):
    """Verify each physical tensor once, optionally copying a subset in order."""
    physical = [name for name in sorted(owners) if owners[name] == name]
    if output is not None and not set(kept).issubset(physical):
        raise ValueError('destination payload is not a subset of original owners')
    whole, verified = hashlib.sha256(), {}
    with (Path(path) / 'weights.bin').open('rb') as file:
        for name in physical:
            spec = meta['tensors'][name]
            if file.tell() != spec['offset']:
                raise ValueError('source payload offset drift')
            copying = output is not None and name in kept
            if copying and output.tell() != kept[name]['offset']:
                raise ValueError('destination payload offset drift')
            digest, remaining = hashlib.sha256(), spec['nbytes']
            while remaining:
                block = file.read(min(8 << 20, remaining))
                if not block:
                    raise ValueError(f'truncated tensor: {name}')
                remaining -= len(block)
                digest.update(block)
                whole.update(block)
                if copying:
                    output.write(block)
            verified[name] = digest.hexdigest()
            if verified[name] != spec['sha256']:
                raise ValueError(f'tensor checksum mismatch: {name}')
        if file.read(1):
            raise ValueError('unaccounted artifact payload')
    return whole.hexdigest(), {name: verified[owner] for name, owner in owners.items()}


def _file_receipt(path, meta, raw_manifest, blob_hash):
    files = {'artifact.json': {'bytes': len(raw_manifest), 'sha256': hashlib.sha256(raw_manifest).hexdigest()},
             'weights.bin': {'bytes': meta['blob_bytes'], 'sha256': blob_hash}}
    for name, spec in meta.get('assets', {}).items():
        digest = file_sha256(Path(path) / name)
        if digest != spec['sha256']:
            raise ValueError(f'asset checksum mismatch: {name}')
        files[name] = {'bytes': (Path(path) / name).stat().st_size, 'sha256': digest}
    return {'payload_bytes': files['weights.bin']['bytes'], 'manifest_bytes': files['artifact.json']['bytes'],
            'asset_bytes': sum(v['bytes'] for k, v in files.items() if k not in ('artifact.json', 'weights.bin')),
            'total_bytes': sum(v['bytes'] for v in files.values()), 'files': files}


def inspect_artifact(path):
    """Verify a complete artifact in bounded memory and report all file bytes/hashes."""
    meta, owners, raw = _artifact_metadata(path)
    blob_hash, verified = _stream_blob(path, meta, owners)
    receipt = _file_receipt(path, meta, raw, blob_hash)
    receipt.update(logical_tensor_count=len(owners), physical_tensor_count=sum(n == o for n, o in owners.items()),
                   table_alias_count=sum(n != o for n, o in owners.items()), all_tensor_hashes_verified=True)
    return receipt


def repack_artifact(source, destination):
    """Share exact immutable tables in a new artifact, without decoding any weight.

    Every original physical tensor is checksum-verified, even when its duplicate
    bytes are omitted. The destination is independently verified on disk. Group
    identities, independent scales, model-state aliases and all assets are kept.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('source and destination artifacts must be separate directories')
    meta, old_owners, raw_manifest = _artifact_metadata(source)
    with (source / 'weights.bin').open('rb') as file:
        def table_values():
            for name in sorted(table_tensor_names(meta)):
                spec = meta['tensors'][name]
                file.seek(spec['offset'])
                raw = file.read(spec['nbytes'])
                if len(raw) != spec['nbytes'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
                    raise ValueError(f'tensor checksum mismatch: {name}')
                yield name, spec['dtype'], spec['shape'], raw
        aliases = _exact_table_aliases(table_values())
    specs = {name: tensor_spec(spec['dtype'], spec['shape']) for name, spec in meta['tensors'].items()}
    semantic = {k: v for k, v in meta.items() if k not in ('format', 'blob', 'blob_bytes', 'tensors')}
    new_meta = artifact_manifest(specs, semantic, tensor_aliases=aliases)
    for name, spec in new_meta['tensors'].items():
        spec['sha256'] = meta['tensors'][name]['sha256']
    new_owners = validate_artifact_layout(new_meta)
    contract = {name: {k: spec[k] for k in ('dtype', 'shape', 'sha256')} for name, spec in meta['tensors'].items()}
    if contract != {name: {k: spec[k] for k in ('dtype', 'shape', 'sha256')} for name, spec in new_meta['tensors'].items()}:
        raise ValueError('logical tensor contract changed')
    destination.mkdir(parents=True)
    try:
        kept = {name: new_meta['tensors'][name] for name, owner in new_owners.items() if name == owner}
        with (destination / 'weights.bin').open('xb') as output:
            original_blob_hash, verified = _stream_blob(source, meta, old_owners, output=output, kept=kept)
        original = _file_receipt(source, meta, raw_manifest, original_blob_hash)
        original.update(logical_tensor_count=len(old_owners),
                        physical_tensor_count=sum(n == owner for n, owner in old_owners.items()),
                        table_alias_count=sum(n != owner for n, owner in old_owners.items()),
                        all_tensor_hashes_verified=True)
        for name in meta.get('assets', {}):
            shutil.copyfile(source / name, destination / name)
        (destination / 'artifact.json').write_bytes(canonical_json(new_meta))
        repacked = inspect_artifact(destination)
        if ((source / 'artifact.json').read_bytes() != raw_manifest
                or (source / 'weights.bin').stat().st_size != meta['blob_bytes']):
            raise ValueError('source artifact changed while repacking')
        if repacked['payload_bytes'] + repacked['manifest_bytes'] != artifact_serialized_bytes(specs, semantic, tensor_aliases=aliases):
            raise ValueError('serialized byte estimate disagrees with actual files')
        if any(original['files'][name] != repacked['files'][name] for name in meta.get('assets', {})):
            raise ValueError('copied assets changed')
        if any(verified[name] != contract[name]['sha256'] for name in contract):
            raise ValueError('original tensor verification incomplete')
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {'source': str(source), 'destination': str(destination), 'operation': 'exact immutable table sharing; no rerounding',
            'original': original, 'repacked': repacked, 'tensor_aliases': aliases, 'logical_tensors': contract,
            'logical_tensor_count': len(contract), 'unchanged_logical_dtype_shape_hash': True,
            'all_original_and_repacked_tensor_hashes_verified': True, 'semantic_metadata_unchanged': True,
            'payload_bytes_saved': original['payload_bytes'] - repacked['payload_bytes'],
            'manifest_bytes_added': repacked['manifest_bytes'] - original['manifest_bytes'],
            'total_bytes_saved': original['total_bytes'] - repacked['total_bytes']}


def normalize_library_record(record):
    """Old dense libraries remain usable; new libraries may retain the exact code entry."""
    if isinstance(record, dict):
        return {'weight': record['weight'], 'bits': float(record['bits']), 'code': record.get('code')}
    if len(record) not in (2, 3):
        raise ValueError('library entry must contain weights, bits, and optionally exact codes')
    return {'weight': record[0], 'bits': float(record[1]), 'code': record[2] if len(record) == 3 else None}


def search_plan(plan, widths, old_bits, new_bits, shapes, *, source, artifact):
    """Refresh derived fields; preserve only the budget constraint and truthful measured costs."""
    new = json.loads(json.dumps(plan))
    new['linears'] = {name: (f'tr{widths[name]}' + ('x' if value.endswith('x') else '')
                             if name in widths else value) for name, value in plan['linears'].items()}
    for key in ('predicted kl', 'summary', 'cost'):
        new.pop(key, None)
    delta = sum(new_bits.values()) - sum(old_bits.values())
    target = plan.get('target spec', {})
    if plan.get('target') == 'disk' and 'cost' in plan:
        new['cost'] = plan['cost'] + delta
        new['cost basis'] = 'modeled stored bits; exact snapshot file bytes reported separately'
    elif target.get('ops') == 0 and 'cost' in plan:
        new['cost'] = plan['cost'] + delta / target.get('batch', 1)
        new['cost basis'] = 'inherited fixed items plus changed stored bits per batch'
    summary = {}
    for name, bits in new_bits.items():
        for key in ('searched linears', 'role ' + name.split('.', 1)[1], 'layer ' + name.split('.', 1)[0]):
            b, n = summary.get(key, (0, 0))
            summary[key] = b + bits, n + math.prod(shapes[name])
    new['summary'] = {key: round(bits / count, 6) for key, (bits, count) in summary.items()}
    new['searched from'] = source
    new['selected artifact'] = artifact
    return new


def save_search_snapshot(path, base_snapshot, weights, widths, *, codes=None, code_template=None, provenance=None):
    """Persist the exact selected tensors. Existing dense-only libraries cannot become compact by assertion.

    base_snapshot carries the original unchanged tensors. All final candidates are
    stored at their current dtype, without converting fp32 to bf16. A complete
    code payload is written separately only when every changed matrix has one.
    """
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    snapshot = dict(base_snapshot)
    snapshot['params'] = {name: tensor.detach().cpu().clone() for name, tensor in base_snapshot['params'].items()}
    snapshot['params'].update({name: tensor.detach().cpu().clone() for name, tensor in weights.items()})
    snapshot['selection'] = {'widths': dict(widths), 'provenance': provenance or {}, 'exact_tensors': True}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    for name, value in snapshot['params'].items():
        if not torch.equal(value, restored['params'][name]) or value.dtype != restored['params'][name].dtype:
            raise RuntimeError(f'selected snapshot changed {name}')
    result = {'path': str(path), 'sha256': file_sha256(path),
              'file_bytes': path.stat().st_size, 'parameters': len(snapshot['params']),
              'exact_tensor_reload': True, 'representation': 'dense selected weights',
              'requires_base_model': True, 'packed_model': False}
    complete = code_template is not None and codes is not None and all(codes.get(name) is not None for name in weights)
    if complete:
        code_path = path.with_suffix(path.suffix + '.codes')
        if code_path.exists():
            raise FileExistsError(code_path)
        payload = dict(code_template)
        payload['matrices'] = {**code_template['matrices'], **codes}
        payload['selection'] = snapshot['selection']
        torch.save(payload, code_path)
        result['exact_codes'] = {'path': str(code_path), 'sha256': file_sha256(code_path),
                                 'file_bytes': code_path.stat().st_size,
                                 'parity_to_selected_dense_weights': 'requires packed reconstruction check'}
    else:
        result['exact_codes'] = None
        result['compact_export_boundary'] = 'Legacy library lacks code entries; no dense-to-code rerounding was performed.'
    return result
