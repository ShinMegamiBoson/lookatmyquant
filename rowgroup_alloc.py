"""Global precision allocation over importance-ranked trellis row groups.

The default partition has at most eight groups per matrix, with boundaries on
64-row multiples. It includes the old 12.5% split as a candidate. Unlike the old
split, bits can move between matrices. The loss is explicitly a proxy supplied
by the caller; this module does not claim model accuracy from an allocation.

Costs come from compiler_artifact.row_group_payload_bytes, not rate_bits().
For a complete-file budget, pass artifact_size(selected_groups), using the
artifact serializer's exact manifest estimator. Without that callback the
budget covers tensor payload plus fixed_bytes only, as reported in the result.
No models or tensor libraries are imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import math
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


class BudgetError(ValueError):
    """No allocation fits the current byte budget."""


@dataclass(frozen=True)
class WidthOption:
    bits: int
    payload_bytes: int
    loss: float


@dataclass(frozen=True)
class RowGroup:
    matrix: str
    index: int
    rows: tuple[int, ...]
    columns: int
    importance: float
    options: tuple[WidthOption, ...]
    rank_start: int

    @property
    def key(self) -> str:
        return f"{self.matrix}/g{self.index}"


@dataclass(frozen=True)
class Allocation:
    selected: tuple[dict, ...]
    payload_bytes: int
    total_bytes: int
    budget_bytes: int
    fixed_bytes: int
    manifest_reserve_bytes: int
    objective: float
    size_scope: str
    solves: int

    def as_dict(self) -> dict:
        return {
            "format": "rowgroup-plan-1",
            "groups": list(self.selected),
            "payload_bytes": self.payload_bytes,
            "total_bytes": self.total_bytes,
            "budget_bytes": self.budget_bytes,
            "fixed_bytes": self.fixed_bytes,
            "manifest_reserve_bytes": self.manifest_reserve_bytes,
            "predicted_loss": self.objective,
            "size_scope": self.size_scope,
            "solves": self.solves,
            "optimality": "exact for the additive loss and final reserved payload budget",
            "qualification": "Proxy allocation; no claim about generated answers or full-model loss.",
        }


def _nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def make_row_groups(
    matrices: Mapping[str, Mapping],
    distortion_by_bits: Mapping[int, float],
    *,
    payload_cost: Callable[[int, int, int], int] | None = None,
) -> tuple[RowGroup, ...]:
    """Make stable groups from {name: {importance, columns, base_bits}}.

    ``importance`` is one nonnegative score per output row, in comparable units
    across matrices. The allowed widths are base_bits-1, base_bits, base_bits+1;
    base_bits must be 2 or 3. Distortion coefficients must be supplied explicitly
    (measured values or a preregistered rate-distortion approximation).

    The default cost helper prices packed codes, fp32 scales, uint32 row maps,
    and a separate fp32 trellis table per group. Shared input signs and all
    unmodified model tensors belong in fixed_bytes/the full-file estimator.
    """
    if payload_cost is None:
        from compiler_artifact import row_group_payload_bytes
        payload_cost = row_group_payload_bytes
    groups = []
    for name, matrix in sorted(matrices.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("matrix names must be nonempty strings")
        scores = tuple(float(x) for x in matrix["importance"])
        columns = _nonnegative_int(matrix["columns"], "columns")
        base = _nonnegative_int(matrix["base_bits"], "base_bits")
        if base not in (2, 3):
            raise ValueError(f"{name}: row allocation requires a 2- or 3-bit baseline")
        if not scores or len(scores) % 64 or not columns:
            raise ValueError(f"{name}: nonempty rows must be a multiple of 64 and columns positive")
        if any(not math.isfinite(x) or x < 0 for x in scores):
            raise ValueError(f"{name}: importance must be finite and nonnegative")
        widths = (base - 1, base, base + 1)
        distortions = {bits: float(distortion_by_bits[bits]) for bits in widths}
        if any(not math.isfinite(x) or x < 0 for x in distortions.values()):
            raise ValueError("distortion coefficients must be finite and nonnegative")
        order = sorted(range(len(scores)), key=lambda row: (-scores[row], row))
        # Include the old one-eighth boundaries exactly; skip empty intervals
        # for narrow matrices rather than creating invalid zero-row groups.
        first_half = [(i * len(scores) // 8) // 64 * 64 for i in range(5)]
        cuts = sorted(set(first_half) | {len(scores) - cut for cut in first_half[:4]})
        for index, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
            rows = tuple(order[lo:hi])
            importance = math.fsum(scores[row] for row in rows)
            options = tuple(WidthOption(bits,
                                        _nonnegative_int(payload_cost(len(rows), columns, bits), "payload cost"),
                                        importance * distortions[bits]) for bits in widths)
            groups.append(RowGroup(name, index, rows, columns, importance, options, lo))
    return tuple(groups)


def _validate(groups: Sequence[RowGroup]) -> None:
    keys, rows_by_matrix, columns_by_matrix = set(), {}, {}
    for group in groups:
        if group.key in keys:
            raise ValueError(f"duplicate group {group.key}")
        keys.add(group.key)
        if not group.rows or len(set(group.rows)) != len(group.rows):
            raise ValueError(f"{group.key}: rows must be nonempty and unique")
        if not math.isfinite(group.importance) or group.importance < 0:
            raise ValueError(f"{group.key}: importance must be finite and nonnegative")
        _nonnegative_int(group.rank_start, "rank_start")
        if any(isinstance(row, bool) or not isinstance(row, (int, np.integer)) or row < 0 for row in group.rows):
            raise ValueError(f"{group.key}: row indices must be nonnegative integers")
        seen = rows_by_matrix.setdefault(group.matrix, set())
        if seen.intersection(group.rows):
            raise ValueError(f"{group.key}: row overlaps another group")
        seen.update(group.rows)
        old_columns = columns_by_matrix.setdefault(group.matrix, group.columns)
        if old_columns != group.columns or _nonnegative_int(group.columns, "columns") == 0:
            raise ValueError(f"{group.key}: inconsistent matrix width")
        if not group.options or len(group.options) > 254:
            raise ValueError(f"{group.key}: requires between 1 and 254 options")
        widths = set()
        for option in group.options:
            if isinstance(option.bits, bool) or option.bits not in (1, 2, 3, 4) or option.bits in widths:
                raise ValueError(f"{group.key}: widths must be distinct and between 1 and 4")
            widths.add(option.bits)
            _nonnegative_int(option.payload_bytes, "option bytes")
            if not math.isfinite(option.loss) or option.loss < 0:
                raise ValueError(f"{group.key}: loss must be finite and nonnegative")


def _nondominated(options: Sequence[WidthOption]) -> tuple[WidthOption, ...]:
    """Drop options which cost no less and have no lower proxy loss."""
    kept, best = [], math.inf
    for option in sorted(options, key=lambda o: (o.payload_bytes, o.loss, o.bits)):
        if option.loss < best:
            kept.append(option)
            best = option.loss
    return tuple(kept)


def _solve(groups: Sequence[RowGroup], budget: int) -> tuple[WidthOption, ...]:
    """Exact multiple-choice knapsack, with integer costs scaled by their GCD."""
    options = tuple(_nondominated(group.options) for group in groups)
    if not options:
        return ()
    minimum = sum(opts[0].payload_bytes for opts in options)
    if minimum > budget:
        raise BudgetError(f"minimum group payload is {minimum} bytes; only {budget} bytes available")
    best = tuple(min(opts, key=lambda o: (o.loss, o.payload_bytes, o.bits)) for opts in options)
    if sum(o.payload_bytes for o in best) <= budget:
        return best
    delta = [[o.payload_bytes - opts[0].payload_bytes for o in opts] for opts in options]
    unit = reduce(math.gcd, (value for group in delta for value in group if value), 0)
    if not unit:
        return best
    room = min((budget - minimum) // unit, sum(max(group) for group in delta) // unit)
    costs = [[value // unit for value in group] for group in delta]
    dp = np.full(room + 1, np.inf, dtype=np.float64)
    dp[0] = 0.0
    # One byte per item and reachable byte bucket. No rounded-up cost bins:
    # a feasible allocation never exceeds the original integer-byte budget.
    pick = np.full((len(options), room + 1), 255, dtype=np.uint8)
    for i, (opts, steps) in enumerate(zip(options, costs)):
        new = np.full(room + 1, np.inf, dtype=np.float64)
        for j, (option, step) in enumerate(zip(opts, steps)):
            if step > room:
                continue
            candidate = dp[:room + 1 - step] + option.loss
            better = candidate < new[step:]
            new[step:][better] = candidate[better]
            pick[i, step:][better] = j
        dp = new
    end = int(np.argmin(dp))  # equal loss: choose the smaller payload
    if not math.isfinite(float(dp[end])):
        raise ValueError("proxy losses overflowed; rescale all importances by one shared factor")
    result = []
    for i in range(len(options) - 1, -1, -1):
        j = int(pick[i, end])
        if j == 255:
            raise RuntimeError("unreachable knapsack backpointer")
        result.append(options[i][j])
        end -= costs[i][j]
    assert end == 0
    return tuple(reversed(result))


def selected_groups(groups: Sequence[RowGroup], widths: Sequence[int]) -> tuple[dict, ...]:
    """Serializer-facing records for explicit widths; also useful for controls."""
    _validate(groups)
    if len(groups) != len(widths):
        raise ValueError("one width is required per group")
    result = []
    for group, bits in zip(groups, widths):
        option = next((o for o in group.options if o.bits == bits), None)
        if option is None:
            raise ValueError(f"{group.key}: width {bits} is not available")
        result.append({"matrix": group.matrix, "group": group.index, "rows": list(group.rows),
                       "columns": group.columns, "bits": option.bits,
                       "payload_bytes": option.payload_bytes, "predicted_loss": option.loss,
                       "importance": group.importance, "rank_start": group.rank_start})
    return tuple(result)


def fixed_split_widths(groups: Sequence[RowGroup], base_bits: Mapping[str, int]) -> tuple[int, ...]:
    """The previous 12.5% up/down split, on make_row_groups()'s partition."""
    _validate(groups)
    total = _row_totals(groups)
    widths = []
    for group in groups:
        n_move = (total[group.matrix] // 8) // 64 * 64
        start, end = group.rank_start, group.rank_start + len(group.rows)
        if start < n_move < end or start < total[group.matrix] - n_move < end:
            raise ValueError("group straddles a fixed-split boundary")
        change = 1 if end <= n_move else -1 if start >= total[group.matrix] - n_move else 0
        widths.append(base_bits[group.matrix] + change)
    return tuple(widths)


def _row_totals(groups: Sequence[RowGroup]) -> dict[str, int]:
    total = {}
    for group in groups:
        total[group.matrix] = total.get(group.matrix, 0) + len(group.rows)
    return total


def load_row_plan(path, shapes: Mapping[str, Sequence[int]]) -> tuple[dict[str, list[dict]], dict]:
    """Validate a serialized allocation against the actual active matrices.

    Matrix keys are the final full parameter names (including '.weight'). A plan
    may change a subset of active matrices, but each listed matrix must cover
    every row exactly once. No unknown names, duplicate rows, truncated groups,
    or unsupported widths are ignored.
    """
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("format") != "rowgroup-plan-1" or not isinstance(data.get("groups"), list) or not data["groups"]:
        raise ValueError("row plan must be a nonempty rowgroup-plan-1 allocation")
    result = {}
    for record in data["groups"]:
        name = record.get("matrix")
        if not isinstance(name, str) or name not in shapes:
            raise ValueError(f"row plan names an unknown or inactive matrix: {name!r}")
        rows, columns = map(int, shapes[name])
        bits = _nonnegative_int(record.get("bits"), "row-plan bits")
        index = _nonnegative_int(record.get("group"), "row-plan group")
        rids = record.get("rows")
        if bits not in (1, 2, 3, 4) or record.get("columns") != columns:
            raise ValueError(f"{name}: unsupported width or wrong column count")
        if not isinstance(rids, list) or not rids or len(rids) % 64:
            raise ValueError(f"{name}: each row group must have a positive multiple of 64 rows")
        if any(isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < rows for row in rids):
            raise ValueError(f"{name}: row index outside the matrix")
        result.setdefault(name, []).append({"group": index, "rows": rids, "bits": bits})
    for name, records in result.items():
        records.sort(key=lambda r: r["group"])
        if len(records) > 8 or [r["group"] for r in records] != list(range(len(records))):
            raise ValueError(f"{name}: use one to eight consecutive group ids starting at zero")
        rows = [row for record in records for row in record["rows"]]
        if sorted(rows) != list(range(int(shapes[name][0]))):
            raise ValueError(f"{name}: row groups must cover every row exactly once")
    manifest = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "matrices": len(result), "groups": len(data["groups"]),
                "allocation": {k: data[k] for k in ("budget_bytes", "total_bytes", "payload_bytes", "size_scope") if k in data},
                "qualification": "Allocation byte fields describe its planned packed artifact; quantizer rate_bits is a separate modeled count."}
    return result, manifest


def allocate(
    groups: Sequence[RowGroup],
    budget_bytes: int,
    *,
    fixed_bytes: int = 0,
    artifact_size: Callable[[tuple[dict, ...]], int] | None = None,
) -> Allocation:
    """Allocate globally, then verify the authoritative complete-artifact size.

    ``fixed_bytes`` covers shared signs and every noncandidate tensor. If
    ``artifact_size`` is supplied, it must return the whole serialized artifact
    size, including those tensors and the manifest. A too-large manifest causes
    a fresh exact solve with its overhead reserved. The reserve never decreases,
    so this is conservative with respect to variable-length header bytes; the
    result is optimal for its final payload budget, not for arbitrary headers.
    A failed solve raises BudgetError and never returns an over-budget plan.
    """
    groups = tuple(groups)
    _validate(groups)
    budget_bytes = _nonnegative_int(budget_bytes, "budget_bytes")
    fixed_bytes = _nonnegative_int(fixed_bytes, "fixed_bytes")
    if fixed_bytes > budget_bytes:
        raise BudgetError("fixed tensors already exceed the complete budget")
    reserve, solves = 0, 0
    while True:
        options = _solve(groups, budget_bytes - fixed_bytes - reserve)
        solves += 1
        selected = selected_groups(groups, [option.bits for option in options])
        payload = sum(option.payload_bytes for option in options)
        total = fixed_bytes + payload
        if artifact_size is not None:
            total = _nonnegative_int(artifact_size(selected), "serialized artifact size")
            if total < fixed_bytes + payload:
                raise ValueError("serializer size is smaller than the priced tensor payload")
        if total <= budget_bytes:
            return Allocation(selected, payload, total, budget_bytes, fixed_bytes, reserve,
                              math.fsum(option.loss for option in options),
                              "serialized_artifact" if artifact_size else "tensor_payload", solves)
        overhead = total - fixed_bytes - payload
        if overhead <= reserve:
            raise RuntimeError("serializer and payload accounting disagree")
        reserve = overhead
        if fixed_bytes + reserve > budget_bytes:
            raise BudgetError("fixed tensors and manifest already exceed the complete budget")
