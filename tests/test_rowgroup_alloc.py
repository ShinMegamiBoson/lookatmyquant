"""Offline allocation correctness: independent exhaustive optima and real byte limits."""

from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rowgroup_alloc import (Allocation, BudgetError, RowGroup, WidthOption,
                           allocate, fixed_split_widths, load_row_plan, make_row_groups, selected_groups)


def group(name, costs, losses, *, rows=(0,), columns=32):
    return RowGroup(name, 0, rows, columns, 1.0,
                    tuple(WidthOption(i + 1, cost, loss) for i, (cost, loss)
                          in enumerate(zip(costs, losses))), 0)


class AllocationTests(unittest.TestCase):
    def test_matches_exhaustive_optimum_for_nonmonotonic_menus(self):
        rng = random.Random(731982)
        for case in range(120):
            groups = tuple(group(f"m{i}",
                                 [rng.randrange(1, 21) * 16 for _ in range(3)],
                                 [rng.randrange(0, 41) / 4 for _ in range(3)])
                           for i in range(rng.randrange(1, 6)))
            budget = rng.randrange(0, 1000)
            feasible = [(sum(o.loss for o in choices), sum(o.payload_bytes for o in choices))
                        for choices in itertools.product(*(g.options for g in groups))
                        if sum(o.payload_bytes for o in choices) <= budget]
            with self.subTest(case=case, budget=budget):
                if not feasible:
                    with self.assertRaises(BudgetError):
                        allocate(groups, budget)
                else:
                    expected = min(feasible)
                    result = allocate(groups, budget)
                    self.assertEqual((result.objective, result.payload_bytes), expected)
                    self.assertLessEqual(result.total_bytes, budget)

    def test_moves_capacity_between_matrices(self):
        groups = (group("sensitive", [10, 20, 30], [100, 25, 6.25]),
                  group("insensitive", [10, 20, 30], [1, .25, .0625]))
        result = allocate(groups, 40)
        self.assertEqual([s["bits"] for s in result.selected], [3, 1])
        self.assertLess(result.objective, 25 + .25)

    def test_integer_cost_boundary_never_rounds_under_budget(self):
        groups = (group("a", [19, 27, 36], [10, 5, 0]),
                  group("b", [11, 21, 29], [9, 4, 0]))
        for budget in range(30, 67):
            result = allocate(groups, budget)
            self.assertLessEqual(result.payload_bytes, budget)
        with self.assertRaises(BudgetError):
            allocate(groups, 29)

    def test_equal_loss_prefers_fewer_bytes_and_stable_width(self):
        groups = (group("a", [10, 10, 20], [2, 2, 2]),)
        first = allocate(groups, 100)
        self.assertEqual(first.payload_bytes, 10)
        self.assertEqual(first.selected[0]["bits"], 1)
        self.assertEqual(first, allocate(groups, 100))

    def test_groups_cover_every_row_with_stable_ties(self):
        matrices = {"b": {"importance": [1.0] * 320, "columns": 257, "base_bits": 3},
                    "a": {"importance": list(range(1024)), "columns": 2560, "base_bits": 2}}
        cost = lambda rows, columns, bits: rows * bits
        groups = make_row_groups(matrices, {1: 1, 2: .25, 3: .0625, 4: .015625}, payload_cost=cost)
        for name in matrices:
            selected = [g for g in groups if g.matrix == name]
            self.assertLessEqual(len(selected), 8)
            self.assertTrue(all(len(g.rows) % 64 == 0 for g in selected))
            rows = [row for g in selected for row in g.rows]
            self.assertEqual(sorted(rows), list(range(len(matrices[name]["importance"]))))
            self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual([r for g in groups if g.matrix == "b" for r in g.rows], list(range(320)))
        self.assertEqual(groups[0].matrix, "a")
        self.assertEqual(groups[0].rows[0], 1023)

    def test_partition_contains_the_exact_old_split(self):
        for rows in (64, 128, 256, 320, 512, 576, 1280, 15360):
            groups = make_row_groups({"m": {"importance": list(range(rows)), "columns": 2560, "base_bits": 2}},
                                     {1: 1, 2: .25, 3: .0625}, payload_cost=lambda m, n, b: m * n * b)
            widths = fixed_split_widths(groups, {"m": 2})
            self.assertLessEqual(len(groups), 8)
            n_move = (rows // 8) // 64 * 64
            self.assertEqual(sum(len(g.rows) for g, bits in zip(groups, widths) if bits == 3), n_move)
            self.assertEqual(sum(len(g.rows) for g, bits in zip(groups, widths) if bits == 1), n_move)
            self.assertEqual(sum(len(g.rows) * bits for g, bits in zip(groups, widths)), 2 * rows)

    def test_serializer_callback_enforces_complete_file_budget(self):
        groups = (group("a", [100, 200, 300], [8, 4, 0]),
                  group("b", [100, 200, 300], [8, 4, 0]))
        fixed = 19

        def serialize(records):
            header = json.dumps({"bits": [r["bits"] for r in records]}, separators=(",", ":")).encode()
            payload = b"\x00" * (fixed + sum(r["payload_bytes"] for r in records))
            return struct.pack("<Q", len(header)) + header + payload

        result = allocate(groups, 419, fixed_bytes=fixed,
                          artifact_size=lambda records: len(serialize(records)))
        self.assertGreater(result.solves, 1)
        self.assertEqual(result.total_bytes, len(serialize(result.selected)))
        self.assertLessEqual(result.total_bytes, 419)
        self.assertEqual(result.size_scope, "serialized_artifact")
        self.assertGreater(result.manifest_reserve_bytes, 0)

    def test_infeasible_complete_artifact_fails_closed(self):
        groups = (group("a", [100, 200], [1, 0]),)
        with self.assertRaises(BudgetError):
            allocate(groups, 105, artifact_size=lambda records: sum(r["payload_bytes"] for r in records) + 20)

    def test_inconsistent_serializer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            allocate((group("a", [100], [0]),), 200, artifact_size=lambda records: 99)

    def test_scope_is_payload_without_serializer(self):
        result = allocate((group("a", [100], [0]),), 200, fixed_bytes=17)
        self.assertEqual(result.total_bytes, 117)
        self.assertEqual(result.size_scope, "tensor_payload")
        self.assertEqual(result.as_dict()["size_scope"], "tensor_payload")

    def test_rejects_overlapping_rows(self):
        a = group("m", [10], [1], rows=(0, 1))
        b = replace(a, index=1, rows=(1, 2), rank_start=2)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            allocate((a, b), 100)

    def test_invalid_options_and_values_are_rejected(self):
        bad = [WidthOption(1, 1.5, 1), WidthOption(1, 1, math.nan), WidthOption(1, -1, 1),
               WidthOption(True, 1, 1), WidthOption(5, 1, 1), WidthOption(1, 1, -1)]
        for option in bad:
            with self.subTest(option=option), self.assertRaises(ValueError):
                allocate((replace(group("m", [10], [1]), options=(option,)),), 100)
        with self.assertRaises(ValueError):
            allocate((group("m", [10], [1]),), 20.5)

    def test_invalid_importance_shape_or_width_is_rejected(self):
        cases = [([1] * 63, 2), ([math.nan] * 64, 2), ([-1] * 64, 2), ([1] * 64, 1)]
        for importance, bits in cases:
            with self.subTest(bits=bits, rows=len(importance)), self.assertRaises(ValueError):
                make_row_groups({"m": {"importance": importance, "columns": 32, "base_bits": bits}},
                                {1: 1, 2: .25, 3: .0625}, payload_cost=lambda m, n, b: m * b)

    def test_explicit_control_selection_rejects_unavailable_width(self):
        groups = (group("m", [10, 20], [1, 0]),)
        with self.assertRaises(ValueError):
            selected_groups(groups, [4])

    def test_empty_allocation_still_prices_fixed_artifact(self):
        self.assertEqual(allocate((), 20, fixed_bytes=12).total_bytes, 12)
        with self.assertRaises(BudgetError):
            allocate((), 20, fixed_bytes=21)

    def test_real_packer_matches_the_allocators_complete_byte_budget(self):
        from compiler_artifact import (artifact_serialized_bytes, encode_entry,
                                       grouped_matrix_specs, write_artifact)
        matrices = {"sensitive": {"importance": list(range(256)), "columns": 35, "base_bits": 2},
                    "insensitive": {"importance": [.01] * 256, "columns": 35, "base_bits": 2}}
        groups = make_row_groups(matrices, {1: 1, 2: .25, 3: .0625})

        def layout(selected):
            specs, metadata = {}, {}
            for name, matrix in sorted(matrices.items()):
                descriptors = [{"rows": len(r["rows"]), "bits": r["bits"]}
                               for r in selected if r["matrix"] == name]
                s, m = grouped_matrix_specs(name, [len(matrix["importance"]), matrix["columns"]], descriptors)
                specs.update(s)
                metadata[name] = m
            return specs, {"matrices": metadata}

        def size(selected):
            return artifact_serialized_bytes(*layout(selected))

        control = selected_groups(groups, fixed_split_widths(groups, {name: 2 for name in matrices}))
        budget = size(control)
        result = allocate(groups, budget, fixed_bytes=70, artifact_size=size)
        tensors, metadata = {}, {}
        rng = np.random.default_rng(19831)
        for name, matrix in sorted(matrices.items()):
            selected = [r for r in result.selected if r["matrix"] == name]
            rows, columns = len(matrix["importance"]), matrix["columns"]
            states = np.empty((rows, columns), dtype=np.int16)
            tables, row_groups = [], []
            for table_index, record in enumerate(selected):
                r, bits = record["rows"], record["bits"]
                state = np.zeros(len(r), dtype=np.int64)
                for column in range(columns):
                    state = ((state << bits) | rng.integers(0, 1 << bits, len(r))) & 4095
                    states[r, column] = state + 4096 * table_index
                tables.append(rng.normal(size=4096).astype(np.float32))
                row_groups.append({"rows": r, "bits": bits, "table_index": table_index})
            entry = {"states": states, "scales": rng.uniform(.5, 2, rows).astype(np.float32),
                     "table": np.concatenate(tables), "in_seed": 0, "row_groups": row_groups}
            t, m = encode_entry(name, entry, signs=np.ones(columns, dtype=np.int8))
            tensors.update(t)
            metadata[name] = m
        self.assertEqual({"matrices": metadata}, layout(result.selected)[1])
        with tempfile.TemporaryDirectory() as temp:
            receipt = write_artifact(Path(temp) / "artifact", tensors, {"matrices": metadata})
        self.assertEqual(receipt["total_bytes"], result.total_bytes)
        self.assertLessEqual(receipt["total_bytes"], budget)

    def test_row_plan_load_validates_shape_and_preserves_original_row_ids(self):
        name = "model.layers.0.mlp.up_proj.weight"
        groups = make_row_groups({name: {"importance": list(range(128)), "columns": 32, "base_bits": 2}},
                                 {1: 1, 2: .25, 3: .0625})
        allocation = allocate(groups, 100_000)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps(allocation.as_dict()))
            loaded, manifest = load_row_plan(path, {name: [128, 32]})
        self.assertEqual(loaded[name][0]["rows"], list(reversed(range(64, 128))))
        self.assertEqual(manifest["groups"], 2)
        self.assertEqual(len(manifest["sha256"]), 64)

    def test_row_plan_rejects_silent_omissions_and_invalid_groups(self):
        name = "m.weight"
        good = {"format": "rowgroup-plan-1", "groups": [
            {"matrix": name, "group": 0, "rows": list(range(64)), "columns": 32, "bits": 1},
            {"matrix": name, "group": 1, "rows": list(range(64, 128)), "columns": 32, "bits": 3}]}
        edits = [lambda d: d["groups"][0].update(matrix="unknown.weight"),
                 lambda d: d["groups"][0].update(rows=[0] * 64),
                 lambda d: d["groups"][0].update(rows=list(range(63))),
                 lambda d: d["groups"][0].update(rows=list(range(129, 193))),
                 lambda d: d["groups"][0].update(bits=5),
                 lambda d: d["groups"][0].update(columns=31),
                 lambda d: d["groups"][1].update(group=0),
                 lambda d: d.update(groups=[]),
                 lambda d: d.update(format="unknown")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            for i, edit in enumerate(edits):
                bad = json.loads(json.dumps(good))
                edit(bad)
                path.write_text(json.dumps(bad))
                with self.subTest(case=i), self.assertRaises(ValueError):
                    load_row_plan(path, {name: [128, 32]})


if __name__ == "__main__":
    unittest.main()
