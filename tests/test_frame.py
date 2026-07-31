"""Backs tests/toybook.py's own docstring claim: `make_frame(**toy_arrays())` equals
`build_toy_frame()` column for column -- names, dtypes AND order -- so the inline builder
(used by every family test for a scenario the 8-event fixture doesn't contain) can never
silently drift from the frozen adapter's real output schema (docs/DATA_CONTRACT.md).

Also covers `edge.features.frame`'s two validation errors (EventFrame._validate,
event_frame_from_table's missing-column checks) and its pyarrow-Table compatibility shim
(`.column()`, `.schema.names`, `.num_rows`), which nothing else in the suite exercises
directly.
"""
import unittest

import numpy as np
import pyarrow as pa

from edge.features.frame import EventFrame, event_frame_from_table
from tests.toybook import (
    TOY_ASK_PX,
    TOY_ASK_SZ,
    TOY_BID_PX,
    TOY_BID_SZ,
    TOY_DIRECTION,
    TOY_EVENT_TYPE,
    TOY_PRICE,
    TOY_SIZE,
    TOY_TS_NS,
    build_toy_frame,
    make_event_frame,
    make_frame,
    toy_arrays,
    toy_event_frame,
)


class TestMakeFrameMatchesBuildToyFrame(unittest.TestCase):
    """tests/toybook.py's docstring: "tests/test_frame.py asserts `make_frame(**toy_arrays())`
    equals `build_toy_frame()` column for column"."""

    def setUp(self):
        self.adapter_table = build_toy_frame()
        self.inline_table = make_frame(**toy_arrays())

    def test_pyarrow_tables_are_exactly_equal(self):
        # pa.Table.equals compares schema (names, types, order) AND data in one shot; this
        # is the single strongest check available, done first so a failure here is
        # unambiguous before the more granular checks below pin down WHAT diverged.
        self.assertTrue(self.adapter_table.equals(self.inline_table))

    def test_column_order_is_identical(self):
        self.assertEqual(self.adapter_table.schema.names, self.inline_table.schema.names)
        # And pinned to the frozen schema order itself (docs/DATA_CONTRACT.md): scalar
        # columns first, then bid_px, bid_sz, ask_px, ask_sz, each block-grouped by level.
        expected = [
            "ts_ns",
            "event_type",
            "direction",
            "price",
            "size",
            "bid_px_1",
            "bid_px_2",
            "bid_sz_1",
            "bid_sz_2",
            "ask_px_1",
            "ask_px_2",
            "ask_sz_1",
            "ask_sz_2",
        ]
        self.assertEqual(self.adapter_table.schema.names, expected)
        self.assertEqual(self.inline_table.schema.names, expected)

    def test_column_dtypes_are_identical(self):
        for name in self.adapter_table.schema.names:
            adapter_type = self.adapter_table.schema.field(name).type
            inline_type = self.inline_table.schema.field(name).type
            self.assertEqual(adapter_type, inline_type, name)
        # event_type / direction are int8 (docs/DATA_CONTRACT.md); everything else int64.
        self.assertEqual(self.adapter_table.schema.field("event_type").type, pa.int8())
        self.assertEqual(self.adapter_table.schema.field("direction").type, pa.int8())
        self.assertEqual(self.adapter_table.schema.field("ts_ns").type, pa.int64())
        self.assertEqual(self.adapter_table.schema.field("price").type, pa.int64())
        self.assertEqual(self.adapter_table.schema.field("size").type, pa.int64())
        self.assertEqual(self.adapter_table.schema.field("bid_px_1").type, pa.int64())

    def test_column_values_are_identical(self):
        for name in self.adapter_table.schema.names:
            self.assertEqual(
                self.adapter_table.column(name).to_pylist(),
                self.inline_table.column(name).to_pylist(),
                name,
            )

    def test_row_count_matches_the_eight_event_fixture(self):
        self.assertEqual(self.adapter_table.num_rows, 8)
        self.assertEqual(self.inline_table.num_rows, 8)

    def test_toy_arrays_values_match_the_module_level_literals(self):
        # toy_arrays() is a mutable COPY -- pin that the untouched copy still matches the
        # literal reference columns tests/toybook.py documents, so a future edit to
        # toy_arrays() that silently diverges from TOY_* is caught here, not three family
        # test files downstream.
        d = toy_arrays()
        self.assertEqual(d["ts_ns"], TOY_TS_NS)
        self.assertEqual(d["event_type"], TOY_EVENT_TYPE)
        self.assertEqual(d["direction"], TOY_DIRECTION)
        self.assertEqual(d["price"], TOY_PRICE)
        self.assertEqual(d["size"], TOY_SIZE)
        self.assertEqual(d["bid_px"], TOY_BID_PX)
        self.assertEqual(d["bid_sz"], TOY_BID_SZ)
        self.assertEqual(d["ask_px"], TOY_ASK_PX)
        self.assertEqual(d["ask_sz"], TOY_ASK_SZ)

    def test_toy_arrays_is_a_mutable_copy_not_a_shared_reference(self):
        # Tweaking the returned dict (the documented use case: "d = toy_arrays();
        # d['event_type'][4] = 1") must not corrupt the module-level TOY_* literals for
        # the next test that calls toy_arrays() again.
        d = toy_arrays()
        d["event_type"][4] = 1
        d["bid_px"][0][1] = 999999
        self.assertEqual(TOY_EVENT_TYPE[4], 4)
        self.assertEqual(TOY_BID_PX[0][1], 100000)
        d2 = toy_arrays()
        self.assertEqual(d2["event_type"][4], 4)
        self.assertEqual(d2["bid_px"][0][1], 100000)

    def test_make_event_frame_and_toy_event_frame_agree_on_the_same_data(self):
        # The EventFrame-typed doors into the same two rows: make_event_frame(**toy_arrays())
        # must equal toy_event_frame() column for column too, not just the raw pa.Table.
        via_inline = make_event_frame(**toy_arrays())
        via_adapter = toy_event_frame()
        np.testing.assert_array_equal(via_inline.ts_ns, via_adapter.ts_ns)
        np.testing.assert_array_equal(via_inline.event_type, via_adapter.event_type)
        np.testing.assert_array_equal(via_inline.direction, via_adapter.direction)
        np.testing.assert_array_equal(via_inline.price, via_adapter.price)
        np.testing.assert_array_equal(via_inline.size, via_adapter.size)
        np.testing.assert_array_equal(via_inline.bid_px, via_adapter.bid_px)
        np.testing.assert_array_equal(via_inline.bid_sz, via_adapter.bid_sz)
        np.testing.assert_array_equal(via_inline.ask_px, via_adapter.ask_px)
        np.testing.assert_array_equal(via_inline.ask_sz, via_adapter.ask_sz)


class TestEventFrameValidationErrors(unittest.TestCase):
    """EventFrame._validate's two hard failures (module docstring: "the two errors that
    would silently corrupt every window feature")."""

    def _kwargs(self, **overrides):
        base = dict(
            ts_ns=[1, 2, 3],
            event_type=[1, 1, 1],
            direction=[1, 1, 1],
            price=[100, 100, 100],
            size=[10, 10, 10],
            bid_px=[[100], [100], [100]],
            bid_sz=[[10], [10], [10]],
            ask_px=[[200], [200], [200]],
            ask_sz=[[10], [10], [10]],
        )
        base.update(overrides)
        return base

    def test_mismatched_scalar_column_shape_raises(self):
        with self.assertRaises(ValueError) as ctx:
            EventFrame(**self._kwargs(event_type=[1, 1]))
        self.assertIn("event_type has shape", str(ctx.exception))

    def test_mismatched_level_column_shape_raises(self):
        with self.assertRaises(ValueError) as ctx:
            EventFrame(**self._kwargs(bid_sz=[[10], [10]]))
        self.assertIn("bid_sz has shape", str(ctx.exception))

    def test_bid_px_not_two_dimensional_raises(self):
        with self.assertRaises(ValueError) as ctx:
            EventFrame(**self._kwargs(bid_px=[100, 100, 100]))
        self.assertIn("bid_px has shape", str(ctx.exception))

    def test_non_decreasing_ts_ns_violation_raises(self):
        with self.assertRaises(ValueError) as ctx:
            EventFrame(**self._kwargs(ts_ns=[1, 3, 2]))
        self.assertIn("non-decreasing", str(ctx.exception))

    def test_single_row_frame_is_trivially_sorted(self):
        # n <= 1 skips the diff check entirely (nothing to compare against) -- must not
        # raise on the degenerate one-row case.
        frame = EventFrame(
            ts_ns=[5],
            event_type=[1],
            direction=[1],
            price=[100],
            size=[10],
            bid_px=[[100]],
            bid_sz=[[10]],
            ask_px=[[200]],
            ask_sz=[[10]],
        )
        self.assertEqual(frame.n_events, 1)

    def test_repeated_ts_ns_ties_are_allowed(self):
        # Non-decreasing, not strictly increasing -- exact ns ties are legal (the toy
        # fixture itself has a tie: events 4 and 5 differ by 1ns, but a genuine tie is
        # still >= 0 under np.diff).
        frame = EventFrame(**self._kwargs(ts_ns=[1, 1, 2]))
        self.assertEqual(frame.ts_ns.tolist(), [1, 1, 2])


class TestEventFrameFromTableErrors(unittest.TestCase):
    def test_missing_scalar_column_raises_with_the_missing_names(self):
        t = pa.table(
            {"ts_ns": pa.array([1, 2], pa.int64()), "event_type": pa.array([1, 1], pa.int8())}
        )
        with self.assertRaises(ValueError) as ctx:
            event_frame_from_table(t)
        msg = str(ctx.exception)
        self.assertIn("direction", msg)
        self.assertIn("price", msg)
        self.assertIn("size", msg)

    def test_no_bid_px_1_column_raises(self):
        t = pa.table(
            {
                "ts_ns": pa.array([1, 2], pa.int64()),
                "event_type": pa.array([1, 1], pa.int8()),
                "direction": pa.array([1, 1], pa.int8()),
                "price": pa.array([100, 100], pa.int64()),
                "size": pa.array([10, 10], pa.int64()),
            }
        )
        with self.assertRaises(ValueError) as ctx:
            event_frame_from_table(t)
        self.assertIn("bid_px_1", str(ctx.exception))

    def test_num_levels_is_inferred_when_omitted(self):
        frame = event_frame_from_table(build_toy_frame())
        self.assertEqual(frame.num_levels, 2)

    def test_num_levels_can_be_capped_explicitly(self):
        frame = event_frame_from_table(build_toy_frame(), num_levels=1)
        self.assertEqual(frame.num_levels, 1)
        self.assertEqual(frame.bid_px.shape[1], 1)


class TestPyarrowCompatibilityShim(unittest.TestCase):
    """EventFrame answers `.column()`, `.schema.names` and `.num_rows` like a pyarrow
    Table, so a table-shaped helper (clocks.column(), for one) accepts either type."""

    def setUp(self):
        self.frame = toy_event_frame()
        self.table = build_toy_frame()

    def test_num_rows_matches_pyarrow(self):
        self.assertEqual(self.frame.num_rows, self.table.num_rows)
        self.assertEqual(self.frame.num_rows, 8)

    def test_schema_names_matches_pyarrow_order(self):
        self.assertEqual(self.frame.schema.names, self.table.schema.names)

    def test_scalar_column_matches_pyarrow(self):
        for name in ("ts_ns", "event_type", "direction", "price", "size"):
            got = self.frame.column(name).to_numpy(zero_copy_only=False)
            want = self.table.column(name).to_numpy(zero_copy_only=False)
            np.testing.assert_array_equal(got, want, err_msg=name)

    def test_level_column_matches_pyarrow(self):
        for name in ("bid_px_1", "bid_px_2", "bid_sz_1", "ask_px_2", "ask_sz_1"):
            got = self.frame.column(name).to_numpy(zero_copy_only=False)
            want = self.table.column(name).to_numpy(zero_copy_only=False)
            np.testing.assert_array_equal(got, want, err_msg=name)

    def test_column_to_pylist_matches_pyarrow(self):
        self.assertEqual(
            self.frame.column("event_type").to_pylist(),
            self.table.column("event_type").to_pylist(),
        )

    def test_unknown_column_name_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.frame.column("not_a_real_column")

    def test_level_beyond_num_levels_raises_keyerror(self):
        # The toy fixture is L=2; bid_px_3 is a well-formed NAME but not a column this
        # 2-level frame carries.
        with self.assertRaises(KeyError):
            self.frame.column("bid_px_3")

    def test_column_view_len_matches_row_count(self):
        self.assertEqual(len(self.frame.column("ts_ns")), self.frame.num_rows)


if __name__ == "__main__":
    unittest.main()
