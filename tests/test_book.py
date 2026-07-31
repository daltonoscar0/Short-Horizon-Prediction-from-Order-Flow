"""Hand-computed tests for the shared book spine (src/edge/features/book.py).

Two toy books, both under 8 events, both literal:

`_toy_frame()` re-states the committed 8-event/2-level fixture
(tests/fixtures/toy_message.csv + toy_orderbook.csv) as an in-module pyarrow table in the
frozen schema, so this file has no dependency on any other test module. Orderbook rows,
0-indexed, `(bid_px_1, bid_sz_1 | ask_px_1, ask_sz_1)`:

  0: (100500, 10 | 101000, 50)   4: (100000, 40 | 101000, 40)
  1: (100500, 30 | 101000, 50)   5: (100000, 40 | 101000, 40)   type-5, book unchanged
  2: (100500, 25 | 101000, 50)   6: (100000, 40 | 101000, 40)
  3: (100000, 40 | 101000, 50)   7: (100000, 40 | 101000, 40)   L2 ask 102000->103000

`_sentinel_frame()` is a 3-event book carrying the LOBSTER no-displayed-level sentinel and
a zero displayed size, neither of which occurs in the 2012-06-21 sample.
"""
import unittest

import numpy as np
import pyarrow as pa

from edge.features.book import (
    book_at_events,
    book_at_stamps,
    event_frame_arrays,
    frame_num_levels,
    ticks,
)
from edge.features.clocks import Stamps, calendar_stamps
from edge.io.lobster import ASK_SENTINEL_PRICE, BID_SENTINEL_PRICE

NUM_LEVELS = 2
GRID_150MS = 150_000_000
SESSION_START = 34_200_000_000_000

# Verbatim from tests/fixtures/toy_message.csv.
TOY_TS_NS = [
    34200000000001,
    34200000000002,
    34200100000000,
    34200100000500,
    34200200000000,
    34200200000001,
    34200300000000,
    34200400000000,
]
TOY_EVENT_TYPE = [1, 1, 2, 3, 4, 5, 1, 3]
TOY_DIRECTION = [1, 1, 1, 1, -1, -1, -1, -1]
TOY_PRICE = [100500, 100500, 100500, 100500, 101000, 101000, 103000, 102000]
TOY_SIZE = [10, 20, 5, 25, 10, 5, 15, 30]

# Verbatim from tests/fixtures/toy_orderbook.csv, de-interleaved into the frozen schema's
# block-grouped order (bid px, bid sz, ask px, ask sz), levels 1..2.
TOY_BID_PX = [[100500, 100000]] * 3 + [[100000, 98000]] * 5
TOY_BID_SZ = [[10, 40], [30, 40], [25, 40]] + [[40, 15]] * 5
TOY_ASK_PX = [[101000, 102000]] * 7 + [[101000, 103000]]
TOY_ASK_SZ = [[50, 30]] * 4 + [[40, 30]] * 3 + [[40, 15]]


def _frame(ts_ns, event_type, direction, price, size, bid_px, bid_sz, ask_px, ask_sz):
    """Event frame in the frozen schema (docs/DATA_CONTRACT.md) from literal rows."""
    columns = {
        "ts_ns": pa.array(ts_ns, pa.int64()),
        "event_type": pa.array(event_type, pa.int8()),
        "direction": pa.array(direction, pa.int8()),
        "price": pa.array(price, pa.int64()),
        "size": pa.array(size, pa.int64()),
    }
    for name, rows in (
        ("bid_px", bid_px),
        ("bid_sz", bid_sz),
        ("ask_px", ask_px),
        ("ask_sz", ask_sz),
    ):
        for lvl in range(NUM_LEVELS):
            columns[f"{name}_{lvl + 1}"] = pa.array([r[lvl] for r in rows], pa.int64())
    return pa.table(columns)


def _toy_frame():
    return _frame(
        TOY_TS_NS,
        TOY_EVENT_TYPE,
        TOY_DIRECTION,
        TOY_PRICE,
        TOY_SIZE,
        TOY_BID_PX,
        TOY_BID_SZ,
        TOY_ASK_PX,
        TOY_ASK_SZ,
    )


def _sentinel_frame():
    """3 events: row 0 normal (L2 ask absent), row 1 sentinel ask, row 2 zero-size bid."""
    return _frame(
        ts_ns=[34200000000001, 34200000000002, 34200000000003],
        event_type=[1, 3, 3],
        direction=[1, -1, 1],
        price=[100000, 101000, 100000],
        size=[40, 50, 40],
        bid_px=[[100000, 99900], [100000, 99900], [100000, 99900]],
        bid_sz=[[40, 10], [40, 10], [0, 10]],
        ask_px=[[101000, ASK_SENTINEL_PRICE], [ASK_SENTINEL_PRICE, ASK_SENTINEL_PRICE], [101000, 101100]],
        ask_sz=[[50, 0], [0, 0], [20, 20]],
    )


class TestTicks(unittest.TestCase):
    def test_raw_units_to_ticks(self):
        # 1 tick = $0.01 = 100 raw units, so a 500-unit spread is 5 ticks.
        self.assertEqual(ticks(500), 5.0)
        np.testing.assert_array_equal(ticks(np.array([100, 250])), np.array([1.0, 2.5]))


class TestEventFrameArrays(unittest.TestCase):
    def test_hand_computed_columns_as_int64(self):
        arrays = event_frame_arrays(_toy_frame())
        self.assertEqual(arrays.ts_ns.tolist(), TOY_TS_NS)
        self.assertEqual(arrays.event_type.tolist(), TOY_EVENT_TYPE)
        self.assertEqual(arrays.direction.tolist(), TOY_DIRECTION)
        self.assertEqual(arrays.price.tolist(), TOY_PRICE)
        self.assertEqual(arrays.size.tolist(), TOY_SIZE)
        # int64, not the frame's int8: under NEP 50 `direction * 100` on an int8 array
        # wraps silently, which would corrupt any signed-volume aggregate.
        for a in arrays:
            self.assertEqual(a.dtype, np.int64)

    def test_returned_arrays_do_not_alias_the_frame(self):
        frame = _toy_frame()
        arrays = event_frame_arrays(frame)
        arrays.price[0] = -1
        self.assertEqual(frame.column("price").to_pylist()[0], 100500)


class TestFrameNumLevels(unittest.TestCase):
    def test_detects_two_levels_in_the_toy_schema(self):
        self.assertEqual(frame_num_levels(_toy_frame()), 2)

    def test_requesting_more_levels_than_the_frame_has_raises(self):
        with self.assertRaises(ValueError):
            book_at_events(_toy_frame(), num_levels=3)


class TestBestLevelAtEvents(unittest.TestCase):
    def setUp(self):
        self.book = book_at_events(_toy_frame())

    def test_hand_computed_best_quotes(self):
        self.assertEqual(self.book.bid_px.tolist(), [100500] * 3 + [100000] * 5)
        self.assertEqual(self.book.bid_sz.tolist(), [10, 30, 25, 40, 40, 40, 40, 40])
        self.assertEqual(self.book.ask_px.tolist(), [101000] * 8)
        self.assertEqual(self.book.ask_sz.tolist(), [50, 50, 50, 50, 40, 40, 40, 40])

    def test_hand_computed_mid_and_spread(self):
        # rows 0-2: (100500 + 101000)/2 = 100750, spread 500 = 5 ticks.
        # rows 3-7: (100000 + 101000)/2 = 100500, spread 1000 = 10 ticks.
        self.assertEqual(self.book.mid.tolist(), [100750.0] * 3 + [100500.0] * 5)
        self.assertEqual(self.book.spread.tolist(), [500.0] * 3 + [1000.0] * 5)
        self.assertEqual(self.book.spread_ticks.tolist(), [5.0] * 3 + [10.0] * 5)

    def test_hand_computed_level_matrices(self):
        self.assertEqual(self.book.bid_px_l.shape, (8, 2))
        # Row 3: the bid touch is deleted, so L1 100500->100000 and L2 100000->98000.
        self.assertEqual(self.book.bid_px_l[3].tolist(), [100000, 98000])
        self.assertEqual(self.book.bid_sz_l[3].tolist(), [40, 15])
        # Row 7: the 102000 ask level is deleted, so L2 102000/30 -> 103000/15.
        self.assertEqual(self.book.ask_px_l[7].tolist(), [101000, 103000])
        self.assertEqual(self.book.ask_sz_l[7].tolist(), [40, 15])
        self.assertTrue(self.book.valid_l.all())


class TestWeightedMidCrossedWeights(unittest.TestCase):
    """FEATURES.md sec. 3: wmid = (P^b*q^a + P^a*q^b)/(q^a + q^b). The weights CROSS."""

    def setUp(self):
        self.book = book_at_events(_toy_frame())

    def test_hand_computed_wmid_row_0_skews_toward_the_bid(self):
        # Row 0: bid 100500 x 10, ask 101000 x 50. mid = 100750.
        # wmid = (100500*50 + 101000*10) / 60 = 6035000/60 = 100583.3333...
        # The ask queue is 5x the bid queue, so the resting sell pressure drags the
        # reference price DOWN, toward the bid. wmid < mid is the whole point.
        self.assertAlmostEqual(self.book.wmid[0], 6035000.0 / 60.0, places=9)
        self.assertLess(self.book.wmid[0], self.book.mid[0])

    def test_uncrossed_weights_would_give_the_opposite_sign(self):
        # The classic error: (P^b*q^b + P^a*q^a)/(q^a+q^b) = (100500*10 + 101000*50)/60
        # = 100916.67 -- ABOVE mid, i.e. a heavy ask read as bullish. Pinned so an
        # "innocuous" refactor that uncrosses the weights fails loudly here.
        uncrossed = (100500.0 * 10 + 101000.0 * 50) / 60.0
        self.assertGreater(uncrossed, self.book.mid[0])
        self.assertNotAlmostEqual(self.book.wmid[0], uncrossed, places=3)

    def test_hand_computed_wmid_row_1_is_exact(self):
        # Row 1: bid 100500 x 30, ask 101000 x 50.
        # wmid = (100500*50 + 101000*30)/80 = 8055000/80 = 100687.5 exactly.
        self.assertEqual(self.book.wmid[1], 100687.5)

    def test_hand_computed_wmid_row_3_skews_toward_the_bid(self):
        # Row 3: bid 100000 x 40, ask 101000 x 50, mid 100500.
        # wmid = (100000*50 + 101000*40)/90 = 9040000/90 = 100444.444...
        self.assertAlmostEqual(self.book.wmid[3], 9040000.0 / 90.0, places=9)
        self.assertLess(self.book.wmid[3], self.book.mid[3])

    def test_equal_queues_give_wmid_equal_to_mid(self):
        # Row 4: bid 100000 x 40, ask 101000 x 40 -- symmetric queues, no skew.
        self.assertEqual(self.book.wmid[4], self.book.mid[4])
        self.assertEqual(self.book.wmid[4], 100500.0)


class TestValidMask(unittest.TestCase):
    def setUp(self):
        self.book = book_at_events(_sentinel_frame())

    def test_sentinel_and_zero_size_rows_are_invalid(self):
        # Row 0 is a real two-sided touch; row 1's ask is the LOBSTER sentinel; row 2's
        # bid carries a real price but zero displayed size.
        self.assertEqual(self.book.valid.tolist(), [True, False, False])

    def test_derived_prices_are_nan_not_sentinel_arithmetic(self):
        # The failure this pins: (100000 + 9999999999)/2 is a perfectly finite number
        # that would silently become a mid of ~$500,000 and poison every downstream mean.
        self.assertTrue(np.isnan(self.book.mid[1]))
        self.assertTrue(np.isnan(self.book.wmid[1]))
        self.assertTrue(np.isnan(self.book.spread[1]))
        self.assertTrue(np.isnan(self.book.spread_ticks[1]))
        # Row 2: zero bid size would be a 0/0 in wmid, not an exception.
        self.assertTrue(np.isnan(self.book.wmid[2]))

    def test_per_level_validity_is_independent_of_the_touch(self):
        # Row 0's touch is fine but its L2 ask is absent (sentinel price, size 0).
        self.assertEqual(self.book.valid_l[0].tolist(), [True, False])
        self.assertEqual(self.book.valid_l[2].tolist(), [False, True])

    def test_raw_integer_columns_are_passed_through_unmasked(self):
        # `valid` is the contract for "is this usable"; the raw int columns stay faithful
        # to the frame so a family can see exactly what LOBSTER printed.
        self.assertEqual(self.book.ask_px[1], ASK_SENTINEL_PRICE)
        self.assertEqual(self.book.bid_px_l[1].tolist(), [100000, 99900])
        self.assertNotEqual(BID_SENTINEL_PRICE, 0)  # guards the constant's sign


class TestBookAtStamps(unittest.TestCase):
    def setUp(self):
        self.frame = _toy_frame()

    def test_stamp_between_two_events_resolves_to_the_earlier_one(self):
        # 150ms grid over [34200.000s, 34200.450s]: grid points .000 (dropped, before the
        # first event), .150, .300, .450 -> last event at or before each is index 3, 6, 7.
        stamps = calendar_stamps(self.frame, SESSION_START, SESSION_START + 450_000_000, GRID_150MS)
        self.assertEqual(stamps.index.tolist(), [3, 6, 7])

        book = book_at_stamps(self.frame, stamps)
        # The .150s stamp sits strictly between event 3 (34200.1000005s) and event 4
        # (34200.2s). Under `<= t` it must report event 3's post-state:
        #   bid_px 100000 (event 2's state was 100500 -- a `< t`/off-by-one backwards)
        #   ask_sz 50     (event 4's state is 40     -- a lookahead by one row)
        self.assertEqual(book.bid_px[0], 100000)
        self.assertEqual(book.ask_sz[0], 50)
        self.assertEqual(book.mid[0], 100500.0)
        self.assertAlmostEqual(book.wmid[0], 9040000.0 / 90.0, places=9)

    def test_stamp_exactly_on_an_event_includes_that_event(self):
        # The `<=` half of the convention: a stamp whose ts equals an event's ts sees it.
        stamps = Stamps(index=np.array([2]), ts_ns=np.array([TOY_TS_NS[2]]))
        book = book_at_stamps(self.frame, stamps)
        self.assertEqual(book.bid_sz[0], 25)  # event 2 is the partial cancel 30 -> 25

    def test_matches_book_at_events_gathered_at_the_same_rows(self):
        stamps = Stamps(index=np.array([0, 3, 7]), ts_ns=np.array(TOY_TS_NS)[[0, 3, 7]])
        at_stamps = book_at_stamps(self.frame, stamps)
        at_events = book_at_events(self.frame)
        for field in ("bid_px", "ask_sz", "mid", "wmid", "spread_ticks"):
            np.testing.assert_array_equal(
                getattr(at_stamps, field), getattr(at_events, field)[[0, 3, 7]]
            )

    def test_empty_stamps_give_empty_arrays_not_an_exception(self):
        book = book_at_stamps(self.frame, Stamps(index=np.array([], dtype=np.int64), ts_ns=np.array([])))
        self.assertEqual(book.mid.shape, (0,))
        self.assertEqual(book.bid_px_l.shape, (0, 2))


class TestNoLookahead(unittest.TestCase):
    """The violation detector: future events must not move a single value at an earlier stamp."""

    def test_appending_future_events_leaves_earlier_stamps_bit_identical(self):
        full = _toy_frame()
        past = full.slice(0, 4)  # events 0-3 only; events 4-7 are "the future"
        end_ns = SESSION_START + GRID_150MS

        stamps_full = calendar_stamps(full, SESSION_START, end_ns, GRID_150MS)
        stamps_past = calendar_stamps(past, SESSION_START, end_ns, GRID_150MS)
        self.assertEqual(stamps_full.index.tolist(), stamps_past.index.tolist(), "same stamp")

        book_full = book_at_stamps(full, stamps_full)
        book_past = book_at_stamps(past, stamps_past)
        for field in ("bid_px", "bid_sz", "ask_px", "ask_sz", "mid", "wmid", "spread", "spread_ticks"):
            np.testing.assert_array_equal(
                getattr(book_full, field),
                getattr(book_past, field),
                err_msg=f"{field} at a past stamp changed when future events were appended",
            )
        np.testing.assert_array_equal(book_full.valid, book_past.valid)
        np.testing.assert_array_equal(book_full.bid_px_l, book_past.bid_px_l)


if __name__ == "__main__":
    unittest.main()
