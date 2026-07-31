"""Hand-computed tests for labels.py (src/edge/features/labels.py) against tests/toybook.py.

labels.py is the one family allowed to look forward, so most tests here build a VARIANT
frame via tests.toybook's make_event_frame/toy_arrays -- the shared 8-event fixture spans
only 400ms (tests/toybook.py), so every 1s+ calendar horizon on it is NaN by construction
(itself one of the required assertions, see TestBaseFixtureCalendarHorizonsAreAllNaN).
Where a test needs a horizon to actually resolve inside the frame, the fixture's book
columns are kept bit-for-bit (same bid/ask px/sz sequence, hence the same known mid/wmid
reference values documented in tests/toybook.py and pinned in tests/test_book.py) and only
`ts_ns` is stretched.

Variant A (used by several classes below), 8 rows, BASE = 34_200_000_000_000:
  ts_ns = [B+1, B+2, B+3, B+500_000_000, B+600_000_000, B+600_000_001, B+700_000_000,
           B+2_000_000_000]
  book columns unchanged from tests/toybook.py, so
    mid  = [100750]*3 + [100500]*5      (row-index identical to the base fixture)
    wmid = [100583.333.., 100687.5, 100666.666.., 100444.444.., 100500, 100500, 100500, 100500]
"""
import unittest

import numpy as np

from edge.features import SESSION_END_NS
from edge.features.book import book_at_stamps
from edge.features.clocks import Stamps, stamps_for_clock
from edge.features.frame import event_frame_from_table
from edge.features.labels import ALPHA_TICKS, classify, compute
from edge.io.lobster import ASK_SENTINEL_PRICE

from tests.toybook import TOY_TS_NS, make_event_frame, make_frame, toy_arrays, toy_event_frame

BASE = 34_200_000_000_000

EXPECTED_COLUMNS = frozenset(
    [f"label_y_mid_{h}" for h in ("1s", "2s", "5s", "10s")]
    + [f"label_y_wmid_{h}" for h in ("1s", "2s", "5s", "10s")]
    + [f"label_c_mid_{h}" for h in ("1s", "2s", "5s", "10s")]
    + [f"label_c_wmid_{h}" for h in ("1s", "2s", "5s", "10s")]
    + [f"label_y_mid_e{n}" for n in (1, 2, 5, 10)]
    + [f"label_y_wmid_e{n}" for n in (1, 2, 5, 10)]
    + [f"label_c_mid_e{n}" for n in (1, 2, 5, 10)]
)


def _variant_a_ts():
    """8 timestamps stretching the fixture across 2s while keeping its book columns."""
    return [
        BASE + 1,
        BASE + 2,
        BASE + 3,
        BASE + 500_000_000,
        BASE + 600_000_000,
        BASE + 600_000_001,
        BASE + 700_000_000,
        BASE + 2_000_000_000,
    ]


def _variant_a_frame():
    d = toy_arrays()
    d["ts_ns"] = _variant_a_ts()
    return make_event_frame(**d)


class TestClassifyDeadband(unittest.TestCase):
    def test_boundary_exactly_plus_and_minus_half_tick_is_flat(self):
        # ALPHA_TICKS = 0.5. Spec: "up if y_h > alpha, down if < -alpha, else flat" --
        # so |y| == alpha exactly must be FLAT (0.0), not up/down. This is the boundary a
        # careless `>=` would get wrong.
        self.assertEqual(ALPHA_TICKS, 0.5)
        y = np.array([0.5, -0.5, 0.500001, -0.500001, 0.0, np.nan])
        c = classify(y)
        np.testing.assert_array_equal(c[:2], [0.0, 0.0])  # exactly +-alpha -> flat
        np.testing.assert_array_equal(c[2:5], [1.0, -1.0, 0.0])
        self.assertTrue(np.isnan(c[5]))

    def test_default_alpha_used_when_omitted(self):
        # classify(2.0) with default alpha=0.5: 2.0 > 0.5 -> up.
        self.assertEqual(classify(np.array([2.0]))[0], 1.0)

    def test_custom_alpha_overrides_the_frozen_default(self):
        # Sanity: the function itself is generic (report stage sweeps alpha); only
        # compute() is pinned to ALPHA_TICKS.
        self.assertEqual(classify(np.array([0.6]), alpha_ticks=1.0)[0], 0.0)


class TestCalendarHorizonHandComputed(unittest.TestCase):
    """Variant A: stamp at row 2 (ts=B+3), mid/wmid known from tests/toybook.py."""

    def setUp(self):
        self.frame = _variant_a_frame()
        ts = _variant_a_ts()
        self.stamps = Stamps(index=np.array([2]), ts_ns=np.array([ts[2]]))

    def test_1s_horizon_resolves_inside_the_frame(self):
        # t = row2 (ts=B+3). target = t + 1s = B+1,000,000,003.
        # Row6 ts = B+700,000,000 <= target; row7 ts = B+2,000,000,000 > target -- so the
        # last qualifying event is row6. mid[6]=100500 (post-transition), row2 mid=100750
        # (pre-transition; the transition is row3's full bid delete, tests/toybook.py).
        # y_mid = (100500 - 100750) raw = -250 raw = -2.5 ticks. |y|=2.5 > alpha=0.5 -> down.
        out = compute(self.frame, self.stamps)
        self.assertAlmostEqual(out["label_y_mid_1s"][0], -2.5, places=9)
        self.assertEqual(out["label_c_mid_1s"][0], -1.0)

        # wmid[2] = 7550000/75 = 100666.666.. (tests/toybook.py); wmid[6] = 100500 (rows
        # 4-7 share wmid=100500, sizes settled after the row4 execution). y_wmid =
        # (100500 - 7550000/75) raw = -12500/75 raw = -12500/7500 ticks = -1.6666666...
        self.assertAlmostEqual(out["label_y_wmid_1s"][0], -12500.0 / 7500.0, places=9)
        self.assertEqual(out["label_c_wmid_1s"][0], -1.0)

    def test_longer_horizons_run_past_the_last_event_and_are_nan(self):
        # frame's last event is row7 (ts=B+2,000,000,000). t=row2 (ts=B+3).
        # 2s target = B+2,000,000,003 > B+2,000,000,000 -> past the last event -> NaN.
        # 5s/10s targets are further out still -> NaN for the same reason. Not clipped to
        # row7's price -- that would manufacture a fake return.
        out = compute(self.frame, self.stamps)
        for h in ("2s", "5s", "10s"):
            self.assertTrue(np.isnan(out[f"label_y_mid_{h}"][0]), h)
            self.assertTrue(np.isnan(out[f"label_y_wmid_{h}"][0]), h)
            self.assertTrue(np.isnan(out[f"label_c_mid_{h}"][0]), h)
            self.assertTrue(np.isnan(out[f"label_c_wmid_{h}"][0]), h)


class TestSessionEndNaN(unittest.TestCase):
    """Isolates the SESSION_END_NS rule from the "past the last event" rule."""

    def test_horizon_within_the_frame_but_past_session_end_is_nan(self):
        # Anchor 100ms before the official close, with a LAST frame event 1s AFTER the
        # official close (synthetic, but legal: labels.py must not assume the frame stops
        # exactly at SESSION_END_NS).
        t_anchor = SESSION_END_NS - 100_000_000
        d = toy_arrays()
        d["ts_ns"] = [
            t_anchor - 2,
            t_anchor - 1,
            t_anchor,
            t_anchor + 100,
            t_anchor + 200,
            t_anchor + 300,
            t_anchor + 400,
            SESSION_END_NS + 1_000_000_000,  # last event, 1s after the official close
        ]
        frame = make_event_frame(**d)
        stamps = Stamps(index=np.array([2]), ts_ns=np.array([t_anchor]))

        # target = t_anchor + 1s = SESSION_END_NS + 900,000,000, which is STILL <= the
        # frame's last event (SESSION_END_NS + 1,000,000,000) -- so the "past the last
        # event" check alone would NOT flag this. It is > SESSION_END_NS, so the
        # independent session-end check must be the one that fires.
        target_ns = t_anchor + 1_000_000_000
        self.assertLessEqual(target_ns, frame.ts_ns[-1])  # would pass the other check
        self.assertGreater(target_ns, SESSION_END_NS)  # but fails this one

        out = compute(frame, stamps)
        self.assertTrue(np.isnan(out["label_y_mid_1s"][0]))
        self.assertTrue(np.isnan(out["label_y_wmid_1s"][0]))


class TestBaseFixtureCalendarHorizonsAreAllNaN(unittest.TestCase):
    """The shared 8-event fixture spans only 400ms, so every 1s+ horizon is NaN there."""

    def test_all_four_calendar_horizons_nan_at_every_stamp(self):
        frame = toy_event_frame()
        stamps = Stamps(index=np.arange(8), ts_ns=np.array(TOY_TS_NS))
        out = compute(frame, stamps)
        for h in ("1s", "2s", "5s", "10s"):
            self.assertTrue(np.all(np.isnan(out[f"label_y_mid_{h}"])), h)
            self.assertTrue(np.all(np.isnan(out[f"label_y_wmid_{h}"])), h)
            self.assertTrue(np.all(np.isnan(out[f"label_c_mid_{h}"])), h)
            self.assertTrue(np.all(np.isnan(out[f"label_c_wmid_{h}"])), h)


class TestEventHorizonHandComputed(unittest.TestCase):
    """Stamps = every one of the base fixture's 8 rows, so book == the known reference table."""

    def setUp(self):
        self.frame = toy_event_frame()
        self.stamps = Stamps(index=np.arange(8), ts_ns=np.array(TOY_TS_NS))
        self.out = compute(self.frame, self.stamps)

    def test_e1_mid_full_array(self):
        # mid = [100750]*3 + [100500]*5 (tests/toybook.py). Diffs mid[i+1]-mid[i]:
        # i0:100750-100750=0  i1:0  i2:100500-100750=-250raw=-2.5ticks  i3..6:0  i7:NaN (i+1=8 OOB)
        np.testing.assert_allclose(
            self.out["label_y_mid_e1"][:7], [0.0, 0.0, -2.5, 0.0, 0.0, 0.0, 0.0]
        )
        self.assertTrue(np.isnan(self.out["label_y_mid_e1"][7]))

    def test_e1_classification_matches_the_deadband(self):
        # 0.0 is flat at any alpha; -2.5 < -0.5 -> down.
        np.testing.assert_array_equal(
            self.out["label_c_mid_e1"][:7], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.assertTrue(np.isnan(self.out["label_c_mid_e1"][7]))

    def test_e1_wmid_two_representative_positions(self):
        # wmid = [6035000/60, 100687.5, 7550000/75, 9040000/90, 100500, 100500, 100500, 100500].
        # pos2: wmid[3]-wmid[2] = 9040000/90 - 7550000/75. LCD 450:
        #   9040000/90 = 45200000/450; 7550000/75 = 45300000/450 -> diff = -100000/450 raw
        #   = -100000/45000 ticks = -2.2222...
        self.assertAlmostEqual(
            self.out["label_y_wmid_e1"][2], -100000.0 / 45000.0, places=9
        )
        # pos3: wmid[4]-wmid[3] = 100500 - 9040000/90 = (9045000-9040000)/90 = 5000/90 raw
        #   = 5000/9000 ticks = 0.5555...
        self.assertAlmostEqual(self.out["label_y_wmid_e1"][3], 5000.0 / 9000.0, places=9)
        self.assertTrue(np.isnan(self.out["label_y_wmid_e1"][7]))

    def test_e2_spot_checks(self):
        # mid[i+2]-mid[i]: i0:mid2-mid0=0  i1:mid3-mid1=100500-100750=-2.5ticks
        # i2:mid4-mid2=100500-100750=-2.5ticks  last N=2 stamps (i6,i7) NaN.
        y = self.out["label_y_mid_e2"]
        self.assertAlmostEqual(y[0], 0.0, places=9)
        self.assertAlmostEqual(y[1], -2.5, places=9)
        self.assertAlmostEqual(y[2], -2.5, places=9)
        self.assertTrue(np.isnan(y[6]))
        self.assertTrue(np.isnan(y[7]))

    def test_e5_spot_checks(self):
        # mid[i+5]-mid[i]: i0:mid5-mid0=100500-100750=-2.5  i1:mid6-mid1=-2.5  i2:mid7-mid2=-2.5
        # last N=5 stamps (i3..i7) NaN.
        y = self.out["label_y_mid_e5"]
        for i in range(3):
            self.assertAlmostEqual(y[i], -2.5, places=9)
        for i in range(3, 8):
            self.assertTrue(np.isnan(y[i]), i)

    def test_e10_entirely_nan(self):
        # n=8 <= N=10: every stamp is within "the last N stamps" (there is no i with
        # i+10 < 8), so the whole column is NaN.
        y = self.out["label_y_mid_e10"]
        self.assertTrue(np.all(np.isnan(y)))
        self.assertTrue(np.all(np.isnan(self.out["label_c_mid_e10"])))


class TestInvalidBookNaN(unittest.TestCase):
    """3-row, 1-level frame; row1's ask is the LOBSTER sentinel with size 0 -> invalid touch.

    Rows (bid_px, bid_sz | ask_px, ask_sz), ts_ns spaced exactly 1s apart:
      0: (100000, 40 | 101000, 50)  valid.  mid=100500, wmid=(100000*50+101000*40)/90=9040000/90
      1: (100000, 40 | SENTINEL, 0) invalid. mid=NaN, wmid=NaN
      2: (100000, 40 | 101000, 60)  valid.  mid=100500, wmid=(100000*60+101000*40)/100=10040000/100=100400.0
    """

    def setUp(self):
        self.ts = [BASE, BASE + 1_000_000_000, BASE + 2_000_000_000]
        self.frame = make_event_frame(
            ts_ns=self.ts,
            event_type=[1, 3, 1],
            direction=[1, -1, 1],
            price=[100000, 101000, 101000],
            size=[40, 50, 60],
            bid_px=[[100000], [100000], [100000]],
            bid_sz=[[40], [40], [40]],
            ask_px=[[101000], [ASK_SENTINEL_PRICE], [101000]],
            ask_sz=[[50], [0], [60]],
        )

    def test_event_horizon_nan_when_target_row_is_invalid(self):
        # stamps over all 3 rows, N=1: pos0's target is row1 (invalid) -> NaN.
        stamps = Stamps(index=np.array([0, 1, 2]), ts_ns=np.array(self.ts))
        out = compute(self.frame, stamps)
        self.assertTrue(np.isnan(out["label_y_mid_e1"][0]))  # target (row1) invalid
        self.assertTrue(np.isnan(out["label_y_wmid_e1"][0]))

    def test_event_horizon_nan_when_anchor_row_is_invalid(self):
        # pos1 (row1 itself) is invalid -> book.mid[1] is NaN -> the label is NaN
        # regardless of the target (row2) being perfectly valid.
        stamps = Stamps(index=np.array([0, 1, 2]), ts_ns=np.array(self.ts))
        book = book_at_stamps(self.frame, stamps)
        self.assertTrue(np.isnan(book.mid[1]))  # confirms the premise
        out = compute(self.frame, stamps, book=book)
        self.assertTrue(np.isnan(out["label_y_mid_e1"][1]))

    def test_calendar_horizon_nan_when_target_row_is_invalid(self):
        # t=row0 (valid, mid=100500). horizon=1s -> target=row1's ts exactly (invalid).
        stamps = Stamps(index=np.array([0]), ts_ns=np.array([self.ts[0]]))
        out = compute(self.frame, stamps)
        self.assertTrue(np.isnan(out["label_y_mid_1s"][0]))
        self.assertTrue(np.isnan(out["label_y_wmid_1s"][0]))

    def test_calendar_horizon_nan_when_anchor_row_is_invalid(self):
        # t=row1 (invalid, mid=NaN). horizon=1s -> target=row2's ts exactly (valid,
        # mid=100500) -- but the anchor is NaN, so the difference must still be NaN.
        stamps = Stamps(index=np.array([1]), ts_ns=np.array([self.ts[1]]))
        out = compute(self.frame, stamps)
        self.assertTrue(np.isnan(out["label_y_mid_1s"][0]))


class TestInvertedLookahead(unittest.TestCase):
    """labels.py's lookahead test is INVERTED: truncating the future must turn a real
    value into NaN, not leave it unchanged (INTERFACE.md sec. 10)."""

    def test_truncating_future_events_turns_the_label_nan(self):
        d = toy_arrays()
        d["ts_ns"] = _variant_a_ts()
        full_table = make_frame(**d)
        past_table = full_table.slice(0, 3)  # rows 0,1,2 only -- row 2 (our stamp) survives

        full_frame = event_frame_from_table(full_table)
        past_frame = event_frame_from_table(past_table)

        stamps = Stamps(index=np.array([2]), ts_ns=np.array([d["ts_ns"][2]]))

        out_full = compute(full_frame, stamps)
        out_past = compute(past_frame, stamps)

        # Established in TestCalendarHorizonHandComputed: with the full frame, the 1s
        # horizon at row2 resolves to row6 and equals -2.5 ticks.
        self.assertAlmostEqual(out_full["label_y_mid_1s"][0], -2.5, places=9)

        # With events 3-7 truncated away, row2 (ts=B+3) is now the frame's OWN last
        # event: target = B+3+1s runs past it -> NaN. The label at the SAME stamp
        # changed from a concrete number to NaN purely because the future changed --
        # which is exactly what a label is supposed to do, and the opposite of every
        # other family's lookahead test.
        self.assertTrue(np.isnan(out_past["label_y_mid_1s"][0]))
        self.assertNotEqual(out_full["label_y_mid_1s"][0], out_past["label_y_mid_1s"][0])


class TestColumnContract(unittest.TestCase):
    def test_exactly_28_columns_all_under_the_label_prefix(self):
        frame = toy_event_frame()
        stamps = Stamps(index=np.arange(8), ts_ns=np.array(TOY_TS_NS))
        out = compute(frame, stamps)
        self.assertEqual(set(out.keys()), EXPECTED_COLUMNS)
        self.assertEqual(len(out), 28)
        self.assertTrue(all(k.startswith("label_") for k in out))

    def test_every_array_is_float64_1d_len_of_stamps(self):
        frame = toy_event_frame()
        stamps = Stamps(index=np.arange(8), ts_ns=np.array(TOY_TS_NS))
        out = compute(frame, stamps)
        for k, v in out.items():
            self.assertEqual(v.dtype, np.float64, k)
            self.assertEqual(v.shape, (8,), k)


class TestBothClocks(unittest.TestCase):
    def test_runs_on_calendar_and_event_clocks(self):
        frame = toy_event_frame()
        for clock in ("calendar", "event"):
            stamps = stamps_for_clock(frame, clock)
            out = compute(frame, stamps)
            n = len(stamps.index)
            self.assertEqual(set(out.keys()), EXPECTED_COLUMNS, clock)
            for k, v in out.items():
                self.assertEqual(v.dtype, np.float64, (clock, k))
                self.assertEqual(v.shape, (n,), (clock, k))

    def test_book_none_matches_explicit_book_bit_identical(self):
        frame = toy_event_frame()
        stamps = stamps_for_clock(frame, "calendar")
        book = book_at_stamps(frame, stamps)
        out_implicit = compute(frame, stamps)
        out_explicit = compute(frame, stamps, book=book)
        for k in EXPECTED_COLUMNS:
            np.testing.assert_array_equal(out_implicit[k], out_explicit[k], err_msg=k)


class TestEmptyStamps(unittest.TestCase):
    def test_empty_stamps_return_length_zero_float64_arrays_not_an_exception(self):
        frame = toy_event_frame()
        stamps = Stamps(index=np.array([], dtype=np.int64), ts_ns=np.array([], dtype=np.int64))
        out = compute(frame, stamps)
        self.assertEqual(set(out.keys()), EXPECTED_COLUMNS)
        for k, v in out.items():
            self.assertEqual(v.shape, (0,), k)
            self.assertEqual(v.dtype, np.float64, k)


if __name__ == "__main__":
    unittest.main()
