"""Hand-computed tests for the calendar and event clocks against tests/toybook.py.

Toy frame ts_ns (ns after midnight): [...001, ...002, 34200100000000, 34200100000500,
34200200000000, 34200200000001, 34200300000000, 34200400000000] at indices 0-7; event
types [1,1,2,3,4,5,1,3] (see tests/toybook.py for the full per-row derivation).
"""
import unittest

import numpy as np

from edge.features.clocks import (
    EVENT_CLOCK_N,
    calendar_grid_stamp_count,
    calendar_stamps,
    event_stamps,
    is_book_changing,
)

from tests.toybook import build_toy_frame, make_frame

SESSION_START_NS = 34_200_000_000_000
SESSION_END_NS = 34_200_400_000_000  # last toy-frame event, so the grid covers it exactly


class TestCalendarStamps(unittest.TestCase):
    def setUp(self):
        self.frame = build_toy_frame()

    def test_hand_computed_stamps(self):
        # Grid points at the default 100ms: 34200.0, .1, .2, .3, .4 s.
        # 34200.0s: first event ts=34200000000001 > 34200000000000 -> no qualifying
        #           event yet -> DROPPED (the <= t convention, per clocks.py docstring).
        # 34200.1s: last event with ts<=...100000000 is index 2 (ts==34200100000000
        #           exactly; index 3 is 500ns later, excluded).
        # 34200.2s: last qualifying is index 4 (ts==34200200000000 exactly; index 5 is
        #           1ns later, excluded).
        # 34200.3s: last qualifying is index 6 (ts==34200300000000 exactly; index 7 is
        #           100ms later, excluded).
        # 34200.4s: last qualifying is index 7 (ts==34200400000000 exactly).
        stamps = calendar_stamps(self.frame, SESSION_START_NS, SESSION_END_NS)
        self.assertEqual(
            stamps.ts_ns.tolist(),
            [34200100000000, 34200200000000, 34200300000000, 34200400000000],
        )
        self.assertEqual(stamps.index.tolist(), [2, 4, 6, 7])

    def test_stamp_before_first_event_is_dropped_not_flagged(self):
        # A naive implementation might emit index=-1 for the 34200.0s grid point (numpy's
        # searchsorted-1 default) -- that silently means "last row of frame" via Python
        # negative indexing, a real (and nasty) leak-adjacent bug, not just a cosmetic one.
        stamps = calendar_stamps(self.frame, SESSION_START_NS, SESSION_END_NS)
        self.assertNotIn(34200000000000, stamps.ts_ns.tolist())
        self.assertTrue((stamps.index >= 0).all())

    def test_index_is_last_qualifying_event_not_merely_a_qualifying_one(self):
        # Catches an off-by-one (e.g. side="left" instead of "right", or a missing -1):
        # for every stamp, frame.ts_ns[index] <= ts_ns, and the NEXT frame row (if any)
        # is strictly after ts_ns -- i.e. index really is the *last* one, not the first.
        frame_ts = self.frame.column("ts_ns").to_numpy()
        stamps = calendar_stamps(self.frame, SESSION_START_NS, SESSION_END_NS)
        for idx, ts in zip(stamps.index, stamps.ts_ns):
            self.assertLessEqual(frame_ts[idx], ts)
            if idx + 1 < len(frame_ts):
                self.assertGreater(frame_ts[idx + 1], ts)

    def test_calendar_grid_stamp_count_full_session(self):
        # (16:00 - 09:30) / 100ms + 1, both ends inclusive.
        self.assertEqual(calendar_grid_stamp_count(), 234_001)


class TestIsBookChanging(unittest.TestCase):
    def test_hand_computed_mask(self):
        # types [1,1,2,3,4,5,1,3] -> only the type-5 (hidden execution, index 5) is False.
        mask = is_book_changing(np.array([1, 1, 2, 3, 4, 5, 1, 3]))
        self.assertEqual(mask.tolist(), [True, True, True, True, True, False, True, True])


class TestEventStamps(unittest.TestCase):
    def setUp(self):
        self.frame = build_toy_frame()

    def test_hand_computed_every_second_book_changing_event(self):
        # Book-changing positions (type != 5): [0,1,2,3,4,6,7] (7 of them). n=2 stamps the
        # 2nd, 4th, 6th book-changing event -> array positions 1, 3, 5 of that list ->
        # frame indices [1, 3, 6].
        stamps = event_stamps(self.frame, n=2)
        self.assertEqual(stamps.index.tolist(), [1, 3, 6])
        self.assertEqual(stamps.ts_ns.tolist(), [34200000000002, 34200100000500, 34200300000000])

    def test_n_equals_1_stamps_every_book_changing_event(self):
        stamps = event_stamps(self.frame, n=1)
        self.assertEqual(stamps.index.tolist(), [0, 1, 2, 3, 4, 6, 7])

    def test_n_must_be_positive(self):
        with self.assertRaises(ValueError):
            event_stamps(self.frame, n=0)

    def test_frozen_single_n_for_both_tickers(self):
        # Frozen 2026-07-30: ONE N for the whole universe, not a per-ticker calibration --
        # a per-ticker N confounds the clock axis with the ticker axis in the ladder table,
        # and the earlier AMZN N=1 made the event clock a relabelling of the event frame.
        # See clocks.py's EVENT_CLOCK_N comment + DECISIONS.md.
        self.assertEqual(EVENT_CLOCK_N, 10)
        self.assertNotIsInstance(EVENT_CLOCK_N, dict)


if __name__ == "__main__":
    unittest.main()


class TestEventClockIndexOnTiedTimestamps(unittest.TestCase):
    """INTERFACE.md sec. 3's "last row with ts <= t" rule, on the EVENT clock.

    The contract test above runs against `calendar_stamps` only, and the shared toy fixture
    has no two events on the same nanosecond -- so the case was unreachable in the suite
    while 10.48% of real INTC event stamps hit it (up to 97 rows skipped). Using the
    triggering event's own row instead of the last tied one leaves a mid move that elapsed
    in zero wall-clock time inside `label_y_mid_*` on the event clock.

    Six events on three distinct timestamps, all book-changing (type 1):

        row:  0    1    2    3    4    5
        ts:   10   10   10   20   20   30

    With n=1 every row is a stamp, and every stamp whose timestamp is shared must resolve to
    the LAST row carrying it: rows 0, 1, 2 all stamp ts=10 and must all report index 2.
    """

    def setUp(self):
        ts = [10, 10, 10, 20, 20, 30]
        n = len(ts)
        self.frame = make_frame(
            ts_ns=ts,
            event_type=[1] * n,
            direction=[1] * n,
            price=[100000] * n,
            size=[10] * n,
            bid_px=[[100000]] * n,
            bid_sz=[[10 * (i + 1)] for i in range(n)],
            ask_px=[[100100]] * n,
            ask_sz=[[50]] * n,
        )

    def test_every_stamp_resolves_to_the_last_row_sharing_its_timestamp(self):
        stamps = event_stamps(self.frame, n=1)
        # ts=10 -> last row with ts<=10 is row 2; ts=20 -> row 4; ts=30 -> row 5. Tied
        # stamps collapse to one sample per timestamp (see the dedupe test below).
        self.assertEqual(stamps.ts_ns.tolist(), [10, 20, 30])
        self.assertEqual(stamps.index.tolist(), [2, 4, 5])

    def test_index_is_last_qualifying_event_not_merely_a_qualifying_one(self):
        # The same invariant the calendar clock is held to, now asserted on the event clock.
        frame_ts = self.frame.column("ts_ns").to_numpy()
        stamps = event_stamps(self.frame, n=1)
        for idx, ts in zip(stamps.index, stamps.ts_ns):
            self.assertLessEqual(frame_ts[idx], ts)
            if idx + 1 < len(frame_ts):
                self.assertGreater(frame_ts[idx + 1], ts)

    def test_holds_for_a_coarser_event_clock_too(self):
        # n=2 stamps rows 1, 3, 5 -> timestamps 10, 20, 30 -> indices 2, 4, 5.
        stamps = event_stamps(self.frame, n=2)
        self.assertEqual(stamps.ts_ns.tolist(), [10, 20, 30])
        self.assertEqual(stamps.index.tolist(), [2, 4, 5])

    def test_tied_stamps_are_collapsed_to_one_sample_per_timestamp(self):
        # Rows 0,1,2 all carry ts=10 and all resolve to index 2, so emitting three stamps
        # there would emit three IDENTICAL samples. INTERFACE.md sec. 3 requires ts_ns
        # strictly increasing; keep the last of each tied run.
        stamps = event_stamps(self.frame, n=1)
        self.assertEqual(stamps.ts_ns.tolist(), [10, 20, 30])
        self.assertEqual(stamps.index.tolist(), [2, 4, 5])
        self.assertTrue((np.diff(stamps.ts_ns) > 0).all())

    def test_no_duplicate_samples_survive(self):
        stamps = event_stamps(self.frame, n=1)
        pairs = set(zip(stamps.ts_ns.tolist(), stamps.index.tolist()))
        self.assertEqual(len(pairs), stamps.ts_ns.shape[0])
