"""Hand-computed tests for windows.py against tests/toybook.py's 8-event frame."""
import unittest

import numpy as np

from edge.features.windows import between_stamps_start_index, trailing_start_index

from tests.toybook import TOY_TS_NS


class TestTrailingStartIndex(unittest.TestCase):
    def test_hand_computed_100ms_window(self):
        # Toy ts_ns: [...001, ...002, 100000000, 100000500, 200000000, 200000001,
        #             300000000, 400000000] (offsets from 34200000000000, indices 0-7).
        # stamp=34200100000000 (34200.1s), window=100ms -> lower_bound=34200000000000.
        #   First frame ts strictly > lower_bound is index 0 (...001 > ...000) -> start=0.
        #   (Matches: window (34200.0s, 34200.1s] should include indices 0,1,2.)
        # stamp=34200200000000 (34200.2s), window=100ms -> lower_bound=34200100000000.
        #   First frame ts strictly > lower_bound: index 2 == lower_bound (not >), index 3
        #   (...100000500) is > -> start=3. (Window (34200.1s, 34200.2s] = indices 3,4.)
        frame_ts = np.array(TOY_TS_NS, dtype=np.int64)
        stamp_ts = np.array([34200100000000, 34200200000000], dtype=np.int64)
        start = trailing_start_index(frame_ts, stamp_ts, window_ns=100_000_000)
        self.assertEqual(start.tolist(), [0, 3])

    def test_left_boundary_is_open_not_closed(self):
        # An event exactly window_ns old must be EXCLUDED (interval is (t-w, t], not
        # [t-w, t]) -- this is what distinguishes side="right" from side="left" and would
        # catch the off-by-one if someone flipped it.
        frame_ts = np.array([100, 200, 300], dtype=np.int64)
        stamp_ts = np.array([300], dtype=np.int64)
        start = trailing_start_index(frame_ts, stamp_ts, window_ns=200)
        # lower_bound = 100, which IS a frame ts -> must be excluded -> start should skip
        # index 0 and land on index 1, not index 0.
        self.assertEqual(start.tolist(), [1])


class TestBetweenStampsStartIndex(unittest.TestCase):
    def test_hand_computed(self):
        # From the calendar-clock example (test_clocks.py): stamps.index = [2, 4, 6, 7].
        # First stamp has no predecessor -> start=0 (from the top of the frame).
        # Subsequent starts are the previous stamp's index + 1.
        stamps_index = np.array([2, 4, 6, 7], dtype=np.int64)
        start = between_stamps_start_index(stamps_index)
        self.assertEqual(start.tolist(), [0, 3, 5, 7])

    def test_empty_input(self):
        self.assertEqual(between_stamps_start_index(np.array([], dtype=np.int64)).tolist(), [])


if __name__ == "__main__":
    unittest.main()
