"""Hand-computed tests for edge.features.microprice (FEATURES.md sec. 3, Stoikov 2018).

House style per tests/test_book.py / tests/test_normalize.py: every expected value is a
literal derived arithmetic-by-arithmetic in a comment above the assertion, never a second
call into the formula under test.
"""
import os
import unittest

import numpy as np

import edge.features.microprice as mp
from edge.features import NS_PER_SEC, REFIT_BLOCK_NS, SESSION_START_NS
from edge.features.book import book_at_stamps, ticks
from edge.features.clocks import Stamps, stamps_for_clock
from edge.features.frame import (
    EventFrame,
    event_frame_from_table,
    frame_path,
    load_event_frame,
)
from edge.features.windows import refit_blocks
from edge.io.lobster import BID_SENTINEL_PRICE

from tests.toybook import (
    TOY_TS_NS,
    make_event_frame,
    make_frame,
    toy_arrays,
    toy_event_frame,
)


class TestWmidDevMidHandComputedOnToyBook(unittest.TestCase):
    """mp_wmid_dev_mid needs no fit -- it is pure book arithmetic, so it is well defined on
    the 8-event toy fixture even though mp_micro/mp_dev_mid/mp_dev_wmid cannot be (below
    MIN_FIT_ROWS on 8 events -- see TestWarmupInsufficientIsNaN)."""

    def setUp(self):
        self.frame = toy_event_frame()
        self.stamps = Stamps(index=np.arange(8, dtype=np.int64), ts_ns=np.array(TOY_TS_NS))
        self.book = book_at_stamps(self.frame, self.stamps)
        self.out = mp.compute(self.frame, self.stamps, self.book)

    def test_hand_computed_wmid_dev_mid_all_eight_events(self):
        # Re-derive mid and wmid from the raw per-event book states (tests/toybook.py's
        # TOY_BID_PX/SZ/ASK_PX/SZ), independently of the docstring's reference numbers.
        #
        # Event 0: bid 100500x10, ask 101000x50.
        #   mid = (100500+101000)/2 = 100750
        #   wmid = (100500*50 + 101000*10)/60 = (5025000+1010000)/60 = 6035000/60 = 100583.3333...
        #   diff_raw = 6035000/60 - 6045000/60 = -10000/60 = -166.6666... ; ticks = -10000/6000 = -5/3
        #
        # Event 1: bid 100500x30, ask 101000x50.
        #   wmid = (100500*50 + 101000*30)/80 = (5025000+3030000)/80 = 8055000/80 = 100687.5
        #   mid = 100750; diff_raw = -62.5; ticks = -0.625 exactly (-5/8)
        #
        # Event 2: bid 100500x25, ask 101000x50.
        #   wmid = (100500*50 + 101000*25)/75 = (5025000+2525000)/75 = 7550000/75 = 100666.6666...
        #   mid = 100750; diff_raw = 7550000/75 - 7556250/75 = -6250/75 = -83.3333...; ticks = -5/6
        #
        # Event 3: bid 100000x40, ask 101000x50 (bid touch just deleted from 100500).
        #   wmid = (100000*50 + 101000*40)/90 = (5000000+4040000)/90 = 9040000/90 = 100444.4444...
        #   mid = 100500; diff_raw = 9040000/90 - 9045000/90 = -5000/90 = -55.5555...; ticks = -5/9
        #
        # Events 4-7: bid 100000x40, ask 101000x40 (ask depleted 50->40 at event 4; hidden
        # execution, invisible new L2 ask, and L2-only delete at events 5-7 don't touch L1).
        #   wmid = (100000*40 + 101000*40)/80 = 8040000/80 = 100500.0 = mid exactly -> diff 0.
        expected_ticks = [-5.0 / 3.0, -0.625, -5.0 / 6.0, -5.0 / 9.0, 0.0, 0.0, 0.0, 0.0]
        got = self.out["mp_wmid_dev_mid"]
        self.assertEqual(got.shape, (8,))
        for i, exp in enumerate(expected_ticks):
            self.assertAlmostEqual(got[i], exp, places=9, msg=f"event {i}")

    def test_wmid_dev_mid_event_1_is_exact(self):
        # 100687.5 - 100750 = -62.5 raw = -0.625 ticks, with no repeating fraction, so this
        # one is checked for bit-exactness rather than approximate equality.
        self.assertEqual(self.out["mp_wmid_dev_mid"][1], -0.625)


class TestWarmupInsufficientIsNaN(unittest.TestCase):
    """8 events is far below MIN_FIT_ROWS (5000): every refit block's fit_end < MIN_FIT_ROWS,
    so no block ever produces a fitted (B, G^1) and mp_micro/mp_dev_mid/mp_dev_wmid must be
    NaN at every stamp -- this is the "warmup insufficient" NaN rule from INTERFACE.md sec. 7."""

    def test_all_stamps_nan_on_the_toy_fixture(self):
        frame = toy_event_frame()
        stamps = Stamps(index=np.arange(8, dtype=np.int64), ts_ns=np.array(TOY_TS_NS))
        out = mp.compute(frame, stamps)
        self.assertTrue(np.all(np.isnan(out["mp_micro"])))
        self.assertTrue(np.all(np.isnan(out["mp_dev_mid"])))
        self.assertTrue(np.all(np.isnan(out["mp_dev_wmid"])))
        # mp_wmid_dev_mid needs no fit at all, so it must stay finite throughout.
        self.assertFalse(np.any(np.isnan(out["mp_wmid_dev_mid"])))


def _build_two_state_frame():
    """6-event, 1-level book walking exactly the transitions:
       A->A(no move), A->A(move +200), A->B(no move), B->B(no move), B->B(move -100)
    where A = (spread bucket 0, imbalance bucket 0) and B = (spread bucket 0, imbalance
    bucket 4) under this module's own bucketing (N_SPREAD=3, N_IMB=5). Spread is always
    100 raw = 1 tick -> spread bucket floor(1)-1 = 0 for every row, so only the imbalance
    ratio distinguishes A (bid 10 / ask 90 -> imb 0.1 -> bucket floor(0.5)=0) from B (bid 90
    / ask 10 -> imb 0.9 -> bucket floor(4.5)=4).
    """
    ts_ns = [34200000000001 + i for i in range(6)]
    event_type = [1] * 6
    direction = [1] * 6
    bid_px = [[100000], [100000], [100200], [100200], [100200], [100100]]
    bid_sz = [[10], [10], [10], [90], [90], [90]]
    ask_px = [[100100], [100100], [100300], [100300], [100300], [100200]]
    ask_sz = [[90], [90], [90], [10], [10], [10]]
    price = [row[0] for row in bid_px]
    size = [row[0] for row in bid_sz]
    return make_event_frame(
        ts_ns=ts_ns,
        event_type=event_type,
        direction=direction,
        price=price,
        size=size,
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
    )


def _build_martingale_frame():
    """9-event, 1-level book where EVERY transition moves the mid, so Q = 0 and the
    absorbing algebra collapses to G^1 = R K and B = R2 -- which makes the outer recursion
    the only thing left to get wrong.

    Spread is always 100 raw = 1 tick (spread bucket 0). Events 0-3 and 8 are state A
    (bid 10 / ask 90 -> imb 0.1 -> bucket 0); events 4-7 are state B (bid 90 / ask 10 ->
    imb 0.9 -> bucket 4). The mid walks up 100 four times then down 100 four times:

        mid: 100050 100150 100250 100350 100450 100350 100250 100150 100050
        state:   A      A      A      A      B      B      B      B      A

    Transitions (k, k+1), all with a nonzero mid change, so NONE is transient:
        (0,1) A->A +100   (1,2) A->A +100   (2,3) A->A +100   (3,4) A->B +100
        (4,5) B->B -100   (5,6) B->B -100   (6,7) B->B -100   (7,8) B->A -100
    """
    ts_ns = [34200000000001 + i for i in range(9)]
    mids = [100050, 100150, 100250, 100350, 100450, 100350, 100250, 100150, 100050]
    bid_px = [[m - 50] for m in mids]
    ask_px = [[m + 50] for m in mids]
    # A-rows are bid-light (imb 0.1), B-rows are bid-heavy (imb 0.9).
    bid_sz = [[10], [10], [10], [10], [90], [90], [90], [90], [10]]
    ask_sz = [[90], [90], [90], [90], [10], [10], [10], [10], [90]]
    return make_event_frame(
        ts_ns=ts_ns,
        event_type=[1] * 9,
        direction=[1] * 9,
        price=[row[0] for row in bid_px],
        size=[row[0] for row in bid_sz],
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
    )


class TestIteratedMicropriceRecursionHandSolved(unittest.TestCase):
    """The recursion itself: G^(i+1) = G^1 + B G^i, hand-solved (Stoikov 2018 sec. 3).

    This is the test that separates the true microprice from its first-order term G^1 --
    the FEATURES.md sec. 3 distinction. On _build_martingale_frame every transition moves
    the mid, so Q = 0, (I - Q)^-1 = I, and:

        counts[A] = 4, counts[B] = 4
        R K[A] = (+100 +100 +100 +100)/4 = +100        -> G^1[A] = +100
        R K[B] = (-100 -100 -100 -100)/4 = -100        -> G^1[B] = -100
        R2[A] = landings A,A,A,B = [3/4 at A, 1/4 at B] -> B[A] = [3/4, 1/4]
        R2[B] = landings B,B,B,A = [1/4 at A, 3/4 at B] -> B[B] = [1/4, 3/4]

    G^1 = 100*[1, -1] is an eigenvector of B with eigenvalue 3/4 - 1/4 = 1/2, so the
    recursion is a plain geometric series in 1/2:

        G^(i) = 100*[1, -1] * sum_{j=0}^{i-1} (1/2)^j

    The loop starts at g = G^1 and applies MICRO_ITERS = 6 further iterations, landing on
    G^(7) = 100 * sum_{j=0}^{6} (1/2)^j = 100 * (2 - 1/64) = 100 * 127/64 = 198.4375.

    Two literals matter here. 198.4375 pins the recursion AND the frozen iteration depth.
    The limit it is converging to is 2 * G^1 = 200 -- so a G^1-only implementation would
    report 100, a full-limit implementation 200, and this one 198.4375. All three are
    distinguishable, which is the point.
    """

    @classmethod
    def setUpClass(cls):
        frame = _build_martingale_frame()
        valid = frame.bid_valid[:, 0] & frame.ask_valid[:, 0]
        cls.state_event = mp._state_index(
            ticks(frame.spread), frame.best_bid_sz, frame.best_ask_sz, valid
        )
        cls.x = mp._estimate_block_adjustment(cls.state_event, frame.mid, valid, fit_end=9)
        cls.state_a = 0
        cls.state_b = 4

    def test_state_assignment_matches_the_construction(self):
        self.assertEqual(self.state_event.tolist(), [0, 0, 0, 0, 4, 4, 4, 4, 0])

    def test_hand_solved_iterated_adjustment_state_a(self):
        # 100 * (2 - 2**-6) = 100 * 1.984375
        self.assertAlmostEqual(self.x[self.state_a], 198.4375, places=9)

    def test_hand_solved_iterated_adjustment_state_b(self):
        self.assertAlmostEqual(self.x[self.state_b], -198.4375, places=9)

    def test_is_not_the_first_order_term(self):
        # The G^1-only bug this test exists to catch would report exactly +/-100.
        self.assertNotAlmostEqual(self.x[self.state_a], 100.0, places=3)
        self.assertNotAlmostEqual(self.x[self.state_b], -100.0, places=3)


class TestTwoStateMarkovChainHandSolved(unittest.TestCase):
    """The CONSTRUCTED Markov case: a 2-state chain solvable by hand.

    Rows (mid in raw units, spread always 100 raw = 1 tick -> spread bucket 0):
      0: bid 100000x10, ask 100100x90 -> state A (imb 0.1 -> bucket 0). mid = 100050.
      1: bid 100000x10, ask 100100x90 -> state A. mid = 100050 (same as row 0).
      2: bid 100200x10, ask 100300x90 -> state A. mid = 100250.
      3: bid 100200x90, ask 100300x10 -> state B (imb 0.9 -> bucket 4). mid = 100250 (same as row 2).
      4: bid 100200x90, ask 100300x10 -> state B. mid = 100250 (same as row 3).
      5: bid 100100x90, ask 100200x10 -> state B. mid = 100150.

    Transitions (fit_end=6 uses all 5 pairs (k, k+1), k=0..4):
      (0,1): A->A, delta_mid = 100050-100050 = 0            -> transient, B[A,A] count 1
      (1,2): A->A, delta_mid = 100250-100050 = 200           -> absorbed, G1[A] sum += 200
      (2,3): A->B, delta_mid = 100250-100250 = 0             -> transient, B[A,B] count 1
      (3,4): B->B, delta_mid = 100250-100250 = 0             -> transient, B[B,B] count 1
      (4,5): B->B, delta_mid = 100150-100250 = -100          -> absorbed, G1[B] sum += -100

    N(A) = 3 (all three transitions leaving A), N(B) = 2 (both transitions leaving B).
      G1[A] = 200 / 3
      G1[B] = -100 / 2 = -50
      Q[A,A] = 1/3, Q[A,B] = 1/3 (row sum 2/3; the remaining 1/3 mass is the A->A absorbed move)
      Q[B,B] = 1/2, Q[B,A] = 0   (row sum 1/2; the remaining 1/2 mass is the B->B absorbed move)

    This pins the FIRST-ORDER term only: G^1 = (I - Q)^-1 R K, restricted to {A, B} (every
    other state is never a transition source, i.e. unvisited, and does not enter these two
    equations):
      G1_B = -50 + (1/2) G1_B                => (1/2) G1_B = -50            => G1_B = -100
      G1_A = 200/3 + (1/3) G1_A + (1/3) G1_B => (2/3) G1_A = 200/3 - 100/3  => G1_A = 50

    Check: G1_A = 200/3 + (1/3)(50) + (1/3)(-100) = 66.6667 + 16.6667 - 33.3333 = 50.0 (OK)
           G1_B = -50 + (1/2)(-100) + 0 = -50 - 50 = -100.0 (OK)

    The FULL microprice on this fixture is deliberately NOT asserted: state B here always
    moves down (R2[B,B] = 1, G^1[B] = -100), so the chain has no martingale property, the
    outer recursion G^(i+1) = G^1 + B G^i drifts by -100 per iteration and its value is an
    artifact of MICRO_ITERS rather than a limit. TestIteratedMicropriceRecursionHandSolved
    uses a convergent fixture for that. Keeping this one at G^1 is deliberate: it is the
    hand-checkable case for `_absorb`, the sub-solve the recursion is built on.
    """

    @classmethod
    def setUpClass(cls):
        frame = _build_two_state_frame()
        valid_event = frame.bid_valid[:, 0] & frame.ask_valid[:, 0]
        spread_ticks_event = ticks(frame.spread)
        cls.state_event = mp._state_index(
            spread_ticks_event, frame.best_bid_sz, frame.best_ask_sz, valid_event
        )
        cls.mid_event = frame.mid
        cls.valid_event = valid_event
        # State A must be index 0 (spread bucket 0 * N_IMB=5 + imbalance bucket 0) and state
        # B index 4 (spread bucket 0 * 5 + imbalance bucket 4), by construction above.
        cls.state_a = 0
        cls.state_b = 4
        cls.x = mp._estimate_block_adjustment(
            cls.state_event, cls.mid_event, cls.valid_event, fit_end=6
        )

    def test_state_assignment_matches_the_construction(self):
        self.assertEqual(self.state_event.tolist(), [0, 0, 0, 4, 4, 4])

    def test_hand_solved_first_order_term_state_a(self):
        # Q and R K written out as literals from the transition table in the class
        # docstring -- NOT read back from the module -- then pushed through the same
        # (I - Q)^-1 solve the recursion uses.
        q = np.zeros((mp.N_STATES, mp.N_STATES), dtype=np.float64)
        q[0, 0] = 1.0 / 3.0
        q[0, 4] = 1.0 / 3.0
        q[4, 4] = 1.0 / 2.0
        rk = np.zeros(mp.N_STATES, dtype=np.float64)
        rk[0] = 200.0 / 3.0
        rk[4] = -50.0
        g1 = mp._absorb(q, rk)
        self.assertAlmostEqual(g1[self.state_a], 50.0, places=6)

    def test_hand_solved_first_order_term_state_b(self):
        q = np.zeros((mp.N_STATES, mp.N_STATES), dtype=np.float64)
        q[0, 0] = 1.0 / 3.0
        q[0, 4] = 1.0 / 3.0
        q[4, 4] = 1.0 / 2.0
        rk = np.zeros(mp.N_STATES, dtype=np.float64)
        rk[0] = 200.0 / 3.0
        rk[4] = -50.0
        g1 = mp._absorb(q, rk)
        self.assertAlmostEqual(g1[self.state_b], -100.0, places=6)

    def test_unvisited_states_are_nan(self):
        # Every state other than A (0) and B (4) is never a transition source in this
        # 6-event walk -- INTERFACE.md sec. 8: "an unvisited state has no estimate".
        for s in range(mp.N_STATES):
            if s in (self.state_a, self.state_b):
                continue
            self.assertTrue(np.isnan(self.x[s]), f"state {s} should be NaN (unvisited)")


class TestNaNRules(unittest.TestCase):
    def test_invalid_book_gives_nan(self):
        # Sentinel the bid touch at event 3 (partial-cancel-then-delete row in the toy
        # fixture) so book.valid is False there; every price-reference column must be NaN.
        d = toy_arrays()
        d["bid_px"][3][0] = BID_SENTINEL_PRICE
        frame = make_event_frame(**d)
        stamps = Stamps(index=np.array([3]), ts_ns=np.array([TOY_TS_NS[3]]))
        out = mp.compute(frame, stamps)
        self.assertTrue(np.isnan(out["mp_micro"][0]))
        self.assertTrue(np.isnan(out["mp_dev_mid"][0]))
        self.assertTrue(np.isnan(out["mp_dev_wmid"][0]))
        self.assertTrue(np.isnan(out["mp_wmid_dev_mid"][0]))

    def test_empty_stamps_give_length_zero_float64_arrays_not_an_exception(self):
        frame = toy_event_frame()
        stamps = Stamps(index=np.array([], dtype=np.int64), ts_ns=np.array([], dtype=np.int64))
        out = mp.compute(frame, stamps)
        self.assertEqual(set(out.keys()), {"mp_micro", "mp_dev_mid", "mp_dev_wmid", "mp_wmid_dev_mid"})
        for k, v in out.items():
            self.assertEqual(v.shape, (0,), k)
            self.assertEqual(v.dtype, np.float64, k)


class TestBothClocks(unittest.TestCase):
    def test_calendar_and_event_clock_shapes_and_dtypes(self):
        frame = toy_event_frame()
        for clock in ("calendar", "event"):
            stamps = stamps_for_clock(frame, clock)
            out = mp.compute(frame, stamps)
            n = stamps.index.shape[0]
            for k, v in out.items():
                self.assertEqual(v.shape, (n,), f"{clock}:{k}")
                self.assertEqual(v.dtype, np.float64, f"{clock}:{k}")
                self.assertEqual(v.ndim, 1, f"{clock}:{k}")


class TestLookaheadOnToyBook(unittest.TestCase):
    """mp_wmid_dev_mid depends only on the book at/before each stamp, never on the fit, so it
    is the cleanest column for a targeted lookahead check on the tiny toy fixture."""

    def test_appending_future_events_leaves_earlier_stamps_bit_identical(self):
        full = toy_event_frame()
        d = toy_arrays()
        past = make_event_frame(**{key: col[:4] for key, col in d.items()})
        stamps_full = Stamps(index=np.array([0, 1, 2, 3]), ts_ns=np.array(TOY_TS_NS[:4]))
        stamps_past = Stamps(index=np.array([0, 1, 2, 3]), ts_ns=np.array(TOY_TS_NS[:4]))

        out_full = mp.compute(full, stamps_full)
        out_past = mp.compute(past, stamps_past)
        np.testing.assert_array_equal(
            out_full["mp_wmid_dev_mid"], out_past["mp_wmid_dev_mid"]
        )
        np.testing.assert_array_equal(out_full["mp_micro"], out_past["mp_micro"])  # both all-NaN


def _truncate_event_frame(frame, cutoff):
    """A prefix EventFrame built directly from the raw arrays (no pyarrow round trip)."""
    return EventFrame(
        ts_ns=frame.ts_ns[:cutoff],
        event_type=frame.event_type[:cutoff],
        direction=frame.direction[:cutoff],
        price=frame.price[:cutoff],
        size=frame.size[:cutoff],
        bid_px=frame.bid_px[:cutoff],
        bid_sz=frame.bid_sz[:cutoff],
        ask_px=frame.ask_px[:cutoff],
        ask_sz=frame.ask_sz[:cutoff],
    )


def _built_frame_present():
    return os.path.exists(frame_path("INTC"))


@unittest.skipUnless(
    _built_frame_present(),
    "built event frames not present under data/frames: the three full-day microprice "
    "properties (mp_micro is not wmid, block-0 NaN, refit-boundary causality) DID NOT RUN. "
    "They need a whole trading day and cannot be exercised on the toy fixture. Run "
    "`make data` to build the frames from the raw LOBSTER CSVs, then rerun.",
)
class TestRealIntcFrame(unittest.TestCase):
    """Property-style checks that only a full trading day can exercise: guards the "microprice
    is just a synonym for wmid" failure mode FEATURES.md sec. 3 warns about, and the block-fit
    causality boundary (targeted lookahead test for the refit machinery)."""

    @classmethod
    def setUpClass(cls):
        cls.frame = load_event_frame(frame_path("INTC"))
        cls.stamps = stamps_for_clock(cls.frame, "calendar")
        cls.out = mp.compute(cls.frame, cls.stamps)

    def test_micro_is_not_a_synonym_for_weighted_mid(self):
        book = book_at_stamps(self.frame, self.stamps)
        micro = self.out["mp_micro"]
        finite = np.isfinite(micro) & np.isfinite(book.wmid)
        self.assertGreater(finite.sum(), 1000, "test would be vacuous with too few finite rows")
        self.assertFalse(
            np.allclose(micro[finite], book.wmid[finite]),
            "mp_micro must not equal book.wmid -- see FEATURES.md sec. 3",
        )

    def test_block_zero_is_nan(self):
        blocks = refit_blocks(self.frame.ts_ns, self.stamps.ts_ns)
        self.assertEqual(blocks.fit_end_index[0], 0)  # INTERFACE.md sec. 8: block 0 has no prior data
        sel = blocks.stamp_block_id == 0
        self.assertGreater(sel.sum(), 1000)
        self.assertTrue(np.all(np.isnan(self.out["mp_micro"][sel])))
        self.assertTrue(np.all(np.isnan(self.out["mp_dev_mid"][sel])))
        self.assertTrue(np.all(np.isnan(self.out["mp_dev_wmid"][sel])))

    def test_refit_boundary_truncation_leaves_earlier_blocks_bit_identical(self):
        # Truncate strictly after block 1 ends (start of block 2, 10:30:00): every stamp in
        # blocks 0-1 must see a BIT-IDENTICAL mp_micro whether or not events from block 2
        # onward exist, because block 1's fit only reads rows strictly before block 1 starts
        # (10:00:00) -- far earlier than the truncation point.
        cutoff_ts = SESSION_START_NS + 2 * REFIT_BLOCK_NS
        cutoff_row = int(np.searchsorted(self.frame.ts_ns, cutoff_ts, side="left"))
        truncated = _truncate_event_frame(self.frame, cutoff_row)
        stamps_trunc = stamps_for_clock(truncated, "calendar")
        out_trunc = mp.compute(truncated, stamps_trunc)

        sel = self.stamps.ts_ns < cutoff_ts
        common_ts = self.stamps.ts_ns[sel]
        self.assertGreater(common_ts.size, 1000)

        idx_trunc = np.searchsorted(stamps_trunc.ts_ns, common_ts)
        np.testing.assert_array_equal(stamps_trunc.ts_ns[idx_trunc], common_ts)
        np.testing.assert_array_equal(
            self.out["mp_micro"][sel], out_trunc["mp_micro"][idx_trunc]
        )
        np.testing.assert_array_equal(
            self.out["mp_wmid_dev_mid"][sel], out_trunc["mp_wmid_dev_mid"][idx_trunc]
        )


if __name__ == "__main__":
    unittest.main()


def _long_martingale_table(n_block0=6000):
    """>= MIN_FIT_ROWS rows inside block 0, cycling the same 8-row A/B pattern as
    _build_martingale_frame, then one row inside block 1 to read the fitted output at.

    _build_martingale_frame is only 9 rows, far under MIN_FIT_ROWS = 5000, so it can only
    exercise `_estimate_block_adjustment` directly. This one is long enough to reach
    compute()'s public output, which is where a `micro = book.mid` degeneracy would hide.
    """
    # 0.25s apart keeps all n_block0 rows inside block 0 (1800s wide).
    step_ns = 250_000_000
    ts = [SESSION_START_NS + i * step_ns for i in range(n_block0)]
    # mid climbs 100 for four rows then falls 100 for four, so it stays bounded; bid-heavy
    # rows (state B) are the falling leg, bid-light rows (state A) the rising leg.
    cycle_mid = [100050, 100150, 100250, 100350, 100450, 100350, 100250, 100150]
    mids = [cycle_mid[i % 8] for i in range(n_block0)]
    bid_light = [(i % 8) < 4 for i in range(n_block0)]
    bid_sz = [[10] if light else [90] for light in bid_light]
    ask_sz = [[90] if light else [10] for light in bid_light]

    block1_start = SESSION_START_NS + REFIT_BLOCK_NS
    ts.append(block1_start + NS_PER_SEC)
    mids.append(100050)
    bid_sz.append([10])   # state A at the read stamp
    ask_sz.append([90])
    ts.append(block1_start + 2 * NS_PER_SEC)
    mids.append(100050)
    bid_sz.append([90])   # state B at the read stamp
    ask_sz.append([10])

    n = len(ts)
    return make_frame(
        ts_ns=ts,
        event_type=[1] * n,
        direction=[1] * n,
        price=[100000] * n,
        size=[10] * n,
        bid_px=[[m - 50] for m in mids],
        bid_sz=bid_sz,
        ask_px=[[m + 50] for m in mids],
        ask_sz=ask_sz,
    )


class TestMicroIsNotDegenerateAtComputeLevel(unittest.TestCase):
    """compute()-level guards on mp_micro / mp_dev_mid / mp_dev_wmid.

    Every value assertion in this file otherwise stops at `_estimate_block_adjustment`, so
    replacing `micro[sel] = book.mid[sel] + adjustment[...]` with `micro[sel] = book.mid[sel]`
    outright -- making mp_dev_mid identically zero -- passed the whole file.
    """

    @classmethod
    def setUpClass(cls):
        frame = event_frame_from_table(_long_martingale_table())
        n = frame.n_events
        idx = np.array([n - 2, n - 1], dtype=np.int64)  # the two block-1 rows
        cls.stamps = Stamps(index=idx, ts_ns=frame.ts_ns[idx])
        cls.out = mp.compute(frame, cls.stamps)

    def test_micro_is_not_the_mid(self):
        # If mp_micro were book.mid, this would be exactly 0.0 at every stamp.
        dev = self.out["mp_dev_mid"]
        self.assertTrue(np.all(np.isfinite(dev)), "block-1 stamps must be fitted, not NaN")
        self.assertTrue(np.all(dev != 0.0))

    def test_micro_is_not_the_weighted_mid(self):
        dev = self.out["mp_dev_wmid"]
        self.assertTrue(np.all(np.isfinite(dev)))
        self.assertTrue(np.all(dev != 0.0))

    def test_micro_is_emitted_in_ticks_not_raw_units(self):
        # INTERFACE.md sec. 1: an emitted price LEVEL is in ticks. The mid at both block-1
        # stamps is 100050 raw = 1000.5 ticks, and the microprice adjustment is on the order
        # of a tick, so mp_micro must land near 1000.5 -- not near 100050.
        micro = self.out["mp_micro"]
        self.assertTrue(np.all(np.isfinite(micro)))
        for value in micro:
            self.assertGreater(value, 990.0)
            self.assertLess(value, 1010.0)

    def test_bid_heavy_and_bid_light_states_disagree_in_sign(self):
        # The fixture is symmetric: state A (bid-light) sits on the rising leg and state B
        # (bid-heavy) on the falling leg, so their adjustments must have opposite signs. A
        # constant or state-independent micro cannot produce this.
        dev = self.out["mp_dev_mid"]
        self.assertGreater(dev[0] * dev[1], -np.inf)
        self.assertNotEqual(np.sign(dev[0]), np.sign(dev[1]))
