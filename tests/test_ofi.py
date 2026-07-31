"""Hand-computed tests for edge.features.ofi against tests/toybook.py's 8-event fixture.

FEATURES.md sec. 1 names three cases explicitly (bid add, bid cancel, aggressive sell
clears the bid) plus, per INTERFACE.md sec. 10, the ask-side mirror of the first so a
sign error on the ask half cannot hide. The toy fixture's 8 events are ~100us-500ms
apart, far tighter than FEATURE_WINDOWS' 100ms/1s/5s -- so a window at any one event's
own stamp would scoop up several neighbours at once. `_retimed_toy_frame` re-times the
SAME 8-event book several seconds apart (a toy_arrays()+make_event_frame() variant, per
tests/toybook.py's own instructions) purely so each case can be read off compute()'s
public output as a single isolated e_n, without re-implementing the formula a second way
to check it.

Depth normalisation and integrated-OFI PCA are FITTED statistics over >= MIN_FIT_ROWS
rows, which is too many to sum by hand digit-by-digit -- so `_multiblock_frame` uses
rows with CONSTANT sizes (closed-form average) to keep the one scaled-OFI literal this
file asserts genuinely hand-derivable, and otherwise tests those two features
structurally (bit-identical under frame truncation, forced-NaN in block 0, forced sign),
exactly the set INTERFACE.md sec. 10 asks for beyond the per-feature literals.
"""
import unittest

import numpy as np

import edge.features.ofi as ofi
from edge.features import NS_PER_SEC, REFIT_BLOCK_NS, SESSION_START_NS
from edge.features.book import book_at_stamps
from edge.features.clocks import Stamps, calendar_stamps, stamps_for_clock
from edge.features.frame import event_frame_from_table
from edge.features.windows import refit_blocks
from edge.io.lobster import ASK_SENTINEL_PRICE

from tests.toybook import make_event_frame, make_frame, toy_arrays, toy_event_frame


def _retimed_toy_frame(gap_ns=2 * NS_PER_SEC):
    """The 8-event toy book, re-timed `gap_ns` apart so a 100ms/1s window anchored at any
    one event's own stamp captures exactly that event -- its immediate predecessor is
    `gap_ns` (>> 5s window) earlier, so it falls outside every FEATURE_WINDOWS lookback."""
    d = toy_arrays()
    d["ts_ns"] = [SESSION_START_NS + i * gap_ns for i in range(8)]
    return make_event_frame(**d)


def _multiblock_table(n_fit=1000, include_future=True):
    """1 level, n_fit CONSTANT rows (bid_sz=100, ask_sz=80) filling block 0 entirely (so
    they are block 1's strictly-prior fit sample), then one bid-add event 1s into block
    1, then optionally two more "future" events (still block 1, then block 2) used only
    to prove block 1's fit does not see rows after its own start."""
    ts = [SESSION_START_NS + i * NS_PER_SEC for i in range(n_fit)]
    bid_sz = [100] * n_fit
    ask_sz = [80] * n_fit

    block1_start = SESSION_START_NS + REFIT_BLOCK_NS
    ts.append(block1_start + NS_PER_SEC)  # +30 bid add, 1s into block 1
    bid_sz.append(130)
    ask_sz.append(80)

    if include_future:
        ts.append(block1_start + 2 * NS_PER_SEC)  # still block 1
        bid_sz.append(200)
        ask_sz.append(80)

        block2_start = SESSION_START_NS + 2 * REFIT_BLOCK_NS
        ts.append(block2_start + NS_PER_SEC)  # block 2
        bid_sz.append(250)
        ask_sz.append(80)

    n = len(ts)
    return make_frame(
        ts_ns=ts,
        event_type=[1] * n,
        direction=[1] * n,
        price=[100000] * n,
        size=list(bid_sz),
        bid_px=[[100000]] * n,
        bid_sz=[[v] for v in bid_sz],
        ask_px=[[100500]] * n,
        ask_sz=[[v] for v in ask_sz],
    )


def _multiblock_frame(n_fit=1000, include_future=True):
    return event_frame_from_table(_multiblock_table(n_fit, include_future))


def _two_level_table(n_fit=1000, include_future=True):
    """TWO levels, with a DIFFERENT covariance structure before and after block 1's start.

    `_multiblock_table` is one level and constant, where the first PC of a single column is
    +-1 no matter what rows it is fitted on -- so a full-sample (leaking) PCA fit, a reversed
    level index and an eigenvector-orientation bug are all arithmetically invisible there.
    This fixture exists to give those three mutations somewhere to show up.

    Prices never move, so both CKS indicators fire at every level and the per-event
    contribution collapses to `e_n^(m) = delta q^b(m) - delta q^a(m)` (ask sizes are held
    constant here, so it is just the bid-size delta).

    Block 0 rows (block 1's strictly-prior fit sample): level 1 alternates by +-10 while
    level 2 alternates by +-100, so level 2 carries ~100x the variance and the first PC
    loads almost entirely on level 2.

    Rows at/after block 1's start: level 1 swings by +-100000, four orders of magnitude
    larger. A fit that (wrongly) includes them is dominated by level 1 and its first PC
    rotates to load almost entirely on level 1 instead. The two fits therefore produce
    visibly different `ofi_integrated_*`, which is what makes the leak detectable.
    """
    ts = [SESSION_START_NS + i * NS_PER_SEC for i in range(n_fit)]
    # Level 1 alternates 100/110 (+-10); level 2 alternates 200/300 (+-100).
    bid_sz = [[100 + 10 * (i % 2), 200 + 100 * (i % 2)] for i in range(n_fit)]

    block1_start = SESSION_START_NS + REFIT_BLOCK_NS
    ts.append(block1_start + NS_PER_SEC)
    bid_sz.append([100, 200])  # the stamp row itself: back to the level-1 baseline

    if include_future:
        # Deliberately violent level-1 swings, AFTER block 1's start. Only a fit sample
        # that reaches past the block boundary can see these.
        ts.append(block1_start + 2 * NS_PER_SEC)
        bid_sz.append([100100, 200])
        ts.append(block1_start + 3 * NS_PER_SEC)
        bid_sz.append([100, 200])
        block2_start = SESSION_START_NS + 2 * REFIT_BLOCK_NS
        ts.append(block2_start + NS_PER_SEC)
        bid_sz.append([100100, 200])

    n = len(ts)
    return make_frame(
        ts_ns=ts,
        event_type=[1] * n,
        direction=[1] * n,
        price=[100000] * n,
        size=[row[0] for row in bid_sz],
        bid_px=[[100000, 99900]] * n,
        bid_sz=bid_sz,
        ask_px=[[100500, 100600]] * n,
        ask_sz=[[80, 90]] * n,
    )


class TestFittedStatisticsAreNotLeaking(unittest.TestCase):
    """The three INTERFACE.md sec. 8 failure modes, on a fixture where they are visible.

    All three of these mutations previously survived the entire 229-test suite because the
    only multi-block fixture was one level deep.
    """

    def _integrated_at_block_one(self, include_future):
        table = _two_level_table(include_future=include_future)
        frame = event_frame_from_table(table)
        stamps = Stamps(index=np.array([1000]), ts_ns=np.array([frame.ts_ns[1000]]))
        return ofi.compute(frame, stamps)

    def test_pca_fit_sample_stops_at_the_block_boundary(self):
        # Block 1's loadings must come from rows strictly before block 1's start. Appending
        # violent level-1 rows AFTER that boundary must not move the block-1 stamp at all.
        # A `raw_win_events[:, :]` (full-sample) fit fails this: the appended rows rotate
        # the first PC from level 2 onto level 1.
        out_truncated = self._integrated_at_block_one(include_future=False)
        out_full = self._integrated_at_block_one(include_future=True)
        for key in ("ofi_integrated_100ms", "ofi_integrated_1s", "ofi_integrated_5s"):
            np.testing.assert_array_equal(out_full[key], out_truncated[key], err_msg=key)

    def test_the_fixture_actually_separates_the_two_fit_samples(self):
        # Guard on the guard: if the appended rows did NOT change what a full-sample fit
        # would produce, the test above would pass vacuously. Fit both samples directly and
        # assert the loadings genuinely differ, so the test above has something to detect.
        table = _two_level_table(include_future=True)
        frame = event_frame_from_table(table)
        contrib = ofi._event_contributions(
            frame.bid_px[:, :2].astype(np.float64),
            frame.bid_sz[:, :2].astype(np.float64),
            frame.ask_px[:, :2].astype(np.float64),
            frame.ask_sz[:, :2].astype(np.float64),
            frame.bid_valid[:, :2],
            frame.ask_valid[:, :2],
        )
        causal = ofi._estimate_first_pc(contrib[:1000, :])
        leaking = ofi._estimate_first_pc(contrib)
        # Causal fit: level 2 dominates -> |loading on level 2| > |loading on level 1|.
        self.assertGreater(abs(causal[1]), abs(causal[0]))
        # Leaking fit: level 1 dominates -> the ordering flips.
        self.assertGreater(abs(leaking[0]), abs(leaking[1]))

    def test_level_index_is_not_reversed(self):
        # ofi_l1 and ofi_l2 must track their OWN level. In the fit sample level 2 moves by
        # +-100 and level 1 by +-10, so the scaled columns cannot be interchangeable; a
        # `scaled[:, num_levels - m]` index reversal swaps them.
        table = _two_level_table(include_future=False)
        frame = event_frame_from_table(table)
        # Row 1000 is the block-1 row. The scaled `ofi_l{m}` columns are FITTED and so are
        # NaN throughout block 0 (no strictly-prior fit sample), which is why this has to be
        # read at a block-1 stamp rather than inside the fit sample itself.
        stamps = Stamps(index=np.array([1000]), ts_ns=np.array([frame.ts_ns[1000]]))
        out = ofi.compute(frame, stamps)
        # Hand-computed: row 999 (odd) is [110, 300], row 1000 is [100, 200], so bid level 1
        # goes 110 -> 100 (delta -10) and level 2 goes 300 -> 200 (delta -100); ask sizes are
        # constant so they contribute 0 at both levels. Row 999 is ~1799s earlier, far
        # outside the 1s window, so this single event is the whole window sum.
        self.assertEqual(out["ofi_raw_l1_1s"][0], -10.0)
        self.assertEqual(out["ofi_raw_l2_1s"][0], -100.0)
        # The SCALED columns are what a `scaled[:, num_levels - m]` reversal swaps, so they
        # have to be pinned too. Level 1's depth normaliser is (avg bid 105 + ask 80)/2 =
        # 92.5 and level 2's is (avg bid 250 + ask 90)/2 = 170, both positive, so the 10x
        # raw gap survives scaling: -10/92.5 = -0.108... vs -100/170 = -0.588...
        self.assertAlmostEqual(out["ofi_l1_1s"][0], -10.0 / 92.5, places=9)
        self.assertAlmostEqual(out["ofi_l2_1s"][0], -100.0 / 170.0, places=9)


class TestAskIndicatorOrientationOnAMovingAsk(unittest.TestCase):
    """The ask half of CKS eq. 4 at an ask price that actually MOVES.

    The existing ask-mirror test uses an unchanged ask price, where both indicators fire and
    swapping `>=` for `<=` is arithmetically a no-op -- so a flipped ask orientation survived
    the whole suite. Only a moving ask separates the two branches:

        ask price RISES  -> 1{Pa_n <= Pa_n-1} = 0, 1{Pa_n >= Pa_n-1} = 1 -> +q^a_n-1
                            (the offer retreats: sell-side pressure withdrawn, OFI positive)
        ask price FALLS  -> 1{Pa_n <= Pa_n-1} = 1, 1{Pa_n >= Pa_n-1} = 0 -> -q^a_n
                            (a new, tighter offer: sell-side pressure added, OFI negative)

    The sizes are chosen so q^a_n and q^a_n-1 differ at every step; if they were equal the
    flip would again be undetectable.
    """

    def setUp(self):
        # 2s apart so each 1s trailing window contains exactly one event. The bid side is
        # held completely constant, so every value below is the ask term alone.
        ts = [SESSION_START_NS + i * 2 * NS_PER_SEC for i in range(3)]
        table = make_frame(
            ts_ns=ts,
            event_type=[1] * 3,
            direction=[1] * 3,
            price=[100000] * 3,
            size=[50] * 3,
            bid_px=[[100000]] * 3,
            bid_sz=[[50]] * 3,
            ask_px=[[100100], [100200], [100050]],  # tied -> rise -> fall
            ask_sz=[[30], [70], [20]],
        )
        self.frame = event_frame_from_table(table)

    def test_ask_price_rise_contributes_the_previous_ask_size(self):
        # Event 1: ask 100100 -> 100200 (a rise), q^a_n-1 = 30, q^a_n = 70.
        # Bid is unchanged at 50 so the bid term is 50 - 50 = 0.
        # e_1 = 0 - 0*70 + 1*30 = +30. A flipped orientation would give -70 instead.
        stamps = Stamps(index=np.array([1]), ts_ns=np.array([self.frame.ts_ns[1]]))
        out = ofi.compute(self.frame, stamps)
        self.assertEqual(out["ofi_best_1s"][0], 30.0)

    def test_ask_price_fall_contributes_minus_the_current_ask_size(self):
        # Event 2: ask 100200 -> 100050 (a fall), q^a_n-1 = 70, q^a_n = 20.
        # e_2 = 0 - 1*20 + 0*70 = -20. A flipped orientation would give +70 instead.
        stamps = Stamps(index=np.array([2]), ts_ns=np.array([self.frame.ts_ns[2]]))
        out = ofi.compute(self.frame, stamps)
        self.assertEqual(out["ofi_best_1s"][0], -20.0)


class TestFirstPCOrientation(unittest.TestCase):
    """`eigvecs[:, -1]` (top eigenVECTOR) vs `eigvecs[-1, :]` (last ROW) -- a classic `eigh`
    slip that is invisible at 2 levels, where the two differ only by a sign the level-1
    normalisation then removes. It takes a 3-column fit matrix to separate them."""

    def test_hand_derived_top_eigenvector_of_a_three_level_fit_matrix(self):
        # x is already mean-zero per column (col sums 0, 0, 0), n - 1 = 3.
        #   x^T x = [[2,2,0],[2,4,2],[0,2,2]]  ->  cov = (1/3) * that
        # Eigenvalues of [[1,1,0],[1,2,1],[0,1,1]] (the 1/2 multiple) factor as
        #   det = (1-L)*L*(L-3)  ->  L in {0, 1, 3}; the top is L = 3.
        # (M - 3I)v = 0:  -2v0 + v1 = 0 -> v1 = 2v0;  v1 - 2v2 = 0 -> v2 = v0
        #   -> v proportional to [1, 2, 1], normalised by sqrt(1+4+1) = sqrt(6).
        x = np.array(
            [[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [0.0, 1.0, 1.0], [0.0, -1.0, -1.0]]
        )
        loadings = ofi._estimate_first_pc(x)
        r6 = np.sqrt(6.0)
        self.assertAlmostEqual(loadings[0], 1.0 / r6, places=9)
        self.assertAlmostEqual(loadings[1], 2.0 / r6, places=9)
        self.assertAlmostEqual(loadings[2], 1.0 / r6, places=9)
        # The last ROW of the eigenvector matrix is [1/sqrt(3), -1/sqrt(2), 1/sqrt(6)] here,
        # so the middle loading is the coordinate that actually separates the two readings.
        self.assertGreater(loadings[1], 0.0)


class TestBestLevelHandCases(unittest.TestCase):
    """FEATURES.md sec. 1's three named cases, plus the ask-side mirror of the first."""

    def test_bid_side_limit_add_at_unchanged_best_is_plus_size(self):
        frame = _retimed_toy_frame()
        # Toy event 1 (0-indexed): bid_px_1 unchanged at 100500, bid_sz_1 10 -> 30; ask
        # unchanged (101000 x 50 both rows).
        # e_1 = 1{100500>=100500}*30 - 1{100500<=100500}*10   (both fire, price tied)
        #     - 1{101000<=101000}*50 + 1{101000>=101000}*50   (both fire, ask tied)
        #     = (30 - 10) - (50 - 50) = 20 - 0 = 20.
        stamps = Stamps(index=np.array([1]), ts_ns=np.array([frame.ts_ns[1]]))
        out = ofi.compute(frame, stamps)
        self.assertEqual(out["ofi_best_100ms"][0], 20.0)
        self.assertEqual(out["ofi_best_1s"][0], 20.0)
        self.assertEqual(out["ofi_raw_l1_1s"][0], 20.0)

    def test_bid_side_partial_cancel_at_unchanged_best_is_minus_size(self):
        frame = _retimed_toy_frame()
        # Toy event 2: bid_px_1 unchanged at 100500, bid_sz_1 30 -> 25; ask unchanged.
        # e_2 = (25 - 30) - (50 - 50) = -5 - 0 = -5.
        stamps = Stamps(index=np.array([2]), ts_ns=np.array([frame.ts_ns[2]]))
        out = ofi.compute(frame, stamps)
        self.assertEqual(out["ofi_best_100ms"][0], -5.0)
        self.assertEqual(out["ofi_best_1s"][0], -5.0)
        self.assertEqual(out["ofi_raw_l1_1s"][0], -5.0)

    def test_aggressive_sell_that_clears_the_bid_is_the_price_improvement_branch(self):
        # Not one of the 8 toy events -- every bid-side price change there comes from a
        # cancel/delete, not a type-4 execution (tests/toybook.py's docstring) -- so this
        # is built inline via make_event_frame per its own instructions.
        # Row 0: bid touch 100500 x 30, ask touch 101000 x 50.
        # Row 1: event_type=4 (visible execution), direction=+1. DATA_CONTRACT.md:
        # direction is the STANDING order's side, so +1 here means a resting BID limit
        # order was executed -- the aggressor was a SELL market order -- price=100500,
        # size=30 fully clears the level, so the bid touch drops to 100000.
        # e_1: Pb 100500 -> 100000 (a DROP) -> 1{100000>=100500}=0, 1{100000<=100500}=1
        #      -> bid term = 0*40 - 1*30 = -30 (the WHOLE prior queue is treated as
        #      removed -- the "price improvement" branch -- not just the traded size;
        #      they coincide here because the trade emptied the level exactly).
        #      Ask side unchanged (101000 x 50 both rows) -> ask term = 50 - 50 = 0.
        #      e_1 = -30 + 0 = -30.
        frame = make_event_frame(
            ts_ns=[SESSION_START_NS, SESSION_START_NS + 2 * NS_PER_SEC],
            event_type=[1, 4],
            direction=[1, 1],
            price=[100500, 100500],
            size=[30, 30],
            bid_px=[[100500, 100000], [100000, 98000]],
            bid_sz=[[30, 40], [40, 15]],
            ask_px=[[101000, 102000], [101000, 102000]],
            ask_sz=[[50, 30], [50, 30]],
        )
        stamps = Stamps(index=np.array([1]), ts_ns=np.array([frame.ts_ns[1]]))
        out = ofi.compute(frame, stamps)
        self.assertEqual(out["ofi_best_100ms"][0], -30.0)
        self.assertEqual(out["ofi_best_1s"][0], -30.0)
        self.assertEqual(out["ofi_raw_l1_1s"][0], -30.0)

    def test_ask_side_limit_add_at_unchanged_best_is_minus_size_the_mirror_of_case_one(self):
        # Row 0: ask touch 101000 x 40, bid touch 100500 x 30 (bid unused here).
        # Row 1: event_type=1 (new limit order), direction=-1 (ask side), price=101000,
        # size=15 -- joins the queue at the unchanged best, ask_sz_1 40 -> 55.
        # e_1: bid unchanged (100500 x 30 both rows) -> bid term = 30 - 30 = 0.
        #      Ask price tied -> both indicators fire:
        #      ask term = -1{101000<=101000}*55 + 1{101000>=101000}*40 = -55 + 40 = -15.
        #      An ask ADD is NEGATIVE OFI -- the mirror of case 1's bid ADD being
        #      positive; a swapped >=/<= on this half would silently give +15 instead.
        #      e_1 = 0 + (-15) = -15.
        frame = make_event_frame(
            ts_ns=[SESSION_START_NS, SESSION_START_NS + 2 * NS_PER_SEC],
            event_type=[1, 1],
            direction=[1, -1],
            price=[100500, 101000],
            size=[30, 15],
            bid_px=[[100500, 100000], [100500, 100000]],
            bid_sz=[[30, 40], [30, 40]],
            ask_px=[[101000, 102000], [101000, 102000]],
            ask_sz=[[40, 30], [55, 30]],
        )
        stamps = Stamps(index=np.array([1]), ts_ns=np.array([frame.ts_ns[1]]))
        out = ofi.compute(frame, stamps)
        self.assertEqual(out["ofi_best_100ms"][0], -15.0)
        self.assertEqual(out["ofi_best_1s"][0], -15.0)
        self.assertEqual(out["ofi_raw_l1_1s"][0], -15.0)


class TestRawLevel1MatchesBest(unittest.TestCase):
    def test_ofi_raw_l1_1s_is_bit_identical_to_ofi_best_1s(self):
        # Both are sliced from the very same w="1s" per-level window-sum array inside
        # compute() -- pinned so a refactor that recomputes one of them separately
        # cannot silently drift.
        frame = toy_event_frame()
        stamps = calendar_stamps(frame, SESSION_START_NS, SESSION_START_NS + 500_000_000)
        out = ofi.compute(frame, stamps)
        np.testing.assert_array_equal(out["ofi_raw_l1_1s"], out["ofi_best_1s"])


class TestEmptyWindowSumIsZeroNotNaN(unittest.TestCase):
    def test_quiet_period_beyond_the_window_gives_zero_not_nan(self):
        frame = toy_event_frame()
        # Last toy event is at 34200.400000000s. A stamp at 34200.800s with a 100ms
        # window looks back to (34200.700s, 34200.800s], which contains no frame rows at
        # all (the last event is 400ms further back than that) -- an empty window's SUM
        # is legitimately 0.0 (INTERFACE.md sec. 7's "sum over an empty window" rule),
        # not NaN.
        stamps = Stamps(index=np.array([7]), ts_ns=np.array([SESSION_START_NS + 800_000_000]))
        out = ofi.compute(frame, stamps)
        self.assertEqual(out["ofi_best_100ms"][0], 0.0)
        self.assertFalse(np.isnan(out["ofi_best_100ms"][0]))


class TestInvalidLevelPoisonsBothAdjacentTransitions(unittest.TestCase):
    def test_sentinel_row_drops_its_own_and_the_next_transition_from_the_sum(self):
        # Row 0: valid touch, bid 100500 x 30 / ask 101000 x 50.
        # Row 1: ask goes to the LOBSTER sentinel with size 0 -- ask_valid[1]=False, so
        # e_1 (row1 vs row0) is undefined -> NaN, per this module's per-level validity
        # rule (frame.bid_valid/ask_valid, joint over both sides at both rows).
        # Row 2: ask is valid again (101000 x 50), bid changed 30 -> 45 -- but e_2
        # compares row2 to row1, and row1 was invalid, so e_2 is ALSO NaN. One bad tick
        # poisons both the transition INTO it and the transition OUT of it.
        frame = make_event_frame(
            ts_ns=[
                SESSION_START_NS,
                SESSION_START_NS + 2 * NS_PER_SEC,
                SESSION_START_NS + 4 * NS_PER_SEC,
            ],
            event_type=[1, 3, 1],
            direction=[1, -1, 1],
            price=[100500, 101000, 100500],
            size=[30, 50, 15],
            bid_px=[[100500, 100000], [100500, 100000], [100500, 100000]],
            bid_sz=[[30, 40], [35, 40], [45, 40]],
            ask_px=[[101000, 102000], [ASK_SENTINEL_PRICE, 102000], [101000, 102000]],
            ask_sz=[[50, 30], [0, 30], [50, 30]],
        )
        stamps = Stamps(index=np.array([2]), ts_ns=np.array([frame.ts_ns[2]]))
        out = ofi.compute(frame, stamps)
        # window="5s" reaches back past all 3 events (lower_bound is before row 0's ts).
        # Only e_0=0 (defined, no predecessor) survives; e_1 and e_2 are NaN and
        # windows.prefix_sums drops NaN contributions from the sum, so the observed raw
        # sum is 0.0 -- the two real transitions vanished silently, not a crash and not
        # a spurious nonzero number.
        self.assertEqual(out["ofi_best_5s"][0], 0.0)


class TestDepthNormaliserAndIntegratedLiteral(unittest.TestCase):
    """Block 1's fit window is 1000 CONSTANT rows (bid_sz=100, ask_sz=80), so its average
    depth has the closed form (100+80)/2 = 90 -- a hand-computable literal even though
    MIN_FIT_ROWS=1000 rows is too many to sum by hand digit-by-digit."""

    def setUp(self):
        self.frame = _multiblock_frame(n_fit=1000, include_future=False)
        # Row 1000 (0-indexed) is 1s into block 1: bid_sz 100 -> 130 (+30 add), price and
        # ask unchanged. Its predecessor (row 999) is 802s earlier (999s vs 1801s after
        # session start), far outside a 1s window, so a 1s window at this stamp isolates
        # exactly this one event: raw ofi_best_1s = 30.0.
        self.stamps = Stamps(index=np.array([1000]), ts_ns=np.array([self.frame.ts_ns[1000]]))

    def test_fit_end_is_exactly_min_fit_rows(self):
        blocks = refit_blocks(self.frame.ts_ns, self.stamps.ts_ns)
        self.assertEqual(blocks.fit_end_index.tolist(), [1000])

    def test_raw_ofi_is_the_isolated_bid_add(self):
        out = ofi.compute(self.frame, self.stamps)
        self.assertEqual(out["ofi_best_1s"][0], 30.0)
        self.assertEqual(out["ofi_raw_l1_1s"][0], 30.0)

    def test_depth_scaled_ofi_matches_the_closed_form_average_depth(self):
        # avg depth = mean_{i=0..999} (bid_sz_i + ask_sz_i)/2 = (100+80)/2 = 90 exactly
        # (every one of the 1000 fit rows is identical, so the mean equals any one row's
        # value -- no need to sum 1000 numbers by hand). ofi_l1_1s = 30 / 90 = 1/3.
        out = ofi.compute(self.frame, self.stamps)
        self.assertAlmostEqual(out["ofi_l1_1s"][0], 30.0 / 90.0, places=12)

    def test_integrated_ofi_equals_the_scaled_ofi_at_l_equals_1(self):
        # L=1: the "first PC" of a single column is that column itself. Its raw
        # direction is +-[1.0] (a 1x1 covariance has no ambiguity beyond sign), and the
        # sign fix forces +1.0, so ofi_integrated_1s must equal ofi_l1_1s exactly.
        out = ofi.compute(self.frame, self.stamps)
        self.assertEqual(out["ofi_integrated_1s"][0], out["ofi_l1_1s"][0])


class TestBlockZeroIsNaN(unittest.TestCase):
    def test_a_stamp_in_block_zero_has_no_prior_data_and_is_nan(self):
        frame = _multiblock_frame(n_fit=1000, include_future=False)
        # Row 500 (500s after session start) is inside block 0 ([0, 1800s)); block 0's
        # fit slice is frame rows strictly before the SESSION START, which is empty.
        stamps = Stamps(index=np.array([500]), ts_ns=np.array([frame.ts_ns[500]]))
        out = ofi.compute(frame, stamps)
        self.assertTrue(np.isnan(out["ofi_l1_1s"][0]))
        self.assertTrue(np.isnan(out["ofi_integrated_1s"][0]))
        # The RAW (unfitted) column is still well-defined in block 0 -- nothing changes
        # across these constant rows, so it is legitimately 0.0, not NaN.
        self.assertEqual(out["ofi_best_1s"][0], 0.0)


class TestRefitBoundaryAndLookahead(unittest.TestCase):
    def test_truncating_the_frame_after_block_one_leaves_block_one_bit_identical(self):
        # This is both required checks at once: the general lookahead-truncation test
        # (surviving stamps bit-identical after appending/removing future rows) AND the
        # refit-boundary test (block 1's fit -- rows strictly before block 1's start --
        # must not see block 1's OWN later rows or block 2 at all).
        full_table = _multiblock_table(n_fit=1000, include_future=True)  # 1003 rows
        truncated_table = full_table.slice(0, 1001)  # rows 0..1000 only

        frame_full = event_frame_from_table(full_table)
        frame_truncated = event_frame_from_table(truncated_table)
        self.assertEqual(frame_truncated.ts_ns[1000], frame_full.ts_ns[1000])

        stamps = Stamps(index=np.array([1000]), ts_ns=np.array([frame_full.ts_ns[1000]]))
        out_full = ofi.compute(frame_full, stamps)
        out_truncated = ofi.compute(frame_truncated, stamps)
        for key in (
            "ofi_best_1s",
            "ofi_deep_1s",
            "ofi_raw_l1_1s",
            "ofi_l1_1s",
            "ofi_l1_100ms",
            "ofi_integrated_1s",
            "ofi_integrated_100ms",
        ):
            np.testing.assert_array_equal(out_full[key], out_truncated[key], err_msg=key)


class TestPCASignStability(unittest.TestCase):
    def test_hand_derived_eigenvector_sign_is_forced_positive_on_level_one(self):
        # x1 = [1,-1,1,-1], x2 = -x1 (perfectly anti-correlated, already mean-zero).
        # cov = x^T x / (n-1): sum(x1^2)=4, sum(x2^2)=4, sum(x1*x2)=-4, n-1=3
        #   -> cov = [[4/3, -4/3], [-4/3, 4/3]].
        # Eigenvalues of [[a,-a],[-a,a]] are 0 and 2a = 8/3. Eigenvector for 2a solves
        #   (a - 2a) v1 - a v2 = 0  ->  -a v1 = a v2  ->  v1 = -v2, direction [1,-1],
        #   normalised: [1/sqrt(2), -1/sqrt(2)].
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
        loadings = ofi._estimate_first_pc(x)
        self.assertAlmostEqual(loadings[0], 1.0 / np.sqrt(2), places=9)
        self.assertAlmostEqual(loadings[1], -1.0 / np.sqrt(2), places=9)

    def test_negated_input_gives_the_identical_signed_output(self):
        # Covariance is invariant under a global sign flip of the data ((-x)^T(-x) =
        # x^T x exactly), so this pins the CONSEQUENCE of the sign fix: two inputs an
        # unfixed PCA could return with opposite eigenvector signs (depending on
        # whatever LAPACK happens to do) must come out bit-identical here, because the
        # fix is a deterministic, data-independent convention -- not a coincidence of
        # this particular covariance.
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
        loadings_pos = ofi._estimate_first_pc(x)
        loadings_neg = ofi._estimate_first_pc(-x)
        np.testing.assert_array_equal(loadings_pos, loadings_neg)
        self.assertGreaterEqual(loadings_pos[0], 0.0)

    def test_forced_nonnegative_on_level_one_holds_across_several_matrices(self):
        rng = np.random.default_rng(0)
        for trial in range(6):
            x = rng.normal(size=(50, 4))
            loadings = ofi._estimate_first_pc(x)
            self.assertGreaterEqual(loadings[0], 0.0, msg=f"trial {trial}")
            self.assertAlmostEqual(
                float(np.linalg.norm(loadings)), 1.0, places=9, msg=f"trial {trial}"
            )


class TestBothClocksAndBookNone(unittest.TestCase):
    def setUp(self):
        self.frame = toy_event_frame()

    def test_runs_on_both_clocks_frozen_dispatcher(self):
        for clock in ("calendar", "event"):
            stamps = stamps_for_clock(self.frame, clock)
            out = ofi.compute(self.frame, stamps)
            n = stamps.index.shape[0]
            for key, arr in out.items():
                self.assertEqual(arr.dtype, np.float64, (clock, key))
                self.assertEqual(arr.shape, (n,), (clock, key))

    def test_event_clock_n10_on_the_8event_toy_fixture_is_legitimately_empty(self):
        # EVENT_CLOCK_N=10 but the toy fixture has only 7 book-changing events (the
        # type-5 row at index 5 doesn't tick the clock) -- event_stamps returns an empty
        # Stamps, and the contract requires that not to raise and to give length-0
        # float64 arrays for every column.
        stamps = stamps_for_clock(self.frame, "event")
        self.assertEqual(stamps.index.shape[0], 0)
        out = ofi.compute(self.frame, stamps)
        for key, arr in out.items():
            self.assertEqual(arr.shape, (0,), key)
            self.assertEqual(arr.dtype, np.float64, key)

    def test_book_none_matches_book_supplied_explicitly(self):
        stamps = calendar_stamps(self.frame, SESSION_START_NS, SESSION_START_NS + 500_000_000)
        book = book_at_stamps(self.frame, stamps)
        out_default = ofi.compute(self.frame, stamps)
        out_explicit = ofi.compute(self.frame, stamps, book=book)
        self.assertEqual(set(out_default.keys()), set(out_explicit.keys()))
        for key in out_default:
            np.testing.assert_array_equal(out_default[key], out_explicit[key], err_msg=key)


class TestColumnContract(unittest.TestCase):
    def setUp(self):
        self.frame = toy_event_frame()  # L=2
        self.stamps = calendar_stamps(self.frame, SESSION_START_NS, SESSION_START_NS + 500_000_000)

    def test_column_count_and_names_at_l_equals_2(self):
        out = ofi.compute(self.frame, self.stamps)
        # 9 + 4*L columns: 3 ofi_best_w, 3 ofi_deep_w, L ofi_raw_l{m}_1s, 3*L
        # ofi_l{m}_w, 3 ofi_integrated_w -- L=2 here -> 9 + 8 = 17.
        expected = set()
        for w in ("100ms", "1s", "5s"):
            expected.add(f"ofi_best_{w}")
            expected.add(f"ofi_deep_{w}")
            expected.add(f"ofi_integrated_{w}")
            for m in (1, 2):
                expected.add(f"ofi_l{m}_{w}")
        expected.add("ofi_raw_l1_1s")
        expected.add("ofi_raw_l2_1s")
        self.assertEqual(set(out.keys()), expected)
        self.assertEqual(len(expected), 17)

    def test_every_column_is_1d_float64_of_stamp_length_and_owns_the_prefix(self):
        out = ofi.compute(self.frame, self.stamps)
        n = self.stamps.index.shape[0]
        for key, arr in out.items():
            self.assertTrue(key.startswith("ofi_"), key)
            self.assertNotIn("_sadj", key)
            self.assertEqual(arr.dtype, np.float64, key)
            self.assertEqual(arr.shape, (n,), key)


if __name__ == "__main__":
    unittest.main()
