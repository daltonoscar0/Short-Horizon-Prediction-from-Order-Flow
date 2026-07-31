"""Hand-computed tests for queue/book imbalance (src/edge/features/queue.py).

Every expected value below is a literal computed by hand in the comment above the assertion,
against `tests/toybook.py`'s shared 8-event/2-level fixture (or a variant built from
`toy_arrays()` for a case the 8 events don't contain), per docs/INTERFACE.md sec. 10.

Toy fixture touch sizes (post-event, from tests/toybook.py / tests/test_book.py):
  bid_sz_1 = [10, 30, 25, 40, 40, 40, 40, 40]
  ask_sz_1 = [50, 50, 50, 50, 40, 40, 40, 40]
  bid_sz_2 = [40, 40, 40, 15, 15, 15, 15, 15]
  ask_sz_2 = [30, 30, 30, 30, 30, 30, 30, 15]
All 8 rows are valid at both levels (tests/test_book.py: `valid_l.all()` is True), so none of
these literals is a NaN case; NaN rules get their own variant-frame tests below.
"""
import unittest

import numpy as np

from edge.features.book import book_at_stamps
from edge.features.clocks import Stamps, calendar_stamps, stamps_for_clock
from edge.features.queue import EXP_DECAY_LAMBDA, compute
from tests.toybook import TOY_TS_NS, make_event_frame, toy_arrays, toy_event_frame

SESSION_START = 34_200_000_000_000
GRID_150MS = 150_000_000


def _all_event_stamps(frame):
    """One stamp per frame row -- the direct `<= t` case, ts_ns[i] == frame.ts_ns[i]."""
    n = frame.n_events
    return Stamps(index=np.arange(n, dtype=np.int64), ts_ns=np.asarray(frame.ts_ns[:n]))


class TestQueueImbalanceTouch(unittest.TestCase):
    """FEATURES.md sec. 2: I = (q^b - q^a) / (q^b + q^a) at the touch."""

    def setUp(self):
        self.frame = toy_event_frame()
        self.out = compute(self.frame, _all_event_stamps(self.frame))

    def test_hand_computed_touch_imbalance_all_eight_events(self):
        # event 0: (10 - 50)/(10 + 50)  = -40/60 = -0.6666666666666666
        # event 1: (30 - 50)/(30 + 50)  = -20/80 = -0.25
        # event 2: (25 - 50)/(25 + 50)  = -25/75 = -0.3333333333333333
        # event 3: (40 - 50)/(40 + 50)  = -10/90 = -0.1111111111111111
        # event 4: (40 - 40)/(40 + 40)  =   0/80 = 0.0
        # event 5: (40 - 40)/(40 + 40)  =   0/80 = 0.0   (type-5 event, book unchanged from 4)
        # event 6: (40 - 40)/(40 + 40)  =   0/80 = 0.0
        # event 7: (40 - 40)/(40 + 40)  =   0/80 = 0.0
        expected = [
            -40 / 60,
            -20 / 80,
            -25 / 75,
            -10 / 90,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        np.testing.assert_allclose(self.out["qi_touch"], expected, rtol=0, atol=1e-15)
        # And the literal float64 values, spelled out (not just allclose):
        self.assertEqual(self.out["qi_touch"][0], -0.6666666666666666)
        self.assertEqual(self.out["qi_touch"][1], -0.25)
        self.assertEqual(self.out["qi_touch"][2], -0.3333333333333333)
        self.assertEqual(self.out["qi_touch"][3], -0.1111111111111111)
        self.assertEqual(self.out["qi_touch"][7], 0.0)


class TestQueueImbalanceSign(unittest.TestCase):
    """sign(I) is its own column (FEATURES.md sec. 2: the Phase-2 L1 baseline)."""

    def setUp(self):
        self.frame = toy_event_frame()
        self.out = compute(self.frame, _all_event_stamps(self.frame))

    def test_hand_computed_sign_all_eight_events(self):
        # sign(-0.6666..)=-1, sign(-0.25)=-1, sign(-0.3333..)=-1, sign(-0.1111..)=-1,
        # sign(0.0)=0.0 (exactly-zero imbalance) for events 4-7.
        self.assertEqual(self.out["qi_sign"].tolist(), [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0])

    def test_exact_zero_imbalance_maps_to_zero_not_a_coin_flip(self):
        # Event 4: bid_sz_1 == ask_sz_1 == 40 -> qi_touch == 0.0 exactly -> qi_sign == 0.0,
        # not +1.0/-1.0. np.sign(0.0) == 0.0 is the guarantee this pins.
        self.assertEqual(self.out["qi_touch"][4], 0.0)
        self.assertEqual(self.out["qi_sign"][4], 0.0)
        # dtype is float64 with values drawn from {-1.0, 0.0, 1.0} only.
        self.assertEqual(self.out["qi_sign"].dtype, np.float64)
        self.assertTrue(set(self.out["qi_sign"].tolist()).issubset({-1.0, 0.0, 1.0}))


class TestQueueImbalanceLevels(unittest.TestCase):
    """qi_l{m}: the same touch ratio computed at level m alone."""

    def setUp(self):
        self.frame = toy_event_frame()
        self.out = compute(self.frame, _all_event_stamps(self.frame))

    def test_qi_l1_is_bit_identical_to_qi_touch(self):
        # book.bid_sz_l[:, 0] IS book.bid_sz (same underlying values), so the level-1 ratio
        # must be bit-identical to the touch ratio, not merely close.
        np.testing.assert_array_equal(self.out["qi_l1"], self.out["qi_touch"])

    def test_hand_computed_level_2_imbalance(self):
        # event 0: bid_sz_2=40, ask_sz_2=30 -> (40-30)/(40+30) = 10/70 = 0.14285714285714285
        # event 3: bid_sz_2=15, ask_sz_2=30 -> (15-30)/(15+30) = -15/45 = -0.3333333333333333
        # event 7: bid_sz_2=15, ask_sz_2=15 -> (15-15)/(15+15) = 0/30 = 0.0
        self.assertEqual(self.out["qi_l2"][0], 10 / 70)
        self.assertEqual(self.out["qi_l2"][0], 0.14285714285714285)
        self.assertEqual(self.out["qi_l2"][3], -15 / 45)
        self.assertEqual(self.out["qi_l2"][3], -0.3333333333333333)
        self.assertEqual(self.out["qi_l2"][7], 0.0)

    def test_no_qi_l3_on_the_two_level_toy_fixture(self):
        # book.n_levels caps the loop at 2 here; a hardcoded 10 would either IndexError or
        # silently emit a bogus qi_l3..qi_l10.
        self.assertEqual(self.frame.num_levels, 2)
        self.assertNotIn("qi_l3", self.out)
        self.assertEqual(sorted(k for k in self.out if k.startswith("qi_l")), ["qi_l1", "qi_l2"])


class TestDepthWeightedImbalance(unittest.TestCase):
    """I_L with the three frozen weight schemes, derived by hand from the L=2 fixture."""

    def setUp(self):
        self.frame = toy_event_frame()
        self.out = compute(self.frame, _all_event_stamps(self.frame))

    def test_hand_computed_inverse_depth_weight_event_0(self):
        # w_1 = 1/1 = 1.0, w_2 = 1/2 = 0.5 (qi_depth_inv weight rule).
        # event 0: bid_l = [10, 40], ask_l = [50, 30].
        # weighted bid = 10*1.0 + 40*0.5 = 10 + 20 = 30.0
        # weighted ask = 50*1.0 + 30*0.5 = 50 + 15 = 65.0
        # numer = 30 - 65 = -35.0, denom = 30 + 65 = 95.0, ratio = -35/95 = -0.3684210526315789
        self.assertEqual(self.out["qi_depth_inv"][0], -35.0 / 95.0)
        self.assertEqual(self.out["qi_depth_inv"][0], -0.3684210526315789)

    def test_hand_computed_inverse_depth_weight_event_3(self):
        # event 3: bid_l = [40, 15], ask_l = [50, 30], w_1=1.0, w_2=0.5.
        # weighted bid = 40*1.0 + 15*0.5 = 40 + 7.5 = 47.5
        # weighted ask = 50*1.0 + 30*0.5 = 50 + 15 = 65.0
        # numer = 47.5 - 65.0 = -17.5, denom = 47.5 + 65.0 = 112.5
        # ratio = -17.5/112.5 = -0.15555555555555556
        self.assertEqual(self.out["qi_depth_inv"][3], -17.5 / 112.5)
        self.assertEqual(self.out["qi_depth_inv"][3], -0.15555555555555556)

    def test_hand_computed_exp_depth_weight_event_0(self):
        # EXP_DECAY_LAMBDA = 3.0 (frozen a priori, see queue.py docstring).
        # w_1 = exp(-(1-1)/3) = exp(0) = 1.0
        # w_2 = exp(-(2-1)/3) = exp(-1/3) = 0.7165313105737893
        self.assertEqual(EXP_DECAY_LAMBDA, 3.0)
        w1 = np.exp(-(1 - 1) / EXP_DECAY_LAMBDA)
        w2 = np.exp(-(2 - 1) / EXP_DECAY_LAMBDA)
        self.assertEqual(w1, 1.0)
        self.assertEqual(w2, 0.7165313105737893)
        # event 0: bid_l = [10, 40], ask_l = [50, 30].
        # weighted bid = 10*1.0 + 40*0.7165313105737893 = 10 + 28.66125242295157 = 38.66125242295157
        # weighted ask = 50*1.0 + 30*0.7165313105737893 = 50 + 21.49593931721368 = 71.49593931721368
        # numer = 38.66125242295157 - 71.49593931721368 = -32.83468689426211
        # denom = 38.66125242295157 + 71.49593931721368 = 110.15719174016525
        # ratio = -32.83468689426211 / 110.15719174016525 = -0.298071205116697
        bid_w = 10 * w1 + 40 * w2
        ask_w = 50 * w1 + 30 * w2
        self.assertAlmostEqual(self.out["qi_depth_exp"][0], (bid_w - ask_w) / (bid_w + ask_w), places=12)
        self.assertAlmostEqual(self.out["qi_depth_exp"][0], -0.298071205116697, places=12)

    def test_hand_computed_uniform_depth_weight_event_0(self):
        # w_1 = w_2 = 1.0 (qi_depth_uniform: full unweighted total depth at each side).
        # event 0: weighted bid = 10 + 40 = 50.0, weighted ask = 50 + 30 = 80.0
        # numer = 50 - 80 = -30.0, denom = 50 + 80 = 130.0, ratio = -30/130 = -0.23076923076923078
        self.assertEqual(self.out["qi_depth_uniform"][0], -30.0 / 130.0)
        self.assertEqual(self.out["qi_depth_uniform"][0], -0.23076923076923078)


class TestNaNRules(unittest.TestCase):
    """One constructed variant per NaN rule this module claims (INTERFACE.md sec. 10)."""

    def test_invalid_touch_gives_nan_touch_and_sign(self):
        # Variant: event 4's sizes zeroed on BOTH sides, BOTH levels (prices left alone) --
        # book.py's validity rule requires bid_sz>0 and ask_sz>0, so this makes the touch
        # (and both levels) invalid without touching any sentinel price.
        d = toy_arrays()
        d["bid_sz"][4] = [0, 0]
        d["ask_sz"][4] = [0, 0]
        frame = make_event_frame(**d)
        out = compute(frame, _all_event_stamps(frame))

        book = book_at_stamps(frame, _all_event_stamps(frame))
        self.assertFalse(bool(book.valid[4]))
        self.assertTrue(np.isnan(out["qi_touch"][4]))
        self.assertTrue(np.isnan(out["qi_sign"][4]))
        # Neighbouring events are untouched by the edit.
        self.assertEqual(out["qi_touch"][3], -10 / 90)

    def test_zero_total_denominator_gives_nan_depth_weighted_not_zero_over_zero(self):
        # Same variant as above: event 4 has zero valid levels on either side, so every
        # depth-weighted sum (uniform/inv/exp) reduces to 0.0/0.0. The safe-ratio guard must
        # turn that into NaN, never a raw 0/0 (which numpy would silently emit as NaN with a
        # warning under an unguarded division, or as 0 under a `where=` that defaulted to 0).
        d = toy_arrays()
        d["bid_sz"][4] = [0, 0]
        d["ask_sz"][4] = [0, 0]
        frame = make_event_frame(**d)
        out = compute(frame, _all_event_stamps(frame))

        self.assertTrue(np.isnan(out["qi_l1"][4]))
        self.assertTrue(np.isnan(out["qi_l2"][4]))
        self.assertTrue(np.isnan(out["qi_depth_uniform"][4]))
        self.assertTrue(np.isnan(out["qi_depth_inv"][4]))
        self.assertTrue(np.isnan(out["qi_depth_exp"][4]))

    def test_single_invalid_level_is_excluded_not_zero_filled(self):
        # Variant: event 0's level-2 ASK size zeroed only -> level 2 invalid, touch/level-1
        # still valid. All three depth-weighted schemes give w_1 = 1.0 (uniform=1, inv=1/1=1,
        # exp=exp(0)=1), so with level 2 excluded entirely they must collapse EXACTLY to the
        # touch ratio: (10-50)/(10+50) = -40/60 = -0.6666666666666666. If the implementation
        # zero-filled the missing level instead of excluding it, weighted ask would include a
        # phantom 0*w_2 term in the denominator's count but not change the sum here (0
        # contributes nothing) -- so this case alone doesn't distinguish "zero-fill" from
        # "exclude" for the SUM, but it does distinguish it for qi_l2, which must be NaN,
        # never 0.0 (0.0 would be a lie: "level 2 has a zero-size ask queue", not "no data").
        d = toy_arrays()
        d["ask_sz"][0][1] = 0
        frame = make_event_frame(**d)
        out = compute(frame, _all_event_stamps(frame))

        book = book_at_stamps(frame, _all_event_stamps(frame))
        self.assertTrue(bool(book.valid[0]))  # touch itself untouched
        self.assertFalse(bool(book.valid_l[0, 1]))  # level 2 now invalid

        self.assertTrue(np.isnan(out["qi_l2"][0]))
        self.assertEqual(out["qi_touch"][0], -40 / 60)
        self.assertEqual(out["qi_l1"][0], -40 / 60)
        self.assertEqual(out["qi_depth_uniform"][0], -40 / 60)
        self.assertEqual(out["qi_depth_inv"][0], -40 / 60)
        self.assertEqual(out["qi_depth_exp"][0], -40 / 60)


class TestLookaheadTruncation(unittest.TestCase):
    """Truncating the frame to "the past" must not move a single earlier stamp's value."""

    def test_appending_future_events_leaves_earlier_stamps_bit_identical(self):
        full = toy_event_frame()

        d = toy_arrays()
        past_d = {k: v[:4] for k, v in d.items()}  # events 0-3 only; 4-7 are "the future"
        past = make_event_frame(**past_d)

        # A 150ms grid stopping at .150s: event 4 (at .200s offset) is strictly beyond this
        # window on BOTH frames, so this reproduces the identical single stamp (index 3) on
        # the full frame and on the truncated one -- see tests/test_book.py's identical setup.
        end_ns = SESSION_START + GRID_150MS
        stamps_full = calendar_stamps(full, SESSION_START, end_ns, GRID_150MS)
        stamps_past = calendar_stamps(past, SESSION_START, end_ns, GRID_150MS)
        self.assertEqual(stamps_full.index.tolist(), [3])
        self.assertEqual(stamps_past.index.tolist(), [3])

        out_full = compute(full, stamps_full)
        out_past = compute(past, stamps_past)
        self.assertEqual(set(out_full), set(out_past))
        for key in out_full:
            np.testing.assert_array_equal(
                out_full[key], out_past[key], err_msg=f"{key} changed when future events were appended"
            )


class TestBothClocks(unittest.TestCase):
    """Runs on both stamps_for_clock outputs with the right shape/dtype, incl. the empty case."""

    def test_calendar_clock_shapes_and_dtypes(self):
        frame = toy_event_frame()
        stamps = stamps_for_clock(frame, "calendar")
        out = compute(frame, stamps)
        n = len(stamps.index)
        self.assertGreater(n, 0)
        expected_keys = {
            "qi_touch",
            "qi_sign",
            "qi_l1",
            "qi_l2",
            "qi_depth_uniform",
            "qi_depth_inv",
            "qi_depth_exp",
        }
        self.assertEqual(set(out), expected_keys)
        for key, arr in out.items():
            self.assertEqual(arr.shape, (n,), key)
            self.assertEqual(arr.dtype, np.float64, key)

    def test_event_clock_is_legally_empty_on_this_fixture_and_still_well_shaped(self):
        # EVENT_CLOCK_N=10 book-changing events, but the toy fixture has only 7 (type-5 at
        # index 5 does not tick the event clock) -- so event_stamps is legitimately empty
        # here. compute() must return length-0 float64 arrays, not raise.
        frame = toy_event_frame()
        stamps = stamps_for_clock(frame, "event")
        self.assertEqual(len(stamps.index), 0)
        out = compute(frame, stamps)
        for key, arr in out.items():
            self.assertEqual(arr.shape, (0,), key)
            self.assertEqual(arr.dtype, np.float64, key)


class TestBookNoneMatchesExplicitBook(unittest.TestCase):
    def test_book_none_is_bit_identical_to_a_precomputed_book(self):
        frame = toy_event_frame()
        stamps = _all_event_stamps(frame)
        book = book_at_stamps(frame, stamps)

        out_implicit = compute(frame, stamps)
        out_explicit = compute(frame, stamps, book)
        self.assertEqual(set(out_implicit), set(out_explicit))
        for key in out_implicit:
            np.testing.assert_array_equal(out_implicit[key], out_explicit[key])


if __name__ == "__main__":
    unittest.main()


class TestDeepLevelsOnATenLevelBook(unittest.TestCase):
    """qi_l3..qi_l10 and the weight tails at m >= 3.

    The shared toy fixture is two levels deep, so eight of the fifteen emitted columns and
    every depth weight beyond w_2 had no test at all. This builds a single-event, TEN-level
    book where the per-level imbalance is a clean fraction at every level.

    Level m carries bid size 10*m and ask size 10*(11-m), so the two sides sum to 110 at
    every level and the imbalance is a whole number of elevenths:

        I_m = (10m - 10(11-m)) / 110 = (20m - 110) / 110 = (2m - 11) / 11

        m:   1     2     3     4     5    6    7    8    9    10
        I_m: -9/11 -7/11 -5/11 -3/11 -1/11 1/11 3/11 5/11 7/11 9/11
    """

    def setUp(self):
        bid_sz = [[10 * m for m in range(1, 11)]]
        ask_sz = [[10 * (11 - m) for m in range(1, 11)]]
        # Prices strictly decreasing on the bid / increasing on the ask, 1 tick apart.
        bid_px = [[100000 - 100 * (m - 1) for m in range(1, 11)]]
        ask_px = [[100100 + 100 * (m - 1) for m in range(1, 11)]]
        frame = make_event_frame(
            ts_ns=[34200000000001],
            event_type=[1],
            direction=[1],
            price=[100000],
            size=[10],
            bid_px=bid_px,
            bid_sz=bid_sz,
            ask_px=ask_px,
            ask_sz=ask_sz,
        )
        stamps = Stamps(index=np.array([0]), ts_ns=np.array([34200000000001]))
        self.out = compute(frame, stamps)

    def test_every_level_imbalance_is_the_hand_computed_fraction(self):
        for m in range(1, 11):
            expected = (2.0 * m - 11.0) / 11.0  # derived in the class docstring
            self.assertAlmostEqual(
                self.out[f"qi_l{m}"][0], expected, places=12, msg=f"qi_l{m}"
            )

    def test_touch_and_sign_agree_with_level_one(self):
        # qi_touch is level 1: (10 - 100)/110 = -9/11 = -0.8181...
        self.assertAlmostEqual(self.out["qi_touch"][0], -9.0 / 11.0, places=12)
        self.assertEqual(self.out["qi_sign"][0], -1.0)

    def test_uniform_depth_weighting_is_exactly_zero(self):
        # Total bid = 10+20+...+100 = 550; total ask = 100+90+...+10 = 550.
        # (550 - 550) / (550 + 550) = 0.0 exactly -- a clean decisive value that a
        # mis-weighted sum cannot reproduce by accident.
        self.assertEqual(self.out["qi_depth_uniform"][0], 0.0)

    def test_deep_levels_are_not_all_the_same_column(self):
        # Guards against a level-index collapse: all ten must be distinct and increasing.
        values = [self.out[f"qi_l{m}"][0] for m in range(1, 11)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), 10)


class TestDepthWeightTails(unittest.TestCase):
    """The 1/m and exp(-(m-1)/lambda) weights at m >= 3, which no test evaluated."""

    def test_inverse_weights_are_the_reciprocals(self):
        from edge.features.queue import _level_weights

        w = _level_weights(10, "inv")
        for m in range(1, 11):
            self.assertAlmostEqual(w[m - 1], 1.0 / m, places=12, msg=f"w_{m}")

    def test_exponential_weights_are_e_folding_at_lambda_three(self):
        from edge.features.queue import _level_weights

        w = _level_weights(10, "exp")
        self.assertEqual(EXP_DECAY_LAMBDA, 3.0)
        # w_1 = e^0 = 1; w_4 = e^-1 = 0.36787944117144233 (one full e-folding);
        # w_10 = e^-3 = 0.049787068367863944. These three are the figures the module
        # docstring quotes, so a wrong docstring and a wrong weight cannot agree silently.
        self.assertEqual(w[0], 1.0)
        self.assertAlmostEqual(w[3], 0.36787944117144233, places=15)
        self.assertAlmostEqual(w[9], 0.049787068367863944, places=15)
