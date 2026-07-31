"""Mechanical structural invariants of the `compute(frame, stamps, book=None)` contract
(docs/INTERFACE.md), run GENERICALLY over every module named in `FAMILY_PREFIXES` via
`importlib` -- so a seventh family module added later inherits every check here for free
instead of a human remembering to wire it in.

Convention pinned here, same as every family module: `<= t` causality (INTERFACE.md sec.
2), except columns under `LABEL_PREFIX`, which are the one documented exception (sec. 2's
"`label_` is the only forward-looking prefix") and are therefore exempt from the lookahead
check below by construction, not by omission.

This file does NOT re-derive any feature's arithmetic -- that lives in each family's own
`tests/test_<family>.py` with hand-computed literals (INTERFACE.md sec. 10). Everything
here is dtype/shape/prefix/causality PLUMBING that is identical in shape across all six
modules, which is exactly what makes it worth asserting once, generically, rather than six
times with six chances to phrase the check slightly differently.
"""
import importlib
import inspect
import unittest

import numpy as np

from edge.features import FAMILY_PREFIXES, LABEL_PREFIX, NS_PER_SEC, SEASONAL_ADJ_SUFFIX, SESSION_START_NS
from edge.features.book import book_at_stamps
from edge.features.clocks import Stamps, event_stamps, stamps_for_clock
from edge.features.frame import EventFrame
from tests.toybook import make_event_frame, toy_event_frame

# Every family module, imported generically by name so this file has no hardcoded list of
# "the six" to fall out of sync with FAMILY_PREFIXES.
MODULES = {name: importlib.import_module(f"edge.features.{name}") for name in FAMILY_PREFIXES}

CLOCKS = ("calendar", "event")

# Event-clock override used throughout this file: the shared toy fixture has only 7
# book-changing events, below the frozen EVENT_CLOCK_N=10, so the frozen dispatcher's event
# clock is legitimately EMPTY there (itself one of the required checks, see
# TestEmptyStamps). To also exercise a NON-empty event clock generically, an extended
# synthetic frame below is stamped with a small override N instead.
EVENT_CLOCK_N_OVERRIDE = 2


def _extended_frame(n=24):
    """n-row, 1-level, always-valid synthetic book at 1s spacing, all book-changing event
    types (1-4) -- built purely to give BOTH clocks a non-trivial multi-stamp span to
    truncate for the generic lookahead check; tests/toybook.py's 8-event fixture spans
    only 400 microseconds, too short for a meaningful multi-stamp calendar/event-clock
    comparison. Not a hand-computed literal fixture (that lives in each family's own test
    file, INTERFACE.md sec. 10) -- this one is exercised only for bit-identical structural
    comparisons across truncation, never for an expected numeric value.
    """
    ts_ns = [SESSION_START_NS + i * NS_PER_SEC for i in range(n)]
    event_type = [1, 2, 3, 4] * (n // 4)
    direction = [1, -1] * (n // 2)
    price = [100_000 + 100 * (i % 5) for i in range(n)]
    size = [50 + i for i in range(n)]
    bid_px = [[100_000 + 100 * (i % 3)] for i in range(n)]
    bid_sz = [[20 + i] for i in range(n)]
    ask_px = [[100_500 + 100 * (i % 3)] for i in range(n)]
    ask_sz = [[30 + i] for i in range(n)]
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


def _stamps(frame, clock):
    """This file's own clock helper: calendar via the frozen dispatcher, event via a small
    override N so a 24-row synthetic frame still yields several event-clock stamps."""
    if clock == "calendar":
        return stamps_for_clock(frame, "calendar")
    return event_stamps(frame, n=EVENT_CLOCK_N_OVERRIDE)


def _assert_dict_bit_identical(test, a, b, keys=None):
    keys = keys if keys is not None else set(a)
    test.assertEqual(set(a), set(b))
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def _snapshot_frame(frame):
    """Copies of every array-valued attribute an EventFrame exposes over its own storage,
    used to prove a family module did not mutate the shared frame in place."""
    return {
        name: np.array(getattr(frame, name), copy=True)
        for name in ("ts_ns", "event_type", "direction", "price", "size", "bid_px", "bid_sz", "ask_px", "ask_sz")
    }


def _assert_frame_unmutated(test, frame, snapshot):
    for name, before in snapshot.items():
        np.testing.assert_array_equal(getattr(frame, name), before, err_msg=name)


def _snapshot_stamps(stamps):
    return Stamps(index=np.array(stamps.index, copy=True), ts_ns=np.array(stamps.ts_ns, copy=True))


def _assert_stamps_unmutated(test, stamps, snapshot):
    np.testing.assert_array_equal(stamps.index, snapshot.index)
    np.testing.assert_array_equal(stamps.ts_ns, snapshot.ts_ns)


def _snapshot_book(book):
    return {field: np.array(getattr(book, field), copy=True) for field in book._fields if field != "n_levels"}


def _assert_book_unmutated(test, book, snapshot):
    for field, before in snapshot.items():
        np.testing.assert_array_equal(getattr(book, field), before, err_msg=field)


class TestPrefixesArePairwiseNonPrefixing(unittest.TestCase):
    """INTERFACE.md sec. 6: "No prefix is a prefix of another"."""

    def test_no_prefix_is_a_prefix_of_another(self):
        prefixes = list(FAMILY_PREFIXES.items())
        for name_a, prefix_a in prefixes:
            for name_b, prefix_b in prefixes:
                if name_a == name_b:
                    continue
                self.assertFalse(
                    prefix_b.startswith(prefix_a),
                    f"{prefix_a!r} ({name_a}) is a prefix of {prefix_b!r} ({name_b})",
                )

    def test_prefixes_are_lowercase_ascii_and_end_in_underscore(self):
        for name, prefix in FAMILY_PREFIXES.items():
            self.assertTrue(prefix.endswith("_"), name)
            self.assertEqual(prefix, prefix.lower(), name)
            self.assertTrue(prefix.replace("_", "").isalpha(), name)


class TestComputeSignature(unittest.TestCase):
    """INTERFACE.md sec. 5: exactly `compute(frame, stamps, book=None)`, no extras."""

    def test_every_module_exposes_exactly_the_frozen_signature(self):
        for name, module in MODULES.items():
            self.assertTrue(hasattr(module, "compute"), name)
            sig = inspect.signature(module.compute)
            params = list(sig.parameters.values())
            self.assertEqual([p.name for p in params], ["frame", "stamps", "book"], name)
            for p in params:
                self.assertEqual(
                    p.kind,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    f"{name}.{p.name} must not be positional-only or keyword-only",
                )
            self.assertEqual(params[0].default, inspect.Parameter.empty, name)
            self.assertEqual(params[1].default, inspect.Parameter.empty, name)
            self.assertIsNone(params[2].default, name)


class TestStructuralInvariantsOnToyFixtureBothClocks(unittest.TestCase):
    """dtype/shape/prefix/no-`_sadj`/no-cross-module-collision, on the toy fixture, on
    both clocks (event clock legitimately empty here -- see TestEmptyStamps)."""

    def test_every_module_every_clock(self):
        frame = toy_event_frame()
        for clock in CLOCKS:
            stamps = stamps_for_clock(frame, clock)
            n = stamps.index.shape[0]
            seen_keys = {}
            for name, module in MODULES.items():
                out = module.compute(frame, stamps)
                prefix = FAMILY_PREFIXES[name]
                for key, arr in out.items():
                    with self.subTest(clock=clock, module=name, key=key):
                        self.assertEqual(arr.dtype, np.float64)
                        self.assertEqual(arr.ndim, 1)
                        self.assertEqual(arr.shape, (n,))
                        self.assertTrue(key.startswith(prefix), f"{key} missing prefix {prefix}")
                        self.assertFalse(key.endswith(SEASONAL_ADJ_SUFFIX), key)
                        self.assertNotIn(
                            key, seen_keys, f"{key} emitted by both {seen_keys.get(key)} and {name}"
                        )
                    seen_keys[key] = name


class TestBookNoneMatchesExplicitBook(unittest.TestCase):
    """INTERFACE.md sec. 5: `book=None` must build `book_at_stamps` internally and give a
    BIT-IDENTICAL result to the caller passing the same book explicitly."""

    def test_every_module_every_clock(self):
        frame = toy_event_frame()
        for clock in CLOCKS:
            stamps = _stamps(frame, clock)
            book = book_at_stamps(frame, stamps)
            for name, module in MODULES.items():
                with self.subTest(clock=clock, module=name):
                    out_implicit = module.compute(frame, stamps)
                    out_explicit = module.compute(frame, stamps, book)
                    _assert_dict_bit_identical(self, out_implicit, out_explicit)


class TestEmptyStamps(unittest.TestCase):
    """INTERFACE.md sec. 3: "An empty stamps is legal and must not raise: return the same
    keys with length-0 float64 arrays." Checked both via a naturally-empty clock (the toy
    fixture's frozen event clock) and a manually-constructed empty Stamps, so the contract
    holds independent of why the caller ended up with zero stamps."""

    def test_naturally_empty_event_clock_on_the_toy_fixture(self):
        frame = toy_event_frame()
        stamps = stamps_for_clock(frame, "event")
        self.assertEqual(stamps.index.shape[0], 0, "toy fixture has < EVENT_CLOCK_N book-changing events")
        reference_stamps = stamps_for_clock(frame, "calendar")
        for name, module in MODULES.items():
            with self.subTest(module=name):
                reference_keys = set(module.compute(frame, reference_stamps).keys())
                out = module.compute(frame, stamps)
                self.assertEqual(set(out.keys()), reference_keys)
                for key, arr in out.items():
                    self.assertEqual(arr.shape, (0,), key)
                    self.assertEqual(arr.dtype, np.float64, key)

    def test_manually_constructed_empty_stamps(self):
        frame = toy_event_frame()
        empty_stamps = Stamps(index=np.array([], dtype=np.int64), ts_ns=np.array([], dtype=np.int64))
        reference_stamps = stamps_for_clock(frame, "calendar")
        for name, module in MODULES.items():
            with self.subTest(module=name):
                reference_keys = set(module.compute(frame, reference_stamps).keys())
                out = module.compute(frame, empty_stamps)
                self.assertEqual(set(out.keys()), reference_keys)
                for key, arr in out.items():
                    self.assertEqual(arr.shape, (0,), key)
                    self.assertEqual(arr.dtype, np.float64, key)
                # book=None must not raise on an empty take either.
                book = book_at_stamps(frame, empty_stamps)
                out_book = module.compute(frame, empty_stamps, book)
                _assert_dict_bit_identical(self, out, out_book)


def _safe_lookahead_pairs(stamps_full, stamps_trunc):
    """Positions (i_full, i_trunc) whose stamps share a ts_ns value AND resolve to the same
    frame row on both the full and the truncated frame -- the only pairs where a causal
    family is guaranteed to have read an identical input slice on both sides, since
    `frame_trunc` is a strict row-prefix of `frame_full`. `Stamps.ts_ns` is strictly
    increasing (clocks.py), so both arrays are already sorted-unique and `intersect1d` is
    exact, not approximate.
    """
    _, i_full, i_trunc = np.intersect1d(stamps_full.ts_ns, stamps_trunc.ts_ns, return_indices=True)
    same_row = stamps_full.index[i_full] == stamps_trunc.index[i_trunc]
    return i_full[same_row], i_trunc[same_row]


class TestGenericLookaheadTruncation(unittest.TestCase):
    """THE generic lookahead test (INTERFACE.md sec. 10): truncating the frame after some
    event k must leave every SURVIVING stamp's non-`label_` output bit-identical, on both
    clocks. `label_` columns are exempt -- they are supposed to change (sec. 2's sole
    forward-looking exception; `labels.py`'s own inverted lookahead test covers that case).
    """

    def test_truncation_leaves_non_label_columns_bit_identical_both_clocks(self):
        full = _extended_frame(n=24)
        for clock in CLOCKS:
            stamps_full = _stamps(full, clock)
            book_full = book_at_stamps(full, stamps_full)
            full_out = {name: module.compute(full, stamps_full, book_full) for name, module in MODULES.items()}

            for cutoff in (9, 17):  # two different truncation points, well inside 24 rows
                # Truncate via a fresh EventFrame over the same arrays' prefix -- cheaper
                # and more direct than round-tripping through pyarrow for a synthetic
                # fixture that was never a Table to begin with.
                truncated = EventFrame(
                    ts_ns=full.ts_ns[:cutoff],
                    event_type=full.event_type[:cutoff],
                    direction=full.direction[:cutoff],
                    price=full.price[:cutoff],
                    size=full.size[:cutoff],
                    bid_px=full.bid_px[:cutoff],
                    bid_sz=full.bid_sz[:cutoff],
                    ask_px=full.ask_px[:cutoff],
                    ask_sz=full.ask_sz[:cutoff],
                )
                stamps_trunc = _stamps(truncated, clock)
                book_trunc = book_at_stamps(truncated, stamps_trunc)

                i_full, i_trunc = _safe_lookahead_pairs(stamps_full, stamps_trunc)
                with self.subTest(clock=clock, cutoff=cutoff):
                    self.assertGreater(
                        i_full.size, 0, "test is vacuous with no comparable stamps -- widen the fixture"
                    )

                for name, module in MODULES.items():
                    trunc_out = module.compute(truncated, stamps_trunc, book_trunc)
                    prefix = FAMILY_PREFIXES[name]
                    for key, arr_full in full_out[name].items():
                        if key.startswith(LABEL_PREFIX):
                            continue  # forward-looking by design; exempt (sec. 2)
                        with self.subTest(clock=clock, cutoff=cutoff, module=name, key=key):
                            np.testing.assert_array_equal(
                                arr_full[i_full],
                                trunc_out[key][i_trunc],
                                err_msg=f"{key} changed at a surviving stamp when future events were removed",
                            )


class TestPurity(unittest.TestCase):
    """INTERFACE.md sec. 5: pure (two calls with equal arguments return equal arrays) and
    never mutates `frame`, `stamps` or `book` (they are shared across all six families)."""

    def test_calling_compute_twice_returns_equal_arrays_and_leaves_inputs_untouched(self):
        frame = toy_event_frame()
        for clock in CLOCKS:
            stamps = _stamps(frame, clock)
            book = book_at_stamps(frame, stamps)

            frame_snap = _snapshot_frame(frame)
            stamps_snap = _snapshot_stamps(stamps)
            book_snap = _snapshot_book(book)

            for name, module in MODULES.items():
                with self.subTest(clock=clock, module=name):
                    out_1 = module.compute(frame, stamps, book)
                    out_2 = module.compute(frame, stamps, book)
                    _assert_dict_bit_identical(self, out_1, out_2)

                    _assert_frame_unmutated(self, frame, frame_snap)
                    _assert_stamps_unmutated(self, stamps, stamps_snap)
                    _assert_book_unmutated(self, book, book_snap)


if __name__ == "__main__":
    unittest.main()
