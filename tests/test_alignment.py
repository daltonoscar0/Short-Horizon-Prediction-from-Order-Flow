"""The alignment invariant: orderbook row k is the book state AFTER message k.

Construction (per the phase-0 spec): find type-3 (full delete) messages. Assert the
displayed size at that price level in row k-1 exceeds row k by exactly the message's size
field; a level that the delete fully empties simply stops appearing in row k, which the
size lookup already treats as size 0. Off-by-one inverts the assertion, so this file also
runs the k -> k+1 comparison on the identical samples and asserts it does NOT hold, at a
threshold tight enough to actually discriminate (see INVERTED_HOLD_RATE_THRESHOLD below).

A price absent from BOTH the prev and next row is undecidable, not a pass: it can mean the
level was never inside the visible depth on either side of the event, so the identity
carries no information about it. Those samples are excluded from the denominator rather
than folded in as an automatic (0 - 0 == 0-size-message) match.

Convention authority: LOBSTER's data-structure page states verbatim that "The k-th row in
the 'message' file describes the limit order event causing the change in the limit order
book from line k-1 to line k in the 'orderbook' file" (archived snapshot, see
DECISIONS.md) -- i.e. orderbook row k is the state AFTER message k, the "after" direction
tested below. Independently, an empirical pre-check on both raw files (logged in
DECISIONS.md) found the k-1 -> k identity holds on the full sample and the inverted
k -> k+1 identity holds on only ~5.5-6.5% of the same sample (chance-level noise from
runs of no-op events, not signal).

Two independent layers of coverage:

  * `TestAlignmentOnCommittedFixture` -- a dozen hand-built messages
    (tests/fixtures/alignment_message.csv + alignment_orderbook.csv), correct alignment
    derived by hand in the class docstring, checked in via git so it runs on every
    checkout with no external data dependency. Also runs the identical check against a
    deliberately +1-row-shifted copy of the same orderbook
    (alignment_orderbook_shifted.csv) and asserts it FAILS -- proof the test has power to
    catch the bug it exists to catch, not just a description of the convention.

  * `TestOrderbookRowIsStateAfterMessage` -- the full-scale check against the real raw
    INTC/AMZN CSVs (gitignored, ~250MB, fetched separately per DECISIONS.md). Stronger
    signal when present, but structurally cannot run on a fresh checkout with no raw data,
    so it is skip-gated; the skip reason says so explicitly rather than passing silently.
"""
import os
import random
import unittest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

RAW_DIR = os.environ.get(
    "EDGE_RAW_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "raw", "lobster")
)

NUM_LEVELS = 10
MIN_SAMPLES = 500
SEED = 0

# Measured (DECISIONS.md, seed 0): "after" (correct) direction holds on 500/500 samples for
# both tickers == 100%; "before" (inverted) direction holds on 110/2000 (5.5%, INTC) and
# 130/2000 (6.5%, AMZN). A bare assertLess(n_holds, n_checked) only requires ONE of the
# sampled events to disagree, so it would still pass at a 499/500 (99.8%) inverted hold
# rate -- nowhere near falsifying. 0.25 sits with ~4x headroom above the measured ~6.5%
# ceiling and ~4x headroom below the correct direction's 100%, so sampling noise cannot
# flip the verdict in either direction while still catching a real off-by-one.
INVERTED_HOLD_RATE_THRESHOLD = 0.25

# Below this fraction of the sampled type-3 deletes actually being decidable (price shown
# in at least one of the two compared rows), the check has too little signal to trust --
# flag it rather than silently passing on a near-empty decidable set.
MIN_DECIDABLE_FRACTION = 0.9

TICKER_FILES = {
    "INTC": (
        os.path.join(RAW_DIR, "INTC_2012-06-21_34200000_57600000_message_10.csv"),
        os.path.join(RAW_DIR, "INTC_2012-06-21_34200000_57600000_orderbook_10.csv"),
    ),
    "AMZN": (
        os.path.join(RAW_DIR, "AMZN_2012-06-21_34200000_57600000_message_10.csv"),
        os.path.join(RAW_DIR, "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"),
    ),
}


def _raw_files_present():
    return all(os.path.exists(p) for paths in TICKER_FILES.values() for p in paths)


def _level_size_or_none(book_np, row_idx, px_cols, sz_cols, price, num_levels):
    """Visible size at `price` in one row; None if not displayed (undecidable), not 0."""
    for lvl in range(num_levels):
        if book_np[px_cols[lvl]][row_idx] == price:
            return int(book_np[sz_cols[lvl]][row_idx])
    return None


def _alignment_counts(
    samples, price_col, size_col, dir_col, book_np,
    ask_px_cols, ask_sz_cols, bid_px_cols, bid_sz_cols, num_levels,
):
    """Core size-drop identity counter shared by the fixture and raw-file checks.

    `samples` is an iterable of (event_row, prev_book_row, next_book_row) index triples.
    Returns (n_checked, n_holds, n_undecidable); a sample whose price is displayed in
    NEITHER row is excluded from n_checked rather than counted as a pass.
    """
    n_checked = 0
    n_holds = 0
    n_undecidable = 0
    for k, prev_row, next_row in samples:
        px_cols, sz_cols = (ask_px_cols, ask_sz_cols) if dir_col[k] == -1 else (bid_px_cols, bid_sz_cols)
        price = int(price_col[k])
        msg_size = int(size_col[k])

        prev_size = _level_size_or_none(book_np, prev_row, px_cols, sz_cols, price, num_levels)
        next_size = _level_size_or_none(book_np, next_row, px_cols, sz_cols, price, num_levels)

        if prev_size is None and next_size is None:
            n_undecidable += 1
            continue

        n_checked += 1
        if (prev_size or 0) - (next_size or 0) == msg_size:
            n_holds += 1
    return n_checked, n_holds, n_undecidable


def _level_cols(num_levels):
    ask_px_cols = [f"ask_px_{i}" for i in range(1, num_levels + 1)]
    ask_sz_cols = [f"ask_sz_{i}" for i in range(1, num_levels + 1)]
    bid_px_cols = [f"bid_px_{i}" for i in range(1, num_levels + 1)]
    bid_sz_cols = [f"bid_sz_{i}" for i in range(1, num_levels + 1)]
    return ask_px_cols, ask_sz_cols, bid_px_cols, bid_sz_cols


_CACHE = {}


def _load(ticker):
    if ticker not in _CACHE:
        from edge.io import lobster  # not yet implemented; import failure is the expected red state

        msg_path, book_path = TICKER_FILES[ticker]
        messages = lobster.read_message_file(msg_path)
        book = lobster.read_orderbook_file(book_path, NUM_LEVELS)
        _CACHE[ticker] = (messages, book)
    return _CACHE[ticker]


class TestAlignmentOnCommittedFixture(unittest.TestCase):
    """Always-on coverage of the alignment invariant, independent of the gitignored raw data.

    Fixture: tests/fixtures/alignment_message.csv + alignment_orderbook.csv, 12 hand-built
    messages, 1 displayed level per side, book state derived by hand alongside the
    messages (not by running the adapter and trusting its output). Three type-3 deletes:

      idx 4 (0-indexed): bid delete of order id=3, price=98, size=4. Price 98 was never
        the best bid (orders at 100 and 99 sit ahead of it) so it is absent from level 1
        in BOTH row 3 (prev) and row 4 (next) -- the undecidable case, must be excluded.
      idx 7: bid delete of order id=1, price=100, size=7 (the only order left at that
        price after msg6's partial cancel). Row 6 shows bid_px_1=100, bid_sz_1=7; the
        delete empties the level and row 7 shows bid_px_1=99 (the next-best order comes
        into view). Decidable: 7 - 0 == 7, holds.
      idx 10: ask delete of order id=101, price=101, size=30 (remaining size after msg9's
        partial cancel). Row 9 shows ask_px_1=101, ask_sz_1=30; row 10 shows
        ask_px_1=102 (101 vanishes). Decidable: 30 - 0 == 30, holds.

    So on the correct fixture: 2 decidable samples, both hold, 1 excluded as undecidable.

    tests/fixtures/alignment_orderbook_shifted.csv is the identical book shifted by one
    row (shifted[k] = correct[k-1], shifted[0] = correct[0]) -- the canonical off-by-one:
    treating row k as the state BEFORE message k instead of after. Re-deriving idx 7 and
    idx 10 by hand against the shifted rows gives 10-7=3 != 7 and 50-30=20 != 30: both
    decidable samples now FAIL, proving this fixture actually has the power to catch the
    bug it exists to catch, not just restate the convention.
    """

    FIXTURE_NUM_LEVELS = 1
    MESSAGE_PATH = os.path.join(FIXTURES_DIR, "alignment_message.csv")
    ORDERBOOK_PATH = os.path.join(FIXTURES_DIR, "alignment_orderbook.csv")
    SHIFTED_ORDERBOOK_PATH = os.path.join(FIXTURES_DIR, "alignment_orderbook_shifted.csv")

    def setUp(self):
        from edge.io import lobster

        self.messages = lobster.read_message_file(self.MESSAGE_PATH)
        self.book_correct = lobster.read_orderbook_file(self.ORDERBOOK_PATH, self.FIXTURE_NUM_LEVELS)
        self.book_shifted = lobster.read_orderbook_file(self.SHIFTED_ORDERBOOK_PATH, self.FIXTURE_NUM_LEVELS)

    def _check(self, book):
        event_type = self.messages.column("event_type").to_numpy()
        size_col = self.messages.column("size").to_numpy()
        price_col = self.messages.column("price").to_numpy()
        dir_col = self.messages.column("direction").to_numpy()
        ask_px_cols, ask_sz_cols, bid_px_cols, bid_sz_cols = _level_cols(self.FIXTURE_NUM_LEVELS)
        book_np = {name: book.column(name).to_numpy() for name in book.column_names}

        type3_idx = [i for i in range(len(event_type)) if event_type[i] == 3 and i > 0]
        self.assertEqual(type3_idx, [4, 7, 10], "fixture derivation drifted from the docstring")
        samples = [(k, k - 1, k) for k in type3_idx]
        return _alignment_counts(
            samples, price_col, size_col, dir_col, book_np,
            ask_px_cols, ask_sz_cols, bid_px_cols, bid_sz_cols, self.FIXTURE_NUM_LEVELS,
        )

    def test_orderbook_row_is_state_after_message_on_fixture(self):
        n_checked, n_holds, n_undecidable = self._check(self.book_correct)
        self.assertEqual(n_undecidable, 1, "expected exactly one undecidable type-3 delete (idx 4)")
        self.assertEqual(n_checked, 2, "expected exactly two decidable type-3 deletes (idx 7, 10)")
        self.assertEqual(n_holds, n_checked, "size-drop identity must hold on every decidable sample")

    def test_shifted_fixture_fails_the_identity(self):
        n_checked, n_holds, n_undecidable = self._check(self.book_shifted)
        self.assertEqual(n_checked, 2, "shift must not change which samples are decidable")
        self.assertLess(
            n_holds,
            n_checked,
            "a deliberately off-by-one orderbook file passed the alignment identity -- "
            "this fixture has no power to catch the bug it exists to catch",
        )


@unittest.skipUnless(
    _raw_files_present(),
    "raw LOBSTER CSVs not present under data/raw/lobster: "
    "the load-bearing full-scale invariant test_orderbook_row_is_state_after_message "
    "(500 sampled type-3 deletes per ticker against the real INTC/AMZN files) DID NOT "
    "RUN in this test run. TestAlignmentOnCommittedFixture above ran the same identity on a "
    "12-row committed fixture, but that is the weaker check -- fetch the raw data "
    "(DECISIONS.md) and rerun before trusting this invariant at full scale.",
)
class TestOrderbookRowIsStateAfterMessage(unittest.TestCase):
    def _run(self, ticker, direction_of_comparison):
        """direction_of_comparison: 'after' checks k-1 -> k (the spec); 'before' checks
        k -> k+1 (the inverted hypothesis, expected to fail)."""
        messages, book = _load(ticker)

        event_type = messages.column("event_type").to_numpy()
        size_col = messages.column("size").to_numpy()
        price_col = messages.column("price").to_numpy()
        dir_col = messages.column("direction").to_numpy()
        n_book_rows = book.num_rows

        ask_px_cols, ask_sz_cols, bid_px_cols, bid_sz_cols = _level_cols(NUM_LEVELS)
        book_np = {name: book.column(name).to_numpy() for name in book.column_names}

        type3_idx = [i for i in range(len(event_type)) if event_type[i] == 3 and i > 0]
        self.assertGreaterEqual(
            len(type3_idx), MIN_SAMPLES, f"{ticker}: not enough type-3 deletes to sample from"
        )
        rng = random.Random(SEED)
        sample = rng.sample(type3_idx, MIN_SAMPLES)

        samples = []
        for k in sample:
            if direction_of_comparison == "after":
                prev_row, next_row = k - 1, k
            else:
                prev_row, next_row = k, k + 1
                if next_row >= n_book_rows:
                    continue
            samples.append((k, prev_row, next_row))

        return _alignment_counts(
            samples, price_col, size_col, dir_col, book_np,
            ask_px_cols, ask_sz_cols, bid_px_cols, bid_sz_cols, NUM_LEVELS,
        )

    def test_orderbook_row_is_state_after_message(self):
        for ticker in TICKER_FILES:
            n_checked, n_holds, n_undecidable = self._run(ticker, "after")
            self.assertGreaterEqual(
                n_checked,
                int(MIN_SAMPLES * MIN_DECIDABLE_FRACTION),
                f"{ticker}: only {n_checked}/{MIN_SAMPLES} sampled type-3 deletes were "
                f"decidable ({n_undecidable} excluded) -- too few to trust the invariant",
            )
            self.assertEqual(
                n_holds,
                n_checked,
                f"{ticker}: k-1->k size-drop identity failed on "
                f"{n_checked - n_holds}/{n_checked} decidable sampled type-3 deletes "
                f"({n_undecidable} excluded as undecidable)",
            )

    def test_inverted_direction_does_not_hold(self):
        for ticker in TICKER_FILES:
            n_checked, n_holds, n_undecidable = self._run(ticker, "before")
            self.assertGreater(n_checked, 0)
            hold_rate = n_holds / n_checked
            self.assertLess(
                hold_rate,
                INVERTED_HOLD_RATE_THRESHOLD,
                f"{ticker}: inverted k->k+1 identity held on {n_holds}/{n_checked} "
                f"({hold_rate:.1%}) decidable samples, at or above the "
                f"{INVERTED_HOLD_RATE_THRESHOLD:.0%} threshold -- too close to the correct "
                f"direction's ~100% to be discriminating; alignment convention may be "
                f"backwards",
            )


if __name__ == "__main__":
    unittest.main()
