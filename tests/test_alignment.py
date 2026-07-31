"""The alignment invariant: orderbook row k is the book state AFTER message k.

Construction (per the phase-0 brief): find type-3 (full delete) messages. Assert the
displayed size at that price level in row k-1 exceeds row k by exactly the message's size
field; a level that the delete fully empties simply stops appearing in row k, which the
size lookup already treats as size 0. Off-by-one inverts the assertion, so this file also
runs the k -> k+1 comparison on the identical samples and asserts it does NOT hold.

Convention authority: LOBSTER's data-structure page
(https://data.lobsterdata.com/info/DataStructure.php) is unreachable from this environment
(SPA shell / dead subdomain as of 2026-07-30, see DECISIONS.md) so it could not be quoted
directly here. In its place: (1) an empirical pre-check on both raw files, logged in
DECISIONS.md, found 2000/2000 sampled type-3 deletes satisfy the k-1 -> k identity below
and 0/2000 satisfy the inverted k -> k+1 one; (2) orderflow-lm's SPEC.md independently
states the same convention ("pre-event book = orderbook file row i-1"). This test is the
executable version of that check, run at full sample size against the actual raw files.
"""
import os
import random
import unittest

RAW_DIR = os.environ.get(
    "EDGE_RAW_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "raw", "lobster")
)

NUM_LEVELS = 10
MIN_SAMPLES = 500
SEED = 0

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


def _level_size(book_np, row_idx, px_cols, sz_cols, price, num_levels):
    """Visible size at `price` on one side of one orderbook row; 0 if not displayed."""
    for lvl in range(num_levels):
        if book_np[px_cols[lvl]][row_idx] == price:
            return int(book_np[sz_cols[lvl]][row_idx])
    return 0


_CACHE = {}


def _load(ticker):
    if ticker not in _CACHE:
        from edge.io import lobster  # not yet implemented; import failure is the expected red state

        msg_path, book_path = TICKER_FILES[ticker]
        messages = lobster.read_message_file(msg_path)
        book = lobster.read_orderbook_file(book_path, NUM_LEVELS)
        _CACHE[ticker] = (messages, book)
    return _CACHE[ticker]


@unittest.skipUnless(_raw_files_present(), "raw LOBSTER CSVs not present under data/raw/lobster")
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

        ask_px_cols = [f"ask_px_{i}" for i in range(1, NUM_LEVELS + 1)]
        ask_sz_cols = [f"ask_sz_{i}" for i in range(1, NUM_LEVELS + 1)]
        bid_px_cols = [f"bid_px_{i}" for i in range(1, NUM_LEVELS + 1)]
        bid_sz_cols = [f"bid_sz_{i}" for i in range(1, NUM_LEVELS + 1)]
        book_np = {name: book.column(name).to_numpy() for name in book.column_names}

        type3_idx = [i for i in range(len(event_type)) if event_type[i] == 3 and i > 0]
        self.assertGreaterEqual(
            len(type3_idx), MIN_SAMPLES, f"{ticker}: not enough type-3 deletes to sample from"
        )
        rng = random.Random(SEED)
        sample = rng.sample(type3_idx, MIN_SAMPLES)

        n_checked = 0
        n_holds = 0
        for k in sample:
            if direction_of_comparison == "after":
                prev_row, next_row = k - 1, k
            else:
                prev_row, next_row = k, k + 1
                if next_row >= n_book_rows:
                    continue

            px_cols, sz_cols = (ask_px_cols, ask_sz_cols) if dir_col[k] == -1 else (bid_px_cols, bid_sz_cols)
            price = int(price_col[k])
            msg_size = int(size_col[k])

            prev_size = _level_size(book_np, prev_row, px_cols, sz_cols, price, NUM_LEVELS)
            next_size = _level_size(book_np, next_row, px_cols, sz_cols, price, NUM_LEVELS)

            n_checked += 1
            if prev_size - next_size == msg_size:
                n_holds += 1

        return n_checked, n_holds

    def test_orderbook_row_is_state_after_message(self):
        for ticker in TICKER_FILES:
            n_checked, n_holds = self._run(ticker, "after")
            self.assertGreaterEqual(n_checked, MIN_SAMPLES)
            self.assertEqual(
                n_holds,
                n_checked,
                f"{ticker}: k-1->k size-drop identity failed on "
                f"{n_checked - n_holds}/{n_checked} sampled type-3 deletes",
            )

    def test_inverted_direction_does_not_hold(self):
        for ticker in TICKER_FILES:
            n_checked, n_holds = self._run(ticker, "before")
            self.assertGreater(n_checked, 0)
            self.assertLess(
                n_holds,
                n_checked,
                f"{ticker}: inverted k->k+1 identity unexpectedly held on all "
                f"{n_checked} samples -- alignment convention may be backwards",
            )


if __name__ == "__main__":
    unittest.main()
