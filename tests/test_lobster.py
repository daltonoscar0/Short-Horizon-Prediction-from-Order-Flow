"""Unit tests for edge.io.lobster against a tiny hand-built fixture.

Fixture: tests/fixtures/toy_message.csv + tests/fixtures/toy_orderbook.csv, 8 messages,
2 price levels, hand-derived so every row obeys the alignment convention (orderbook row k
= state after message k) -- see docs/DATA_CONTRACT.md for the full derivation. Message 4
(index 3, type 3) fully deletes bid 100500; message 8 (index 7, type 3) fully deletes ask
102000, exercising the level-vanishes-from-view case on both sides.
"""
import os
import unittest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MESSAGE_PATH = os.path.join(FIXTURES_DIR, "toy_message.csv")
ORDERBOOK_PATH = os.path.join(FIXTURES_DIR, "toy_orderbook.csv")
NUM_LEVELS = 2


class TestReadMessageFile(unittest.TestCase):
    def setUp(self):
        from edge.io import lobster  # not yet implemented: import failure is the expected red state

        self.table = lobster.read_message_file(MESSAGE_PATH)

    def test_row_count(self):
        self.assertEqual(self.table.num_rows, 8)

    def test_ns_exact_timestamps_from_decimal_strings(self):
        # Ns must come from string split + integer padding, never float64 multiplication
        # (float rounding can perturb ns ties) -- "34200.1" has to pad to 100000000 ns, not
        # decay through a float64 intermediate.
        expected = [
            34200000000001,
            34200000000002,
            34200100000000,
            34200100000500,
            34200200000000,
            34200200000001,
            34200300000000,
            34200400000000,
        ]
        self.assertEqual(self.table.column("ts_ns").to_pylist(), expected)

    def test_event_type_passthrough(self):
        self.assertEqual(self.table.column("event_type").to_pylist(), [1, 1, 2, 3, 4, 5, 1, 3])

    def test_direction_passthrough(self):
        self.assertEqual(self.table.column("direction").to_pylist(), [1, 1, 1, 1, -1, -1, -1, -1])

    def test_price_and_size_are_ints(self):
        self.assertEqual(
            self.table.column("price").to_pylist(),
            [100500, 100500, 100500, 100500, 101000, 101000, 103000, 102000],
        )
        self.assertEqual(self.table.column("size").to_pylist(), [10, 20, 5, 25, 10, 5, 15, 30])


class TestReadOrderbookFile(unittest.TestCase):
    def setUp(self):
        from edge.io import lobster

        self.table = lobster.read_orderbook_file(ORDERBOOK_PATH, NUM_LEVELS)

    def test_row_count(self):
        self.assertEqual(self.table.num_rows, 8)

    def test_columns_present(self):
        expected_cols = {
            "ask_px_1", "ask_sz_1", "bid_px_1", "bid_sz_1",
            "ask_px_2", "ask_sz_2", "bid_px_2", "bid_sz_2",
        }
        self.assertEqual(set(self.table.column_names), expected_cols)

    def test_full_delete_empties_the_level_bid_side(self):
        # msg4 (1-based) = type-3 delete of 25 at bid 100500: row idx 2 (after msg3) has
        # it at full size, row idx 3 (after msg4) no longer shows that price on either
        # displayed bid level.
        prior = {c: self.table.column(c)[2].as_py() for c in self.table.column_names}
        post = {c: self.table.column(c)[3].as_py() for c in self.table.column_names}
        self.assertEqual(prior["bid_px_1"], 100500)
        self.assertEqual(prior["bid_sz_1"], 25)
        self.assertNotIn(100500, (post["bid_px_1"], post["bid_px_2"]))

    def test_full_delete_empties_the_level_ask_side(self):
        # msg8 (1-based) = type-3 delete of 30 at ask 102000: row idx 6 (after msg7) has
        # it, row idx 7 (after msg8) does not; level 3 (103000) shifts up into view.
        prior = {c: self.table.column(c)[6].as_py() for c in self.table.column_names}
        post = {c: self.table.column(c)[7].as_py() for c in self.table.column_names}
        self.assertEqual(prior["ask_px_2"], 102000)
        self.assertEqual(prior["ask_sz_2"], 30)
        self.assertNotIn(102000, (post["ask_px_1"], post["ask_px_2"]))
        self.assertEqual(post["ask_px_2"], 103000)


class TestBuildEventFrame(unittest.TestCase):
    def test_schema_and_join(self):
        from edge.io import lobster

        frame = lobster.build_event_frame(MESSAGE_PATH, ORDERBOOK_PATH, NUM_LEVELS)
        expected_cols = [
            "ts_ns", "event_type", "direction", "price", "size",
            "bid_px_1", "bid_px_2", "bid_sz_1", "bid_sz_2",
            "ask_px_1", "ask_px_2", "ask_sz_1", "ask_sz_2",
        ]
        self.assertEqual(frame.column_names, expected_cols)
        self.assertEqual(frame.num_rows, 8)


if __name__ == "__main__":
    unittest.main()
