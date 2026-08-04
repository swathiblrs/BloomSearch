import json
import tempfile
import unittest
from pathlib import Path

from bloom_filter import BloomFilter, BloomFilterConfig


class BloomFilterTests(unittest.TestCase):
    def test_configuration_matches_standard_formula(self):
        config = BloomFilterConfig.calculate(capacity=1_000, error_rate=0.01)
        self.assertEqual(config.bit_count, 9586)
        self.assertEqual(config.hash_count, 7)

    def test_inserted_values_have_no_false_negatives(self):
        bloom = BloomFilter(capacity=100, error_rate=0.01)
        values = ["geeks", "nerd", "bloom", "filter"]
        bloom.add_many(values)
        for value in values:
            self.assertIn(value, bloom)

    def test_absent_value_is_usually_rejected(self):
        bloom = BloomFilter(capacity=100, error_rate=0.001)
        bloom.add_many(["geeks", "nerd"])
        self.assertNotIn("definitely-absent", bloom)

    def test_strings_and_bytes_are_equivalent(self):
        bloom = BloomFilter(capacity=10)
        bloom.add("hello")
        self.assertIn(b"hello", bloom)

    def test_save_and_load_round_trip(self):
        bloom = BloomFilter(capacity=50, error_rate=0.02)
        bloom.add_many(["alpha", "beta"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filter.json"
            bloom.save(path)
            restored = BloomFilter.load(path)
        self.assertIn("alpha", restored)
        self.assertIn("beta", restored)
        self.assertEqual(restored.items_added, 2)
        self.assertEqual(restored.stats(), bloom.stats())

    def test_rejects_invalid_parameters(self):
        for capacity, rate in [(0, 0.1), (10, 0), (10, 1), (10, -0.5)]:
            with self.subTest(capacity=capacity, rate=rate):
                with self.assertRaises(ValueError):
                    BloomFilter(capacity, rate)

    def test_serialized_file_is_readable_json(self):
        bloom = BloomFilter(capacity=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filter.json"
            bloom.save(path)
            data = json.loads(path.read_text())
        self.assertEqual(data["format_version"], 1)
        self.assertIn("bits", data)


if __name__ == "__main__":
    unittest.main()

