import unittest

from bloom_filter import CountingBloomFilter
from bloom_search.research import (
    CertaintyAwareCascade,
    LogisticSegmentClassifier,
    build_synthetic_benchmark,
    evaluate_benchmark,
)


class CountingBloomFilterTests(unittest.TestCase):
    def test_inserted_values_have_no_false_negatives(self):
        counting = CountingBloomFilter(capacity=100, error_rate=0.01)
        values = ["bloom", "filter", "search", "certainty"]
        counting.add_many(values)
        for value in values:
            self.assertIn(value, counting)
            self.assertGreater(counting.certainty(value), 0)

    def test_remove_known_value(self):
        counting = CountingBloomFilter(capacity=100, error_rate=0.001)
        counting.add("known")
        self.assertTrue(counting.remove("known"))
        self.assertNotIn("known", counting)
        self.assertFalse(counting.remove("never-added"))

    def test_certainty_is_bounded(self):
        counting = CountingBloomFilter(capacity=10, error_rate=0.1)
        counting.add_many(["one", "two", "three"])
        self.assertGreaterEqual(counting.certainty("one"), 0)
        self.assertLessEqual(counting.certainty("one"), 1)


class LearnedCascadeTests(unittest.TestCase):
    def test_logistic_classifier_learns_separable_rows(self):
        classifier = LogisticSegmentClassifier(feature_count=1)
        classifier.fit([((0.0,), 0), ((0.1,), 0), ((0.9,), 1), ((1.0,), 1)])
        self.assertLess(classifier.predict_probability((0.1,)), 0.5)
        self.assertGreater(classifier.predict_probability((0.9,)), 0.5)

    def test_cascade_selects_relevant_segment(self):
        index, train, test = build_synthetic_benchmark(documents_per_segment=6)
        cascade = CertaintyAwareCascade(index)
        cascade.fit(train)
        relevant_id = next(iter(test[0].relevant_ids))
        selected = cascade.select_segments(test[0].query)
        selected_ids = {
            document.id for segment_index in selected for document in index.segments[segment_index].documents
        }
        self.assertIn(relevant_id, selected_ids)

    def test_reproducible_benchmark_preserves_recall_and_reduces_work(self):
        results = evaluate_benchmark()["systems"]
        standard = results["standard_bloom_bm25"]
        cascade = results["certainty_cascade_bm25"]
        self.assertEqual(cascade["recall_at_10"], standard["recall_at_10"])
        self.assertLess(cascade["avg_segments_searched"], standard["avg_segments_searched"])
        self.assertLess(
            cascade["avg_estimated_reranker_tokens"],
            standard["avg_estimated_reranker_tokens"],
        )


if __name__ == "__main__":
    unittest.main()

