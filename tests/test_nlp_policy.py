import tempfile
import unittest
from pathlib import Path

from bloom_search.index import Document, SearchIndex
from bloom_search.nlp_policy import JointActionPolicy, SoftmaxActionClassifier


class JointActionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.index = SearchIndex.build(
            [
                Document("bloom", "Bloom filters", "probabilistic membership and hashing"),
                Document("bm25", "BM25 ranking", "lexical information retrieval scoring"),
                Document("python", "Python", "programming language and web development"),
            ],
            segment_size=1,
        )
        self.policy = JointActionPolicy()

    def test_clear_supported_query_is_kept(self):
        decision = self.policy.decide_layer_one(self.index, "How does BM25 ranking work?")
        self.assertEqual(decision.action, "keep")
        self.assertGreater(decision.features.bm25_confidence, 0)

    def test_underspecified_query_requests_clarification(self):
        decision = self.policy.decide_layer_one(self.index, "What is it?")
        self.assertEqual(decision.action, "clarify")
        self.assertIn("clarify", self.policy.clarification_question("What is it?").lower())

    def test_context_dependent_query_is_rewritten_when_context_exists(self):
        decision = self.policy.decide_layer_one(
            self.index,
            "How does it work?",
            context="We were discussing Bloom-filter membership checks.",
        )
        self.assertEqual(decision.action, "rewrite")

    def test_layer_two_reranks_close_candidates(self):
        decision = self.policy.decide_layer_two([(object(), 2.0), (object(), 1.95)])
        self.assertTrue(decision.should_rerank)


class SoftmaxActionClassifierTests(unittest.TestCase):
    def test_fit_predict_and_round_trip(self):
        classifier = SoftmaxActionClassifier(feature_count=2)
        rows = [
            ((1.0, 0.0), "keep"),
            ((0.9, 0.1), "keep"),
            ((0.0, 1.0), "rewrite"),
            ((0.1, 0.9), "rewrite"),
            ((0.0, 0.0), "clarify"),
            ((0.05, 0.05), "clarify"),
        ]
        classifier.fit(rows, epochs=1_000)
        self.assertEqual(classifier.predict((1.0, 0.0))[0], "keep")
        self.assertEqual(classifier.predict((0.0, 1.0))[0], "rewrite")
        self.assertEqual(classifier.predict((0.0, 0.0))[0], "clarify")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            classifier.save(path)
            restored = SoftmaxActionClassifier.load(path)
        self.assertEqual(restored.predict((1.0, 0.0))[0], "keep")


if __name__ == "__main__":
    unittest.main()
