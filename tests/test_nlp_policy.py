import tempfile
import unittest
from pathlib import Path

from bloom_search.index import Document, SearchIndex
from bloom_search.ambiguity import AmbiguityResult
from bloom_search.nlp_policy import (
    BM25MarginRerankPredictor,
    JointActionPolicy,
    SoftmaxActionClassifier,
)


class FakeAmbiguityDetector:
    def __init__(self, score=0.95, needs_clarification=True):
        self.score = score
        self.needs_clarification = needs_clarification

    def predict(self, query):
        return AmbiguityResult(
            self.score,
            self.needs_clarification,
            "test_ambiguity_detector",
        )


class FakeRewriteClient:
    def chat_json(self, *, system, user, temperature=0):
        return {"rewritten_query": "How do Bloom filter membership checks work?"}


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

    def test_layer_one_can_reuse_precomputed_bm25_candidates(self):
        query = "How does BM25 ranking work?"
        candidates, _ = self.index.search_bm25(query, limit=10)
        ordinary = self.policy.decide_layer_one(self.index, query)
        reused = self.policy.decide_layer_one(
            self.index,
            query,
            initial_candidates=candidates,
        )
        self.assertEqual(reused.action, ordinary.action)
        self.assertEqual(reused.features, ordinary.features)

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

    def test_trained_ambiguity_detector_feeds_joint_policy(self):
        policy = JointActionPolicy(ambiguity_detector=FakeAmbiguityDetector())
        decision = policy.decide_layer_one(self.index, "Who played it?")
        self.assertEqual(decision.action, "clarify")
        self.assertEqual(decision.source, "test_ambiguity_detector")
        self.assertIsNotNone(decision.clarification_question)

    def test_joint_policy_executes_selected_hosted_rewrite(self):
        policy = JointActionPolicy(
            ambiguity_detector=FakeAmbiguityDetector(0.1, False)
        )
        outcome = policy.apply_layer_one(
            self.index,
            "How does it work?",
            context="We were discussing Bloom filter membership checks.",
            hosted_client=FakeRewriteClient(),
        )
        self.assertEqual(outcome.decision.action, "rewrite")
        self.assertTrue(outcome.rewrite.accepted)
        self.assertEqual(
            outcome.effective_query,
            "How do Bloom filter membership checks work?",
        )

    def test_scifact_margin_rule_is_used_for_layer_two(self):
        policy = JointActionPolicy(
            rerank_predictor=BM25MarginRerankPredictor(margin_threshold=1.0)
        )
        close = policy.decide_layer_two([(object(), 4.0), (object(), 3.5)])
        separated = policy.decide_layer_two([(object(), 4.0), (object(), 2.0)])
        self.assertTrue(close.should_rerank)
        self.assertFalse(separated.should_rerank)
        self.assertEqual(close.source, "scifact_margin_rule")


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
