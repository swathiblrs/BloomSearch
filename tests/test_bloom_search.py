import tempfile
import unittest
from pathlib import Path

from bloom_search import BloomSearchEngine, Document, SearchIndex


class FakeHostedModel:
    def __init__(self):
        self.calls = 0

    def chat_json(self, *, system, user, temperature=0):
        self.calls += 1
        if self.calls == 1:
            return {"rewritten_query": "bloom filter web crawler duplicate urls"}
        return {
            "results": [
                {"id": "crawler", "relevance_score": 0.98, "explanation": "Direct answer."},
                {"id": "bloom", "relevance_score": 0.72, "explanation": "Explains the structure."},
            ]
        }


class SearchIndexTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document("bloom", "Bloom filter", "Probabilistic membership lookup"),
            Document("crawler", "Web crawler", "Bloom filters avoid duplicate URLs"),
            Document("ranking", "Search ranking", "BM25 ranks keyword matches"),
        ]

    def test_bm25_and_segment_filter(self):
        index = SearchIndex.build(self.documents, segment_size=1)
        results, skipped = index.search_bm25("duplicate URLs")
        self.assertEqual(results[0][0].id, "crawler")
        self.assertGreaterEqual(skipped, 1)

    def test_exact_duplicate_documents_are_removed(self):
        duplicate = Document("different-id", "Bloom filter", "Probabilistic membership lookup")
        index = SearchIndex.build([*self.documents, duplicate])
        self.assertEqual(index.document_count, 3)

    def test_index_round_trip(self):
        index = SearchIndex.build(self.documents, segment_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            index.save(path)
            restored = SearchIndex.load(path)
        results, _ = restored.search_bm25("keyword ranking")
        self.assertEqual(results[0][0].id, "ranking")

    def test_search_uses_remote_model_for_rewrite_and_ranking(self):
        client = FakeHostedModel()
        engine = BloomSearchEngine(SearchIndex.build(self.documents), client)
        output = engine.search("how avoid same website")
        self.assertEqual(output["rewritten_query"], "bloom filter web crawler duplicate urls")
        self.assertEqual(output["results"][0].document.id, "crawler")
        self.assertEqual(client.calls, 2)

    def test_snippet_focuses_on_matching_text(self):
        document = Document(
            "long",
            "Long document",
            "Introduction " * 40 + "Bloom filters avoid unnecessary lookups. " + "Ending " * 40,
        )
        snippet = SearchIndex.snippet(document, "Bloom lookups", length=100)
        self.assertIn("Bloom filters", snippet)
        self.assertLessEqual(len(snippet), 106)


if __name__ == "__main__":
    unittest.main()
