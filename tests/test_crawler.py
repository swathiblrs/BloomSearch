import tempfile
import unittest
from pathlib import Path

from bloom_search.collections import CollectionStore
from bloom_search.crawler import FetchedPage, WebCrawler, canonicalize_url
from bloom_search.index import Document, has_meaningful_match


class FakeWebsite:
    def __init__(self):
        self.pages = {
            "https://example.com/": """
                <html><head><title>Search Home</title></head><body>
                <h1>Bloom search documentation</h1>
                <a href="/bm25">BM25 guide</a>
                <a href="https://outside.example/private">Outside</a>
                </body></html>
            """,
            "https://example.com/bm25": """
                <html><head><title>BM25 Guide</title></head><body>
                BM25 ranks documents using term frequency and document length.
                <a href="/">Home</a>
                </body></html>
            """,
        }

    def __call__(self, url: str) -> FetchedPage:
        normalized = canonicalize_url(url)
        if normalized not in self.pages:
            raise OSError("not found")
        return FetchedPage(normalized, "text/html", self.pages[normalized].encode())


class CrawlerTests(unittest.TestCase):
    def test_crawl_stays_on_domain_and_stops_at_limits(self):
        documents = WebCrawler(
            max_pages=10,
            max_depth=2,
            delay_seconds=0,
            fetcher=FakeWebsite(),
        ).crawl("https://example.com/")
        self.assertEqual(len(documents), 2)
        self.assertEqual({document.title for document in documents}, {"Search Home", "BM25 Guide"})
        self.assertTrue(all(document.url.startswith("https://example.com") for document in documents))

    def test_crawled_pages_can_be_persisted_and_searched(self):
        documents = WebCrawler(
            max_pages=10,
            max_depth=2,
            delay_seconds=0,
            fetcher=FakeWebsite(),
        ).crawl("https://example.com/")
        with tempfile.TemporaryDirectory() as directory:
            store = CollectionStore(Path(directory), segment_size=1)
            store.save_documents("docs", documents)
            restored = store.load_index("docs")
            results, _ = restored.search_bm25("term frequency document length")
            self.assertEqual(results[0][0].title, "BM25 Guide")
            self.assertEqual(store.list()[0].documents, 2)

    def test_collection_name_cannot_escape_storage_root(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CollectionStore(directory)
            with self.assertRaises(ValueError):
                store.save_documents("../escape", [])

    def test_automatically_selects_best_matching_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CollectionStore(directory)
            store.save_documents(
                "python-docs",
                [Document("python", "Python", "Functions and modules")],
            )
            store.save_documents(
                "garden-docs",
                [Document("garden", "Gardening", "Tomato soil and watering")],
            )
            self.assertEqual(store.best_collection("How should I water tomatoes?"), "garden-docs")

    def test_rejects_match_supported_only_by_common_words(self):
        document = Document("bloom", "Bloom filters", "A Bloom filter is a data structure")
        self.assertFalse(has_meaningful_match("What is my name?", document))
        self.assertTrue(has_meaningful_match("What is a Bloom filter?", document))


if __name__ == "__main__":
    unittest.main()
