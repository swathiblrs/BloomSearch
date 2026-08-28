"""Persistent named document collections for BloomSearch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from .index import Document, SearchIndex


COLLECTION_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    name: str
    documents: int
    segments: int


class CollectionStore:
    """Store source documents and built indexes under a controlled directory."""

    def __init__(self, root: str | Path, *, segment_size: int = 100) -> None:
        self.root = Path(root)
        self.segment_size = segment_size
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_name(name: str) -> str:
        normalized = name.strip().lower()
        if not COLLECTION_NAME.fullmatch(normalized):
            raise ValueError(
                "collection names must contain only lowercase letters, numbers, hyphens, and underscores"
            )
        return normalized

    def _directory(self, name: str) -> Path:
        return self.root / self.validate_name(name)

    def _documents_path(self, name: str) -> Path:
        return self._directory(name) / "documents.json"

    def _index_path(self, name: str) -> Path:
        return self._directory(name) / "index.json"

    def save_documents(self, name: str, documents: Iterable[Document]) -> SearchIndex:
        normalized = self.validate_name(name)
        directory = self._directory(normalized)
        directory.mkdir(parents=True, exist_ok=True)
        materialized = list(documents)
        index = SearchIndex.build(materialized, segment_size=self.segment_size)
        unique = [document for segment in index.segments for document in segment.documents]
        self._documents_path(normalized).write_text(
            json.dumps([asdict(document) for document in unique], indent=2) + "\n",
            encoding="utf-8",
        )
        index.save(self._index_path(normalized))
        return index

    def append_documents(self, name: str, documents: Iterable[Document]) -> SearchIndex:
        existing = self.load_documents(name) if self._documents_path(name).exists() else []
        return self.save_documents(name, [*existing, *documents])

    def load_documents(self, name: str) -> list[Document]:
        path = self._documents_path(name)
        if not path.exists():
            raise KeyError(f"collection not found: {name}")
        return [Document(**item) for item in json.loads(path.read_text(encoding="utf-8"))]

    def load_index(self, name: str) -> SearchIndex:
        path = self._index_path(name)
        if not path.exists():
            raise KeyError(f"collection not found: {name}")
        return SearchIndex.load(path)

    def list(self) -> list[CollectionInfo]:
        collections: list[CollectionInfo] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or not (path / "index.json").exists():
                continue
            index = SearchIndex.load(path / "index.json")
            collections.append(CollectionInfo(path.name, index.document_count, len(index.segments)))
        return collections

