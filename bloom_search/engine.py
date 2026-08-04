"""Hybrid search orchestration using only remotely hosted language models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from .index import Document, SearchIndex


class JSONModelClient(Protocol):
    def chat_json(self, *, system: str, user: str, temperature: float = 0) -> object: ...


@dataclass(frozen=True, slots=True)
class SearchResult:
    document: Document
    bm25_score: float
    relevance_score: float
    explanation: str


class BloomSearchEngine:
    def __init__(self, index: SearchIndex, model_client: JSONModelClient) -> None:
        self.index = index
        self.model_client = model_client

    def _rewrite_query(self, query: str) -> str:
        response = self.model_client.chat_json(
            system=(
                "You improve search queries. Preserve the user's intent and names. "
                "Return JSON only with one string field named rewritten_query."
            ),
            user=query,
        )
        if not isinstance(response, dict) or not isinstance(response.get("rewritten_query"), str):
            raise ValueError("hosted model returned an invalid query rewrite")
        return response["rewritten_query"].strip() or query

    def search(self, query: str, *, limit: int = 5, candidate_limit: int = 20) -> dict[str, object]:
        rewritten = self._rewrite_query(query)
        candidates, skipped_segments = self.index.search_bm25(rewritten, candidate_limit)
        if not candidates:
            return {
                "query": query,
                "rewritten_query": rewritten,
                "segments_skipped": skipped_segments,
                "results": [],
            }
        candidate_payload = [
            {
                "id": document.id,
                "title": document.title,
                "text": document.text[:1_500],
                "bm25_score": score,
            }
            for document, score in candidates
        ]
        response = self.model_client.chat_json(
            system=(
                "You are a search reranker hosted behind an API. Rank documents only by relevance "
                "to the query. Return JSON with a results array. Every item must contain id, "
                "relevance_score from 0 to 1, and a short explanation. Do not invent ids."
            ),
            user=json.dumps({"query": query, "documents": candidate_payload}),
        )
        if not isinstance(response, dict) or not isinstance(response.get("results"), list):
            raise ValueError("hosted model returned an invalid reranking response")
        by_id = {document.id: (document, score) for document, score in candidates}
        ranked: list[SearchResult] = []
        seen: set[str] = set()
        for item in response["results"]:
            if not isinstance(item, dict) or item.get("id") not in by_id or item["id"] in seen:
                continue
            document, bm25_score = by_id[item["id"]]
            try:
                relevance = min(1.0, max(0.0, float(item["relevance_score"])))
            except (KeyError, TypeError, ValueError):
                continue
            ranked.append(
                SearchResult(document, bm25_score, relevance, str(item.get("explanation", "")))
            )
            seen.add(document.id)
        ranked.sort(key=lambda result: (result.relevance_score, result.bm25_score), reverse=True)
        return {
            "query": query,
            "rewritten_query": rewritten,
            "segments_skipped": skipped_segments,
            "results": ranked[:limit],
        }

