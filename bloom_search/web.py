"""FastAPI application for free lexical search and optional user-funded reranking."""

from __future__ import annotations

from dataclasses import asdict
import ipaddress
import json
import os
from pathlib import Path
import socket
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, HttpUrl, SecretStr

from .api import HostedModelClient, ModelAPIError
from .collections import CollectionStore
from .crawler import WebCrawler
from .engine import BloomSearchEngine
from .index import Document, SearchIndex, has_meaningful_match
from .nlp_policy import JointActionPolicy
from .weather import WeatherService, is_weather_query, weather_location


ROOT = Path(os.getenv("BLOOMSEARCH_ROOT", Path.cwd())).resolve()
EXAMPLE_DOCUMENTS = ROOT / "examples" / "documents.json"
WEB_PAGE = ROOT / "web" / "index.html"
COLLECTIONS_ROOT = Path(
    os.getenv("BLOOMSEARCH_COLLECTIONS_ROOT", ROOT / "data" / "collections")
).resolve()


def build_default_index() -> SearchIndex:
    raw = json.loads(EXAMPLE_DOCUMENTS.read_text(encoding="utf-8"))
    return SearchIndex.build((Document(**item) for item in raw), segment_size=2)


STORE = CollectionStore(COLLECTIONS_ROOT)
if not (COLLECTIONS_ROOT / "demo" / "index.json").exists():
    raw = json.loads(EXAMPLE_DOCUMENTS.read_text(encoding="utf-8"))
    STORE.save_documents("demo", (Document(**item) for item in raw))
app = FastAPI(
    title="Adaptive BloomSearch",
    description="Bloom-filter-assisted collection search with optional user-supplied model APIs",
    version="0.3.0",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    collection: str = Field(default="auto", pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    api_base: HttpUrl | None = None
    api_key: SecretStr | None = None
    model: str | None = Field(default=None, min_length=1, max_length=300)
    limit: int = Field(default=10, ge=1, le=50)
    location: str = Field(default="", max_length=200)
    context: str = Field(default="", max_length=2_000)


class CrawlRequest(BaseModel):
    start_url: HttpUrl
    collection: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    max_pages: int = Field(default=50, ge=1, le=500)
    max_depth: int = Field(default=2, ge=0, le=5)
    replace: bool = False


def validate_public_provider_url(url: str) -> None:
    """Reject non-HTTPS and private-network API targets on public deployments."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("the model API base URL must use public HTTPS")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise ValueError("private model API hostnames are not allowed")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as error:
        raise ValueError("the model API hostname could not be resolved") from error
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:
            raise ValueError("the model API must resolve only to public IP addresses")


@app.middleware("http")
async def prevent_sensitive_caching(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return WEB_PAGE.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/collections")
def collections() -> list[dict[str, object]]:
    return [asdict(item) for item in STORE.list()]


@app.get("/api/stats")
def stats(collection: str = "demo") -> dict[str, object]:
    try:
        index = STORE.load_index(collection)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "collection": collection,
        "documents": index.document_count,
        "segments": len(index.segments),
        "term_bloom_filters": [segment.term_filter.stats() for segment in index.segments],
    }


@app.post("/api/crawl")
def crawl(request: CrawlRequest) -> dict[str, object]:
    """Crawl one public domain and persist the result as a searchable collection."""
    try:
        crawler = WebCrawler(
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            delay_seconds=0.25,
        )
        documents = crawler.crawl(str(request.start_url))
        if not documents:
            raise ValueError("the crawl did not produce any searchable HTML pages")
        index = (
            STORE.save_documents(request.collection, documents)
            if request.replace
            else STORE.append_documents(request.collection, documents)
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "collection": request.collection,
        "pages_crawled": len(documents),
        "documents": index.document_count,
        "segments": len(index.segments),
    }


@app.post("/api/search")
def search(request: SearchRequest) -> JSONResponse:
    try:
        if is_weather_query(request.query):
            weather = WeatherService().current(weather_location(request.query, request.location))
            return JSONResponse(
                {
                    "mode": "live-weather",
                    "collection": "Open-Meteo live data",
                    "query": request.query,
                    "rewritten_query": request.query,
                    "segments_skipped": 0,
                    "results": [
                        {
                            "document": {
                                "id": "live-weather",
                                "title": f"Current weather for {weather.location}",
                                "text": weather.summary,
                                "url": "https://open-meteo.com/",
                            },
                            "bm25_score": 0.0,
                            "relevance_score": None,
                            "explanation": weather.summary,
                        }
                    ],
                }
            )
        selected_collection = (
            STORE.best_collection(request.query)
            if request.collection == "auto"
            else request.collection
        )
        index = STORE.load_index(selected_collection)
        policy = JointActionPolicy()
        layer_one = policy.decide_layer_one(index, request.query, context=request.context)
        api_fields = (request.api_base, request.api_key, request.model)
        if any(api_fields) and not all(api_fields):
            raise ValueError("api_base, api_key, and model must be supplied together")
        if layer_one.action == "clarify":
            question = policy.clarification_question(request.query)
            return JSONResponse(
                {
                    "mode": "clarification",
                    "collection": selected_collection,
                    "query": request.query,
                    "rewritten_query": request.query,
                    "segments_skipped": 0,
                    "results": [],
                    "clarification_question": question,
                    "nlp_policy": {
                        "layer_one": layer_one.to_dict(),
                        "layer_two": None,
                        "hosted_calls": 0,
                    },
                }
            )
        if all(api_fields):
            # The credential exists only for this request and is never persisted.
            validate_public_provider_url(str(request.api_base))
            client = HostedModelClient(
                base_url=str(request.api_base),
                api_key=request.api_key.get_secret_value(),  # type: ignore[union-attr]
                model=request.model,
            )
            engine = BloomSearchEngine(index, client)
            hosted_calls = 0
            effective_query = request.query
            if layer_one.action == "rewrite":
                effective_query = engine.rewrite_query(request.query)
                hosted_calls += 1
            candidates, skipped = index.search_bm25(effective_query, max(20, request.limit))
            layer_two = policy.decide_layer_two(candidates)
            if layer_two.should_rerank:
                ranked = engine.rerank(effective_query, candidates, limit=request.limit)
                hosted_calls += 1
                results = [asdict(result) for result in ranked]
                mode = "hosted-selective"
            else:
                filtered = [
                    (document, score)
                    for document, score in candidates[: request.limit]
                    if has_meaningful_match(effective_query, document)
                ]
                results = [
                    {
                        "document": asdict(document),
                        "bm25_score": score,
                        "relevance_score": None,
                        "explanation": index.snippet(document, effective_query),
                    }
                    for document, score in filtered
                ]
                mode = "hosted-not-needed" if hosted_calls == 0 else "hosted-rewrite-only"
            return JSONResponse(
                {
                    "mode": mode,
                    "collection": selected_collection,
                    "query": request.query,
                    "rewritten_query": effective_query,
                    "segments_skipped": skipped,
                    "results": results,
                    "nlp_policy": {
                        "layer_one": layer_one.to_dict(),
                        "layer_two": layer_two.to_dict(),
                        "hosted_calls": hosted_calls,
                    },
                }
            )

        results, skipped = index.search_bm25(request.query, request.limit)
        results = [
            (document, score)
            for document, score in results
            if has_meaningful_match(request.query, document)
        ]
        layer_two = policy.decide_layer_two(results)
        return JSONResponse(
            {
                "mode": "lexical",
                "collection": selected_collection,
                "query": request.query,
                "rewritten_query": request.query,
                "segments_skipped": skipped,
                "results": [
                    {
                        "document": asdict(document),
                        "bm25_score": score,
                        "relevance_score": None,
                        "explanation": index.snippet(document, request.query),
                    }
                    for document, score in results
                ],
                "nlp_policy": {
                    "layer_one": layer_one.to_dict(),
                    "layer_two": layer_two.to_dict(),
                    "hosted_calls": 0,
                    "execution_note": (
                        "A hosted API was not supplied, so rewrite and rerank recommendations "
                        "fall back to lexical BM25."
                        if layer_one.action == "rewrite" or layer_two.should_rerank
                        else "The policy selected the free BM25 path."
                    ),
                },
            }
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ModelAPIError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
