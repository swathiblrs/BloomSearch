"""FastAPI web application for user-funded BloomSearch deployments."""

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
from .engine import BloomSearchEngine
from .index import Document, SearchIndex


ROOT = Path(os.getenv("BLOOMSEARCH_ROOT", Path.cwd())).resolve()
EXAMPLE_DOCUMENTS = ROOT / "examples" / "documents.json"
WEB_PAGE = ROOT / "web" / "index.html"


def build_default_index() -> SearchIndex:
    raw = json.loads(EXAMPLE_DOCUMENTS.read_text(encoding="utf-8"))
    return SearchIndex.build((Document(**item) for item in raw), segment_size=2)


INDEX = build_default_index()
app = FastAPI(
    title="BloomSearch",
    description="Bloom-filter-assisted search using a user-supplied hosted model API",
    version="0.2.0",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    api_base: HttpUrl
    api_key: SecretStr
    model: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=5, ge=1, le=10)


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


@app.get("/api/stats")
def stats() -> dict[str, object]:
    return {
        "documents": INDEX.document_count,
        "segments": len(INDEX.segments),
        "term_bloom_filters": [segment.term_filter.stats() for segment in INDEX.segments],
    }


@app.post("/api/search")
def search(request: SearchRequest) -> JSONResponse:
    # The caller's credential exists only for this request. It is never persisted
    # or included in responses, index files, environment variables, or logs.
    try:
        validate_public_provider_url(str(request.api_base))
        client = HostedModelClient(
            base_url=str(request.api_base),
            api_key=request.api_key.get_secret_value(),
            model=request.model,
        )
        output = BloomSearchEngine(INDEX, client).search(request.query, limit=request.limit)
    except (ModelAPIError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    serializable = dict(output)
    serializable["results"] = [asdict(result) for result in output["results"]]
    return JSONResponse(serializable)
