"""Small, bounded and robots-aware web crawler for searchable collections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
import ipaddress
import socket
from time import sleep
from typing import Callable
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from bloom_filter import BloomFilter

from .index import Document


MAX_RESPONSE_BYTES = 2_000_000


def validate_public_web_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("crawl URLs must use public HTTP or HTTPS")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise ValueError("private crawl hostnames are not allowed")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM
            )
        }
    except socket.gaierror as error:
        raise ValueError("crawl hostname could not be resolved") from error
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("crawl URL must resolve only to public IP addresses")


def canonicalize_url(url: str) -> str:
    clean, _ = urldefrag(url)
    parsed = urlsplit(clean)
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._title = False
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        if lowered == "title":
            self._title = True
        if lowered == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1
        if lowered == "title":
            self._title = False

    def handle_data(self, data: str) -> None:
        compact = " ".join(data.split())
        if not compact or self._ignored:
            return
        self.text_parts.append(compact)
        if self._title:
            self.title_parts.append(compact)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def text(self) -> str:
        return " ".join(self.text_parts).strip()


@dataclass(frozen=True, slots=True)
class FetchedPage:
    url: str
    content_type: str
    body: bytes


Fetcher = Callable[[str], FetchedPage]


def fetch_page(url: str) -> FetchedPage:
    validate_public_web_url(url)
    request = Request(url, headers={"User-Agent": "AdaptiveBloomSearch/0.3 (+educational crawler)"})
    with urlopen(request, timeout=15) as response:
        final_url = canonicalize_url(response.geturl())
        validate_public_web_url(final_url)
        content_type = response.headers.get_content_type()
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("page exceeds the crawler response-size limit")
    return FetchedPage(final_url, content_type, body)


class WebCrawler:
    def __init__(
        self,
        *,
        max_pages: int = 100,
        max_depth: int = 2,
        delay_seconds: float = 0.2,
        fetcher: Fetcher = fetch_page,
        respect_robots: bool = True,
    ) -> None:
        if not 1 <= max_pages <= 1_000:
            raise ValueError("max_pages must be between 1 and 1000")
        if not 0 <= max_depth <= 10:
            raise ValueError("max_depth must be between 0 and 10")
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay_seconds = max(0.0, delay_seconds)
        self.fetcher = fetcher
        self.respect_robots = respect_robots

    def crawl(self, start_url: str, allowed_domains: set[str] | None = None) -> list[Document]:
        start = canonicalize_url(start_url)
        validate_public_web_url(start) if self.fetcher is fetch_page else None
        start_host = (urlsplit(start).hostname or "").lower()
        allowed = {item.lower().strip(".") for item in (allowed_domains or {start_host})}
        if start_host not in allowed:
            raise ValueError("the starting hostname must be in allowed_domains")

        robots: dict[str, RobotFileParser] = {}
        visited_filter = BloomFilter(max(1_000, self.max_pages * 20), 0.001)
        visited_exact: set[str] = set()
        queued: set[str] = {start}
        queue = deque([(start, 0)])
        documents: list[Document] = []

        while queue and len(documents) < self.max_pages:
            url, depth = queue.popleft()
            if url in visited_filter and url in visited_exact:
                continue
            visited_filter.add(url)
            visited_exact.add(url)
            if self.respect_robots and self.fetcher is fetch_page and not self._allowed_by_robots(url, robots):
                continue
            try:
                page = self.fetcher(url)
            except (OSError, ValueError):
                continue
            if page.content_type != "text/html":
                continue
            final_host = (urlsplit(page.url).hostname or "").lower()
            if final_host not in allowed:
                # Do not index or follow a redirect that leaves the approved domain.
                continue
            parser = PageParser()
            parser.feed(page.body.decode("utf-8", errors="replace"))
            if parser.text:
                final_url = canonicalize_url(page.url)
                documents.append(
                    Document(
                        id=sha256(final_url.encode("utf-8")).hexdigest()[:24],
                        title=parser.title or final_url,
                        text=parser.text,
                        url=final_url,
                    )
                )
            if depth < self.max_depth:
                for href in parser.links:
                    candidate = canonicalize_url(urljoin(page.url, href))
                    parsed = urlsplit(candidate)
                    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower() not in allowed:
                        continue
                    if candidate not in queued and candidate not in visited_exact:
                        queued.add(candidate)
                        queue.append((candidate, depth + 1))
            if self.delay_seconds:
                sleep(self.delay_seconds)
        return documents

    @staticmethod
    def _allowed_by_robots(url: str, cache: dict[str, RobotFileParser]) -> bool:
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in cache:
            robots = RobotFileParser(f"{origin}/robots.txt")
            try:
                robots.read()
            except OSError:
                # A missing or unreachable robots file does not prohibit crawling.
                robots.parse([])
            cache[origin] = robots
        return cache[origin].can_fetch("AdaptiveBloomSearch", url)
