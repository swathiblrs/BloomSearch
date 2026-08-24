"""CLI for building an index and searching it through a hosted model API."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .api import HostedModelClient, ModelAPIError
from .engine import BloomSearchEngine
from .index import Document, SearchIndex
from .research import write_results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bloom-filter-assisted API-first search engine")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build an index from a JSON document list")
    build.add_argument("documents", type=Path)
    build.add_argument("index", type=Path)
    build.add_argument("--segment-size", type=int, default=100)
    build.add_argument("--error-rate", type=float, default=0.01)
    search = commands.add_parser("search", help="search using the configured hosted model API")
    search.add_argument("index", type=Path)
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    stats = commands.add_parser("stats", help="show index and Bloom-filter statistics")
    stats.add_argument("index", type=Path)
    benchmark = commands.add_parser(
        "benchmark", help="run the certainty-aware synthetic research benchmark"
    )
    benchmark.add_argument(
        "--output", type=Path, default=Path("research/results/latest.json")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "benchmark":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results = write_results(args.output)
        print(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")
        return 0

    if args.command == "build":
        raw = json.loads(args.documents.read_text(encoding="utf-8"))
        index = SearchIndex.build(
            (Document(**item) for item in raw),
            segment_size=args.segment_size,
            error_rate=args.error_rate,
        )
        index.save(args.index)
        print(f"Indexed {index.document_count} unique documents in {len(index.segments)} segments")
        return 0

    index = SearchIndex.load(args.index)
    if args.command == "stats":
        print(
            json.dumps(
                {
                    "documents": index.document_count,
                    "segments": len(index.segments),
                    "term_bloom_filters": [segment.term_filter.stats() for segment in index.segments],
                },
                indent=2,
            )
        )
        return 0

    try:
        output = BloomSearchEngine(index, HostedModelClient()).search(args.query, limit=args.limit)
    except (ValueError, ModelAPIError) as error:
        parser.error(str(error))
    serializable = dict(output)
    serializable["results"] = [asdict(result) for result in output["results"]]
    print(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
