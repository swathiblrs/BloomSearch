"""Command-line interface for creating and exploring Bloom filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import BloomFilter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and query persistent Bloom filters")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create an empty filter")
    create.add_argument("file", type=Path)
    create.add_argument("--capacity", type=int, required=True)
    create.add_argument("--error-rate", type=float, default=0.01)

    add = commands.add_parser("add", help="add one or more values")
    add.add_argument("file", type=Path)
    add.add_argument("values", nargs="+")

    check = commands.add_parser("check", help="test one or more values")
    check.add_argument("file", type=Path)
    check.add_argument("values", nargs="+")

    stats = commands.add_parser("stats", help="show filter parameters and usage")
    stats.add_argument("file", type=Path)

    demo = commands.add_parser("demo", help="run the geeks/nerd example from the slides")
    demo.add_argument("--capacity", type=int, default=100)
    demo.add_argument("--error-rate", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "create":
        bloom = BloomFilter(args.capacity, args.error_rate)
        bloom.save(args.file)
        print(f"Created {args.file} ({bloom.config.bit_count} bits, {bloom.config.hash_count} hashes)")
        return 0

    if args.command == "demo":
        bloom = BloomFilter(args.capacity, args.error_rate)
        bloom.add_many(["geeks", "nerd"])
        for value in ("geeks", "nerd", "cat"):
            verdict = "probably present" if value in bloom else "definitely not present"
            print(f"{value}: {verdict}")
        print(json.dumps(bloom.stats(), indent=2))
        return 0

    bloom = BloomFilter.load(args.file)
    if args.command == "add":
        bloom.add_many(args.values)
        bloom.save(args.file)
        print(f"Added {len(args.values)} value(s); total add operations: {bloom.items_added}")
    elif args.command == "check":
        for value in args.values:
            verdict = "probably present" if value in bloom else "definitely not present"
            print(f"{value}: {verdict}")
    elif args.command == "stats":
        print(json.dumps(bloom.stats(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

