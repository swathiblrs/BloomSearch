"""Counting Bloom filter with a collision-based certainty signal."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Iterator

from .core import BloomFilterConfig


class CountingBloomFilter:
    """Approximate membership using small counters instead of individual bits.

    ``certainty`` is a bounded heuristic derived from counter congestion. A
    positive whose hashed counters are mostly 1 receives a higher score than a
    positive whose counters are shared by many inserted values. It is useful
    for prioritization, not a calibrated probability or correctness guarantee.
    """

    def __init__(self, capacity: int, error_rate: float = 0.01, counter_max: int = 65_535) -> None:
        if counter_max <= 0:
            raise ValueError("counter_max must be greater than zero")
        self.config = BloomFilterConfig.calculate(capacity, error_rate)
        self.counter_max = counter_max
        self._counters = [0] * self.config.bit_count
        self.items_added = 0

    @staticmethod
    def _to_bytes(value: str | bytes) -> bytes:
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, bytes):
            return value
        raise TypeError("Counting Bloom filter values must be str or bytes")

    def _indices(self, value: str | bytes) -> Iterator[int]:
        digest = hashlib.blake2b(
            self._to_bytes(value), digest_size=16, person=b"bloom-filter"
        ).digest()
        first = int.from_bytes(digest[:8], "big")
        second = int.from_bytes(digest[8:], "big") or 1
        for number in range(self.config.hash_count):
            yield (first + number * second) % self.config.bit_count

    def add(self, value: str | bytes) -> None:
        for index in self._indices(value):
            self._counters[index] = min(self.counter_max, self._counters[index] + 1)
        self.items_added += 1

    def add_many(self, values: Iterable[str | bytes]) -> int:
        count = 0
        for value in values:
            self.add(value)
            count += 1
        return count

    def remove(self, value: str | bytes) -> bool:
        """Remove a value known to be inserted; return False if definitely absent.

        Removing a false-positive value can decrement another value's shared
        counters, so callers must verify ownership outside the filter first.
        """
        indices = list(self._indices(value))
        if any(self._counters[index] == 0 for index in indices):
            return False
        for index in indices:
            self._counters[index] -= 1
        self.items_added = max(0, self.items_added - 1)
        return True

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, (str, bytes)):
            return False
        return all(self._counters[index] > 0 for index in self._indices(value))

    def probe_counts(self, value: str | bytes) -> tuple[int, ...]:
        return tuple(self._counters[index] for index in self._indices(value))

    def certainty(self, value: str | bytes) -> float:
        """Return a collision-certainty heuristic in [0, 1].

        Zero means definitely absent. Positive values are the geometric mean
        of reciprocal counter loads, so congested/collision-heavy probes rank
        below sparse probes.
        """
        counts = self.probe_counts(value)
        if not counts or any(count == 0 for count in counts):
            return 0.0
        return math.exp(sum(math.log(1.0 / count) for count in counts) / len(counts))

    @property
    def counters_set(self) -> int:
        return sum(counter > 0 for counter in self._counters)

    @property
    def fill_ratio(self) -> float:
        return self.counters_set / self.config.bit_count

    @property
    def logical_byte_count(self) -> int:
        """Portable 16-bit representation size, not Python object overhead."""
        return self.config.bit_count * 2

    def stats(self) -> dict[str, int | float]:
        return {
            "capacity": self.config.capacity,
            "configured_error_rate": self.config.error_rate,
            "counter_count": self.config.bit_count,
            "logical_byte_count": self.logical_byte_count,
            "hash_count": self.config.hash_count,
            "items_added": self.items_added,
            "counters_set": self.counters_set,
            "fill_ratio": self.fill_ratio,
        }
