"""Space-efficient probabilistic set membership.

Bloom filters can return false positives, but never false negatives for values
that were inserted into an unchanged filter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class BloomFilterConfig:
    """Calculated parameters for a Bloom filter."""

    capacity: int
    error_rate: float
    bit_count: int
    hash_count: int

    @classmethod
    def calculate(cls, capacity: int, error_rate: float) -> "BloomFilterConfig":
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        if not 0 < error_rate < 1:
            raise ValueError("error_rate must be between zero and one")

        bit_count = max(8, math.ceil(-capacity * math.log(error_rate) / math.log(2) ** 2))
        hash_count = max(1, round((bit_count / capacity) * math.log(2)))
        return cls(capacity, error_rate, bit_count, hash_count)


class BloomFilter:
    """A Bloom filter backed by a compact bytearray.

    Values may be strings or bytes. Hash positions are produced with double
    hashing, giving k deterministic positions from two BLAKE2b digests.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        capacity: int,
        error_rate: float = 0.01,
        *,
        _bits: bytes | bytearray | None = None,
        _items_added: int = 0,
    ) -> None:
        self.config = BloomFilterConfig.calculate(capacity, error_rate)
        byte_count = (self.config.bit_count + 7) // 8
        self._bits = bytearray(_bits) if _bits is not None else bytearray(byte_count)
        if len(self._bits) != byte_count:
            raise ValueError("stored bit array does not match the filter configuration")
        if _items_added < 0:
            raise ValueError("items_added cannot be negative")
        self.items_added = _items_added

    @staticmethod
    def _to_bytes(value: str | bytes) -> bytes:
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, bytes):
            return value
        raise TypeError("Bloom filter values must be str or bytes")

    def _indices(self, value: str | bytes) -> Iterator[int]:
        data = self._to_bytes(value)
        digest = hashlib.blake2b(data, digest_size=16, person=b"bloom-filter").digest()
        first = int.from_bytes(digest[:8], "big")
        second = int.from_bytes(digest[8:], "big") or 1
        for number in range(self.config.hash_count):
            yield (first + number * second) % self.config.bit_count

    def add(self, value: str | bytes) -> bool:
        """Insert a value and return True if at least one bit changed."""
        changed = False
        for index in self._indices(value):
            byte_index, bit_index = divmod(index, 8)
            mask = 1 << bit_index
            if not self._bits[byte_index] & mask:
                changed = True
                self._bits[byte_index] |= mask
        self.items_added += 1
        return changed

    def add_many(self, values: Iterable[str | bytes]) -> int:
        """Insert values and return the number of add operations performed."""
        count = 0
        for value in values:
            self.add(value)
            count += 1
        return count

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, (str, bytes)):
            return False
        return all(
            self._bits[byte_index] & (1 << bit_index)
            for byte_index, bit_index in (divmod(index, 8) for index in self._indices(value))
        )

    @property
    def bits_set(self) -> int:
        return sum(byte.bit_count() for byte in self._bits)

    @property
    def fill_ratio(self) -> float:
        return self.bits_set / self.config.bit_count

    @property
    def estimated_false_positive_rate(self) -> float:
        m = self.config.bit_count
        k = self.config.hash_count
        n = self.items_added
        return (1 - math.exp(-k * n / m)) ** k

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.FORMAT_VERSION,
            "config": asdict(self.config),
            "items_added": self.items_added,
            "bits": base64.b64encode(self._bits).decode("ascii"),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BloomFilter":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported Bloom filter file version")
        config = data["config"]
        instance = cls(
            capacity=int(config["capacity"]),
            error_rate=float(config["error_rate"]),
            _bits=base64.b64decode(data["bits"], validate=True),
            _items_added=int(data["items_added"]),
        )
        if instance.config.bit_count != int(config["bit_count"]):
            raise ValueError("stored bit_count is inconsistent")
        if instance.config.hash_count != int(config["hash_count"]):
            raise ValueError("stored hash_count is inconsistent")
        return instance

    def stats(self) -> dict[str, int | float]:
        return {
            "capacity": self.config.capacity,
            "configured_error_rate": self.config.error_rate,
            "bit_count": self.config.bit_count,
            "byte_count": len(self._bits),
            "hash_count": self.config.hash_count,
            "items_added": self.items_added,
            "bits_set": self.bits_set,
            "fill_ratio": self.fill_ratio,
            "estimated_false_positive_rate": self.estimated_false_positive_rate,
        }

