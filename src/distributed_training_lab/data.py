"""Byte-level training data: a pinned TinyStories file or a generated corpus, sharded by rank.

The TinyStories validation file comes from one fixed dataset commit (the same file
tiny-transformer-scratch uses) and is refused unless its size and SHA-256 match.
"""

from __future__ import annotations

import hashlib
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # dataset commit of 2024-08-12
URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    f"{REVISION}/TinyStories-valid.txt"
)
SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
SIZE = 19_447_282
FILENAME = "TinyStories-valid.txt"
VAL_FRACTION = 0.05


class IntegrityError(ValueError):
    """The bytes do not match the pinned size or SHA-256."""


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(dest: Path, url: str = URL, sha256: str = SHA256, size: int = SIZE) -> Path:
    """Fetch the pinned file into dest over HTTPS; keep it only if size and hash match."""
    if not url.startswith("https://"):
        raise ValueError("only https downloads are allowed")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / FILENAME
    if target.exists() and target.stat().st_size == size and sha256_file(target) == sha256:
        return target
    partial = target.with_suffix(".part")
    digest, got = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(partial, "wb") as out:  # noqa: S310
            while chunk := resp.read(1 << 20):
                got += len(chunk)
                if got > size:
                    raise IntegrityError("download is larger than the pinned size")
                digest.update(chunk)
                out.write(chunk)
        if got != size or digest.hexdigest() != sha256:
            raise IntegrityError("download does not match the pinned size and SHA-256")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def load_bytes(path: Path, max_bytes: int = SIZE) -> Tensor:
    """Read at most max_bytes of a text file as a uint8 token tensor."""
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


_NAMES = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Sue"]
_VERBS = ["sees", "finds", "likes", "wants", "holds", "draws"]
_ADJS = ["red", "big", "small", "happy", "blue", "old"]
_NOUNS = ["ball", "cat", "tree", "box", "kite", "dog", "cake"]


def synthetic_corpus(n_bytes: int, seed: int = 0) -> Tensor:
    """A deterministic story-like byte corpus with learnable structure, for tests and CI."""
    if n_bytes < 1:
        raise ValueError("n_bytes must be positive")
    rng = random.Random(seed)  # noqa: S311 - reproducible data, not security
    parts: list[str] = []
    total = 0
    while total < n_bytes:
        s = (
            f"{rng.choice(_NAMES)} {rng.choice(_VERBS)} the "
            f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)}. "
        )
        parts.append(s)
        total += len(s)
    raw = "".join(parts).encode("ascii")[:n_bytes]
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


@dataclass(frozen=True)
class Splits:
    """Training and held-out token streams."""

    train: Tensor
    val: Tensor


def split(tokens: Tensor, val_fraction: float = VAL_FRACTION) -> Splits:
    """Hold out the tail of the stream for evaluation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    cut = int(len(tokens) * (1.0 - val_fraction))
    return Splits(tokens[:cut], tokens[cut:])


def global_starts(n_tokens: int, seq_len: int, global_batch: int, step: int, seed: int) -> Tensor:
    """Window start offsets for the whole global batch of one step (same on every rank)."""
    if n_tokens <= seq_len + 1:
        raise ValueError("token stream is shorter than one training window")
    gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
    return torch.randint(0, n_tokens - seq_len - 1, (global_batch,), generator=gen)


def rank_batch(
    tokens: Tensor, seq_len: int, global_batch: int, step: int, seed: int, rank: int, world: int
) -> tuple[Tensor, Tensor]:
    """This rank's contiguous slice of the global batch: inputs and next-token targets."""
    if global_batch % world:
        raise ValueError("global_batch must be a multiple of world")
    if not 0 <= rank < world:
        raise ValueError("rank out of range")
    per = global_batch // world
    starts = global_starts(len(tokens), seq_len, global_batch, step, seed)[
        rank * per : (rank + 1) * per
    ]
    offsets = torch.arange(seq_len + 1)
    windows = tokens[starts.unsqueeze(1) + offsets].long()
    return windows[:, :-1], windows[:, 1:]
