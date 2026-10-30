"""Data: deterministic sharding of the global batch and the verified download."""

import hashlib
from pathlib import Path

import pytest
import torch

from distributed_training_lab import data


def test_rank_slices_concatenate_to_the_single_process_batch() -> None:
    tokens = data.synthetic_corpus(5000, seed=3)
    full_x, full_y = data.rank_batch(tokens, 32, 12, step=5, seed=1, rank=0, world=1)
    for world in (2, 3, 4):
        parts = [data.rank_batch(tokens, 32, 12, 5, 1, r, world) for r in range(world)]
        assert torch.equal(torch.cat([p[0] for p in parts]), full_x)
        assert torch.equal(torch.cat([p[1] for p in parts]), full_y)


def test_targets_are_inputs_shifted_by_one() -> None:
    tokens = data.synthetic_corpus(2000)
    x, y = data.rank_batch(tokens, 16, 4, 0, 0, 0, 1)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batches_change_with_step_and_repeat_with_seed() -> None:
    a = data.global_starts(10_000, 32, 8, step=0, seed=0)
    assert torch.equal(a, data.global_starts(10_000, 32, 8, step=0, seed=0))
    assert not torch.equal(a, data.global_starts(10_000, 32, 8, step=1, seed=0))


def test_uneven_split_and_bad_rank_rejected() -> None:
    tokens = data.synthetic_corpus(2000)
    with pytest.raises(ValueError, match="multiple"):
        data.rank_batch(tokens, 16, 6, 0, 0, 0, 4)
    with pytest.raises(ValueError, match="rank"):
        data.rank_batch(tokens, 16, 8, 0, 0, 4, 4)


def test_synthetic_corpus_is_deterministic_ascii() -> None:
    a = data.synthetic_corpus(1000, seed=7)
    assert torch.equal(a, data.synthetic_corpus(1000, seed=7))
    assert len(a) == 1000
    assert int(a.max()) < 128


def test_split_holds_out_the_tail() -> None:
    s = data.split(torch.arange(100, dtype=torch.uint8), 0.1)
    assert len(s.train) == 90
    assert int(s.val[0]) == 90


def _serve(tmp: Path, payload: bytes) -> str:
    """A local file:// URL, which download must refuse."""
    src = tmp / "src.txt"
    src.write_bytes(payload)
    return src.as_uri()


class _FakeResponse:
    """Minimal context-manager response for urlopen."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, n: int) -> bytes:
        chunk, self.payload = self.payload[:n], self.payload[n:]
        return chunk


def test_download_accepts_matching_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"Once upon a time" * 100
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))
    digest = hashlib.sha256(payload).hexdigest()
    path = data.download(tmp_path, "https://example.test/x", digest, len(payload))
    assert path.read_bytes() == payload
    assert data.load_bytes(path).numel() == len(payload)


@pytest.mark.parametrize("extra", [b"", b"!"])
def test_download_refuses_tampered_or_oversized_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: bytes
) -> None:
    good = b"Once upon a time" * 100
    bad = good[:-1] + b"X" + extra
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(bad))
    with pytest.raises(data.IntegrityError):
        data.download(
            tmp_path, "https://example.test/x", hashlib.sha256(good).hexdigest(), len(good)
        )
    assert not (tmp_path / data.FILENAME).exists()
    assert not (tmp_path / "TinyStories-valid.part").exists()


def test_download_refuses_non_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="https"):
        data.download(tmp_path, _serve(tmp_path, b"x"), "0" * 64, 1)
