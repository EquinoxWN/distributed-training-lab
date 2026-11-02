"""Bucketing and ring cost accounting (no processes needed)."""

import pytest
import torch
from torch import nn

from distributed_training_lab.comm import make_buckets, ring_bytes_sent_per_rank
from distributed_training_lab.config import PRESETS
from distributed_training_lab.model import GPT


def test_buckets_cover_every_parameter_once_in_reverse_order() -> None:
    params = list(GPT(PRESETS["scaled"]).parameters())
    buckets = make_buckets(params, 4 << 20)
    flat = [p for b in buckets for p in b]
    assert [id(p) for p in flat] == [id(p) for p in reversed(params)]
    assert len(buckets) > 1


def test_bucket_size_cap_respected_unless_one_tensor_is_bigger() -> None:
    params = [nn.Parameter(torch.zeros(n)) for n in (10, 20, 30, 500, 5)]
    for cap in (40, 100, 200, 4000):
        for b in make_buckets(params, cap):
            nbytes = sum(p.numel() * 4 for p in b)
            assert nbytes <= cap or len(b) == 1


def test_parameters_that_exactly_fill_the_cap_share_one_bucket() -> None:
    # Reversed order: 100 + 60 bytes fill the 160-byte cap exactly, then 40 bytes start a new bucket.
    params = [nn.Parameter(torch.zeros(n)) for n in (10, 15, 25)]
    assert [len(b) for b in make_buckets(params, 160)] == [2, 1]
    assert [len(b) for b in make_buckets(params, 200)] == [3]


def test_bucket_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_buckets([nn.Parameter(torch.zeros(1))], 0)


@pytest.mark.parametrize(("world", "expected"), [(1, 0), (2, 400), (4, 600), (3, 544)])
def test_ring_sends_two_n_minus_one_over_n_of_the_payload(world: int, expected: int) -> None:
    # 100 float32 = 400 bytes; ring sends 2 (N-1) chunks of ceil(100 / N) elements.
    assert ring_bytes_sent_per_rank(100, world) == expected
