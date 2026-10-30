"""Check the ring all-reduce against a locally computed sum and against gloo's own all-reduce."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .comm import ring_all_reduce
from .train import free_port

SIZES = (1, 2, 3, 5, 7, 64, 1000, 4097, 100_003)


def _rank_tensor(rank: int, size: int) -> torch.Tensor:
    """The tensor rank contributes for a given size (reproducible on every rank)."""
    gen = torch.Generator().manual_seed(rank * 7919 + size)
    return torch.randn(size, generator=gen, dtype=torch.float64)


def _check(rank: int, world: int, port: int, out: str) -> None:
    """Run every size on this rank and record the worst error."""
    os.environ.setdefault("USE_LIBUV", "0")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}?use_libuv=0", rank=rank, world_size=world
    )
    try:
        worst_expected = 0.0
        worst_builtin = 0.0
        for size in SIZES:
            expected = sum(_rank_tensor(r, size) for r in range(world))
            ring = _rank_tensor(rank, size)
            ring_all_reduce(ring)
            builtin = _rank_tensor(rank, size)
            dist.all_reduce(builtin)
            worst_expected = max(worst_expected, float((ring - expected).abs().max()))
            worst_builtin = max(worst_builtin, float((ring - builtin).abs().max()))
        errs = torch.tensor([worst_expected, worst_builtin], dtype=torch.float64)
        dist.all_reduce(errs, op=dist.ReduceOp.MAX)
        if rank == 0:
            result = {
                "world_size": world,
                "sizes": list(SIZES),
                "max_abs_err_vs_expected": float(errs[0]),
                "max_abs_err_vs_gloo": float(errs[1]),
            }
            Path(out).write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def verify_allreduce(world: int, out: Path) -> dict[str, Any]:
    """Spawn world ranks, run the check, return rank 0's result."""
    if not 2 <= world <= 8:
        raise ValueError("world must be in 2..8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    mp.spawn(  # type: ignore[attr-defined,no-untyped-call]
        _check, args=(world, free_port(), str(out)), nprocs=world, join=True
    )
    result: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return result
