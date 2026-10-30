"""Data-parallel training: one process per rank, gloo process group, results as JSON."""

from __future__ import annotations

import json
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from . import data
from .comm import GradientAllReducer, max_replica_divergence
from .config import TrainConfig
from .model import GPT, model_state_bytes, param_count

DATA_DIR = Path(".tmp") / "data"


def free_port() -> int:
    """A TCP port on localhost that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def load_tokens(cfg: TrainConfig) -> data.Splits:
    """Token streams for the configured data source."""
    if cfg.data == "tinystories":
        path = DATA_DIR / data.FILENAME
        if not path.exists() or path.stat().st_size != data.SIZE:
            raise FileNotFoundError(f"{path} missing: run `dtlab data` first")
        return data.split(data.load_bytes(path))
    return data.split(data.synthetic_corpus(400_000, seed=1234))


def _device(cfg: TrainConfig, rank: int) -> torch.device:
    """Device for this rank; every rank shares GPU 0 when only one GPU exists."""
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device cuda requested but CUDA is not available")
        return torch.device("cuda", rank % torch.cuda.device_count())
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work so timers measure it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model: nn.Module, val: Tensor, cfg: TrainConfig, device: torch.device) -> float:
    """Mean loss on fixed held-out windows (same windows for every run)."""
    model.eval()
    losses = []
    for i in range(cfg.eval_batches):
        x, y = data.rank_batch(val, cfg.seq_len, cfg.global_batch, i, 10_007, 0, 1)
        logits = model(x.to(device))
        losses.append(
            float(
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)
                )
            )
        )
    model.train()
    return statistics.fmean(losses)


def worker(rank: int, cfg: TrainConfig, port: int, out: str, init_seed_offset: int) -> None:
    """Train on one rank; rank 0 writes the run summary to out."""
    os.environ.setdefault("USE_LIBUV", "0")
    torch.set_num_threads(cfg.threads_per_rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}?use_libuv=0",
        rank=rank,
        world_size=cfg.world_size,
    )
    try:
        summary = _train(rank, cfg, init_seed_offset)
        if rank == 0:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _train(rank: int, cfg: TrainConfig, init_seed_offset: int) -> dict[str, Any]:
    """The training loop for one rank."""
    device = _device(cfg, rank)
    splits = load_tokens(cfg)
    # A per-rank init seed proves the broadcast: replicas only match because rank 0's weights win.
    torch.manual_seed(cfg.seed + init_seed_offset * rank)
    model = GPT(cfg.model).to(device)
    reducer: GradientAllReducer | None = None
    net: nn.Module
    if cfg.strategy == "torch":
        # PyTorch DDP also broadcasts rank 0's weights when it wraps the model.
        net = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            bucket_cap_mb=cfg.bucket_mb,
        )
    else:
        reducer = GradientAllReducer(model, cfg.algorithm, int(cfg.bucket_mb * (1 << 20)))
        net = model
    # The fused CUDA kernel replaces hundreds of small launches per step (launch-bound here).
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.0, fused=device.type == "cuda"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    step_s: list[float] = []
    compute_s: list[float] = []
    for step in range(cfg.steps):
        x, y = data.rank_batch(
            splits.train, cfg.seq_len, cfg.global_batch, step, cfg.seed, rank, cfg.world_size
        )
        x, y = x.to(device), y.to(device)
        _sync(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = net(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()  # type: ignore[no-untyped-call]
        _sync(device)
        t1 = time.perf_counter()
        if reducer is not None:
            reducer.sync()
        opt.step()
        _sync(device)
        step_s.append(time.perf_counter() - t0)
        compute_s.append(t1 - t0)
        global_loss = loss.detach().to("cpu").reshape(1).clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        losses.append(float(global_loss.item()) / cfg.world_size)
    divergence = max_replica_divergence(model)
    val_loss = evaluate(model, splits.val, cfg, device)
    w = cfg.warmup_steps
    mean_step = statistics.fmean(step_s[w:])
    tokens_per_step = cfg.global_batch * cfg.seq_len
    comm = reducer.stats.per_step[w:] if reducer is not None else []
    summary: dict[str, Any] = {
        "config": cfg.to_dict(),
        "params": param_count(model),
        "model_state_bytes_per_rank": model_state_bytes(model),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "losses": losses,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "val_loss": val_loss,
        "replica_max_abs_diff": divergence,
        "mean_step_s": mean_step,
        "mean_compute_s": statistics.fmean(compute_s[w:]),
        "mean_allreduce_s": statistics.fmean(comm) if comm else None,
        "allreduce_calls_per_step": (reducer.stats.calls // cfg.steps) if reducer else None,
        "tokens_per_s": tokens_per_step / mean_step,
        "cuda_peak_mem_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "final_param_checksum": float(
            sum(p.detach().double().sum().item() for p in model.parameters())
        ),
    }
    return summary


def run(cfg: TrainConfig, out: Path, init_seed_offset: int = 1) -> dict[str, Any]:
    """Launch world_size ranks (world 1 runs in this process) and return rank 0's summary."""
    port = free_port()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    if cfg.world_size == 1:
        worker(0, cfg, port, str(out), init_seed_offset)
    else:
        mp.spawn(  # type: ignore[attr-defined,no-untyped-call]
            worker,
            args=(cfg, port, str(out), init_seed_offset),
            nprocs=cfg.world_size,
            join=True,
        )
    result: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return result
