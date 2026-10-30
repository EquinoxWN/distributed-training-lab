"""Probe: one-rank CUDA step time with the default AdamW versus the fused CUDA AdamW.

Run from the repository root after `dtlab data`:
    .venv/Scripts/python docs/results/fused_adamw_probe.py
Same model, data and batch as the GPU runs in m1.md ("scaled", TinyStories, 32 x 128 tokens).
"""

from __future__ import annotations

import platform
import statistics
import time

import torch

from distributed_training_lab import data
from distributed_training_lab.config import PRESETS, TrainConfig
from distributed_training_lab.model import GPT
from distributed_training_lab.train import load_tokens

STEPS = 20
WARMUP = 3


def mean_step_ms(fused: bool, cfg: TrainConfig, train: torch.Tensor) -> tuple[float, list[float]]:
    """Mean step time in ms after warm-up, and every step's time."""
    device = torch.device("cuda", 0)
    torch.manual_seed(cfg.seed)
    model = GPT(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0, fused=fused)
    times: list[float] = []
    for step in range(STEPS):
        x, y = data.rank_batch(train, cfg.seq_len, cfg.global_batch, step, cfg.seed, 0, 1)
        x, y = x.to(device), y.to(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.fmean(times[WARMUP:]), times


def main() -> None:
    """Measure both optimizers back to back in one process and print the raw step times."""
    cfg = TrainConfig(
        model=PRESETS["scaled"],
        device="cuda",
        data="tinystories",
        steps=STEPS,
        global_batch=32,
        seq_len=128,
        lr=1e-3,
        warmup_steps=WARMUP,
    )
    train = load_tokens(cfg).train
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
    print(f"python {platform.python_version()}, {platform.platform()}")
    print(
        f"model scaled, batch {cfg.global_batch} x {cfg.seq_len}, {STEPS} steps, first {WARMUP} excluded"
    )
    results = {}
    for fused in (False, True):
        mean, times = mean_step_ms(fused, cfg, train)
        results[fused] = mean
        name = "fused=True " if fused else "fused=False"
        print(f"{name} step times ms: " + " ".join(f"{t:.1f}" for t in times))
        print(f"{name} mean step after warm-up: {mean:.1f} ms")
    print(f"speedup: {results[False] / results[True]:.2f}x")


if __name__ == "__main__":
    main()
