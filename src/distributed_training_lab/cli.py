"""Command line: download data, train one run, sweep world sizes, verify the all-reduce."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import data
from .config import PRESETS, TrainConfig
from .train import DATA_DIR, run
from .verify import verify_allreduce

RUNS = Path(".tmp") / "runs"


def _add_train_args(p: argparse.ArgumentParser) -> None:
    """Options shared by train and scale."""
    p.add_argument("--model", choices=sorted(PRESETS), default="tiny")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--strategy", choices=["manual", "torch"], default="manual")
    p.add_argument("--algorithm", choices=["ring", "builtin"], default="ring")
    p.add_argument("--data", choices=["synthetic", "tinystories"], default="synthetic")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--global-batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bucket-mb", type=float, default=4.0)
    p.add_argument("--threads-per-rank", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=3)


def _config(a: argparse.Namespace, world: int) -> TrainConfig:
    """Build a validated TrainConfig from parsed options."""
    return TrainConfig(
        model=PRESETS[a.model],
        world_size=world,
        device=a.device,
        strategy=a.strategy,
        algorithm=a.algorithm,
        data=a.data,
        steps=a.steps,
        global_batch=a.global_batch,
        seq_len=a.seq_len,
        lr=a.lr,
        seed=a.seed,
        bucket_mb=a.bucket_mb,
        threads_per_rank=a.threads_per_rank,
        warmup_steps=a.warmup_steps,
    )


def _safe_out(name: str) -> Path:
    """Result path under .tmp/runs; the name may not leave that folder."""
    if not name or Path(name).name != name or name.startswith("."):
        raise ValueError("--name must be a plain file name")
    return RUNS / f"{name}.json"


def _line(r: dict[str, Any]) -> str:
    """One human-readable summary line for a run."""
    c = r["config"]
    comm = r["mean_allreduce_s"]
    comm_txt = f" allreduce {comm * 1000:.1f} ms" if comm is not None else ""
    return (
        f"world={c['world_size']} device={r['device_name']} params={r['params']:,} "
        f"loss {r['first_loss']:.3f} -> {r['final_loss']:.3f} val {r['val_loss']:.3f} "
        f"step {r['mean_step_s'] * 1000:.1f} ms{comm_txt} tokens/s {r['tokens_per_s']:,.0f} "
        f"replica_diff {r['replica_max_abs_diff']:.1e}"
    )


def scaling_table(results: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Strong-scaling speedup and efficiency against the smallest world size."""
    base = results[0]
    n0 = base["config"]["world_size"]
    rows = []
    for r in results:
        n = r["config"]["world_size"]
        speedup = base["mean_step_s"] / r["mean_step_s"]
        rows.append(
            {
                "world_size": n,
                "mean_step_ms": r["mean_step_s"] * 1000,
                "tokens_per_s": r["tokens_per_s"],
                "speedup": speedup,
                "efficiency": speedup * n0 / n,
                "val_loss": r["val_loss"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Entry point for the dtlab command."""
    parser = argparse.ArgumentParser(prog="dtlab", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download the pinned TinyStories validation file (19 MB)")
    t = sub.add_parser("train", help="one data-parallel run")
    _add_train_args(t)
    t.add_argument("--world-size", type=int, default=2)
    t.add_argument("--name", default="train")
    s = sub.add_parser("scale", help="the same run at several world sizes")
    _add_train_args(s)
    s.add_argument("--world-sizes", default="1,2,4")
    s.add_argument("--name", default="scale")
    v = sub.add_parser("verify", help="ring all-reduce against gloo and the expected sum")
    v.add_argument("--world-size", type=int, default=4)
    a = parser.parse_args(argv)
    try:
        if a.cmd == "data":
            print(f"verified {data.download(DATA_DIR)}")
            return 0
        if a.cmd == "verify":
            r = verify_allreduce(a.world_size, RUNS / f"verify-{a.world_size}.json")
            ok = r["max_abs_err_vs_expected"] < 1e-9 and r["max_abs_err_vs_gloo"] < 1e-9
            print(json.dumps(r))
            print(f"ring all-reduce, {a.world_size} ranks: {'PASS' if ok else 'FAIL'}")
            return 0 if ok else 1
        out = _safe_out(a.name)
        if a.cmd == "train":
            r = run(_config(a, a.world_size), out)
            print(_line(r))
            return 0
        worlds = [int(w) for w in a.world_sizes.split(",")]
        results = []
        for w in worlds:
            r = run(_config(a, w), out.with_name(f"{out.stem}-w{w}.json"))
            print(_line(r), flush=True)
            results.append(r)
        rows = scaling_table(results)
        out.write_text(json.dumps({"runs": results, "scaling": rows}, indent=2), encoding="utf-8")
        print("| ranks | step ms | tokens/s | speedup | efficiency | val loss |")
        print("|---|---|---|---|---|---|")
        for row in rows:
            print(
                f"| {row['world_size']} | {row['mean_step_ms']:.1f} | {row['tokens_per_s']:,.0f} "
                f"| {row['speedup']:.2f}x | {row['efficiency']:.0%} | {row['val_loss']:.4f} |"
            )
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
