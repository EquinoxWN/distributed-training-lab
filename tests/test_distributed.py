"""Real multi-process runs over gloo on this machine's CPU (2 to 4 ranks)."""

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_training_lab.comm import GradientAllReducer, max_replica_divergence
from distributed_training_lab.config import PRESETS, TrainConfig
from distributed_training_lab.train import free_port, run
from distributed_training_lab.verify import verify_allreduce


def _cfg(**kw: Any) -> TrainConfig:
    """A small, fast run."""
    base: dict[str, Any] = {
        "model": PRESETS["tiny"],
        "steps": 6,
        "global_batch": 8,
        "seq_len": 32,
        "threads_per_rank": 1,
        "warmup_steps": 1,
        "eval_batches": 2,
    }
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def single(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The reference run: one process, whole global batch."""
    return run(_cfg(world_size=1), tmp_path_factory.mktemp("w1") / "r.json")


@pytest.mark.parametrize("world", [2, 3, 4])
def test_ring_all_reduce_matches_expected_sum_and_gloo(world: int, tmp_path: Path) -> None:
    r = verify_allreduce(world, tmp_path / "v.json")
    assert r["max_abs_err_vs_expected"] < 1e-9
    assert r["max_abs_err_vs_gloo"] < 1e-9


@pytest.mark.parametrize(("world", "algorithm"), [(2, "ring"), (4, "ring"), (2, "builtin")])
def test_ddp_matches_single_process_training(
    world: int, algorithm: str, single: dict[str, Any], tmp_path: Path
) -> None:
    r = run(_cfg(world_size=world, algorithm=algorithm), tmp_path / "r.json")
    # Replicas start from different seeds, so equality proves the broadcast and the all-reduce.
    assert r["replica_max_abs_diff"] == 0.0
    assert r["losses"] == pytest.approx(single["losses"], rel=1e-4, abs=1e-4)
    assert r["final_param_checksum"] == pytest.approx(single["final_param_checksum"], rel=1e-3)
    assert r["final_loss"] < r["first_loss"]
    assert r["allreduce_calls_per_step"] >= 1


def test_manual_ddp_matches_pytorch_ddp(single: dict[str, Any], tmp_path: Path) -> None:
    r = run(_cfg(world_size=2, strategy="torch"), tmp_path / "r.json")
    assert r["replica_max_abs_diff"] == 0.0
    assert r["losses"] == pytest.approx(single["losses"], rel=1e-4, abs=1e-4)
    assert r["mean_allreduce_s"] is None


def test_single_process_summary_fields(single: dict[str, Any]) -> None:
    assert single["params"] > 0
    assert single["model_state_bytes_per_rank"] == 16 * single["params"]
    assert single["tokens_per_s"] > 0
    assert len(single["losses"]) == 6
    assert single["mean_step_s"] >= single["mean_compute_s"]


def _divergence_worker(rank: int, port: int, out: str) -> None:
    """Ranks 1 and 2 nudge one weight by 0.25; every rank records the divergence it sees."""
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}?use_libuv=0", rank=rank, world_size=3
    )
    try:
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3)
        if rank > 0:
            with torch.no_grad():
                model.weight[2, 1] += 0.25
        Path(f"{out}.{rank}").write_text(str(max_replica_divergence(model)), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def test_replica_check_reports_a_diverged_rank_on_every_rank(tmp_path: Path) -> None:
    out = str(tmp_path / "div")
    mp.spawn(_divergence_worker, args=(free_port(), out), nprocs=3, join=True)  # type: ignore[attr-defined,no-untyped-call]
    for rank in range(3):
        assert float(Path(f"{out}.{rank}").read_text(encoding="utf-8")) == pytest.approx(0.25)


def _average_worker(rank: int, port: int, out: str) -> None:
    """Each rank sets its own gradients, syncs, and records what it ends up with."""
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}?use_libuv=0", rank=rank, world_size=3
    )
    try:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 1))
        reducer = GradientAllReducer(model, "ring", bucket_bytes=16)  # several buckets
        params = list(model.parameters())
        params[0].grad = torch.full_like(params[0], float(rank + 1))
        params[1].grad = None if rank == 0 else torch.full_like(params[1], 3.0)
        params[2].grad = torch.full_like(params[2], float(rank))
        reducer.sync()
        got = [float(p.grad.mean()) for p in params if p.grad is not None]
        Path(f"{out}.{rank}").write_text(" ".join(map(str, got)), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def test_gradients_are_averaged_and_missing_ones_count_as_zero(tmp_path: Path) -> None:
    # Adam is almost invariant to a constant gradient scale, so loss equality alone cannot
    # tell a sum from a mean: check the synchronised gradients directly.
    out = str(tmp_path / "avg")
    mp.spawn(_average_worker, args=(free_port(), out), nprocs=3, join=True)  # type: ignore[attr-defined,no-untyped-call]
    for rank in range(3):
        got = [float(x) for x in Path(f"{out}.{rank}").read_text(encoding="utf-8").split()]
        assert got == pytest.approx([2.0, 2.0, 1.0])
