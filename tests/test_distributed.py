"""Real multi-process runs over gloo on this machine's CPU (2 to 4 ranks)."""

from pathlib import Path
from typing import Any

import pytest

from distributed_training_lab.config import PRESETS, TrainConfig
from distributed_training_lab.train import run
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
