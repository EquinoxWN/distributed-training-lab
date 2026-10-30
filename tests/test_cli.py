"""The dtlab command: argument checks and an end-to-end sweep over 1 and 2 ranks."""

import json
from pathlib import Path

import pytest

from distributed_training_lab import cli
from distributed_training_lab.config import TrainConfig


def test_scale_sweep_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    args = ["scale", "--world-sizes", "1,2", "--steps", "5", "--global-batch", "8"]
    args += ["--seq-len", "32", "--threads-per-rank", "1", "--warmup-steps", "1", "--name", "s"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "| ranks |" in out
    saved = json.loads((tmp_path / "runs" / "s.json").read_text(encoding="utf-8"))
    assert [row["world_size"] for row in saved["scaling"]] == [1, 2]
    assert saved["scaling"][0]["efficiency"] == pytest.approx(1.0)


def test_scaling_table_math() -> None:
    def fake(n: int, step: float) -> dict[str, object]:
        return {
            "config": {"world_size": n},
            "mean_step_s": step,
            "tokens_per_s": 1.0,
            "val_loss": 2.0,
        }

    rows = cli.scaling_table([fake(1, 1.0), fake(2, 0.6), fake(4, 0.5)])
    assert rows[1]["speedup"] == pytest.approx(1 / 0.6)
    assert rows[2]["efficiency"] == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["../x", "a/b", ".hidden", ""])
def test_output_name_cannot_escape_runs_folder(name: str) -> None:
    with pytest.raises(ValueError, match="plain file name"):
        cli._safe_out(name)


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--world-size", "3", "--global-batch", "8"],
        ["train", "--world-size", "9"],
        ["train", "--seq-len", "500"],
        ["train", "--steps", "0"],
    ],
)
def test_invalid_runs_exit_with_code_2(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(args) == 2
    assert "error:" in capsys.readouterr().err


def test_tinystories_requires_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["train", "--world-size", "1", "--data", "tinystories", "--name", "t"]) == 2
    assert "dtlab data" in capsys.readouterr().err


def test_local_batch() -> None:
    assert TrainConfig(world_size=4, global_batch=16).local_batch == 4
