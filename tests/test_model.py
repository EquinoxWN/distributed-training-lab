"""The transformer: shapes, causality, parameter accounting and presets."""

import pytest
import torch

from distributed_training_lab.config import PRESETS, ModelConfig
from distributed_training_lab.model import GPT, model_state_bytes, param_count


def test_logits_shape() -> None:
    model = GPT(PRESETS["tiny"])
    out = model(torch.randint(0, 256, (3, 17)))
    assert out.shape == (3, 17, 256)


def test_causal_future_tokens_do_not_change_past_logits() -> None:
    torch.manual_seed(0)
    model = GPT(PRESETS["tiny"]).eval()
    x = torch.randint(0, 256, (1, 20))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 256
    with torch.no_grad():
        assert torch.allclose(model(x)[0, :10], model(y)[0, :10], atol=1e-6)


def test_scaled_preset_is_about_39x_tiny() -> None:
    tiny, scaled = param_count(GPT(PRESETS["tiny"])), param_count(GPT(PRESETS["scaled"]))
    assert 38 < scaled / tiny < 40
    assert 4_700_000 < scaled < 5_000_000


def test_model_state_bytes_counts_params_grads_and_two_adam_moments() -> None:
    model = GPT(PRESETS["tiny"])
    assert model_state_bytes(model) == 16 * param_count(model)


def test_initial_loss_near_uniform_and_one_step_reduces_it() -> None:
    torch.manual_seed(0)
    model = GPT(PRESETS["tiny"])
    x = torch.randint(0, 256, (4, 32))
    loss = model.loss(x[:, :-1], x[:, 1:])
    assert abs(loss.item() - torch.log(torch.tensor(256.0)).item()) < 0.2
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss.backward()  # type: ignore[no-untyped-call]
    opt.step()
    assert model.loss(x[:, :-1], x[:, 1:]).item() < loss.item()


def test_rejects_sequences_longer_than_max() -> None:
    model = GPT(ModelConfig(max_seq_len=8))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.zeros(1, 9, dtype=torch.long))


@pytest.mark.parametrize(
    "kwargs",
    [{"d_model": 30, "n_heads": 4}, {"n_layers": 0}, {"d_model": 8192}, {"vocab_size": -1}],
)
def test_invalid_model_config_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=r"must|larger"):
        ModelConfig(**kwargs)
