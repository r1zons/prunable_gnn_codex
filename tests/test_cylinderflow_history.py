"""Check that rollout feeds back predictions and preserves boundary values."""
from types import SimpleNamespace
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_cylinderflow_positions import forecast


def test_feedback_without_using_future_and_with_fixed_boundary():
    mesh = SimpleNamespace(active=torch.tensor([True, False]))
    model = SimpleNamespace(step=lambda m, v: v + m.active[:, None])
    frames = torch.tensor([[[0.], [5.]], [[2.], [7.]], [[99.], [99.]], [[99.], [99.]]])
    two = forecast(model, mesh, frames, 2, [0., 1.], collect=True)
    assert two.tolist() == [[[4.], [7.]], [[6.], [7.]]]
    assert torch.equal(forecast(model, mesh, frames, 2), forecast(model, mesh, frames, 2, [1.]))
