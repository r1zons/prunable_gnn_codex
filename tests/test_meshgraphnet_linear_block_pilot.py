"""Check the experimental ridge fit and processor replacement without datasets."""

from pathlib import Path
import sys

import numpy as np
import torch

from gnn_pruning.models.meshgraphnet import MeshGraphNet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_cylinderflow_linear_block_pilot import AffinePair, check_variants, fit_ridge, variant


def test_centered_ridge_matches_closed_form():
    x = np.array([[1., 2.], [3., 2.], [1., 4.], [3., 4.]])
    y = x * [3., 4.] + [5., 6.]
    w, bias, penalty = fit_ridge([x], [y], .5)
    assert penalty == .5
    torch.testing.assert_close(w, torch.diag(torch.tensor([2., 8. / 3.])))
    torch.testing.assert_close(bias, torch.tensor([7., 10.]))


def test_bypass_and_zero_affine_match():
    from gnn_pruning.data.cylinderflow import mesh_from_triangles

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        mesh = mesh_from_triangles(torch.tensor([[0, 1, 2]]),
                                   torch.tensor([[0., 0.], [1., 0.], [0., 1.]]),
                                   torch.tensor([0, 4, 5]))
        dense = MeshGraphNet().eval()
        pair = AffinePair()
        with torch.no_grad():
            for p in pair.parameters():
                p.zero_()
        models = {"A": dense, "B": variant(dense, 6, 7), "C": variant(dense, 6, 7, pair)}
        initial = torch.zeros(3, 2)
        counts = check_variants(models, dense, mesh, initial, 6, 7)
        assert {k: sum(v) for k, v in counts.items()} == {"A": 15, "B": 13, "C": 14}
        with torch.inference_mode():
            torch.testing.assert_close(models["B"].step(mesh, initial),
                                       models["C"].step(mesh, initial), rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)
