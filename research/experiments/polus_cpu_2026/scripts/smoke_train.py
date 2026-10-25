"""FP32 CUDA training check for PyTorch 1.2+, with no external dataset."""
from __future__ import print_function

import json
import os
import platform
import time

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check LSF allocation, --nv, and host driver.")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = torch.device("cuda:0")
    metadata = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "visible_gpus": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    print(json.dumps(metadata, sort_keys=True), flush=True)

    # Exercise cuDNN forward/backward separately from the convergence check.
    convolution = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    convolution(torch.randn(4, 3, 32, 32, device=device)).pow(2).mean().backward()
    torch.cuda.synchronize()
    del convolution

    x = torch.randn(512, 64, device=device)
    teacher = torch.randn(64, 4, device=device)
    y = x.mm(teacher).argmax(dim=1)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        initial = criterion(model(x), y).item()
    started = time.time()
    for step in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss at step {}".format(step))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    with torch.no_grad():
        final = criterion(model(x), y).item()
        expected = model(x).detach().clone()
    if final >= initial * 0.5:
        raise RuntimeError("Loss did not decrease sufficiently: {} -> {}".format(initial, final))

    checkpoint_path = os.environ.get("POLUS_CHECKPOINT", "smoke-checkpoint.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": 100, "seed": 42}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    restored = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4)
    ).to(device)
    restored.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        if not torch.allclose(restored(x), expected):
            raise RuntimeError("Checkpoint restore changed model output")
    print(json.dumps({"status": "PASS", "initial_loss": initial, "final_loss": final,
                      "seconds": time.time() - started,
                      "checkpoint": checkpoint_path}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
