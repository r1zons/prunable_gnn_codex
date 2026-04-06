"""Dense trainer for node-classification models."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn


@dataclass
class TrainResult:
    """Result object returned by trainer.fit."""

    best_epoch: int
    epochs_ran: int
    best_val_loss: float
    final_train_loss: float
    final_val_loss: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "best_epoch": float(self.best_epoch),
            "epochs_ran": float(self.epochs_ran),
            "best_val_loss": float(self.best_val_loss),
            "final_train_loss": float(self.final_train_loss),
            "final_val_loss": float(self.final_val_loss),
        }


class DenseTrainer:
    """Trainer implementing dense node-classification training with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        early_stopping_patience: int,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.max_epochs = int(max_epochs)
        self.early_stopping_patience = int(early_stopping_patience)

    def fit(
        self,
        data: Any,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> TrainResult:
        """Train model with early stopping and optional resume state."""
        data = data.to(self.device)
        train_idx = train_idx.to(self.device)
        val_idx = val_idx.to(self.device)

        start_epoch = 0
        best_epoch = -1
        best_val_loss = float("inf")
        best_state_dict = None

        if resume_state is not None:
            self.model.load_state_dict(resume_state["model_state_dict"])
            self.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            start_epoch = int(resume_state.get("epoch", -1)) + 1
            best_epoch = int(resume_state.get("best_epoch", -1))
            best_val_loss = float(resume_state.get("best_val_loss", float("inf")))

        epochs_without_improvement = 0
        final_train_loss = float("inf")
        final_val_loss = float("inf")
        last_epoch = start_epoch - 1

        for epoch in range(start_epoch, self.max_epochs):
            epoch_start = time.perf_counter()
            last_epoch = epoch
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(data)
            train_loss = self.loss_fn(logits[train_idx], data.y[train_idx])
            train_loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                logits = self.model(data)
                val_loss = self.loss_fn(logits[val_idx], data.y[val_idx])
                train_acc = float((logits[train_idx].argmax(dim=-1) == data.y[train_idx]).float().mean().item())
                val_acc = float((logits[val_idx].argmax(dim=-1) == data.y[val_idx]).float().mean().item())

            final_train_loss = float(train_loss.item())
            final_val_loss = float(val_loss.item())

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "epoch": float(epoch + 1),
                        "max_epochs": float(self.max_epochs),
                        "train_loss": final_train_loss,
                        "val_loss": final_val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "best_epoch": float(best_epoch + 1 if best_epoch >= 0 else 0),
                        "early_stopping_counter": float(epochs_without_improvement),
                        "elapsed_sec": float(time.perf_counter() - epoch_start),
                    }
                )

            if epochs_without_improvement >= self.early_stopping_patience:
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        if self.max_epochs <= 0:
            epochs_ran = 0
        elif last_epoch >= 0:
            epochs_ran = last_epoch + 1
        else:
            epochs_ran = start_epoch
        return TrainResult(
            best_epoch=best_epoch,
            epochs_ran=epochs_ran,
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
        )
