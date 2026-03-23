"""Tests for pruning abstraction and registry."""

from __future__ import annotations

from gnn_pruning.pruning import (
    PRUNER_REGISTRY,
    BasePruner,
    PruningContext,
    PruningPlan,
    list_pruners,
    register_pruner,
)


class DummyPruner(BasePruner):
    name = "dummy"
    category = "magnitude"
    supports_unstructured = True
    supports_structured = True

    def score(self, model, context):
        return {"score": 1.0}

    def apply(self, model, scores, context, target_sparsity, structured):
        _ = scores, context, structured
        return model, PruningPlan(name=self.name, category=self.category, target_sparsity=target_sparsity)


def test_pruning_plan_object_fields() -> None:
    plan = PruningPlan(
        name="dummy",
        category="magnitude",
        target_sparsity=0.7,
        details={"layer": "conv1"},
        achieved_sparsity=0.68,
    )

    assert plan.name == "dummy"
    assert plan.category == "magnitude"
    assert plan.details["layer"] == "conv1"
    assert plan.achieved_sparsity == 0.68


def test_pruner_registry_and_metadata_listing() -> None:
    existing = dict(PRUNER_REGISTRY)
    PRUNER_REGISTRY.clear()
    register_pruner(DummyPruner)

    rows = list_pruners()

    assert len(rows) == 1
    assert rows[0]["name"] == "dummy"
    assert rows[0]["supports_unstructured"] is True
    assert rows[0]["supports_structured"] is True

    PRUNER_REGISTRY.clear()
    PRUNER_REGISTRY.update(existing)


def test_pruning_context_fields() -> None:
    context = PruningContext(config={"foo": "bar"}, data={"graph": 1}, device="cpu", seed=123)

    assert context.config["foo"] == "bar"
    assert context.device == "cpu"
    assert context.seed == 123
