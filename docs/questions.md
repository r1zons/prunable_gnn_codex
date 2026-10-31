# Questions integration and study protocol

`load_dataset('questions', root=...)` returns PyG's native
`HeterophilousGraphDataset(name='Questions')`; the registry and dataset configuration
use the same public API as Cora/CiteSeer/PubMed. No new dataset wrapper, feature
normalization, label remapping or additional undirected transform is introduced.
The factory validates finite floating features, binary int64 labels, dimensions,
edge bounds, unique directed entries, symmetry and all ten official partitions.

## Graph identity and direction

The measured release has 48,921 nodes, 301 features and labels 0/1. Its raw file has
153,540 undirected edges, each stored once; PyG 2.6.1 converts them to 307,080 unique
symmetric directed entries. The loaded graph has no self-loops. The factory does
not append reverse edges again. GCN's existing internal normalization/self-loops
and GraphSAGE's root path remain model operations, unchanged by this integration.
See the [authors' release](https://github.com/yandex-research/heterophilous-graphs)
and [versioned PyG processor](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/datasets/heterophilous_graph_dataset.html).

Raw Questions SHA256:
`757ebd772bab1475c4dd951ca9e364400c6db161656cff9d21780ee874cf3074`.
Cache location: `<data-root>/heterophilous/questions/{raw,processed}`.
Transfer/pin the raw input and process it in the target PyG environment; do not
assume a local processed-cache pickle is portable to POLUS.

## Primary generated splits versus official folds

The official `train_mask`, `val_mask`, `test_mask` tensors are boolean `[N,10]`.
Each official column is disjoint and exhaustive, with 24,460/12,230/12,231 nodes
(approximately 50/25/25). These masks are preserved as provenance; the factory
selects no column.

The **primary cross-dataset study uses the existing generated 60/20/20 protocol**:
Python `random.Random(seed).shuffle` over all node IDs, floor 60% train, floor 20%
validation, and the remainder test. It is not stratified. Questions seed 101 gives
29,352/9,784/9,785 nodes. Explicit indices and their saved hash govern training and
frozen pruning, as before; tests demonstrate that no official column is substituted.
`configs/datasets/questions.yaml` makes these ratios explicit. Choosing official
folds later would require a separately identified protocol, not a silent fallback.

## Imbalance and evaluation conventions

The release contains 47,461 class 0 nodes and1,460 class 1 nodes (2.9844% positive).
Report accuracy **and** macro-F1, positive-class recall/F1, and supplementary
ROC-AUC. ROC-AUC must use continuous positive-class scores, not argmax labels;
our two-logit models use `softmax(logits)[:,1]`. The readiness helper reports `None`
if a split has only one class. Legacy classification-metric APIs remain unchanged.

The [authors' implementation](https://github.com/yandex-research/heterophilous-graphs/blob/main/datasets.py)
uses one-logit BCE and ROC-AUC for binary tasks; its
[logger](https://github.com/yandex-research/heterophilous-graphs/blob/main/utils.py)
selects the maximum validation metric over official folds. Our framework keeps
**two-logit cross-entropy, minimum validation CE loss and generated splits** for
cross-dataset consistency. Our pilot is not a reproduction of the official
benchmark, and its results must not be ranked against published ten-fold scores.

Neither supplementary AUC nor test labels change training selection or pruning.
The pilot masks test labels before the unchanged DenseTrainer and fixed pruning
smoke. Class distributions are inspected as dataset characteristics; no test
performance is computed, no threshold is optimized and no class weight is selected.

## Bounded local readiness check

The explicit opt-in `gnn_pruning.experiments.questions_readiness` entry point
prepares pinned inputs and then runs one chosen model per invocation:

```bash
python -m gnn_pruning.experiments.questions_readiness --root /fresh/external/root
python -m gnn_pruning.experiments.questions_readiness --root /fresh/external/root --model gcn
python -m gnn_pruning.experiments.questions_readiness --root /fresh/external/root --model graphsage
```

The approved check fixes seed 101, three layers, widths 32/32, dropout 0.5, Adam
lr 0.01, weight decay 0.0005, 200 epochs and patience 50, using the existing defaults
and DenseTrainer. It restores minimum-validation-loss weights. Each model receives
one fixed magnitude-based `prune:0:1` action, followed by physical count/reload and
validation checks. This is not a pruning-policy comparison or a target matrix.
Fresh directories prevent replacement of prior evidence. Raw inputs, histories,
checkpoints and caches belong outside Git. Results/provenance are in the
[readiness report](../research/experiments/questions_readiness_20261031/REPORT.md).

Dataset integration is technically ready. Inclusion in the full POLUS quality
campaign is currently **NO-GO at these baselines**: GCN collapses to the majority
class and GraphSAGE has only 1.58% minority recall at fixed argmax despite useful
ranking. A validation-only, separately declared convergence/imbalance diagnostic
is needed before scaling; no wider model or different loss was silently trained.
The threshold-versus-exact-controller distinction from the Lookahead audit also
continues to apply. No Greedy, Lookahead or Q-learning algorithm was changed here.
