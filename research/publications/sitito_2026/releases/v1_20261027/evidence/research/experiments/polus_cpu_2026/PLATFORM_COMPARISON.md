# Local CPU versus POLUS CPU — seed 101

This is a cross-platform consistency note, not a pooled statistical analysis.

## What matched

- Same research commit: `72293149d8fff186dd268789e7b8fea5848c9360`.
- Same six dataset/model conditions, seven methods and two requested targets.
- Same seed-101 split SHA256 per dataset on both platforms:
  - Cora: `59a0209a8e2a73a5e70a720d327f8c024d36d184a6d00be820ced0e1a3eb08bc`
  - CiteSeer: `cb75b92838f59cfa51336965c0e4442086e4cde30e691a6ca9163a30a4d6c72c`
  - PubMed: `9699d19a758ec8dcd55a78f34c4ebf0d289c2f6e15d99e9f03cd6e8c39458b27`
- All 84 POLUS method-target rows and all corresponding local seed-101 rows are present.
- Static-method achieved parameter reduction was identical across platforms. Architecture-derived parameter counts therefore remained stable.

## What differed

- Dense validation accuracy differed in every condition by POLUS minus local values from -0.00150 to +0.00555. Dense checkpoints are not byte-identical.
- Post-prune validation differences ranged from -0.25878 to +0.14431 across the 84 pairs. Sensitive methods and high-compression points can diverge materially even with the same split and nominal seed.
- Achieved reduction differed in 11/84 pairs: four Adaptive Layerwise and seven Q-learning rows. The Q-learning difference ranged from -0.15635 to +0.12435 because learned actions/stopping paths changed.
- Adaptive Layerwise missed both requested targets on both platforms. Local seed-101 achieved 0.00619–0.04399 with 6/12 validation-budget violations; POLUS achieved 0.00675–0.04399 with 2/12 violations.

Likely contributors include numerical/library/platform differences between the Mac CPU environment and the POWER8 PyTorch/PyG stack. The evidence establishes divergence, not a single isolated causal mechanism. Do not select the more favorable platform result or average the platforms together.
