# V3 Q-learning ablation sprint — launch protocol

Researcher-authorized bounded sprint, 2026-10-27, starting from
`3925278d5824f35a7f0256c68b75da9b586030a0`.
This task implements/validates and submits jobs only. After both platform
submission attempts and receipt recording, stop: no monitoring, collection,
aggregation, plots, scientific interpretation or second round in this session.

## Infrastructure recovered and reused

- Canonical local checkout and adjacent `prunable_gnn_artifacts/`; existing
  `/private/tmp/gnn-v2-validation-py312-20261027/bin/python`, Python3.12.9,
  Torch2.2.2/PyG2.6.1. Local host has6 physical/12 logical cores and16GiB RAM.
- Existing local launchers orchestrate two independent Python subprocesses with
  two CPU threads each. `launch_local.py` adapts that pattern, adding `nohup`, a
  new process session, persistent stdout/stderr, atomic submission claims and PID
  receipts. The unit test confirms a detached child outlives its launcher parent.
- Historical POLUS `activate-polus.sh`, `submit-pilot.sh` and `pilot-cpu.lsf` are
  reference procedures. Reuse the working environment without reinstalling:
  Python3.10.22, Torch2.0.1, PyG2.4.0, ppc64le/POWER8, four CPU threads.
  Replace the old72293149 source/campaign invocation explicitly for V3.
- POLUS LSF remains `normal`,4 slots, `span[hosts=1]`, exclusive allocation,
  `polus-c4-ib.bmc.hpc.cs.msu.ru`, max03:00. A separate job per model/target
  reuses the existing immutable INTENT/ACCEPTED/response/job-id convention.
- Current route/network was inspected before SSH. Outline was active; POLUS
  incorrectly followed the VPN route. The researcher restored the documented
  host exception using the verified current physical gateway. The corrected
  route was verified before SSH. Access values stay in existing/private local
  configuration, never tracked. Default SSH alias selected a different course
  account; the research account was recovered from frozen executed provenance.
- The historical remote repo was clean at72293149 and the original environment
  activated successfully. A separate Git worktree at the pinned V3 source commit
  isolates execution without overwriting historical checkout/results.

## Separate experimental identifiers (single changes against R2)

| ID | Change |
|---|---|
| `r2_reference` | Direct call to the original `adaptive_v2.run_q_learning`; no monkeypatch or modified R2 default/update. |
| `q_stop_exact_v3` | H1: STOP value is exactly the known immediate terminal penalty P(s), including selection and next-state bootstrap. STOP entries are saved analytically; other entries retain the R2 alpha/Bellman formula. |
| `q_conservative_v3` | H2: unseen entries use a finite-horizon conservative lower bound; learned entries are not clamped. This changes only the explicit unseen-value initialization/lookup convention. |
| `q_random_ties_v3` | H3: exact equal maximum Q values are resolved uniformly with the separate policy RNG; unique maxima and epsilon exploration retain their ordinary meaning. |
| `q_target_stop_v3` | H4a: hard termination after a successful removal crosses s≥target−tolerance; this may overshoot the upper band. |
| `q_reachable_v3` | H4b: exact future-width/horizon filtering, no off-band STOP, automatic termination in the exact target band. |
| `random_reachable_v3` | H6 Random: uniform sampling among the same reachability-legal integer actions as H4b Q. It randomizes actions, not the original static Random's channel masks. |
| `greedy_reachable_v3` | H6 Greedy: evaluate the same legal actions and choose immediate R plus any terminal penalty. |

H5 multi-step credit is deliberately omitted. These are experimental ablations,
not silently adopted production defaults. No variant combines H1/H2/H3 changes.
V1, R2 production files, the registry and approved R2 specification are unchanged.

H2 uses `-.4*sum(gamma**j,j=0..remaining-1) - lambda*max_gap`, where
`max_gap=max(0,max(target,1-target)-tolerance)`. Each pruning reward is at least−.4;
charging the entire worst terminal penalty immediately gives a conservative lower
bound on any remaining discounted return. It uses no new tuned scalar constant.

Reachability counts integer weight/bias parameters exactly. It enumerates target
width pairs using rational arithmetic, and computes the minimum count of allowed
k=1,2,4,8 deletions independently in each layer. A width pair is reachable iff the
sum fits the remaining successful-removal horizon. Intermediate widths cannot
violate the minimum because deletion is monotonic. No labels or accuracy enter
this calculation. Every retained action preserves a feasible continuation when
initially feasible. An infeasible initial band/horizon yields the explicit
`initial_target_infeasible` terminal event, without widening tolerance or applying
a fallback. Existing max_steps/no_valid_actions priorities precede new target
termination; each terminal penalty is paid once. Local Surgery is unchanged.

## Frozen scientific matrix

Cora; GCN and mean GraphSAGE, three layers/hidden32/32, dropout0.5; targets0.30/0.50.
Training seed101 is pinned independently of policy seeds101/5001/5002.
Q uses100 episodes,max20 successful deletions,lambda2,gamma0.9,alpha0.1,epsilon0.2;
minimum4,tolerance0.01. Controls execute once per condition/policy seed.
No dense training, fine-tuning or dataset download.

- Local:8 variants ×2 models ×2 targets ×3 policy seeds =96 tasks, plus2 fixed
  cross-platform probes. Workers2, threads2 each, at most4 simultaneous CPU threads.
- POLUS: the same8 variants ×2 models ×2 targets for paired policy seed101 =32
  tasks plus2 fixed probes, partitioned into4 LSF jobs (9/8/9/8 tasks). One worker
  per allocation, four threads. Additional policy seeds remain local to bound the
  remote batch. No platform pooling; this is an explicitly paired subset.
- Per-task timeout900s; batch wall budget10800s. The resource choice must pass
  measured local and remote preflight before submission; gates/receipts record
  actual search costs. No unverified/oversized batch may be submitted.

`prepare_inputs.py` verifies the prior original-graph cache/hash and exports only
plain x/edge_index/y tensors, byte-identical dense checkpoints and the shared
frozen split. Both platforms consume the same input bundle and graph tensor hash.
This avoids a PyG2.6→2.4 processed-object compatibility assumption and preserves
historical row-sorted edge and node ordering. The recorded original graph lineage
remains reconstructed from raw provenance; no contemporaneous training tensor
hash is invented. All input file/tensor/split/architecture signatures fail closed.

The runner masks every label outside validation to−1 before adaptive evaluation.
No test metrics are generated. Source commit, variant, training/policy seeds,
platform, runtime/backend/threads and graph/split/checkpoint hashes are separate.
The platform-independent algorithm identity excludes paths/thread counts but
includes all scientific settings, RNG seeds, source and frozen input hashes.

## Comparison artifacts generated by the queued batch

Each task saves actual physical model, original retained IDs, config, validation
metrics, full action/episode traces, Q-table, parameter counts, provenance, search
cost and runtime backend. Paired seed101 tasks additionally benchmark final-model
CPU inference with5 warmups/20 timed passes, separately from search.
Two fixed probes save full dense/pruned logits and inference measurements. Their
shared deterministic sequence deletes highest current IDs: layer0 k1, layer1 k2,
layer0 k4, ending at widths27/30. No importance ranking affects this sequence.

A later session can distinguish dense-logit/numerical differences, validation
metrics, retained channels/checkpoints, learned actions, search cost and inference
latency. Torch/PyG/backend versions and thread counts differ; differences must not
be attributed to hardware alone or pooled into one statistical sample.
No such result comparison is performed in this submission session.

## Collision protection and safe continuation

Configs and fresh roots are prepared before launching. `validate` checks pins,
settings, uniqueness and initial target feasibility without running a batch.
Source pin checks require executable `src/` to match the source commit; later
receipt-only commits may advance branch HEAD without changing the pinned code.
Any executable source edit makes subsequent tasks fail closed.

A batch owns an atomic PID/host lock. Completed tasks are hash-verified and skipped
on explicit resume. Failed/interrupted attempts are retained under unique attempt
directories. A live/ambiguous owner or child PID blocks resubmission; stale local
claims are archived only under explicit resume after an inactive same-host owner
is established. No accepted LSF submission is automatically retried.

Raw files: adjacent `${GNN_ARTIFACT_ROOT}/local-structured-v3-sprint-20261027/` and
`${POLUS_TASK_ROOT}/results/local-structured-v3-sprint-20261027/`.
Exact config/manifests, preflight records and immutable launcher snapshots are
indexed there. `launch_receipt.json` will contain acceptance/PID/job IDs and log
locations; it is a submission record, not a result-completion claim.
