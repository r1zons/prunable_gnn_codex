"""Opt-in bounded depth-1/native-Greedy check on the verified offline Cora archive.

No dataset loader/download, training, depth-2 Cora search, or campaign launch.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from gnn_pruning.evaluation.metrics import classification_metrics
from gnn_pruning.experiments.v3_sprint import load_frozen
from gnn_pruning.pruning.adaptive_v2 import run_greedy
from gnn_pruning.pruning.lookahead import run_lookahead

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def frozen_manifest():
    declared = os.environ.get('GNN_LOOKAHEAD_CORA_MANIFEST')
    if not declared:
        pytest.skip('Archived Cora gate not enabled: set GNN_LOOKAHEAD_CORA_MANIFEST to verified offline inputs')
    path = Path(declared).expanduser().resolve()
    assert path.is_file(), 'Explicitly supplied frozen manifest is missing; this is not a PASS or optional skip'
    anchor = json.loads((REPO/'research/experiments/polus_v4_diagnostic_20261029/results/provenance.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == anchor['input_manifest_sha256']
    return path


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('target',[.30,.50])
def test_real_frozen_cora_depth_one_matches_native_greedy(frozen_manifest,tmp_path,kind,target):
    dense,data,val,checkpoint,manifest = load_frozen(frozen_manifest,kind)
    before = {k:v.clone() for k,v in dense.state_dict().items()}
    nonval = torch.ones(data.num_nodes,dtype=torch.bool)
    nonval[val] = False
    assert (data.y[nonval] == -1).all()
    calls = []
    def evaluate(candidate):
        assert (data.y[nonval] == -1).all()
        candidate.eval()
        with torch.no_grad():
            value = (candidate(data).argmax(-1)[val] == data.y[val]).float().mean().item()
        calls.append(value)
        return value
    reference = run_greedy(dense,evaluate,target=target,penalty_lambda=2.,max_steps=20,
                           min_channels=4,tolerance=.01)
    reference_calls = len(calls)
    actual = run_lookahead(dense,evaluate,target,depth=1,penalty_lambda=2.,max_steps=20,
                           min_channels=4,tolerance=.01)
    assert actual.trace == reference.trace
    assert actual.kept_indices == reference.kept_indices
    assert actual.statistics['candidate_evaluations'] == reference.statistics['candidate_evaluations']
    assert actual.statistics['validation_evaluations'] == reference_calls
    assert len(calls)-reference_calls == reference_calls
    assert actual.statistics['terminal_reason'] == reference.statistics['terminal_reason']
    assert actual.statistics['termination_policy'] == 'greedy_threshold'
    assert all(torch.equal(v,actual.model.state_dict()[k]) for k,v in reference.model.state_dict().items())
    assert all(torch.equal(v,dense.state_dict()[k]) for k,v in before.items())
    def metrics(model):
        with torch.no_grad():
            pred = model(data).argmax(-1)[val]
        return classification_metrics(data.y[val].numpy(),pred.numpy())
    actual_metrics, reference_metrics = metrics(actual.model),metrics(reference.model)
    assert actual_metrics == reference_metrics
    record = dict(status='PASS',check='bounded archived depth1/native Greedy correctness',
        model=kind,target=target,training_seed=manifest['training_seed'],
        manifest_sha256=hashlib.sha256(frozen_manifest.read_bytes()).hexdigest(),
        checkpoint_sha256=manifest['models'][kind]['sha256'],split_sha256=manifest['split']['sha256'],
        graph_tensor_sha256=manifest['graph_tensor_sha256'],
        actions=[t['action'] for t in actual.trace],retained_ids=actual.kept_indices,
        validation=actual_metrics,reference_validation=reference_metrics,
        parameters_before=sum(p.numel() for p in dense.parameters()),
        parameters_after=sum(p.numel() for p in actual.model.parameters()),
        achieved_sparsity=actual.statistics['achieved_sparsity'],
        overshoot=actual.statistics['overshoot'],target_gap=actual.statistics['target_gap'],
        target_violation=actual.statistics['target_violation'],
        terminal_reason=actual.statistics['terminal_reason'],
        validation_evaluations_per_policy=reference_calls,
        reporting_forward_calls=2,identical_traces=True,identical_ids=True,identical_tensors=True,
        labels_exposed='validation_only',depth2_cora_executed=False)
    output = Path(os.environ.get('GNN_LOOKAHEAD_AUDIT_OUTPUT',str(tmp_path)))
    output.mkdir(parents=True,exist_ok=True)
    path = output/f'cora-{kind}-t{round(100*target)}.json'
    assert not path.exists(), 'Refusing to overwrite earlier verification evidence'
    path.write_text(json.dumps(record,indent=2)+'\n')
