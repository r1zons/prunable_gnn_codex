"""Frozen synthetic checkpoint integration; existing datasets are named, never loaded."""
import hashlib
import json

import pytest
import torch
import yaml

from gnn_pruning.models import build_model
from gnn_pruning.pruning import get_pruner
from gnn_pruning.pruning import workflow_v2
from gnn_pruning.pruning.workflow import prune_from_checkpoint
from gnn_pruning.pruning.adaptive_v2 import LocalV2Env, run_greedy
from gnn_pruning.pruning.lookahead import LookaheadPlanner
from test_local_v2_integration import fixture_files
from test_lookahead import without_times


def configure(tmp_path, kind, dataset='cora', depth=2):
    dense, data, checkpoint, path, cfg = fixture_files(tmp_path,'lookahead_local_v4',kind)
    cfg['dataset'] = dataset
    cfg['pruning']['target_sparsity'] = .44
    cfg['v2']['max_steps'] = 3
    cfg['lookahead'] = dict(depth=depth,gamma=.9)
    payload = torch.load(checkpoint,weights_only=True)
    payload['run_signature']['dataset'] = dataset
    torch.save(payload,checkpoint)
    cfg['v2']['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(cfg))
    return dense,data,checkpoint,path,cfg


def assert_projection(dense, final, keeps):
    inputs = [list(range(dense.in_channels))]+keeps
    outputs = keeps+[list(range(dense.out_channels))]
    for i,(source,result) in enumerate(zip(dense.convs,final.convs)):
        if hasattr(source,'lin_l'):
            for name in ('lin_l','lin_r'):
                a,b = getattr(source,name),getattr(result,name)
                assert torch.equal(b.weight,a.weight[outputs[i]][:,inputs[i]])
                if a.bias is not None:
                    assert torch.equal(b.bias,a.bias[outputs[i]])
        else:
            assert torch.equal(result.lin.weight,source.lin.weight[outputs[i]][:,inputs[i]])
            assert torch.equal(result.bias,source.bias[outputs[i]])


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('dataset',['cora','citeseer','pubmed'])
def test_existing_prune_interface_checkpoints_trees_metrics_and_provenance(tmp_path,monkeypatch,kind,dataset):
    dense,data,checkpoint,path,cfg = configure(tmp_path,kind,dataset)
    def load(name,**kwargs):
        assert name == dataset
        return [data]
    monkeypatch.setattr(workflow_v2,'load_dataset',load)
    artifact = prune_from_checkpoint(str(checkpoint),str(path))
    payload = torch.load(artifact.pruned_checkpoint_path,weights_only=True)
    final = build_model(payload['model_name'],**payload['model_config']).eval()
    final.load_state_dict(payload['model_state_dict'],strict=True)
    report = json.loads(artifact.pruning_metrics_path.read_text())
    decisions = json.loads(artifact.adaptive_trace_path.read_text())
    trace = json.loads(artifact.rl_trace_path.read_text())
    assert report['provenance']['protocol'] == 'LOCAL-STRUCTURED-LOOKAHEAD-GREEDY-THRESHOLD-V1'
    assert report['provenance']['checkpoint_sha256'] == cfg['v2']['checkpoint_sha256']
    assert report['provenance']['dataset'] == dataset
    assert report['statistics']['depth'] == 2 and report['statistics']['gamma'] == .9
    assert report['statistics']['terminal_reason'] in ('target_reached','max_steps','no_valid_actions')
    assert not trace['training'] and artifact.q_table_path is None
    assert len(decisions) == len(trace['deployment'])
    assert [d['selected_action'] for d in decisions] == [t['action'] for t in trace['deployment']]
    assert report['original_kept_ids'] == payload['original_kept_ids']
    assert_projection(dense,final,payload['original_kept_ids'])
    assert sum(p.numel() for p in final.parameters()) == report['parameters_after'] < report['parameters_before']
    assert report['achieved_sparsity'] == 1-report['parameters_after']/report['parameters_before']
    assert report['target_gap'] == pytest.approx(abs(report['achieved_sparsity']-.44))
    assert report['target_violation'] == (report['target_gap'] > .01)
    assert report['statistics']['termination_policy'] == 'greedy_threshold'
    assert report['overshoot'] == max(0.,report['achieved_sparsity']-.44)
    assert report['signed_target_deviation'] == report['achieved_sparsity']-.44
    assert report['overshoot_beyond_tolerance'] == (report['achieved_sparsity'] > .45)
    assert report['threshold_reached'] == (report['achieved_sparsity'] >= .44-.01)
    assert report['reporting_evaluations'] == 4
    assert 'test' in report['final_metrics'] and 'macro_f1' in report['final_metrics']['val']
    with torch.no_grad():
        val = (final(data).argmax(-1)[[1,3]] == data.y[[1,3]]).float().mean().item()
    assert report['final_metrics']['val']['accuracy'] == val
    assert 'local_structured_lookahead' in str(artifact.pruned_checkpoint_path)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_nonvalidation_labels_cannot_change_actions_tree_or_final_tensors(tmp_path,monkeypatch,kind):
    _,data,checkpoint,path,cfg = configure(tmp_path,kind)
    monkeypatch.setattr(workflow_v2,'load_dataset',lambda *a,**k:[data])
    pruner = get_pruner('lookahead_local_v4')
    original = pruner.plan
    def checked(self,model,scores,context,target):
        assert context.data['data'].y[[5,6]].tolist() == [-1,-1]
        assert 'evaluate_test' not in context.data
        return original(self,model,scores,context,target)
    monkeypatch.setattr(pruner,'plan',checked)
    first = prune_from_checkpoint(str(checkpoint),str(path))
    # Intentional synthetic graph revision is independently pinned, same inputs/split/weights.
    data.y[[0,2,4,5,6]] = (data.y[[0,2,4,5,6]]+1)%3
    cfg['v2']['graph_sha256'] = workflow_v2.tensor_hash({'x':data.x,'edge_index':data.edge_index,'y':data.y})
    path.write_text(yaml.safe_dump(cfg))
    second = prune_from_checkpoint(str(checkpoint),str(path))
    a,b = [torch.load(x.pruned_checkpoint_path,weights_only=True) for x in (first,second)]
    assert a['original_kept_ids'] == b['original_kept_ids']
    assert all(torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items())
    assert first.rl_trace_path.read_text() == second.rl_trace_path.read_text()
    assert without_times(json.loads(first.adaptive_trace_path.read_text())) == without_times(json.loads(second.adaptive_trace_path.read_text()))
    assert first.pruned_checkpoint_path != second.pruned_checkpoint_path and first.pruned_checkpoint_path.exists()


def test_configuration_reference_depth_and_invalid_options(tmp_path,monkeypatch):
    _,data,checkpoint,path,cfg = configure(tmp_path,'gcn',depth=1)
    monkeypatch.setattr(workflow_v2,'load_dataset',lambda *a,**k:[data])
    artifact = prune_from_checkpoint(str(checkpoint),str(path))
    assert json.loads(artifact.pruning_metrics_path.read_text())['statistics']['depth'] == 1
    cfg['lookahead']['beam_width'] = 2
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Unknown Lookahead'):
        prune_from_checkpoint(str(checkpoint),str(path))


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_depth_one_real_validation_greedy_equivalence_including_reported_f1(tmp_path,monkeypatch,kind):
    dense,data,checkpoint,path,cfg = configure(tmp_path,kind,depth=1)
    monkeypatch.setattr(workflow_v2,'load_dataset',lambda *a,**k:[data])
    artifact = prune_from_checkpoint(str(checkpoint),str(path))
    report = json.loads(artifact.pruning_metrics_path.read_text())
    payload = torch.load(artifact.pruned_checkpoint_path,weights_only=True)
    val = torch.tensor([1,3])
    def evaluate(candidate):
        with torch.no_grad():
            return (candidate(data).argmax(-1)[val] == data.y[val]).float().mean().item()
    reference = run_greedy(dense,evaluate,target=.44,penalty_lambda=2.,max_steps=3)
    assert reference.kept_indices == payload['original_kept_ids']
    assert reference.trace == json.loads(artifact.rl_trace_path.read_text())['deployment']
    assert report['final_metrics']['val'] == workflow_v2._metrics(reference.model,data,val)
    assert all(torch.equal(v,payload['model_state_dict'][k]) for k,v in reference.model.state_dict().items())
    assert report['statistics']['validation_evaluations'] == reference.statistics['validation_evaluations']


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_depth_two_actual_forward_independent_path_search_and_physical_projection(tmp_path,kind):
    from lookahead_forward_oracle import Node, enumerate_paths
    dense,data,_,_,_ = configure(tmp_path,kind)
    validation = torch.tensor([1,3])
    hidden = torch.ones(data.num_nodes,dtype=torch.bool)
    hidden[validation] = False
    data.y[hidden] = -1
    def evaluate(candidate):
        assert (data.y[hidden] == -1).all()
        candidate.eval()
        with torch.no_grad():
            predictions = candidate(data).argmax(-1)
            return float((predictions[validation] == data.y[validation]).float().mean())
    p0 = sum(p.numel() for p in dense.parameters())
    dense_accuracy = evaluate(dense)
    calls = []
    def counted(candidate):
        calls.append(1)
        return evaluate(candidate)
    env = LocalV2Env(dense,counted,.44,2.,greedy=True,max_steps=3)
    planner = LookaheadPlanner(depth=2)
    evaluations = 1
    while not env.terminal:
        root = Node(env.model,env.original_kept_ids,env.steps,env.sparsity,env.accuracy)
        possible = enumerate_paths(root,evaluate,p0,dense_accuracy,.44)
        best = max(possible,key=lambda path:path['score'])
        selected = planner.plan(env)
        rows = selected.decision['candidate_paths']
        keys = lambda path:[f'prune:{a}:{k}' for a,k in path]
        assert [p['actions'] for p in rows] == [keys(p['actions']) for p in possible]
        assert [p['score'] for p in rows] == pytest.approx([p['score'] for p in possible],abs=1e-14)
        assert selected.decision['selected_action'] == keys(best['actions'])[0]
        assert selected.decision['best_planned_path'] == keys(best['actions'])
        nodes = {tuple(keys(p['actions'][:i+1])):node
                 for p in possible for i,node in enumerate(p['nodes'])}
        for recorded in selected.decision['candidate_transitions']:
            node = nodes[tuple(recorded['path'])]
            transition = recorded['transition']
            assert transition['original_kept_ids'] == node.ids
            assert transition['reward'] == pytest.approx(node.reward,abs=1e-14)
            assert transition['validation_accuracy'] == node.accuracy
            assert transition['sparsity'] == node.sparsity
            assert transition['terminal_penalty'] == node.penalty
            assert transition['terminal_reason'] == node.reason
            assert recorded['parameters_after'] == sum(p.numel() for p in node.model.parameters())
        assert len(selected.next_env.trace) == len(env.trace)+1
        assert_projection(dense,selected.next_env.model,selected.next_env.original_kept_ids)
        assert selected.next_env.original_kept_ids == best['nodes'][0].ids
        evaluations += selected.evaluations
        env = selected.next_env
    assert len(env.trace) <= 3 and len(calls) == evaluations
