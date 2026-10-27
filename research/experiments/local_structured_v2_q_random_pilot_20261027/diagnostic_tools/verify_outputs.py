"""Independent physical/validation verification of existing pilot checkpoints only."""
import json,math
from pathlib import Path
import torch,numpy as np
from sklearn.metrics import f1_score
from pilot_runner import ROOT,load_inputs,sha
from summarize import count

torch.set_num_threads(2);checks=[]
for kind in ('gcn','graphsage'):
    dense,data,idx,source,entry,prov=load_inputs(kind)
    assert int((data.y>=0).sum())==len(idx)
    for group in ('q_runs','random_runs'):
        for result_path in sorted((ROOT/group).glob(f'{kind}_*/result.json')):
            report=json.loads(result_path.read_text());checkpoint_path=Path(report['checkpoint']);c=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            from gnn_pruning.models import build_model
            model=build_model(c['model_name'],**c['model_config']).eval();model.load_state_dict(c['model_state_dict'],strict=True)
            ids=c['original_kept_ids'];assert ids==report['original_kept_ids']
            for name,value in model.state_dict().items():
                layer=int(name.split('.')[1]);original=source['model_state_dict'][name]
                rows=ids[layer] if layer<2 else list(range(7))
                expected=original[rows]
                if expected.ndim==2:
                    cols=ids[layer-1] if layer>0 else list(range(1433));expected=expected[:,cols]
                assert torch.equal(value,expected),name
            assert model.hidden_channel_dims==list(map(len,ids))==report['hidden_widths']
            P=sum(p.numel() for p in model.parameters());assert P==count(kind,list(map(len,ids)))==report['parameters_after']
            assert math.isclose(1-P/count(kind,[32,32]),report['achieved_sparsity'],abs_tol=1e-12)
            with torch.no_grad():predictions=model(data)
            assert predictions.shape==(2708,7) and torch.isfinite(predictions).all()
            true=data.y[idx].numpy();pred=predictions.argmax(1)[idx].numpy()
            assert math.isclose(float((true==pred).mean()),report['validation_accuracy'],abs_tol=1e-12)
            assert math.isclose(float(f1_score(true,pred,average='macro',zero_division=0)),report['validation_macro_f1'],abs_tol=1e-12)
            assert c['checkpoint_training_seed']==101 and c['checkpoint_sha256']==entry['checkpoint_sha256']==sha(entry['checkpoint']) and c['split_sha256']==entry['split_sha256'] and c['graph_tensor_sha256']==prov['graph_tensor_sha256']
            checks.append(dict(kind=group,model=kind,checkpoint=str(checkpoint_path),checkpoint_sha256=sha(checkpoint_path),status='PASS'))
old=json.loads((ROOT/'protected_files_before.json').read_text());changed=[p for p,h in old.items() if not Path(p).is_file() or sha(p)!=h];assert not changed,changed
(ROOT/'verification.json').write_text(json.dumps(dict(status='PASS',physical_checkpoints_verified=len(checks),checkpoints=checks,protected_files_verified=len(old),previous_artifacts_modified=changed,validation_only=True,labels_outside_validation=-1,scope='Strict reload, independent dense tensor slicing, finite full graph forward, parameter counts, actual compression, validation accuracy and macro-F1, frozen source/split/graph pins. All Q RNG/reward/penalty/Bellman/counter checks are replayed separately in summarize.py.'),indent=2))
print(f'PASS {len(checks)} physical checkpoints; {len(old)} old files unchanged; validation only')
