"""Bounded native compatibility assertions on a 20-node synthetic graph, not Cora."""
import argparse
from pathlib import Path
import torch
from torch_geometric.data import Data
from gnn_pruning.config import load_yaml
from gnn_pruning.experiments.complete_campaign import METHODS, search, verify_projection, tensor_hash, write
from gnn_pruning.models import build_model


def smoke(protocol):
    protocol['adaptive']['max_steps']=2
    protocol['q_learning']['episodes']=1
    records=[]
    for kind in ('gcn','graphsage'):
        torch.manual_seed(72)
        model=build_model(kind,in_channels=5,out_channels=3,hidden_channels=8,num_layers=3,dropout=.5).eval()
        data=Data(x=torch.randn(20,5),edge_index=torch.tensor([list(range(20)),[(i+1)%20 for i in range(20)]]),y=torch.arange(20)%3)
        split=dict(train=list(range(12)),val=list(range(12,16)),test=list(range(16,20)))
        frozen=tensor_hash(model.state_dict())
        for method in METHODS:
            run=dict(method=method,target=.3,policy_seed=101)
            final,keeps,evidence=search(model,data,split,run,protocol)
            verify_projection(model,final,keeps)
            assert tensor_hash(model.state_dict())==frozen
            changed=data.clone();changed.y[split['test']]=2-changed.y[split['test']]
            other,ids,second=search(model,changed,split,run,protocol)
            assert ids==keeps and tensor_hash(other.state_dict())==tensor_hash(final.state_dict())
            assert evidence['trace']==second['trace'] and not evidence['test_labels_exposed']
            assert evidence['observed_validation_evaluations']==evidence['statistics']['validation_evaluations']
            rebuilt=build_model(kind,**{k:v for k,v in final.export_architecture_config().items() if k!='name'}).eval()
            rebuilt.load_state_dict(final.state_dict(),strict=True)
            with torch.no_grad():assert torch.equal(rebuilt(data),final(data))
            records.append(dict(model=kind,method=method,status='PASS',validation_evaluations=evidence['observed_validation_evaluations']))
    return dict(status='PASS',scope='20-node synthetic unit/regression smoke, not scientific pilot',cases=records,
                torch=str(torch.__version__),threads=torch.get_num_threads())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);r=smoke(load_yaml(a.protocol));write(a.output,r);print('NATIVE_SMOKE_PASS',len(r['cases']),flush=True)
