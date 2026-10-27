"""Verify observational wrappers leave the original routine's outputs and RNG unchanged."""
import random,torch
from torch_geometric.data import Data
from gnn_pruning.models import build_model
from gnn_pruning.pruning import adaptive_v2 as a
from pilot_runner import observe_q

torch.set_num_threads(2)
for kind in ('gcn','graphsage'):
    torch.manual_seed(501);model=build_model(kind,in_channels=5,hidden_channels=8,out_channels=3,num_layers=3,dropout=.5).eval()
    data=Data(x=torch.randn(7,5),edge_index=torch.tensor([[0,1,2,3,4,5,6,1,2],[1,2,3,4,5,6,0,0,1]]),y=torch.tensor([0,1,2,0,1,2,-1]));idx=torch.tensor([1,4])
    def evaluate(m):
        with torch.no_grad():return float((m(data).argmax(1)[idx]==data.y[idx]).float().mean())
    settings=dict(target=.3,penalty_lambda=2.,episodes=3,seed=101,max_steps=3)
    before=torch.random.get_rng_state();plain=a.run_q_learning(model,evaluate,**settings);after_plain=torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    with observe_q(3) as audit:observed=a.run_q_learning(model,evaluate,**settings)
    assert torch.equal(after_plain,torch.random.get_rng_state())
    assert plain.q_table==observed.q_table and plain.trace==observed.trace and plain.training_trace==observed.training_trace and plain.kept_indices==observed.kept_indices
    for k,v in plain.model.state_dict().items():assert torch.equal(v,observed.model.state_dict()[k])
    assert len(audit['resets'])==4
    print(kind,'observer equivalence PASS')
