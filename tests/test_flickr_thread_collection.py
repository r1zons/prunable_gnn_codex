"""Independent arithmetic oracles for the offline thread-profile audit."""
import importlib.util
import itertools
from pathlib import Path
from fractions import Fraction

import pytest
import torch

path=Path(__file__).parents[1]/'research/experiments/flickr_cpu_thread_scaling_20261101/collection_20261102/audit.py'
spec=importlib.util.spec_from_file_location('thread_collection_audit',path)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


@pytest.mark.parametrize('widths',[(256,256,256),(111,230,219),(244,256,256),(4,5,6)])
def test_parameter_geometry_is_independent_layer_sum(widths):
    dims=[500,*widths,7]
    expected=sum(2*x*y+y for x,y in zip(dims,dims[1:]))
    assert a.count(widths)==expected


@pytest.mark.parametrize('target',[.5,.98,.99])
def test_interval_reachability_matches_brute_force_all_small_descendants(target):
    for widths in itertools.product(range(4,8),repeat=3):
        expected=any(abs(Fraction(a.P0-a.count(w),a.P0)-Fraction(str(target)))<=Fraction('.01')
                     for w in itertools.product(*(range(4,n+1) for n in widths)))
        assert a.reachable(widths,target)==expected


def test_target_band_is_not_prefix_completion():
    assert a.goal((111,230,219),.5)
    assert not a.goal((244,256,256),.3)
    assert not a.goal((255,255,256),.5)


def test_projection_preserves_original_channel_identity_and_incident_weights():
    # Distinguishable exact tensors; no model inference or policy execution.
    dims=[500,256,256,256,7];dense={}
    for layer,(x,y) in enumerate(zip(dims,dims[1:])):
        for name in ['lin_l','lin_r']:
            dense[f'convs.{layer}.{name}.weight']=torch.arange(x*y,dtype=torch.float32).reshape(y,x)
        dense[f'convs.{layer}.lin_l.bias']=torch.arange(y,dtype=torch.float32)
    kept=[[1,3,8,11],[0,5,7,90,254],[2,9,32,64,127,255]]
    projected=a.projected_state(dense,kept)
    assert torch.equal(projected['convs.0.lin_l.weight'],dense['convs.0.lin_l.weight'][kept[0]])
    assert torch.equal(projected['convs.1.lin_r.weight'],dense['convs.1.lin_r.weight'][kept[1]][:,kept[0]])
    assert torch.equal(projected['convs.3.lin_l.weight'],dense['convs.3.lin_l.weight'][:,kept[2]])
    assert sum(v.numel() for v in projected.values())==a.count(tuple(map(len,kept)))
