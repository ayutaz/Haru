from copy import deepcopy

import pytest

from test_codec_adapter_trial import specification


def diagnostic(tmp_path):
    spec=specification(tmp_path)
    spec.update(kind='haru-codec-diagnostic',max_steps=300,max_seconds=300,
                milestones=[0,30,100,300],training_ids=['a','b','c'])
    return spec


def test_diagnostic_is_separate_from_original_thirty_update_trial(tmp_path):
    from haru_training.codec_adapter_diagnostic import validate_diagnostic
    from haru_training.codec_adapter_trial import validate_trial
    spec=diagnostic(tmp_path);original=deepcopy(spec)
    assert validate_diagnostic(spec,root=tmp_path)==spec
    assert spec==original
    with pytest.raises(ValueError):validate_trial(spec,root=tmp_path)


@pytest.mark.parametrize('key,value',[('kind','trial'),('max_steps',301),('max_steps',True),
    ('max_seconds',301),('max_seconds',float('nan')),('rank',32),
    ('milestones',[0,30,100]),('training_ids',['a','a','b']),('training_ids',['a'])])
def test_diagnostic_cannot_expand_or_silently_change_the_registered_small_experiment(tmp_path,key,value):
    from haru_training.codec_adapter_diagnostic import validate_diagnostic
    spec=diagnostic(tmp_path);spec[key]=value
    with pytest.raises(ValueError):validate_diagnostic(spec,root=tmp_path)


def test_subset_preserves_dataset_order_and_refuses_nontrain_or_unknown_ids():
    from haru_training.codec_adapter_diagnostic import select_training_pairs
    from types import SimpleNamespace
    pairs=[SimpleNamespace(metadata={'id':name,'split':'train'}) for name in 'abcd']
    assert [p.metadata['id'] for p in select_training_pairs(pairs,['c','a','b'])]==['a','b','c']
    with pytest.raises(ValueError):select_training_pairs(pairs,['a','b','z'])
    pairs[2].metadata['split']='reference'
    with pytest.raises(ValueError):select_training_pairs(pairs,['a','b','c'])


def test_stages_observe_initial_actual_updates_and_terminal_with_frozen_base():
    import torch
    from test_codec_adapter_training import trainer
    from haru_training.codec_adapter import get_adapter
    from haru_training.codec_adapter_diagnostic import run_stages
    t=trainer();before={n:p.clone() for n,p in t.model.named_parameters() if not p.requires_grad}
    seen=[]
    def observe(step,reason):
        seen.append((step,reason,get_adapter(t.model.codec).up.weight.detach().clone()))
    history=run_stages(t,[0,1,3],observe)
    assert len(history)==3 and [(s,r) for s,r,w in seen]==[(0,'initial'),(1,'milestone'),(3,'milestone')]
    assert not torch.equal(seen[0][2],seen[-1][2])
    assert all(torch.equal(before[n],p) for n,p in t.model.named_parameters() if n in before)


def test_time_limit_does_not_claim_the_requested_terminal_checkpoint():
    from types import SimpleNamespace
    from haru_training.codec_adapter_diagnostic import run_stages
    class Stopped:
        step=0
        spec=SimpleNamespace(max_steps=300)
        def fit(self,until_step): self.step=7;return [{'step':i} for i in range(1,8)]
    seen=[]
    run_stages(Stopped(),[0,30,100,300],lambda s,r:seen.append((s,r)))
    assert seen==[(0,'initial'),(7,'time_limit')]


def test_roundtrip_inputs_must_match_the_original_three_training_groups(tmp_path):
    from hashlib import sha256
    from haru_training.codec_adapter_diagnostic import validate_roundtrip_inputs
    spec={'training_ids':['a','b','c'],'roundtrip':[]}
    for name in spec['training_ids']:
        for side in ('source','target'):
            p=tmp_path/(name+side+'.wav');p.write_bytes((name+side).encode())
            spec['roundtrip'].append(dict(id=name,side=side,path=p.name,sha256=sha256(p.read_bytes()).hexdigest()))
    assert len(validate_roundtrip_inputs(spec,root=tmp_path))==6
    spec['roundtrip'][0]['id']='reference'
    with pytest.raises(ValueError):validate_roundtrip_inputs(spec,root=tmp_path)
