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


def test_time_limit_at_observed_boundary_does_not_overwrite_the_same_checkpoint():
    from types import SimpleNamespace
    from haru_training.codec_adapter_diagnostic import run_stages
    class Stopped:
        step=0
        spec=SimpleNamespace(max_steps=300)
        def fit(self,until_step):
            if self.step==0:self.step=30;return [{'step':i} for i in range(1,31)]
            return []
    seen=[]
    def observe(step,reason):
        assert step not in [s for s,r in seen], 'existing checkpoint would be overwritten'
        seen.append((step,reason))
    assert len(run_stages(Stopped(),[0,30,100,300],observe))==30
    assert [s for s,r in seen]==[0,30]


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


def test_small_training_requires_complete_hash_bound_independent_codec_gate():
    from haru_training.codec_adapter_diagnostic import validate_training_gate
    spec={'training_ids':['a','b','c']}
    gate=dict(small_training_gate_passed=True,spec_sha256='a'*64,roundtrip_report_sha256='b'*64,
        cases=[dict(id=name,side=side,status='supported_for_small_training',reasons=[])
               for name in 'abc' for side in ('source','target')])
    assert validate_training_gate(spec,gate,spec_sha256='a'*64,report_sha256='b'*64)==gate
    for mutation in ('failed','wrong_spec','wrong_report','missing','held'):
        changed=deepcopy(gate)
        if mutation=='failed':changed['small_training_gate_passed']=False
        if mutation=='wrong_spec':changed['spec_sha256']='c'*64
        if mutation=='wrong_report':changed['roundtrip_report_sha256']='c'*64
        if mutation=='missing':changed['cases'].pop()
        if mutation=='held':changed['cases'][0]['status']='hold'
        with pytest.raises(ValueError):validate_training_gate(spec,changed,spec_sha256='a'*64,report_sha256='b'*64)


def test_rank32_correction_changes_capacity_only_and_preserves_its_parent_gate(tmp_path):
    import json
    from hashlib import sha256
    from haru_training.codec_adapter_diagnostic import validate_correction
    parent=diagnostic(tmp_path)
    path=tmp_path/'diagnostic.json';path.write_text(json.dumps(parent),encoding='utf-8')
    correction=deepcopy(parent)
    correction.update(kind='haru-codec-rank-correction',rank=32,
        parent=dict(path=path.name,sha256=sha256(path.read_bytes()).hexdigest()))
    assert validate_correction(correction,root=tmp_path)==correction
    for field,value in [('rank',64),('training_ids',['x','y','z']),('learning_rate',.1),('max_steps',301)]:
        changed=deepcopy(correction);changed[field]=value
        with pytest.raises(ValueError):validate_correction(changed,root=tmp_path)
    path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError):validate_correction(correction,root=tmp_path)


def test_full_data_correction_uses_all_fifteen_existing_train_groups_without_other_changes(tmp_path,monkeypatch):
    import json
    from hashlib import sha256
    from types import SimpleNamespace
    from haru_training.codec_adapter_diagnostic import validate_full_data_correction
    parent=diagnostic(tmp_path)
    first=tmp_path/'diagnostic.json';first.write_text(json.dumps(parent),encoding='utf-8')
    rank32=deepcopy(parent);rank32.update(kind='haru-codec-rank-correction',rank=32,
        parent=dict(path=first.name,sha256=sha256(first.read_bytes()).hexdigest()))
    second=tmp_path/'rank32.json';second.write_text(json.dumps(rank32),encoding='utf-8')
    ids=['a','b','c']+[f'other-{i}' for i in range(12)]
    pairs=[SimpleNamespace(metadata=dict(id=name,split='train')) for name in ids]
    monkeypatch.setattr('haru_training.codec_adapter_dataset.load_dataset',lambda path:(dict(dataset_sha256=parent['dataset_sha256']),pairs))
    full=deepcopy(rank32);full.update(kind='haru-codec-full-data-correction',training_ids=ids,
        parent=dict(path=second.name,sha256=sha256(second.read_bytes()).hexdigest()),
        roundtrip_parent_sha256=rank32['parent']['sha256'])
    assert validate_full_data_correction(full,root=tmp_path)==full
    for field,value in [('training_ids',ids[:-1]),('learning_rate',.1),('rank',64),('roundtrip_parent_sha256','d'*64)]:
        changed=deepcopy(full);changed[field]=value
        with pytest.raises(ValueError):validate_full_data_correction(changed,root=tmp_path)
    pairs[-1].metadata['split']='reference'
    with pytest.raises(ValueError):validate_full_data_correction(full,root=tmp_path)
