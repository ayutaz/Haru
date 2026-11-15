from dataclasses import replace

import pytest
import torch

from test_preserving_pilot import records
from test_training_forward import tiny_model


def corpus():
    source=records('source')[0]
    train=[replace(source,record_id=f't{i:03}') for i in range(80)]
    validation=[replace(source,record_id=f'v{i:03}') for i in range(10)]
    splits=dict(train=[r.record_id for r in train],validation=[r.record_id for r in validation],
        test=[f'e{i}' for i in range(8)],reference=['r0','r1'])
    return train,validation,splits


def specification(train,val,splits):
    from haru_training.full_authentic import full_authentic_spec
    return full_authentic_spec(train,val,splits=splits,reference='r1',preservation_weight=1.,
        split_sha256='a'*64,cache_sha256='b'*64)


def test_full_recipe_is_three_epochs_of_exact80_and_preserves_test_and_reference_exclusion():
    from haru_training.trainer import dataset_fingerprint
    train,val,splits=corpus();spec=specification(train,val,splits)
    assert spec.max_steps==240 and spec.max_seconds==1200 and spec.validation_interval==240
    assert spec.dataset_sha256==dataset_fingerprint(train) and spec.validation_sha256==dataset_fingerprint(val)
    assert spec.preservation_target=='authentic_audio' and spec.training_audio_history=='recorded'
    assert spec.learning_rate==1e-6 and spec.feedback_mode=='generated_audio'


@pytest.mark.parametrize('invalid',['missing_train','reference_leak','test_leak','validation_leak','unknown_reference'])
def test_full_recipe_cannot_substitute_or_drop_any_corpus_record(invalid):
    from haru_training.full_authentic import full_authentic_spec
    train,val,splits=corpus();reference='r1'
    if invalid=='missing_train':train=train[:-1]
    elif invalid=='reference_leak':train[0]=replace(train[0],record_id='r0')
    elif invalid=='test_leak':train[0]=replace(train[0],record_id='e0')
    elif invalid=='validation_leak':val[0]=replace(val[0],record_id=train[0].record_id)
    else:reference='t000'
    with pytest.raises(ValueError):full_authentic_spec(train,val,splits=splits,reference=reference,
        preservation_weight=1.,split_sha256='a'*64,cache_sha256='b'*64)


def test_full_runner_covers_every_record_and_exports_the_fixed_terminal_model(tmp_path):
    from haru_training.full_authentic import run_full_authentic
    from haru_training.checkpoint import read_checkpoint
    import json
    torch.set_num_threads(1);torch.manual_seed(5)
    train,val,splits=corpus();spec=replace(specification(train,val,splits),base_model='test/tiny-haru',base_revision='a'*40)
    rendered=[]
    result=run_full_authentic(tiny_model(),train,val,spec,tmp_path,splits=splits,reference='r1',
        registered_weight=1.,render=lambda m,p,s:rendered.append(s))
    assert result['complete'] and result['completed_updates']==240 and result['frozen_weights_unchanged']
    assert result['all_training_records_seen'] and not result['final_test_used']
    assert rendered==[0,240]
    events=[json.loads(s) for s in (tmp_path/'trainer/events.jsonl').read_text().splitlines()]
    ids=[i for e in events if e['event']=='train' for i in e['record_ids']]
    assert set(ids)==set(splits['train']) and all(ids.count(i)==3 for i in splits['train'])
    bundle=read_checkpoint(tmp_path/'inference/step-00000240')
    assert bundle['step']==240 and bundle['spec']['preservation_target']=='authentic_audio'


@pytest.mark.parametrize('change',[dict(max_steps=239),dict(learning_rate=1e-5),dict(preservation_weight=.5)])
def test_full_runner_rejects_unregistered_changes_before_any_output(tmp_path,change):
    from haru_training.full_authentic import run_full_authentic
    train,val,splits=corpus();spec=replace(specification(train,val,splits),**change)
    with pytest.raises(ValueError):run_full_authentic(tiny_model(),train,val,spec,tmp_path/'run',splits=splits,
        reference='r1',registered_weight=1.,render=lambda *a:None)
    assert not (tmp_path/'run').exists()


def test_pilot_stays_bounded_to_three_records_and100_updates():
    from haru_training.authentic_feedback import authentic_spec
    train,val,splits=corpus()
    with pytest.raises(ValueError):authentic_spec(train,val,preservation_weight=1.,split_sha256='a'*64,cache_sha256='b'*64)
    with pytest.raises(ValueError):authentic_spec(train[:3],val[:3],preservation_weight=1.,split_sha256='a'*64,cache_sha256='b'*64,max_steps=240)
