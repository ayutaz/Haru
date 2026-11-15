"""One bounded full-data diagnostic, distinct from the three-record pilot.

Three complete epochs of the immutable80/10 corpus split. This experiment does
not certify voice identity or establish that the auxiliary loss beats a full-data
control. Final test recordings and final dialogue cases never enter training.
"""
from dataclasses import replace
import json
import math
from pathlib import Path

from .authentic_feedback import AuthenticFeedbackLoss
from .content_preservation import _execute_head_pilot
from .trainer import RunSpec,dataset_fingerprint


def validate_full_membership(splits,*,train,validation,reference):
    try:
        groups=[list(splits[k]) for k in ['train','validation','test','reference']]
        all_ids=[i for group in groups for i in group]
        if ([len(g) for g in groups]!=[80,10,8,2] or len(set(all_ids))!=100
                or not all(isinstance(i,str) and i for i in all_ids)
                or [r.record_id for r in train]!=groups[0]
                or [r.record_id for r in validation]!=groups[1] or reference not in groups[3]):
            raise ValueError('full diagnostic requires the immutable80/10/8/2 corpus split')
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('incomplete full-corpus membership') from error


def full_authentic_spec(train,validation,*,splits,reference,preservation_weight,split_sha256,cache_sha256):
    validate_full_membership(splits,train=train,validation=validation,reference=reference)
    if (type(preservation_weight) not in (int,float) or not math.isfinite(preservation_weight)
            or not 0<preservation_weight<=1000):
        raise ValueError('full diagnostic requires a fixed positive authentic-feedback coefficient')
    return RunSpec(base_model='higurashi-ai/Haru-4B',base_revision='8d6e83bcd3701a76fc27a764965665da1df7da78',
        dataset_sha256=dataset_fingerprint(train),validation_sha256=dataset_fingerprint(validation),
        split_sha256=split_sha256,cache_sha256=cache_sha256,mode='head',optimizer_precision='fp32_master',
        feedback_mode='generated_audio',training_audio_history='recorded',preservation_target='authentic_audio',
        preservation_weight=preservation_weight,learning_rate=1e-6,seed=20261114,
        max_steps=240,max_seconds=1200,validation_interval=240,checkpoint_interval=0)


def run_full_authentic(model,train,validation,spec,destination,*,splits,reference,registered_weight,render):
    expected=replace(full_authentic_spec(train,validation,splits=splits,reference=reference,
        preservation_weight=registered_weight,split_sha256=spec.split_sha256,cache_sha256=spec.cache_sha256),
        base_model=spec.base_model,base_revision=spec.base_revision)
    if spec!=expected:raise ValueError('not the registered full-data diagnostic')
    result=_execute_head_pilot(model,train,validation,spec,destination,render=render,objective_factory=AuthenticFeedbackLoss)
    root=Path(destination)
    events=[json.loads(s) for s in (root/'trainer/events.jsonl').read_text(encoding='utf-8').splitlines()]
    seen={i for e in events if e['event']=='train' for i in e['record_ids']}
    result.pop('expansion_authorized',None)
    result.update(training_data_scope='full_train80',training_record_count=80,validation_record_count=10,
        all_training_records_seen=seen==set(splits['train']),adoption_authorized=False,
        candidate_rule='Fixed terminal240, never selected by development score')
    (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result
