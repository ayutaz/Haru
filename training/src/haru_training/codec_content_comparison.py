"""Controlled content-loss comparisons bound to saved listening evidence."""
from copy import deepcopy
from hashlib import sha256
import json
import re


def _fingerprint(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _validate_teacher(ctc):
    try:
        if (not re.fullmatch(r'[\w.-]+/[\w.-]+',ctc['repo'])
                or not re.fullmatch('[a-f0-9]{40}',ctc['revision'])
                or type(ctc['blank']) is not int or ctc['blank']<0
                or not {'model.safetensors','vocab.json'}<=set(ctc['files'])
                or any(not re.fullmatch(r'[\w.-]+',name) or name in ('.','..') for name in ctc['files'])
                or any(not re.fullmatch('[a-f0-9]{64}',v) for v in ctc['files'].values())):
            raise ValueError('content teacher must retain its pinned identity')
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('incomplete pinned content teacher') from error


def prepare_comparison(parent, review, ctc):
    """Prepare settings only; never loads weights, rents a GPU, or trains.

    The caller must decode the actual hash-bound exported response first.
    Summary pass flags are ignored; support is recomputed from all three rows.
    """
    try:
        if (parent['kind']!='haru-codec-rank-correction' or parent['rank']!=32
                or type(parent['seed']) is not int or parent['seed']!=7
                or type(parent['max_steps']) is not int or parent['max_steps']!=300
                or parent['max_seconds']!=300 or parent['learning_rate']!=.001
                or len(parent['training_ids'])!=3 or len(set(parent['training_ids']))!=3
                or not re.fullmatch('[a-f0-9]{40}',parent['base_revision'])
                or not re.fullmatch('[a-f0-9]{64}',parent['dataset_sha256'])):
            raise ValueError('comparison requires the registered rank32 three-pair parent')
        expected=[c['id'] for c in parent['cases']]
        rows=review['cases']
        if (len(expected)!=3 or len(set(expected))!=3 or len(rows)!=3
                or {r['id'] for r in rows}!=set(expected)
                or sum(r['preferred_condition']=='candidate' for r in rows)<2
                or any(r['preferred_condition'] not in ('candidate',None) for r in rows)
                or any(r['content_by_condition']['candidate'] not in ('聞き取れる','一部不明') for r in rows)):
            raise ValueError('three-case listening support is missing or insufficient')
        _validate_teacher(ctc)
    except (KeyError,TypeError) as error:
        raise ValueError('incomplete comparison parent, listening evidence, or teacher') from error
    common={key:deepcopy(parent[key]) for key in (
        'base_model','base_revision','dataset_sha256','target_speaker','training_ids',
        'rank','seed','max_steps','max_seconds','learning_rate','cases','reference','system_prompt')}
    arms=[]
    for name,weight,strength in [('control',0.,0.),('uniform',.1,0.),('transition',.1,2.)]:
        arms.append(dict(deepcopy(common),id=name,kind='haru-codec-content-comparison-arm',
            start_from='pinned_base',content_weight=weight,transition_strength=strength,
            ctc=deepcopy(ctc) if weight else None))
    return dict(version=1,kind='haru-codec-content-comparison',arms=arms,
        parent_spec_fingerprint=_fingerprint(parent),listening_review_fingerprint=_fingerprint(review),
        requires_equal_completed_updates=True,training_started=False,voice_similarity_confirmed=False,
        partial_intelligibility_cases=[r['id'] for r in rows if r['content_by_condition']['candidate']=='一部不明'],
        gate_scope='Exploratory relative listening support; no absolute voice identity certification',
        interpretation='Unequal completed updates or missing arms cannot establish the proposed loss advantage.')


def create_arm_trainer(model,pairs,arm,*,content=None):
    """Connect a verified arm to the existing trainer without weakening freezing."""
    from .codec_adapter_training import CodecTrainer,TrainSpec
    expected={'control':(0.,0.),'uniform':(.1,0.),'transition':(.1,2.)}
    if (arm.get('id') not in expected or
            (arm.get('content_weight'),arm.get('transition_strength'))!=expected[arm['id']]):
        raise ValueError('unregistered content-loss arm')
    if arm['content_weight']:
        teacher=arm['ctc']
        provenance=getattr(content,'provenance',{})
        if (content is None or content.training or any(p.requires_grad for p in content.parameters())
                or any(provenance.get(k)!=teacher[k] for k in ('repo','revision'))
                or any(p.metadata.get('ctc_revision')!=teacher['revision'] for p in pairs)):
            raise ValueError('content encoder must be frozen and match every prepared teacher')
    elif content is not None:
        raise ValueError('control arm must not load a content encoder')
    spec=TrainSpec(**{k:arm[k] for k in ('base_model','base_revision','target_speaker','max_steps',
        'max_seconds','learning_rate','seed','content_weight','transition_strength')})
    if spec.max_steps>300 or spec.max_seconds>300:
        raise ValueError('content comparison exceeds the finite training budget')
    return CodecTrainer(model,pairs,spec,content=content,rank=arm['rank'])


def load_pinned_content(teacher,*,device='cuda'):
    """Load exactly the preparation teacher; CUDA remains Vast-only."""
    _validate_teacher(teacher)
    from pathlib import Path
    import torch
    from .codec_adapter_trial import require_vast_cuda,file_hash
    from .codec_content_encoder import FrozenCTC
    if torch.device(device).type=='cuda':require_vast_cuda()
    from huggingface_hub import snapshot_download
    directory=Path(snapshot_download(repo_id=teacher['repo'],revision=teacher['revision'],allow_patterns=list(teacher['files'])))
    for name,expected in teacher['files'].items():
        # HF snapshots may symlink into sibling blobs; validate the flat name
        # above and the pinned bytes here, without rejecting legitimate links.
        path=directory/name
        if file_hash(path)!=expected:
            raise ValueError('content teacher file identity differs from preparation')
    from transformers import Wav2Vec2ForCTC
    model=Wav2Vec2ForCTC.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False).to(device)
    result=FrozenCTC(model)
    result.provenance.update(repo=teacher['repo'],revision=teacher['revision'],files=deepcopy(teacher['files']))
    return result


def prepare_execution(parent_path,*,presentation_path,responses_path,answer_key_path,
                      comparisons_root,adapter_path,arm_id):
    """Recheck actual saved evidence before entering the paid training runner."""
    from pathlib import Path
    from .codec_adapter_diagnostic import validate_correction,select_training_pairs
    from .codec_adapter_dataset import load_dataset
    from .codec_adapter_training import _data_digest
    from .codec_adapter_evaluation import _verify,_tensor_hash
    from .codec_adapter_inference import _validate
    from .codec_adapter_trial import file_hash
    from .checkpoint import read_checkpoint
    from .listening_review import decode_review
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8-sig'))
    parent_path=Path(parent_path);root=parent_path.parent
    parent=validate_correction(read(parent_path),root=root)
    presentation=read(presentation_path);key=read(answer_key_path)
    review=decode_review(presentation,read(responses_path),key)
    if presentation['reference_sha256']!=parent['reference']['sha256']:
        raise ValueError('listening reference differs from the registered target')
    ctc=read(root/parent['membership']['path'])['ctc']
    plan=prepare_comparison(parent,review,ctc)
    arms={a['id']:a for a in plan['arms']}
    if arm_id not in arms:raise ValueError('unregistered comparison arm')
    data,pairs=load_dataset(root/parent['dataset']['path'])
    selected=select_training_pairs(pairs,parent['training_ids'])
    if (data['dataset_sha256']!=parent['dataset_sha256']
            or [p.metadata['id'] for p in selected]!=[p.metadata['id'] for p in pairs[:3]]
            or any(p.metadata.get('ctc_revision')!=ctc['revision'] for p in selected)):
        raise ValueError('comparison data or CTC teacher differs')
    artifact=read_checkpoint(adapter_path);identity=_validate(artifact)
    if (identity['base_model']!=parent['base_model'] or identity['base_revision']!=parent['base_revision']
            or identity['rank']!=32 or identity['target_speaker']!=parent['target_speaker']
            or identity['dataset_sha256']!=_data_digest(selected)):
        raise ValueError('listened adapter differs from the fixed rank32 parent')
    adapter_hash=sha256(''.join(k+_tensor_hash(v) for k,v in sorted(artifact['weights'].items())).encode()).hexdigest()
    keys={row['id']:row for row in key['cases']}
    comparison_hashes={}
    for case in parent['cases']:
        directory=Path(comparisons_root)/case['id'];info=_verify(directory)
        if (info['adapter_sha256']!=adapter_hash or info['codec_sha256']!=artifact['codec_sha256']
                or info['base_revision']!=parent['base_revision'] or info['input_sha256']!=case['sha256']
                or info['target_speaker']!=parent['target_speaker']
                or info['dataset_input_manifest_sha256']!=parent['membership']['sha256']
                or info['system_prompt']!=parent['system_prompt'] or info['config']['seed']!=parent['seed']
                or any(info['files'][side+'/response.wav']!=keys[case['id']]['audio_sha256'][side]
                       for side in ('baseline','candidate'))):
            raise ValueError('listened audio does not belong to the registered candidate')
        comparison_hashes[case['id']]=file_hash(directory/'comparison.json')
    evidence={name+'_sha256':file_hash(path) for name,path in (
        ('parent',parent_path),('presentation',presentation_path),('responses',responses_path),
        ('answer_key',answer_key_path),('candidate_manifest',Path(adapter_path)/'manifest.json'))}
    evidence['comparisons']=comparison_hashes
    return dict(arm=arms[arm_id],review=review,evidence=evidence)


def main():
    import argparse
    from .codec_adapter_diagnostic import run_small_training
    parser=argparse.ArgumentParser(description='Run one bounded content-loss arm after verifying saved listening evidence.')
    parser.add_argument('parent');parser.add_argument('destination')
    for name in ('presentation','responses','answer-key','comparisons-root','adapter','arm','gate','roundtrip-report'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    return run_small_training(args.parent,args.destination,gate_path=args.gate,roundtrip_report=args.roundtrip_report,
        comparison=dict(presentation_path=args.presentation,responses_path=args.responses,answer_key_path=args.answer_key,
            comparisons_root=args.comparisons_root,adapter_path=args.adapter,arm_id=args.arm))


if __name__=='__main__':main()
