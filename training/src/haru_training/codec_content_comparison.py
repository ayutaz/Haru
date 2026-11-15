"""Inert preparation of controlled content-loss comparisons, after listening."""
from copy import deepcopy
from hashlib import sha256
import json
import re


def _fingerprint(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


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
        if (not re.fullmatch(r'[\w.-]+/[\w.-]+',ctc['repo'])
                or not re.fullmatch('[a-f0-9]{40}',ctc['revision'])
                or type(ctc['blank']) is not int or ctc['blank']<0
                or not {'model.safetensors','vocab.json'}<=set(ctc['files'])
                or any(not re.fullmatch('[a-f0-9]{64}',v) for v in ctc['files'].values())):
            raise ValueError('content teacher must retain its pinned identity')
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
