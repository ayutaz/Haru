"""One bounded mixed-data epoch with separate held-out validation roles.

The caller authenticates data, reviews replay eligibility, chooses a stable
controlled-trial condition and provisions vast.ai. This module does none of those
actions. A validation candidate is not a perceptual or publication approval.
"""

from dataclasses import asdict
from copy import deepcopy
import math
from pathlib import Path

from .diagnostics import preserved_inference
from .inference import export_inference
from .trials import _json, assess_response


SELECTION_POLICY = {'speech_text_increase_max': .2,
                    'dialogue_text_increase_max': .2,
                    'dialogue_audio_ratio_max': 1.25,
                    'rank': 'lowest speech validation audio loss, earlier step breaks ties',
                    'scope': 'validation candidate only; voice quality remains unknown'}


def run_adaptation(trainer, destination, generate, *, speech_ids, dialogue_ids, interval=8,
                   boundary_review=None):
    """Save step zero and every observation; consume at most one full epoch.

    Rendering is frozen, but checkpoint/export occurs after gradient flags and
    RNG have been restored. Validation IDs must exactly partition the supplied
    held-out validation set; arbitrary training/test IDs cannot be requested.
    When supplied, boundary_review receives a copy of the saved observation and
    must return {'accepted': bool, 'reason': nonempty str} before more updates.
    It may block for external inspection. Pending/invalid reviews never permit
    training; acceptance does not override the automatic rejection rules.
    """
    speech_ids, dialogue_ids = list(speech_ids), list(dialogue_ids)
    combined = speech_ids+dialogue_ids
    if (not speech_ids or not dialogue_ids or len(set(combined))!=len(combined)
            or set(combined)!={r.record_id for r in trainer.validation}):
        raise ValueError('speech/dialogue validation roles must partition held-out validation')
    width=trainer.spec.micro_batch_size*trainer.spec.accumulation_steps
    if (trainer.state.step!=0 or trainer.failed
            or trainer.spec.max_steps!=math.ceil(len(trainer.train)/width)):
        raise ValueError('adaptation requires a fresh trainer bounded to one epoch')
    if type(interval) is not int or interval<=0 or trainer.spec.checkpoint_interval:
        raise ValueError('observation interval must be positive; this runner owns checkpoints')
    root=Path(destination)
    root.mkdir(parents=True,exist_ok=False)
    _json(root/'spec.json',asdict(trainer.spec))
    _json(root/'selection-policy.json',SELECTION_POLICY)
    observations=[]
    stopped=False
    while True:
        step=trainer.state.step
        metrics={role:trainer.validate(record_ids=ids)
                 for role,ids in (('speech',speech_ids),('dialogue',dialogue_ids))}
        folder=root/f'step-{step:08d}'
        with preserved_inference(trainer.model):
            responses=generate(trainer.model,folder)
        if not responses:
            raise ValueError('at least one saved response is required at every observation')
        baseline=observations[0]['metrics'] if observations else metrics
        reasons={name:sorted(set(reason for role in metrics
                     for reason in assess_response(baseline[role],metrics[role],response)))
                 for name,response in responses.items()}
        stopped=any(reasons.values())
        trainer.save()
        export_inference(trainer,root/'inference'/f'step-{step:08d}')
        row={'step':step,'metrics':metrics,'rejection_reasons':reasons,
             'texts':{name:response['text'] for name,response in responses.items()},
             'response_directory':folder.name,'voice_quality_evaluated':False,
             'boundary_review':{'status':'pending' if boundary_review else 'not_requested'}}
        if boundary_review is not None:
            _json(root/f'review-request-{step:08d}.json',row)
            with preserved_inference(trainer.model):
                decision=boundary_review(deepcopy(row))
            if (not isinstance(decision,dict) or type(decision.get('accepted')) is not bool
                    or not isinstance(decision.get('reason'),str) or not decision['reason'].strip()):
                raise ValueError('boundary review requires explicit acceptance and a nonempty reason')
            row['boundary_review']={'status':'accepted' if decision['accepted'] else 'rejected',
                                    'reason':decision['reason'].strip()}
            stopped=stopped or not decision['accepted']
        _json(root/f'observation-{step:08d}.json',row)
        observations.append(row)
        if stopped or step==trainer.spec.max_steps:
            break
        trainer.fit(until_step=min(step+interval,trainer.spec.max_steps))
        if trainer.state.step==step:
            break
    base=observations[0]['metrics']
    eligible=[]
    for row in observations[1:]:
        speech,dialogue=row['metrics']['speech'],row['metrics']['dialogue']
        if (not any(row['rejection_reasons'].values())
                and speech['audio_loss']<base['speech']['audio_loss']
                and speech['text_loss']<=base['speech']['text_loss']+SELECTION_POLICY['speech_text_increase_max']
                and dialogue['text_loss']<=base['dialogue']['text_loss']+SELECTION_POLICY['dialogue_text_increase_max']
                and dialogue['audio_loss']<=max(base['dialogue']['audio_loss'],1e-12)*SELECTION_POLICY['dialogue_audio_ratio_max']):
            eligible.append(row)
    complete=trainer.state.step==trainer.spec.max_steps and trainer.state.epoch==1
    best=min(eligible,key=lambda r:(r['metrics']['speech']['audio_loss'],r['step'])) if eligible and complete and not stopped else None
    frozen_unchanged=trainer._frozen_digest()==trainer.frozen_digest
    if not frozen_unchanged:
        raise ValueError('frozen base parameters changed')
    result={'steps':trainer.state.step,'completed_epoch':complete,'stopped_for_quality':stopped,
            'candidate_step':best['step'] if best else None,'voice_quality_evaluated':False,
            'frozen_parameters_unchanged':frozen_unchanged,'observations':observations,
            'conclusion':'validation candidate only' if best else 'no candidate under fixed validation policy'}
    _json(root/'result.json',result)
    return result
