"""Explicit conversation cards and lossless system-prefix replacement."""

import torch

from haru.prompt import card_templates,render_card
from .data import StreamSequence,_validate


def character_prompt(role):
    if role not in ('target','dialogue'):
        raise ValueError('unknown conversation role')
    card={'persona':'相手の話を聞き、日本語で自然に会話する。',
          'name':'つくよみちゃん' if role=='target' else '話し手',
          'other_name':'話し相手','relationship':'親しい知人',
          'situation':'電話で日常のことや予定を話している。'}
    if role=='target': card['gender']='female'
    return render_card(card,card_templates()['templates']['ja'])


def replace_system(sequence,system_ids):
    _validate(sequence)
    if not system_ids or any(type(token) is not int or token<0 for token in system_ids):
        raise ValueError('system prefix requires nonempty token IDs')
    reference=torch.nonzero(sequence.model_mask,as_tuple=False).flatten()
    if not len(reference): raise ValueError('missing reference after system prefix')
    count=int(reference[0])
    if (sequence.user_mask[:count].any() or sequence.supervise[:count].any()
            or (sequence.frame_indices[:count]!=-1).any()
            or sequence.model_latents[:count].any() or sequence.user_latents[:count].any()):
        raise ValueError('system prefix contains audio or supervision')
    values={'record_id':sequence.record_id}
    for key,value in vars(sequence).items():
        if key=='record_id': continue
        if key=='tokens': prefix=torch.tensor(system_ids,dtype=value.dtype,device=value.device)
        else:
            prefix=torch.full((len(system_ids),*value.shape[1:]),-1 if key=='frame_indices' else 0,
                              dtype=value.dtype,device=value.device)
        values[key]=torch.cat([prefix,value[count:]])
    result=StreamSequence(**values)
    _validate(result)
    return result
