"""Cross-speaker pairs and explicit temporal correspondence for codec adaptation."""
from copy import deepcopy
from dataclasses import dataclass
import re

import numpy as np
import torch

from .checkpoint import read_checkpoint,write_checkpoint
from .speech_quality import _normalized


@dataclass(frozen=True)
class CodecPair:
    metadata: dict
    latents: torch.Tensor
    spectrum: torch.Tensor
    posteriors: torch.Tensor


def make_pair(metadata,latents,spectrum,posteriors,*,splits):
    m=deepcopy(metadata)
    try:
        if m['kind']!='haru-codec-pair' or m['version']!=1:
            raise ValueError('wrong pair artifact kind/version')
        if (m['split'] not in ('train','validation') or m['group_id']!=m['id']
                or m['group_id'] not in splits.get(m['split'],[])
                or sum(m['group_id'] in values for values in splits.values())!=1):
            raise ValueError('recording group crosses or disagrees with frozen split')
        if (not m['source_speaker'] or not m['target_speaker']
                or m['source_speaker']==m['target_speaker']
                or m['source_sha256']==m['target_sha256']):
            raise ValueError('adaptation needs distinct source and target speech')
        if any(not re.fullmatch('[a-f0-9]{64}',m[key]) for key in ('source_sha256','target_sha256')):
            raise ValueError('invalid source identities')
        if not re.fullmatch('[a-f0-9]{40}',m['codec_revision']):
            raise ValueError('codec revision must be pinned')
        if (not _normalized(m['source_text']) or _normalized(m['source_text'])!=_normalized(m['target_text'])
                or m['representation'] not in ('raw_codec_mean','raw_codec_sample') or m['sample_rate']!=24000):
            raise ValueError('unaligned text, normalized latent, or invalid rate')
        z,s,p=[torch.as_tensor(value).detach().clone().float() for value in (latents,spectrum,posteriors)]
        count,warmup,hop=m['valid_samples'],m['warmup_samples'],m['spectral_hop']
        if (z.ndim!=2 or z.shape[1]!=64 or not len(z)
                or any(type(value) is not int for value in (count,warmup,hop,m['ctc_frames']))
                or not 0<=warmup<count<=len(z)*3200 or hop<=0
                or s.ndim!=2 or s.shape!=(count//hop+1,257)
                or p.ndim!=2 or p.shape[0]!=m['ctc_frames'] or p.shape[1]<2
                or not all(torch.isfinite(value).all() for value in (z,s,p))
                or (s<0).any() or (p<0).any() or (p>1).any()
                or not torch.allclose(p.sum(-1),torch.ones(len(p)),atol=1e-5,rtol=1e-5)):
            raise ValueError('invalid raw latents, lengths, or aligned teacher features')
        a=m['alignment']
        if a['status']!='accepted' or a['method'] not in ('ctc-token-centers-v1','observed-ctc-anchors-v1'):
            raise ValueError('temporal correspondence has not been accepted')
        if a['method']=='observed-ctc-anchors-v1':
            q=m['quality']
            if (not re.fullmatch('[a-f0-9]{40}',m['ctc_revision'])
                    or q['matched_tokens']<12 or min(q['source_coverage'],q['target_coverage'])<2/3
                    or q['minimum_anchor_confidence']<.5 or q['maximum_anchor_gap_seconds']>1.):
                raise ValueError('observed anchor quality is insufficient')
        left,right=np.asarray(a['source_seconds']),np.asarray(a['target_seconds'])
        if (left.ndim!=1 or len(left)<2 or right.shape!=left.shape
                or not np.isfinite(left).all() or not np.isfinite(right).all()
                or np.any(np.diff(left)<=0) or np.any(np.diff(right)<=0)
                or left[0]!=0 or right[0]!=0 or not np.isclose(left[-1],count/24000)):
            raise ValueError('invalid monotonic time map')
        return CodecPair(m,z,s,p)
    except (KeyError,TypeError) as error:
        raise ValueError('incomplete pair metadata') from error


def save_pair(path,pair):
    return write_checkpoint(path,dict(metadata=pair.metadata,latents=pair.latents,
                                      spectrum=pair.spectrum,posteriors=pair.posteriors))


def load_pair(path,*,splits,codec_revision,expected_sources=None):
    data=read_checkpoint(path)
    if data['metadata']['codec_revision']!=codec_revision:
        raise ValueError('pair belongs to a different codec base')
    if expected_sources is not None and tuple(expected_sources)!=tuple(
            data['metadata'][key] for key in ('source_sha256','target_sha256')):
        raise ValueError('pair source recordings differ from input manifest')
    return make_pair(**data,splits=splits)


def align_ctc(log_probs,tokens,*,blank):
    """Viterbi CTC alignment, including required blanks between repeated symbols.

    Returned token spans are encoder-frame estimates, not phoneme ground truth.
    Acoustic confidence and independent text QC must be checked by the caller.
    """
    probs=np.asarray(log_probs,dtype=np.float64)
    if (probs.ndim!=2 or not probs.size or not np.isfinite(probs).all()
            or type(blank) is not int or not 0<=blank<probs.shape[1]
            or not tokens or any(type(t) is not int or not 0<=t<probs.shape[1] or t==blank for t in tokens)):
        raise ValueError('invalid CTC observations or labels')
    minimum=len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:]))
    if len(probs)<minimum:
        raise ValueError('CTC path cannot cover the transcript')
    states=np.full(2*len(tokens)+1,blank,dtype=np.int64);states[1::2]=tokens
    previous=np.full(len(states),-np.inf);previous[0]=0.
    back=np.empty((len(probs),len(states)),dtype=np.int8)
    skip=np.zeros(len(states),dtype=bool)
    skip[2:]=(states[2:]!=blank)&(states[2:]!=states[:-2])
    for t in range(len(probs)):
        one=np.r_[-np.inf,previous[:-1]]
        two=np.r_[-np.inf,-np.inf,previous[:-2]];two[~skip]=-np.inf
        choices=np.stack([previous,one,two])
        back[t]=np.argmax(choices,axis=0)
        previous=choices.max(axis=0)+probs[t,states]
    end=len(states)-2+int(previous[-1]>previous[-2])
    if not np.isfinite(previous[end]):
        raise ValueError('CTC alignment has no finite path')
    path=np.empty(len(probs),dtype=np.int64)
    for t in range(len(probs)-1,-1,-1):
        path[t]=end;end-=int(back[t,end])
    result=[]
    for index,token in enumerate(tokens):
        frames=np.flatnonzero(path==2*index+1)
        if not len(frames):raise ValueError('CTC transcript token was skipped')
        result.append(dict(token=token,start=int(frames[0]),end=int(frames[-1])+1,
                           confidence=float(np.exp(probs[frames,token].mean()))))
    return result


def warp_features(values,target_times,query_times,source_anchors,target_anchors):
    """Interpolate feature frames only; this never resynthesizes a warped WAV."""
    values=np.asarray(values,dtype=np.float32)
    times,query,left,right=[np.asarray(value,dtype=np.float64)
                            for value in (target_times,query_times,source_anchors,target_anchors)]
    if (values.ndim!=2 or times.ndim!=1 or len(times)!=len(values) or not len(times)
            or query.ndim!=1 or left.ndim!=1 or len(left)<2 or right.shape!=left.shape
            or any(not np.isfinite(value).all() for value in (values,times,query,left,right))
            or np.any(np.diff(times)<=0) or np.any(np.diff(left)<=0) or np.any(np.diff(right)<=0)):
        raise ValueError('invalid feature clock or nonmonotonic temporal map')
    mapped=np.interp(query,left,right)
    return np.stack([np.interp(mapped,times,column) for column in values.T],axis=1).astype(np.float32)
