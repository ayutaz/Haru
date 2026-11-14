"""Monotone anchors from independently observed matching CTC characters.

Do not force a canonical character with negligible acoustic probability. Content
against the known script is checked separately before these anchors are used.
"""
import math

import numpy as np


def _spans(log_probs,blank,ignored):
    probs=np.asarray(log_probs,dtype=np.float64)
    if probs.ndim!=2 or not probs.size or not np.isfinite(probs).all():
        raise ValueError('invalid observed CTC evidence')
    if not 0<=blank<probs.shape[1] or not np.allclose(np.exp(probs).sum(-1),1,atol=1e-5):
        raise ValueError('invalid CTC blank or posterior normalization')
    best=probs.argmax(-1);rows=[];start=0
    for stop in range(1,len(best)+1):
        if stop<len(best) and best[stop]==best[start]:continue
        token=int(best[start])
        if token!=blank and token not in ignored:
            frame=start+int(probs[start:stop,token].argmax())
            rows.append(dict(token=token,frame=frame,confidence=float(np.exp(probs[frame,token]))))
        start=stop
    return rows


def observed_anchors(source_log_probs,target_log_probs,*,source_seconds,target_seconds,blank,ignore_tokens=()):
    source=_spans(source_log_probs,blank,ignore_tokens)
    target=_spans(target_log_probs,blank,ignore_tokens)
    if (not all(math.isfinite(v) and v>0 for v in (source_seconds,target_seconds))
            or np.shape(source_log_probs)[1]!=np.shape(target_log_probs)[1] or not source or not target):
        raise ValueError('missing or incompatible content observations')
    # Longest common subsequence matches observed characters, without inserting
    # unsupported characters into either acoustic path.
    scores=np.zeros((len(source)+1,len(target)+1),dtype=np.int32)
    for i,left in enumerate(source,1):
        for j,right in enumerate(target,1):
            scores[i,j]=scores[i-1,j-1]+1 if left['token']==right['token'] else max(scores[i-1,j],scores[i,j-1])
    pairs=[];i=len(source);j=len(target)
    while i and j:
        if source[i-1]['token']==target[j-1]['token']:
            if min(source[i-1]['confidence'],target[j-1]['confidence'])>=.5:
                pairs.append((source[i-1],target[j-1]))
            i-=1;j-=1
        elif scores[i-1,j]>=scores[i,j-1]:i-=1
        else:j-=1
    pairs.reverse()
    coverage=[len(pairs)/len(source),len(pairs)/len(target)]
    if len(pairs)<12 or min(coverage)<2/3:
        raise ValueError('insufficient observed character coverage')
    anchors=[]
    for side,duration in enumerate((source_seconds,target_seconds)):
        times=np.array([0.,*[(p[side]['frame']*320+200)/16000 for p in pairs],duration])
        if np.any(np.diff(times)<=0):raise ValueError('nonmonotone observed clock')
        if np.diff(times).max()>1.:
            raise ValueError('unobserved alignment gap exceeds one second')
        anchors.append(times)
    quality=dict(matched_tokens=len(pairs),source_observed_tokens=len(source),target_observed_tokens=len(target),
        source_coverage=coverage[0],target_coverage=coverage[1],
        minimum_anchor_confidence=min(p[s]['confidence'] for p in pairs for s in (0,1)),
        maximum_anchor_gap_seconds=max(float(np.diff(a).max()) for a in anchors),
        human_quality_confirmed=False,scope='Observed character anchors; not ground-truth phoneme boundaries')
    return *anchors,quality
