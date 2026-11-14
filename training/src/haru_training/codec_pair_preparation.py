"""Measured CTC correspondence and spectral targets, not raw waveform matching."""
from copy import deepcopy

import numpy as np
import torch

from .codec_adapter_data import align_ctc,make_pair,warp_features


def token_error_rate(expected,actual):
    previous=list(range(len(actual)+1))
    for i,left in enumerate(expected,1):
        current=[i]
        for j,right in enumerate(actual,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(left!=right)))
        previous=current
    return previous[-1]/len(expected)


def collapse_ctc(log_probs,blank):
    result=[];previous=None
    for value in np.asarray(log_probs).argmax(-1).tolist():
        if value!=blank and value!=previous:result.append(value)
        previous=value
    return result


def log_spectrum(audio):
    value=torch.as_tensor(np.ascontiguousarray(audio),dtype=torch.float32)
    result=torch.stft(value,n_fft=512,hop_length=160,win_length=512,
                      window=torch.hann_window(512),center=True,return_complex=True)
    return torch.log1p(result.abs()).T


def ctc_times(frames):
    # Pinned XLSR convolution stride=320, receptive field=400 at 16 kHz.
    return (np.arange(frames,dtype=np.float64)*320+200)/16000


def prepare_pair(metadata,source_audio,target_audio,latents,source_log_probs,target_log_probs,
                 tokens,*,blank,splits,ignore_tokens=()):
    m=deepcopy(metadata)
    source,target=[np.asarray(value,dtype=np.float32) for value in (source_audio,target_audio)]
    if any(value.ndim!=1 or len(value)<=3200 or not np.isfinite(value).all() for value in (source,target)):
        raise ValueError('preparation needs finite mono 24 kHz source and target')
    content_tokens=[value for value in tokens if value not in ignore_tokens]
    if not content_tokens:raise ValueError('content reference is empty')
    spans=[];errors=[]
    for audio,probs in ((source,source_log_probs),(target,target_log_probs)):
        probs=np.asarray(probs,dtype=np.float64)
        if probs.ndim!=2 or not np.isfinite(probs).all():raise ValueError('invalid CTC content evidence')
        if len(probs)!=((len(audio)*2+2)//3-400)//320+1:
            raise ValueError('CTC clock differs from pinned convolution configuration')
        # Keep acoustic observations intact; ignore formatting-only delimiter
        # tokens solely when comparing the independently decoded kana text.
        decoded=[value for value in collapse_ctc(probs,blank) if value not in ignore_tokens]
        errors.append(token_error_rate(content_tokens,decoded))
        if errors[-1]>.35:raise ValueError('content mismatch before forced alignment')
        spans.append(align_ctc(probs,list(map(int,tokens)),blank=int(blank)))
    confidence=min(row['confidence'] for rows in spans for row in rows)
    if confidence<.01:raise ValueError('forced alignment acoustic confidence is insufficient')
    clocks=[ctc_times(len(value)) for value in (source_log_probs,target_log_probs)]
    anchors=[]
    for audio,rows in zip((source,target),spans):
        middle=[((row['start']+row['end']-1)/2*320+200)/16000 for row in rows]
        anchors.append(np.array([0.,*middle,len(audio)/24000],dtype=np.float64))
    slopes=np.diff(anchors[1])/np.diff(anchors[0])
    ratio=len(target)/len(source)
    if (not .5<=ratio<=2 or np.any(slopes<=0) or np.quantile(slopes,.95)>8
            or np.quantile(slopes,.05)<.125):
        raise ValueError('temporal correspondence has excessive local stretching')
    target_spectrum=log_spectrum(target).numpy()
    source_spectral_times=np.arange(len(source)//160+1)*160/24000
    target_spectral_times=np.arange(len(target_spectrum))*160/24000
    aligned_spectrum=warp_features(target_spectrum,target_spectral_times,source_spectral_times,*anchors)
    aligned_p=warp_features(np.exp(target_log_probs),clocks[1],clocks[0],*anchors)
    aligned_p/=aligned_p.sum(-1,keepdims=True)
    m.update(kind='haru-codec-pair',version=1,representation='raw_codec_mean',sample_rate=24000,
             valid_samples=len(source),warmup_samples=3200,spectral_hop=160,ctc_frames=len(source_log_probs),
             alignment=dict(status='accepted',method='ctc-token-centers-v1',
                            source_seconds=anchors[0].tolist(),target_seconds=anchors[1].tolist(),
                            scope='Automatic CTC character-time correspondence, not ground-truth phoneme timing'),
             quality=dict(source_token_error_rate=errors[0],target_token_error_rate=errors[1],
                          minimum_alignment_confidence=confidence,duration_ratio=ratio,
                          local_slope_05=float(np.quantile(slopes,.05)),local_slope_95=float(np.quantile(slopes,.95)),
                          human_quality_confirmed=False))
    return make_pair(m,latents,aligned_spectrum,aligned_p,splits=splits)
