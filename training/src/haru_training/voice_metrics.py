"""Fixed voiced-region preprocessing and within-experiment speaker similarity.

The caller supplies a pinned VAD and speaker encoder. Imports never download
models or allocate GPU resources. Similarity does not certify perceptual quality.
"""

import math

import numpy as np
from scipy.signal import resample_poly


def voiced_audio(audio, sample_rate, vad, *, minimum_seconds=1.):
    """16kHz, 512-sample VAD frames, threshold .5 and one-frame context.

    Independent recordings reset recurrent VAD state. Only selected regions are
    concatenated. RMS is normalized to .1, with a peak ceiling of .99; there is
    no clipping or invented speech when the detector finds too little speech.
    """
    original=np.asarray(audio,dtype=np.float32)
    if (original.ndim!=1 or not len(original) or not np.isfinite(original).all()
            or type(sample_rate) is not int or not 1<=sample_rate<=192000
            or not math.isfinite(minimum_seconds) or minimum_seconds<=0):
        raise ValueError('expected finite mono audio and a valid sample rate')
    divisor=math.gcd(sample_rate,16000)
    signal=resample_poly(original,16000//divisor,sample_rate//divisor) if sample_rate!=16000 else original.copy()
    resample_gain=min(1.,.99/max(float(np.max(np.abs(signal))),1e-12))
    signal=signal*resample_gain
    count=math.ceil(len(signal)/512)
    padded=np.pad(signal,(0,count*512-len(signal)))
    vad.reset_states()
    probabilities=np.array([float(vad(frame.copy())) for frame in padded.reshape(-1,512)])
    if not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('VAD returned an invalid speech probability')
    positive=probabilities>=.5
    selected=positive.copy()
    selected[1:] |= positive[:-1]
    selected[:-1] |= positive[1:]
    value=signal[np.repeat(selected,512)[:len(signal)]]
    if len(value)<math.ceil(minimum_seconds*16000):
        raise ValueError('insufficient voiced audio; similarity remains unknown')
    rms=float(np.sqrt(np.mean(value.astype(np.float64)**2)))
    if rms<=1e-12:
        raise ValueError('selected voiced audio is silent; similarity remains unknown')
    gain=min(.1/rms,.99/float(np.max(np.abs(value))))
    result=(value*gain).astype(np.float32)
    return result,{'input_samples':len(original),'input_sample_rate':sample_rate,'sample_rate':16000,
                   'resampled_samples':len(signal),'selected_samples':len(result),
                   'vad_positive_frames':int(positive.sum()),'selected_frames':int(selected.sum()),
                   'threshold':.5,'padding_frames':1,'resample_gain':resample_gain,'rms_gain':gain,
                   'scope':'VAD-selected regions, not phoneme or pronunciation verification'}


def speaker_cosine(first,second):
    first,second=np.asarray(first,dtype=np.float64),np.asarray(second,dtype=np.float64)
    if (first.ndim!=1 or second.shape!=first.shape or not len(first)
            or not np.isfinite(first).all() or not np.isfinite(second).all()
            or np.linalg.norm(first)==0 or np.linalg.norm(second)==0):
        raise ValueError('speaker embeddings must be finite, nonzero and have matching dimensions')
    return float(np.clip(np.dot(first,second)/(np.linalg.norm(first)*np.linalg.norm(second)),-1.,1.))
