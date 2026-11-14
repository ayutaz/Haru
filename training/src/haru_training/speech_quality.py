"""Real question inputs and independent ASR checks; no voice-quality claims."""
from hashlib import sha256
import math
from pathlib import Path
import unicodedata

import numpy as np


def _normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


def transcript_comparison(generated_text, recognized_text):
    reference,hypothesis=map(_normalized,(generated_text,recognized_text))
    previous=list(range(len(hypothesis)+1))
    for i,left in enumerate(reference,1):
        current=[i]
        for j,right in enumerate(hypothesis,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(left!=right)))
        previous=current
    return {'generated_text':generated_text,'recognized_text':recognized_text,
            'character_error_rate':previous[-1]/len(reference) if reference else None,
            'reference_characters':len(reference),'voice_similarity_confirmed':False,
            'scope':'ASR text agreement only; semantic coherence and voice identity are not assessed'}


def transcribe_comparison(path, generated_text, recognizer, *, expected_sha256):
    path=Path(path)
    actual=sha256(path.read_bytes()).hexdigest()
    if actual!=expected_sha256:raise ValueError('audio differs from the declared response')
    from .data import read_wave
    # The ndarray interface expects 16 kHz mono PCM. Decode our WAVs with the
    # same reader as training, avoiding version-dependent PyAV file decoding.
    samples=read_wave(path,sample_rate=16000).mean(axis=1,dtype=np.float32)
    segments,info=recognizer.transcribe(samples,language='ja',beam_size=5,
                                      condition_on_previous_text=False,vad_filter=True)
    rows=[{'start':s.start,'end':s.end,'text':s.text} for s in segments]
    for row in rows:
        if (not math.isfinite(row['start']) or not math.isfinite(row['end'])
                or not 0<=row['start']<=row['end'] or not isinstance(row['text'],str)):
            raise ValueError('invalid ASR segment')
    result=transcript_comparison(generated_text,''.join(row['text'] for row in rows).strip())
    result.update(wav_sha256=actual,segments=rows,language=getattr(info,'language',None))
    return result


def question_audio(audio, sample_rate, *, user_channel, start_seconds, end_seconds,
                   lead_seconds=.4, tail_seconds=8.):
    """Keep one recorded user turn, then silence, rather than a read-speech probe."""
    audio=np.asarray(audio,dtype=np.float32)
    if (audio.ndim!=2 or not len(audio) or not np.isfinite(audio).all()
            or type(sample_rate) is not int or sample_rate<=0
            or type(user_channel) is not int or not 0<=user_channel<audio.shape[1]
            or not all(math.isfinite(x) for x in (start_seconds,end_seconds,lead_seconds,tail_seconds))
            or not 0<=start_seconds<end_seconds<=len(audio)/sample_rate
            or lead_seconds<0 or tail_seconds<0):
        raise ValueError('invalid question audio, channel, or timing')
    start=round(max(0,start_seconds-lead_seconds)*sample_rate)
    end=round(end_seconds*sample_rate)
    return np.concatenate([audio[start:end,user_channel],np.zeros(round(tail_seconds*sample_rate),np.float32)])
