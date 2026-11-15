"""Real question inputs and independent ASR checks; no voice-quality claims."""
from hashlib import sha256
from copy import deepcopy
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
            'character_edit_distance':previous[-1],
            'character_error_rate':previous[-1]/len(reference) if reference else None,
            'reference_characters':len(reference),'voice_similarity_confirmed':False,
            'scope':'ASR text agreement only; semantic coherence and voice identity are not assessed'}


def _asr_pcm(path):
    from .data import read_wave
    return read_wave(path,sample_rate=16000).mean(axis=1,dtype=np.float32)


def reuse_transcription(path, generated_text, cached, *, source, model, revision, settings):
    """Reuse an observed transcript only for byte-identical recognizer input.

    WAV container metadata can differ even when decoded samples are identical.
    The old WAV hash and recognizer protocol still have to match their record.
    No tolerance or audio similarity heuristic substitutes for exact PCM equality.
    """
    if (cached.get('generated_text')!=generated_text or cached.get('model')!=model
            or cached.get('revision')!=revision or cached.get('settings')!=settings):
        return None
    source,path=Path(source),Path(path)
    if sha256(source.read_bytes()).hexdigest()!=cached.get('wav_sha256'):
        raise ValueError('cached transcription source differs from its recorded WAV hash')
    before,after=_asr_pcm(source),_asr_pcm(path)
    if not np.array_equal(before,after):return None
    expected=transcript_comparison(generated_text,cached['recognized_text'])
    if any(cached.get(k)!=v for k,v in expected.items()):
        raise ValueError('cached transcription summary is inconsistent')
    result=deepcopy(cached)
    result['wav_sha256']=sha256(path.read_bytes()).hexdigest()
    result['transcription_reuse']=dict(identical_asr_input_pcm=True,
        source_wav_sha256=cached['wav_sha256'],input_pcm_sha256=sha256(after.tobytes()).hexdigest(),
        sample_rate=16000,input_samples=len(after),scope='Exact recognizer input and protocol, not an approximate audio match')
    return result


def transcribe_comparison(path, generated_text, recognizer, *, expected_sha256):
    path=Path(path)
    actual=sha256(path.read_bytes()).hexdigest()
    if actual!=expected_sha256:raise ValueError('audio differs from the declared response')
    # The ndarray interface expects 16 kHz mono PCM. Decode our WAVs with the
    # same reader as training, avoiding version-dependent PyAV file decoding.
    samples=_asr_pcm(path)
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


def turn_response_text(response, *, user_end_seconds):
    """Split replay text by how much user PCM the model had actually consumed.

    Frame index is playback time; Frame.seconds is compute time. A chunk may
    span a spoken turn boundary, so this is text timing, not forced alignment.
    The caller supplies the measured user-turn end and retains its provenance.
    """
    rate, size, samples = (response[k] for k in ('sample_rate', 'frame_samples', 'input_samples'))
    frames = response['frames']
    if (any(type(x) is not int or x <= 0 for x in (rate, size, samples))
            or not math.isfinite(user_end_seconds) or not 0 <= user_end_seconds <= samples / rate
            or len(frames) != (samples + size - 1) // size
            or any(f['index'] != i or not isinstance(f['text'], str) for i, f in enumerate(frames))):
        raise ValueError('invalid replay or user-turn timing')
    end_sample = user_end_seconds * rate
    before, after = [], []
    first = None
    for frame in frames:
        observed = min(frame['index'] * size, samples)
        (after if observed >= end_sample else before).append(frame['text'])
        if first is None and frame['text'].strip():
            first = frame['index'] * size / rate
    return dict(text_before_user_end=''.join(before).strip(), text_after_user_end=''.join(after).strip(),
                first_text_seconds=first, user_turn_fully_observed=(len(frames)-1)*size >= end_sample,
                question_answered=None, voice_similarity_confirmed=False,
                scope='Playback-frame text timing only; relevance, audibility and voice identity are not assessed')


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
