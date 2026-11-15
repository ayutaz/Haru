from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
import soundfile as sf

from haru_training.speech_quality import transcript_comparison


def inputs(tmp_path):
    audio=(.1*np.sin(np.arange(24000)*2*np.pi*220/24000)).astype(np.float32)
    source,target=tmp_path/'source.wav',tmp_path/'target.wav'
    sf.write(source,audio,24000,format='WAV',subtype='FLOAT')
    sf.write(target,audio,24000,format='WAVEX',subtype='FLOAT')
    spec=dict(model='test/recognizer',revision='a'*40,settings=dict(language='ja',beam_size=5))
    cache=dict(transcript_comparison('こんにちは','こんにちは'),**spec,
        wav_sha256=sha256(source.read_bytes()).hexdigest(),segments=[dict(start=0.,end=1.,text='こんにちは')])
    return source,target,cache,spec


def test_reuse_requires_identical_asr_pcm_but_allows_different_wav_containers(tmp_path):
    from haru_training.speech_quality import reuse_transcription
    source,target,cache,spec=inputs(tmp_path);before=deepcopy(cache)
    assert sha256(target.read_bytes()).hexdigest()!=cache['wav_sha256']
    result=reuse_transcription(target,'こんにちは',cache,source=source,**spec)
    assert result['recognized_text']==cache['recognized_text'] and cache==before
    assert result['wav_sha256']==sha256(target.read_bytes()).hexdigest()
    assert result['transcription_reuse']['identical_asr_input_pcm']
    assert result['transcription_reuse']['source_wav_sha256']==cache['wav_sha256']


def test_reuse_rejects_a_different_waveform(tmp_path):
    from haru_training.speech_quality import reuse_transcription
    source,target,cache,spec=inputs(tmp_path)
    audio,rate=sf.read(target,dtype='float32');audio[100:200]*=-1
    sf.write(target,audio,rate,subtype='FLOAT')
    assert reuse_transcription(target,'こんにちは',cache,source=source,**spec) is None


@pytest.mark.parametrize('change',['text','model','revision','settings'])
def test_reuse_does_not_cross_text_or_recognizer_protocols(tmp_path,change):
    from haru_training.speech_quality import reuse_transcription
    source,target,cache,spec=inputs(tmp_path);text='こんにちは'
    if change=='text':text='こんばんは'
    elif change=='settings':spec['settings']=dict(language='en',beam_size=5)
    else:spec[change]='different'
    assert reuse_transcription(target,text,cache,source=source,**spec) is None


def test_reuse_checks_the_actual_cached_source_wav_hash(tmp_path):
    from haru_training.speech_quality import reuse_transcription
    source,target,cache,spec=inputs(tmp_path);cache['wav_sha256']='0'*64
    with pytest.raises(ValueError,match='source'):
        reuse_transcription(target,'こんにちは',cache,source=source,**spec)
