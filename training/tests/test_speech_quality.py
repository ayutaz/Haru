from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


def test_japanese_character_error_is_not_a_voice_similarity_score():
    from haru_training.speech_quality import transcript_comparison
    result=transcript_comparison('今日は、晴れです。','今日は 晴れです')
    assert result['character_error_rate']==0
    assert result['voice_similarity_confirmed'] is False
    assert transcript_comparison('あいう','あえう')['character_error_rate']==pytest.approx(1/3)
    assert transcript_comparison('', 'はい')['character_error_rate'] is None
    assert transcript_comparison('はい', '')['character_error_rate']==1


def test_asr_reads_real_audio_without_the_model_text_as_a_prompt(tmp_path):
    from haru_training.speech_quality import transcribe_comparison
    path=tmp_path/'response.wav'
    signal=np.sin(np.arange(24000,dtype=np.float32)*2*np.pi*440/24000)
    sf.write(path,signal,24000,subtype='FLOAT')
    calls=[]
    class Recognizer:
        def transcribe(self,path,**kwargs):
            calls.append((path,kwargs))
            return iter([SimpleNamespace(start=0.,end=1.,text='こんにちは')]),SimpleNamespace(language='ja')
    result=transcribe_comparison(path,'違う生成テキスト',Recognizer(),expected_sha256=sha256(path.read_bytes()).hexdigest())
    samples,kwargs=calls[0]
    assert isinstance(samples,np.ndarray)
    assert samples.shape==(16000,) and samples.dtype==np.float32
    assert np.isfinite(samples).all() and np.std(samples)>.5
    assert kwargs==dict(language='ja',beam_size=5,condition_on_previous_text=False,vad_filter=True)
    assert result['recognized_text']=='こんにちは'
    assert result['generated_text']=='違う生成テキスト'
    assert result['character_error_rate']>0
    assert result['wav_sha256']==sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='audio'):
        transcribe_comparison(path,'text',Recognizer(),expected_sha256=result['wav_sha256'])
    assert len(calls)==1


def test_question_uses_only_the_user_turn_and_gives_time_to_reply():
    from haru_training.speech_quality import question_audio
    audio=np.stack([np.full(100,99.),np.arange(100)],axis=1).astype('float32')
    result=question_audio(audio,10,user_channel=1,start_seconds=2.,end_seconds=3.,lead_seconds=.4,tail_seconds=2.)
    np.testing.assert_array_equal(result[:14],np.arange(16,30,dtype='float32'))
    np.testing.assert_array_equal(result[14:],np.zeros(20,dtype='float32'))
    assert result.ndim==1 and result.dtype==np.float32


@pytest.mark.parametrize('changes',[{'user_channel':2},{'start_seconds':4.,'end_seconds':3.},{'end_seconds':11.},{'tail_seconds':-1}])
def test_invalid_question_input_fails_before_rendering(changes):
    from haru_training.speech_quality import question_audio
    args=dict(user_channel=1,start_seconds=2.,end_seconds=3.);args.update(changes)
    with pytest.raises(ValueError):question_audio(np.ones((100,2),dtype='float32'),10,**args)
