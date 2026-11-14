import numpy as np
import pytest


class Vad:
    def __init__(self, probabilities):
        self.probabilities=probabilities
        self.resets=0
    def reset_states(self):
        self.cursor=0
        self.resets+=1
    def __call__(self, frame):
        assert frame.shape==(512,) and np.isfinite(frame).all()
        value=self.probabilities[self.cursor]
        self.cursor+=1
        return value


def test_voice_regions_resample_then_pad_and_normalize_without_clipping():
    from haru_training.voice_metrics import voiced_audio
    audio=(np.sin(np.arange(24000)*.05)*.2).astype(np.float32)
    vad=Vad([0]*5+[1]*20+[0]*7)
    value,meta=voiced_audio(audio,24000,vad,minimum_seconds=.5)
    assert len(value)==22*512 and meta['selected_samples']==len(value)
    assert meta['input_samples']==24000 and meta['sample_rate']==16000
    assert np.max(np.abs(value))<1 and np.sqrt(np.mean(value**2))==pytest.approx(.1,rel=1e-5)
    assert vad.resets==1 and meta['vad_positive_frames']==20
    repeated,_=voiced_audio(audio,24000,vad,minimum_seconds=.5)
    np.testing.assert_array_equal(value,repeated)
    assert vad.resets==2


def test_no_speech_or_nonfinite_vad_is_unknown_rather_than_zero_similarity():
    from haru_training.voice_metrics import voiced_audio
    with pytest.raises(ValueError,match='voiced'):
        voiced_audio(np.zeros(16000,np.float32),16000,Vad([0]*32))
    with pytest.raises(ValueError,match='probability'):
        voiced_audio(np.ones(16000,np.float32)*.1,16000,Vad([float('nan')]*32))
    with pytest.raises(ValueError,match='finite mono'):
        voiced_audio(np.ones((16000,2),np.float32),16000,Vad([1]*32))


def test_cosine_uses_full_embedding_and_rejects_zero_or_mismatched_vectors():
    from haru_training.voice_metrics import speaker_cosine
    assert speaker_cosine([1.,0.],[1.,1.])==pytest.approx(2**-.5)
    assert speaker_cosine([1.,0.],[-1.,0.])==pytest.approx(-1.)
    for first,second in (([0.,0.],[1.,0.]),([1.],[1.,2.]),([float('nan')],[1.])):
        with pytest.raises(ValueError,match='embedding'):
            speaker_cosine(first,second)
