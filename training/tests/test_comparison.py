import numpy as np
import pytest

from haru_training.comparison import measure_repeatability, compare_outputs


def output(value=0.):
    return {'audio': np.array([0., value], np.float32),
            'logits': np.array([1., 2.], np.float32),
            'latent': np.array([3.], np.float32), 'tokens': [3, 4]}


def test_repeatability_sets_each_tolerance_before_candidate_comparison():
    a, b = output(), output(.125)
    policy = measure_repeatability(a, b)
    assert policy == {'audio': .125, 'logits': 0., 'latent': 0., 'tokens': 'exact'}
    assert compare_outputs(a, b, policy)['passed']
    assert not compare_outputs(a, output(.25), policy)['passed']
    assert policy['audio'] == .125


def test_logit_latent_and_token_errors_cannot_hide_behind_audio_tolerance():
    policy = measure_repeatability(output(), output())
    for key in ('audio', 'logits', 'latent', 'tokens'):
        changed = output()
        changed[key][-1] += 1
        assert not compare_outputs(output(), changed, policy)['passed']
    changed = output()
    changed['audio'] = np.ones(3)
    with pytest.raises(ValueError): compare_outputs(output(), changed, policy)
    changed = output()
    changed['logits'][0] = float('nan')
    with pytest.raises(ValueError): measure_repeatability(output(), changed)


def test_nondeterministic_baseline_tokens_require_investigation():
    changed = output()
    changed['tokens'][0] = 5
    with pytest.raises(ValueError): measure_repeatability(output(), changed)


def wav_payload(values, rate=24000):
    import io
    import soundfile as sf
    stream=io.BytesIO()
    sf.write(stream,np.asarray(values,dtype=np.float32),rate,format='WAV',subtype='FLOAT')
    return stream.getvalue()


def test_waveform_parity_ignores_container_metadata_but_not_pcm_changes():
    import struct
    from haru_training.comparison import compare_waveforms
    original=wav_payload([0.,.25,-.125])
    changed=bytearray(original[:12]+b'JUNK'+struct.pack('<I',4)+b'abcd'+original[12:])
    struct.pack_into('<I',changed,4,len(changed)-8)
    result=compare_waveforms(original,bytes(changed))
    assert result['passed'] and not result['file_bytes_equal']
    assert result['first_pcm_sha256']==result['second_pcm_sha256']
    assert result['max_absolute_error']==0 and result['samples']==3
    mismatch=compare_waveforms(original,wav_payload([0.,.250001,-.125]))
    assert not mismatch['passed'] and mismatch['max_absolute_error']>0


@pytest.mark.parametrize('other',[
    wav_payload([.1,.2],16000),wav_payload([.1]),wav_payload([float('nan'),.2]),
    wav_payload([[.1,.2],[.1,.2]])],ids=['rate','length','nonfinite','stereo'])
def test_waveform_parity_rejects_invalid_or_differently_shaped_audio(other):
    from haru_training.comparison import compare_waveforms
    with pytest.raises(ValueError): compare_waveforms(wav_payload([.1,.2]),other)
