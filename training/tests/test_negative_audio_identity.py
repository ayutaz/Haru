import numpy as np
import soundfile as sf

from haru_training.direct_speaker_evaluation import evaluate_direct_manifest
from test_direct_speaker_evaluation import manifest
from test_evaluation import digest,reseal,speaker_panels


def test_decoded_identity_ignores_container_metadata_but_preserves_audio_changes(tmp_path):
    from haru_training.speech_quality import decoded_audio_sha256
    signal=np.sin(np.arange(24000)*.02).astype(np.float32)*.1
    first,second=tmp_path/'first.wav',tmp_path/'second.wav'
    sf.write(first,signal,24000,subtype='FLOAT');sf.write(second,signal,24000,format='WAVEX',subtype='FLOAT')
    assert digest(first)!=digest(second) and decoded_audio_sha256(first)==decoded_audio_sha256(second)
    sf.write(second,signal*.5,24000,subtype='FLOAT')
    assert decoded_audio_sha256(first)!=decoded_audio_sha256(second)


def test_known_human_negative_cannot_evade_veto_with_a_different_wav_container(manifest,tmp_path):
    from haru_training.speech_quality import decoded_audio_sha256
    wav=tmp_path/'adapted/response.wav';negative_wav=digest(wav);negative_pcm=decoded_audio_sha256(wav)
    signal,rate=sf.read(wav,dtype='float32');sf.write(wav,signal,rate,format='WAVEX',subtype='FLOAT')
    assert digest(wav)!=negative_wav
    reseal(wav.parent);manifest['cases'][0]['candidate']['asr']['wav_sha256']=digest(wav)
    manifest['cases'][0]['speaker_panels']=speaker_panels(manifest,tmp_path)
    report=evaluate_direct_manifest(manifest,root=tmp_path,known_negative_wav_sha256=[negative_wav],
        known_negative_audio_sha256=[negative_pcm])
    assert report['decision']=='hold' and 'known_human_voice_negative' in report['cases'][0]['reasons']
    assert not report['voice_similarity_confirmed']
