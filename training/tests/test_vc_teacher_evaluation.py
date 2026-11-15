"""External VC diagnostics are not permission to adopt a Haru adapter."""
from copy import deepcopy
import json

import numpy as np
import pytest
import soundfile as sf

from test_evaluation import manifest, speaker_panels, digest, dump, reseal


def teacher_manifest(manifest, root):
    result=deepcopy(manifest);case=result['cases'][0]
    folder=root/case['candidate']['directory']
    sf.write(folder/'response.wav',.1*np.sin(np.arange(22050)*.12),22050,subtype='FLOAT')
    metadata=json.loads((folder/'response.json').read_text(encoding='utf-8'))
    metadata.update(sample_rate=22050,feedback_mode='external_vc_teacher_only')
    metadata['provenance'].update(teacher_conversion_sha256='d'*64,teacher_original_wav_sha256='e'*64)
    dump(folder/'response.json',metadata);reseal(folder)
    case['candidate']['asr']['wav_sha256']=digest(folder/'response.wav')
    case['speaker_panels']=speaker_panels(result,root)
    conversion=dict(expected_cases=list(result['expected_cases']),cases=[dict(id=case['id'],
        source_sha256=digest(root/case['baseline']['directory']/'response.wav'),output_sha256='e'*64)],
        training_started=False)
    return result,conversion


def screen(value,conversion,root):
    from haru_training.vc_teacher_evaluation import screen_vc_teachers
    return screen_vc_teachers(value,conversion,root=root,conversion_sha256='d'*64)


def test_native_vc_rates_get_content_and_speaker_checks_without_relaxing_codec_gate(manifest,tmp_path):
    from haru_training.evaluation import evaluate_manifest
    value,conversion=teacher_manifest(manifest,tmp_path)
    original=evaluate_manifest(value,root=tmp_path)
    assert original['cases'][0]['reasons']==['unpaired_audio_format']
    report=screen(value,conversion,tmp_path)
    case=report['cases'][0]
    assert 'unpaired_audio_format' not in case['reasons']
    assert case['content']['status']=='no_detected_regression'
    assert case['source_content']['character_error_rate']==0
    assert len(case['speaker']['encoders'])==2
    assert report['student_training_authorized'] is False
    assert report['voice_similarity_confirmed'] is False


@pytest.mark.parametrize('mutation',['source','output','delivery','mode','missing'])
def test_unbound_or_missing_teacher_cannot_be_screened_as_usable(manifest,tmp_path,mutation):
    value,conversion=teacher_manifest(manifest,tmp_path)
    if mutation=='source':conversion['cases'][0]['source_sha256']='f'*64
    elif mutation=='output':conversion['cases'][0]['output_sha256']='f'*64
    elif mutation=='missing':conversion['cases']=[]
    else:
        path=tmp_path/'adapted/response.json';metadata=json.loads(path.read_text(encoding='utf-8'))
        if mutation=='delivery':metadata['provenance']['teacher_conversion_sha256']='f'*64
        else:metadata['feedback_mode']='generated_audio'
        dump(path,metadata);reseal(path.parent)
    report=screen(value,conversion,tmp_path)
    assert report['decision']=='hold'
    assert report['counts']['hold']==1
    assert report['student_training_authorized'] is False


def test_content_changes_against_source_are_not_hidden_by_generated_text_agreement(manifest,tmp_path):
    value,conversion=teacher_manifest(manifest,tmp_path);case=value['cases'][0]
    # The generated text may already disagree with what the source says.
    # The conversion must not silently change the source toward that text.
    source='今日は近くのお店でお昼ご飯を食べましょう'
    different='明日は遠くの公園で朝ご飯を食べましょう'
    for side in ('baseline','candidate'):
        path=tmp_path/case[side]['directory']/'response.json'
        metadata=json.loads(path.read_text(encoding='utf-8'));metadata['text']=different
        dump(path,metadata);reseal(path.parent)
        case[side]['asr'].update(generated_text=different,recognized_text=different if side=='candidate' else source)
    report=screen(value,conversion,tmp_path)
    assert report['decision']=='reject'
    assert 'source_asr_regression' in report['cases'][0]['reasons']
    assert report['cases'][0]['source_content']['character_error_rate']>.15


def test_frozen_expected_cases_keep_missing_teacher_in_denominator(manifest,tmp_path):
    value,conversion=teacher_manifest(manifest,tmp_path)
    value['expected_cases'].append('missing-teacher');conversion['expected_cases'].append('missing-teacher')
    report=screen(value,conversion,tmp_path)
    assert report['expected_count']==2
    assert report['cases'][1]['reasons']==['missing_case']
