"""Human teacher evidence can veto a metric improvement; never authorizes training."""
from copy import deepcopy

import pytest

from haru_training.listening_review import make_presentation


def evidence():
    cases=[dict(id=name,audio_sha256={'A':str(i+1)*64,'B':str(i+3)*64})
           for i,name in enumerate(['probe-one','probe-two'])]
    presentation=make_presentation(cases,reference_sha256='a'*64,include_voice_match=True)
    answers=dict(version=1,presentation_sha256=presentation['presentation_sha256'],answers={})
    key=dict(presentation_sha256=presentation['presentation_sha256'],cases=[])
    automatic=dict(expected_count=2,evaluated_count=2,cases=[])
    for i,case in enumerate(cases):
        order=['baseline','candidate'] if i==0 else ['candidate','baseline']
        hashes={side:case['audio_sha256'][label] for label,side in zip(['A','B'],order)}
        key['cases'].append(dict(id=case['id'],order=order,audio_sha256=hashes))
        automatic['cases'].append(dict(id=case['id'],decision='candidate_for_listening',reasons=[],
            artifacts={side:{'response.wav':value} for side,value in hashes.items()}))
        for side in ['A','B']:answers['answers'][case['id']+':'+side+'_content']='聞き取れる'
        answers['answers'][case['id']+':voice']='差が分からない'
        answers['answers'][case['id']+':voice_match']='両方似ていない'
    return presentation,answers,key,automatic


def assess(parts):
    from haru_training.vc_teacher_review import assess_teacher_listening
    return assess_teacher_listening(*parts)


def positive(parts):
    p,a,key,auto=parts
    for case in key['cases']:
        label=['A','B'][case['order'].index('candidate')]
        a['answers'][case['id']+':voice']=label
        a['answers'][case['id']+':voice_match']=label
    return parts


def test_actual_negative_pattern_vetoes_two_automatic_candidates():
    result=assess(evidence())
    assert result['decision']=='failed'
    assert result['relative_improvement_count']==0 and result['candidate_target_match_count']==0
    assert result['candidate_intelligible_count']==2
    assert result['teacher_listening_supported'] is False
    assert result['training_authorized'] is False
    assert all('no_perceived_voice_gain' in r['reasons'] for r in result['cases'])


def test_positive_listening_is_not_codec_timing_or_training_approval():
    result=assess(positive(evidence()))
    assert result['decision']=='supported_for_teacher_preflight'
    assert result['teacher_listening_supported'] is True
    assert result['fine_timing_supported'] is False and result['codec_roundtrip_supported'] is False
    assert result['training_authorized'] is False


def test_relative_gain_does_not_certify_absolute_identity():
    parts=positive(evidence());p,a,key,auto=parts
    for case in p['cases']:a['answers'][case['id']+':voice_match']='両方似ていない'
    result=assess(parts)
    assert result['relative_improvement_count']==2
    assert result['candidate_target_match_count']==0
    assert result['voice_similarity_confirmed'] is False


@pytest.mark.parametrize('kind',['voice','content'])
def test_uncertain_human_judgment_remains_held(kind):
    parts=positive(evidence());p,a,key,auto=parts
    label=['A','B'][key['cases'][0]['order'].index('candidate')]
    field='voice' if kind=='voice' else label+'_content'
    a['answers'][p['cases'][0]['id']+':'+field]='判定できない'
    result=assess(parts)
    assert result['decision']=='hold' and result['teacher_listening_supported'] is False


@pytest.mark.parametrize('answer',['一部不明','意味が分からない'])
def test_voice_gain_cannot_hide_unintelligible_teacher(answer):
    parts=positive(evidence());p,a,key,auto=parts
    label=['A','B'][key['cases'][0]['order'].index('candidate')]
    a['answers'][p['cases'][0]['id']+':'+label+'_content']=answer
    result=assess(parts)
    assert result['decision']=='failed'
    assert 'teacher_not_intelligible' in result['cases'][0]['reasons']


@pytest.mark.parametrize('status',['hold','reject','no_metric_gain'])
def test_human_preference_does_not_overwrite_automatic_gate(status):
    parts=positive(evidence());parts[3]['cases'][0]['decision']=status
    result=assess(parts)
    assert result['decision']!='supported_for_teacher_preflight'
    assert result['teacher_listening_supported'] is False


@pytest.mark.parametrize('mutation',['wrong_page','audio','missing_auto','extra_auto','incomplete','count'])
def test_mixed_or_incomplete_evidence_is_rejected(mutation):
    parts=positive(evidence());p,a,key,auto=parts
    if mutation=='wrong_page':a['presentation_sha256']='f'*64
    elif mutation=='audio':auto['cases'][0]['artifacts']['candidate']['response.wav']='f'*64
    elif mutation=='missing_auto':auto['cases'].pop()
    elif mutation=='extra_auto':auto['cases'].append(deepcopy(auto['cases'][0]))
    elif mutation=='count':auto['expected_count']=3
    else:a['answers'].pop(next(iter(a['answers'])))
    with pytest.raises(ValueError):assess(parts)
