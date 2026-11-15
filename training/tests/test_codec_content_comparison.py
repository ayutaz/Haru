"""Comparison preparation is inert and cannot invent a passed listening gate."""
from copy import deepcopy

import pytest


def inputs():
    parent=dict(kind='haru-codec-rank-correction',base_model='higurashi-ai/Haru-4B',
        base_revision='a'*40,dataset_sha256='b'*64,target_speaker='target',rank=32,seed=7,
        max_steps=300,max_seconds=300,learning_rate=.001,training_ids=['train-a','train-b','train-c'],
        cases=[dict(id=f'dev-{i}',sha256=str(i+1)*64,path=f'{i}.wav',text='question') for i in range(3)],
        reference={'sha256':'d'*64},system_prompt='fixed')
    review=dict(voice_similarity_confirmed=False,cases=[dict(id=f'dev-{i}',
        preferred_condition='candidate' if i<2 else None,candidate_voice_match=False,
        content_by_condition={'baseline':'聞き取れる','candidate':'聞き取れる'}) for i in range(3)])
    ctc=dict(repo='test/ctc',revision='c'*40,blank=85,files={'model.safetensors':'e'*64,'vocab.json':'f'*64})
    return parent,review,ctc


def test_prepare_three_equal_budget_arms_without_loading_or_training_models():
    from haru_training.codec_content_comparison import prepare_comparison
    parent,review,ctc=inputs();before=deepcopy((parent,review,ctc))
    plan=prepare_comparison(parent,review,ctc)
    assert (parent,review,ctc)==before
    assert plan['training_started'] is False and plan['voice_similarity_confirmed'] is False
    assert [(a['id'],a['content_weight'],a['transition_strength']) for a in plan['arms']]==[
        ('control',0.,0.),('uniform',.1,0.),('transition',.1,2.)]
    for key in ('base_revision','dataset_sha256','training_ids','rank','seed','max_steps','max_seconds','learning_rate','cases','reference','system_prompt'):
        assert all(a[key]==parent[key] for a in plan['arms'])
    assert all(a['start_from']=='pinned_base' for a in plan['arms'])
    assert plan['requires_equal_completed_updates'] is True


@pytest.mark.parametrize('mutation',['no_review','claimed_pass','only_one_gain','baseline_better','unintelligible','unknown','missing','duplicate'])
def test_missing_or_unsupportive_human_evidence_does_not_open_the_comparison(mutation):
    from haru_training.codec_content_comparison import prepare_comparison
    parent,review,ctc=inputs()
    if mutation=='no_review':review=None
    if mutation=='claimed_pass':review=dict(voice_similarity_confirmed=True,relative_improvement_count=3)
    if mutation=='only_one_gain':review['cases'][1]['preferred_condition']=None
    if mutation=='baseline_better':review['cases'][2]['preferred_condition']='baseline'
    if mutation=='unintelligible':review['cases'][0]['content_by_condition']['candidate']='意味が分からない'
    if mutation=='unknown':review['cases'][0]['content_by_condition']['candidate']='判定できない'
    if mutation=='missing':review['cases'].pop()
    if mutation=='duplicate':review['cases'][2]['id']='dev-0'
    with pytest.raises(ValueError):prepare_comparison(parent,review,ctc)


@pytest.mark.parametrize('field,value',[('rank',64),('max_steps',301),('max_seconds',301),('seed',True)])
def test_preparation_refuses_unregistered_expansion_of_the_parent_budget(field,value):
    from haru_training.codec_content_comparison import prepare_comparison
    parent,review,ctc=inputs();parent[field]=value
    with pytest.raises(ValueError):prepare_comparison(parent,review,ctc)


def test_same_ctc_identity_is_pinned_for_both_content_arms():
    from haru_training.codec_content_comparison import prepare_comparison
    parent,review,ctc=inputs();plan=prepare_comparison(parent,review,ctc)
    assert plan['arms'][0]['ctc'] is None
    assert plan['arms'][1]['ctc']==plan['arms'][2]['ctc']==ctc
    ctc['revision']='main'
    with pytest.raises(ValueError):prepare_comparison(parent,review,ctc)
