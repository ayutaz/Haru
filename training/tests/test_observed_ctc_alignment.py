import numpy as np
import pytest


def observations(replacement=None):
    labels=np.zeros(100,dtype=int)
    labels[np.arange(20)*4+8]=np.arange(1,21)
    if replacement is not None:labels[replacement*4+8]=25
    probs=np.full((100,32),.001/31)
    probs[np.arange(100),labels]=.999
    return np.log(probs)


def test_only_observed_matches_become_anchors_not_forced_mismatched_characters():
    from haru_training.observed_ctc_alignment import observed_anchors
    source,target,quality=observed_anchors(observations(),observations(10),
        source_seconds=2.1,target_seconds=2.1,blank=0)
    assert quality['matched_tokens']==19
    assert quality['source_coverage']==pytest.approx(.95)
    assert len(source)==21 and np.array_equal(source,target)
    assert np.all(np.diff(source)>0)


def test_known_scaled_clock_is_recovered_and_content_corruption_is_held():
    from haru_training.observed_ctc_alignment import observed_anchors
    src=observations()
    target=np.repeat(src,2,axis=0)
    left,right,quality=observed_anchors(src,target,source_seconds=2.1,target_seconds=4.2,blank=0)
    np.testing.assert_allclose(right[1:-1],2*left[1:-1]-.0125,atol=.02)
    bad=np.roll(src,15,axis=1)
    with pytest.raises(ValueError):observed_anchors(src,bad,source_seconds=2.1,target_seconds=2.1,blank=0)
    weak=np.full_like(src,-np.log(32))
    with pytest.raises(ValueError):observed_anchors(src,weak,source_seconds=2.1,target_seconds=2.1,blank=0)


def test_no_long_unobserved_region_is_filled_by_interpolation():
    from haru_training.observed_ctc_alignment import observed_anchors
    src=observations()
    src=np.vstack([src[:48],np.tile(np.log(np.r_[.999,np.full(31,.001/31)]),(70,1)),src[48:]])
    with pytest.raises(ValueError,match='gap'):
        observed_anchors(src,src,source_seconds=3.5,target_seconds=3.5,blank=0)


def test_pair_preparation_keeps_content_gate_and_records_observed_anchor_scope():
    import torch
    from haru_training.codec_pair_preparation import prepare_observed_pair
    audio=(.1*np.sin(np.arange(48120)*.04)).astype('float32')
    meta=dict(id='line',group_id='line',split='train',source_speaker='source',target_speaker='target',
        source_sha256='a'*64,target_sha256='b'*64,source_text='今日は晴れです',target_text='今日は晴れです',
        codec_revision='c'*40,ctc_revision='d'*40)
    args=(meta,audio,audio*.8,torch.ones(16,64),observations(),observations(10),list(range(1,21)))
    item=prepare_observed_pair(*args,blank=0,splits={'train':['line']})
    assert item.metadata['alignment']['method']=='observed-ctc-anchors-v1'
    assert item.metadata['quality']['matched_tokens']==19
    assert item.metadata['quality']['target_token_error_rate']==.05
    assert item.metadata['quality']['human_quality_confirmed'] is False
    torch.testing.assert_close(item.posteriors.sum(-1),torch.ones(100))
    bad=list(args);bad[5]=np.roll(observations(),15,axis=1)
    with pytest.raises(ValueError,match='content'):prepare_observed_pair(*bad,blank=0,splits={'train':['line']})
