from copy import deepcopy

import numpy as np
import pytest
import torch


def metadata():
    return dict(kind='haru-codec-pair',version=1,id='sentence-a',group_id='sentence-a',split='train',
                source_speaker='source',target_speaker='target',source_sha256='a'*64,target_sha256='b'*64,
                source_text='今日は晴れです。',target_text='今日は晴れです',
                codec_revision='c'*40,representation='raw_codec_mean',sample_rate=24000,
                valid_samples=6400,warmup_samples=3200,spectral_hop=160,ctc_frames=20,
                alignment=dict(status='accepted',method='ctc-token-centers-v1',
                               source_seconds=[0.,.1,.2,6400/24000],target_seconds=[0.,.2,.3,.4]))


def pair(meta=None):
    from haru_training.codec_adapter_data import make_pair
    return make_pair(meta or metadata(),torch.zeros(2,64),torch.ones(41,257),
                     torch.full((20,5),.2),splits={'train':['sentence-a'],'validation':['sentence-b'],
                                                  'test':['sentence-c'],'reference':['sentence-d']})


def test_pair_is_cross_speaker_raw_latents_with_aligned_feature_targets():
    result=pair()
    assert result.latents.shape==(2,64)
    assert result.spectrum.shape==(41,257)
    assert result.posteriors.shape==(20,5)
    assert result.metadata['valid_samples']==6400


@pytest.mark.parametrize('changes',[
    {'split':'test'},{'group_id':'sentence-c'},{'source_speaker':'target'},
    {'source_sha256':'b'*64},{'source_text':'別の文章です'},
    {'representation':'normalized_haru'},{'valid_samples':6401},{'warmup_samples':6400},
    {'alignment':{'status':'unknown','method':'ctc-token-centers-v1'}},
    {'codec_revision':'main'},
])
def test_bad_or_leaking_teacher_pairs_are_rejected(changes):
    meta=metadata();meta.update(changes)
    with pytest.raises(ValueError):pair(meta)


def test_pair_tensor_shapes_finiteness_and_probability_contract():
    from haru_training.codec_adapter_data import make_pair
    args=[metadata(),torch.zeros(2,64),torch.ones(41,257),torch.full((20,5),.2)]
    split={'train':['sentence-a']}
    for index,value in [(1,torch.zeros(2,63)),(2,torch.ones(40,257)),
                        (3,torch.ones(20,5)),(1,torch.full((2,64),float('nan')))]:
        changed=list(args);changed[index]=value
        with pytest.raises(ValueError):make_pair(*changed,splits=split)


def test_feature_cache_roundtrip_binds_input_split_and_base(tmp_path):
    from haru_training.codec_adapter_data import save_pair,load_pair
    item=pair();path=tmp_path/'pair'
    save_pair(path,item)
    loaded=load_pair(path,splits={'train':['sentence-a']},codec_revision='c'*40)
    torch.testing.assert_close(loaded.latents,item.latents,rtol=0,atol=0)
    torch.testing.assert_close(loaded.posteriors,item.posteriors,rtol=0,atol=0)
    with pytest.raises(ValueError):load_pair(path,splits={'train':['sentence-a']},codec_revision='d'*40)
    with pytest.raises(ValueError):load_pair(path,splits={'test':['sentence-a']},codec_revision='c'*40)
    with pytest.raises(FileExistsError):save_pair(path,item)
    (path/'tensors.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError):load_pair(path,splits={'train':['sentence-a']},codec_revision='c'*40)


def test_ctc_alignment_handles_repeated_symbols_and_rejects_impossible_paths():
    from haru_training.codec_adapter_data import align_ctc
    labels=[0,1,0,1,0,2,0]
    logits=np.full((7,3),-12.);logits[np.arange(7),labels]=0
    logits=logits-np.log(np.exp(logits).sum(axis=1,keepdims=True))
    spans=align_ctc(logits,[1,1,2],blank=0)
    assert [(r['token'],r['start'],r['end']) for r in spans]==[(1,1,2),(1,3,4),(2,5,6)]
    assert all(r['confidence']>.99 for r in spans)
    with pytest.raises(ValueError):align_ctc(logits[:2],[1,1,2],blank=0)
    with pytest.raises(ValueError):align_ctc(logits,[0],blank=0)
    with pytest.raises(ValueError):align_ctc(np.full((7,3),np.nan),[1],blank=0)


def test_time_mapping_is_monotonic_and_does_not_warp_waveform_samples():
    from haru_training.codec_adapter_data import warp_features
    values=np.stack([np.arange(5),2*np.arange(5)],axis=1).astype('float32')
    actual=warp_features(values,np.arange(5)*.1,np.array([0.,.1,.2]),
                         [0.,.2],[0.,.4])
    np.testing.assert_allclose(actual,[[0,0],[2,4],[4,8]])
    with pytest.raises(ValueError):warp_features(values,np.arange(5)*.1,[0.,.1],[0.,0.],[0.,.4])
    with pytest.raises(ValueError):warp_features(values,np.arange(5)*.1,[0.,.1],[0.,.2],[.4,0.])


def test_cache_cannot_be_reused_after_input_recording_changes(tmp_path):
    from haru_training.codec_adapter_data import save_pair,load_pair
    path=tmp_path/'pair';item=pair();save_pair(path,item)
    load_pair(path,splits={'train':['sentence-a']},codec_revision='c'*40,
              expected_sources=('a'*64,'b'*64))
    with pytest.raises(ValueError,match='source'):
        load_pair(path,splits={'train':['sentence-a']},codec_revision='c'*40,
                  expected_sources=('d'*64,'b'*64))
