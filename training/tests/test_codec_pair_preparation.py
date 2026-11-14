import numpy as np
import pytest
import torch


def observations():
    logits=np.full((13,4),-12.)
    labels=[0,0,1,1,0,2,2,0,3,3,0,0,0]
    logits[np.arange(13),labels]=0
    return logits-np.log(np.exp(logits).sum(1,keepdims=True))


def prepare(source=None,target=None):
    from haru_training.codec_pair_preparation import prepare_pair
    audio=.1*np.sin(np.arange(6400)*.1).astype('float32')
    metadata=dict(id='line',group_id='line',split='train',source_speaker='other',target_speaker='target',
                  source_sha256='a'*64,target_sha256='b'*64,source_text='こんにちは',target_text='こんにちは',
                  codec_revision='c'*40)
    return prepare_pair(metadata,audio,audio*.8,torch.ones(2,64),
                        observations() if source is None else source,
                        observations() if target is None else target,
                        [1,2,3],blank=0,splits={'train':['line']})


def test_preparation_has_real_time_map_normalized_posteriors_and_no_waveform_l1():
    result=prepare()
    assert result.metadata['alignment']['status']=='accepted'
    assert result.metadata['quality']['source_token_error_rate']==0
    assert result.metadata['quality']['target_token_error_rate']==0
    assert result.spectrum.shape==(41,257)
    torch.testing.assert_close(result.posteriors.sum(-1),torch.ones(13))
    assert result.metadata['alignment']['source_seconds'][0]==0
    assert result.metadata['alignment']['source_seconds'][-1]==6400/24000


def test_wrong_spoken_tokens_are_not_accepted_by_forced_alignment_alone():
    bad=observations()[:,[0,3,2,1]]
    with pytest.raises(ValueError,match='content'):
        prepare(target=bad)


def test_rejected_alignment_never_creates_a_training_pair():
    bad=np.full((13,4),-np.log(4.))
    with pytest.raises(ValueError):prepare(source=bad)


@pytest.mark.parametrize('tokens',[[1,2,3],[1,4,2,4,3]])
def test_word_delimiters_are_not_counted_as_pronunciation_errors(tokens):
    from haru_training.codec_pair_preparation import prepare_pair
    data=observations()
    data=np.pad(data,((0,0),(0,1)),constant_values=-12.)
    data[[4,7]]= -12.;data[[4,7],4]=0.
    data=data-np.log(np.exp(data).sum(1,keepdims=True))
    item=prepare()
    result=prepare_pair(item.metadata,np.ones(6400,dtype='float32')*.1,
                         np.ones(6400,dtype='float32')*.2,item.latents,data,data,tokens,
                         blank=0,splits={'train':['line']},ignore_tokens=[4])
    assert result.metadata['quality']['source_token_error_rate']==0
