from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from test_codec_adapter_checkpoint import base


def comparison(tmp_path,train_sha256='1'*64):
    from haru_training.codec_adapter import attach_adapter
    from haru_training.codec_adapter_evaluation import save_decoder_comparison
    model=base();adapter=attach_adapter(model,rank=4)
    with torch.no_grad():adapter.up.bias.fill_(.02)
    model.requires_grad_(False)
    latent=torch.ones(1,2,64)
    training=tmp_path/'training-inputs.json'
    training.write_text(json.dumps({'records':[dict(group_id='train-group',split='train',
        target_speaker='target',source_sha256='2'*64,target_sha256=train_sha256)]}))
    metadata=dict(input_sha256='a'*64,base_model='test/tiny-haru',base_revision='c'*40,
        target_speaker='target',dataset_input_manifest_sha256=sha256(training.read_bytes()).hexdigest(),
        representation='raw_codec_sample',text='今日は晴れです。',config={'seed':7},system_prompt='test')
    save_decoder_comparison(tmp_path/'comparison',model,latent,metadata)
    return tmp_path/'comparison'


def test_comparison_binds_same_raw_latent_input_and_weights_and_drives_existing_evaluator(tmp_path):
    from haru_training.codec_adapter_evaluation import evaluate_comparison
    p=comparison(tmp_path)
    bundle=json.loads((p/'comparison.json').read_text(encoding='utf-8'))
    left=json.loads((p/'baseline/response.json').read_text(encoding='utf-8'))
    right=json.loads((p/'candidate/response.json').read_text(encoding='utf-8'))
    assert left['provenance']['raw_latent_sha256']==right['provenance']['raw_latent_sha256']
    assert left['provenance']['input_sha256']==right['provenance']['input_sha256']
    assert bundle['codec_sha256'] and bundle['adapter_sha256']
    assert (p/'raw-latents/tensors.safetensors').exists()
    assert not np.array_equal(sf.read(p/'baseline/response.wav')[0],sf.read(p/'candidate/response.wav')[0])
    report=evaluate_comparison(p,{'version':1,'expected_cases':['comparison'],'cases':[{
        'id':'comparison','baseline':{'directory':'baseline'},'candidate':{'directory':'candidate'}}]})
    assert report['decision']=='hold' and report['voice_similarity_confirmed'] is False
    (p/'candidate/response.wav').write_bytes(b'wrong')
    with pytest.raises(ValueError):evaluate_comparison(p,{'version':1,'expected_cases':[],'cases':[]})


def test_blind_html_has_embedded_audio_and_separate_key_without_scores_or_conditions(tmp_path):
    from haru_training.codec_adapter_evaluation import write_blind_html
    p=comparison(tmp_path)
    ref=tmp_path/'ref.wav';sf.write(ref,np.ones(3200)*.1,24000)
    digest=sha256(ref.read_bytes()).hexdigest()
    args=dict(reference_sha256=digest,seed=1,training_manifest=tmp_path/'training-inputs.json',
        reference_info=dict(group_id='reference-group',split='reference',speaker='target',wav_sha256=digest))
    write_blind_html(p,ref,**args)
    content=(p/'listen.html').read_text(encoding='utf-8')
    assert content.count('data:audio/wav;base64,')==3
    assert 'cosine' not in content and 'baseline' not in content and 'candidate' not in content
    assert set(json.loads((p/'answer-key.json').read_text())['order'])=={'baseline','candidate'}
    with pytest.raises(FileExistsError):write_blind_html(p,ref,**args)


@pytest.mark.parametrize('bad',['training_group','training_wave','wrong_speaker','wrong_role'])
def test_blind_reference_requires_an_untrained_target_recording(tmp_path,bad):
    from haru_training.codec_adapter_evaluation import write_blind_html
    ref=tmp_path/'ref.wav';sf.write(ref,np.ones(3200)*.1,24000)
    digest=sha256(ref.read_bytes()).hexdigest()
    p=comparison(tmp_path,train_sha256=digest if bad=='training_wave' else '1'*64)
    reference=dict(group_id='reference-group',split='reference',speaker='target',wav_sha256=digest)
    if bad=='training_group':reference['group_id']='train-group'
    if bad=='wrong_speaker':reference['speaker']='other'
    if bad=='wrong_role':reference['split']='train'
    with pytest.raises(ValueError):
        write_blind_html(p,ref,reference_sha256=digest,seed=1,reference_info=reference,
            training_manifest=tmp_path/'training-inputs.json')
    assert not (p/'listen.html').exists()


def test_collection_contains_three_questions_and_no_condition_or_score_leaks(tmp_path):
    from haru_training.codec_adapter_evaluation import write_blind_collection
    entries=[]
    for i in range(3):
        folder=tmp_path/str(i);folder.mkdir()
        entries.append(dict(id=f'dev-{i}',question='<質問>',directory=comparison(folder)))
    ref=tmp_path/'ref.wav';sf.write(ref,np.ones(3200)*.1,24000)
    digest=sha256(ref.read_bytes()).hexdigest()
    args=dict(reference_sha256=digest,seed=7,training_manifest=tmp_path/'0/training-inputs.json',
        reference_info=dict(group_id='reference-group',split='reference',speaker='target',wav_sha256=digest))
    page=write_blind_collection(tmp_path/'collection',entries,ref,**args)
    content=page.read_text(encoding='utf-8')
    assert content.count('<audio')==7 and content.count('&lt;質問&gt;')==3
    assert '<質問>' not in content and 'baseline' not in content and 'candidate' not in content
    assert 'cosine' not in content and '両方似ていない' in content
    assert (page.parent/'answer-key.json').is_file()
    key=json.loads((page.parent/'answer-key.json').read_text(encoding='utf-8'))
    assert len({tuple(row['order']) for row in key['cases']})>1
    with pytest.raises(FileExistsError):write_blind_collection(tmp_path/'collection',entries,ref,**args)
