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


@pytest.mark.parametrize('name,weight,strength',[('control',0.,0.),('uniform',.1,0.),('transition',.1,2.)])
def test_arm_runs_actual_updates_with_adapter_only_gradients(name,weight,strength):
    import torch
    from test_codec_adapter_checkpoint import base
    from test_codec_adapter_data import pair
    from test_codec_adapter_training import Content
    from haru_training.codec_content_comparison import create_arm_trainer
    model=base();item=pair();item.metadata['ctc_revision']='d'*40
    arm=dict(id=name,base_model='test/tiny-haru',base_revision='c'*40,target_speaker='target',
        max_steps=3,max_seconds=30.,learning_rate=.01,seed=7,rank=4,
        content_weight=weight,transition_strength=strength,
        ctc=dict(repo='test/ctc',revision='d'*40) if weight else None)
    encoder=None
    if weight:
        torch.manual_seed(9);encoder=Content().eval().requires_grad_(False)
        encoder.provenance=dict(repo='test/ctc',revision='d'*40)
    before={n:p.clone() for n,p in model.named_parameters()}
    trainer=create_arm_trainer(model,[item],arm,content=encoder)
    history=trainer.fit()
    changed=[n for n,p in model.named_parameters()
        if n.replace('codec.decoder.stem.base.','codec.decoder.stem.') in before
        and not torch.equal(before[n.replace('codec.decoder.stem.base.','codec.decoder.stem.')],p)]
    assert changed==[]
    assert len(history)==3 and trainer.step==3
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in trainer.optimizer.param_groups[0]['params'])
    from haru_training.codec_adapter import get_adapter
    assert get_adapter(model.codec).up.weight.count_nonzero()>0
    if encoder:
        assert all(p.grad is None for p in encoder.parameters())
        assert all(r['content']>0 for r in history)
    else:assert all(r['content']==0 for r in history)


def test_arm_refuses_a_different_or_trainable_content_encoder_before_attaching_adapter():
    from test_codec_adapter_checkpoint import base
    from test_codec_adapter_data import pair
    from test_codec_adapter_training import Content
    from haru_training.codec_content_comparison import create_arm_trainer
    item=pair();item.metadata['ctc_revision']='d'*40
    arm=dict(id='uniform',base_model='test/tiny-haru',base_revision='c'*40,target_speaker='target',
        max_steps=3,max_seconds=30.,learning_rate=.01,seed=7,rank=4,content_weight=.1,transition_strength=0.,
        ctc=dict(repo='test/ctc',revision='d'*40))
    for state in ('missing','wrong_revision','trainable'):
        model=base();encoder=Content().eval().requires_grad_(False)
        encoder.provenance=dict(repo='test/ctc',revision='e'*40 if state=='wrong_revision' else 'd'*40)
        if state=='missing':encoder=None
        if state=='trainable':encoder.requires_grad_(True)
        with pytest.raises(ValueError):create_arm_trainer(model,[item],arm,content=encoder)
        assert not getattr(model.codec,'_codec_adapter_attached',False)


def test_content_loader_requires_a_revision_before_any_download(monkeypatch):
    from haru_training.codec_content_comparison import load_pinned_content
    def forbidden(**kwargs):raise AssertionError('unpinned teacher attempted a download')
    monkeypatch.setattr('huggingface_hub.snapshot_download',forbidden)
    with pytest.raises(ValueError):load_pinned_content(dict(repo='test/ctc',revision='main',blank=0,files={}),device='cpu')


@pytest.mark.parametrize('name',['../outside','/outside',r'..\outside'])
def test_teacher_file_names_cannot_escape_the_snapshot_before_download(monkeypatch,name):
    from haru_training.codec_content_comparison import load_pinned_content
    _,_,ctc=inputs();ctc['files'][name]='e'*64
    def forbidden(**kwargs):raise AssertionError('unsafe file list attempted a download')
    monkeypatch.setattr('huggingface_hub.snapshot_download',forbidden)
    with pytest.raises(ValueError):load_pinned_content(ctc,device='cpu')


@pytest.mark.parametrize('symlink_snapshot',[False,True])
def test_verified_content_loader_keeps_actual_waveform_gradients_and_frozen_parameters(tmp_path,monkeypatch,symlink_snapshot):
    import torch
    from hashlib import sha256
    from transformers import Wav2Vec2Config,Wav2Vec2ForCTC
    from haru_training.codec_content_comparison import load_pinned_content
    directory=tmp_path/'snapshot' if symlink_snapshot else tmp_path
    directory.mkdir(exist_ok=True)
    files={}
    for name in ('model.safetensors','vocab.json'):
        p=tmp_path/name;p.write_bytes(name.encode());files[name]=sha256(p.read_bytes()).hexdigest()
        if symlink_snapshot:
            try:(directory/name).symlink_to(p)
            except OSError:pytest.skip('Creating Hugging Face style cache symlinks is unavailable')
    teacher=dict(repo='test/ctc',revision='d'*40,blank=0,files=files)
    config=Wav2Vec2Config(hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,
        conv_dim=[4]*7,conv_kernel=[10,3,3,3,3,2,2],conv_stride=[5,2,2,2,2,2,2],
        num_conv_pos_embeddings=8,num_conv_pos_embedding_groups=2,feat_extract_norm='layer',
        vocab_size=5,pad_token_id=0,mask_time_prob=0.)
    calls=[]
    def snapshot(**kwargs):
        assert kwargs['revision']=='d'*40 and set(kwargs['allow_patterns'])==set(files)
        return str(directory)
    def load(path,**kwargs):calls.append(path);return Wav2Vec2ForCTC(config)
    monkeypatch.setattr('huggingface_hub.snapshot_download',snapshot)
    monkeypatch.setattr(Wav2Vec2ForCTC,'from_pretrained',load)
    content=load_pinned_content(teacher,device='cpu')
    wave=torch.randn(6400,requires_grad=True)
    content(wave).square().mean().backward()
    assert wave.grad.abs().sum()>0 and all(p.grad is None for p in content.parameters())
    assert content.provenance['revision']=='d'*40
    (tmp_path/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError):load_pinned_content(teacher,device='cpu')
    assert len(calls)==1


def execution_inputs(tmp_path,monkeypatch):
    import json
    import torch
    from hashlib import sha256
    from test_codec_adapter_diagnostic import diagnostic
    from test_codec_adapter_checkpoint import base
    from test_codec_adapter_data import pair
    from haru_training.codec_adapter import attach_adapter
    from haru_training.codec_adapter_training import _data_digest
    from haru_training.codec_adapter_inference import export_adapter
    from haru_training.codec_adapter_evaluation import save_decoder_comparison
    from haru_training.listening_review import make_presentation
    parent=diagnostic(tmp_path);parent.update(base_revision='c'*40,target_speaker='target')
    parent['reference']['speaker']='target'
    _,_,ctc=inputs()
    membership=tmp_path/parent['membership']['path']
    membership.write_text(json.dumps(dict(ctc=ctc)),encoding='utf-8')
    parent['membership']['sha256']=sha256(membership.read_bytes()).hexdigest()
    original=tmp_path/'diagnostic.json';original.write_text(json.dumps(parent),encoding='utf-8')
    parent.update(kind='haru-codec-rank-correction',rank=32,
        parent=dict(path=original.name,sha256=sha256(original.read_bytes()).hexdigest()))
    parent_path=tmp_path/'rank32.json';parent_path.write_text(json.dumps(parent),encoding='utf-8')
    pairs=[]
    for name in parent['training_ids']:
        p=pair();p.metadata.update(id=name,ctc_revision=ctc['revision']);pairs.append(p)
    monkeypatch.setattr('haru_training.codec_adapter_dataset.load_dataset',
        lambda path:(dict(dataset_sha256=parent['dataset_sha256']),pairs))
    model=base();attach_adapter(model,rank=32);model.requires_grad_(False)
    artifact=export_adapter(model,tmp_path/'adapter',base_model=parent['base_model'],
        base_revision=parent['base_revision'],target_speaker='target',dataset_sha256=_data_digest(pairs))
    public=[];keys=[];answers={}
    for case in parent['cases']:
        dest=tmp_path/'comparisons'/case['id']
        save_decoder_comparison(dest,model,torch.ones(1,2,64),dict(representation='raw_codec_sample',
            input_sha256=case['sha256'],base_revision=parent['base_revision'],target_speaker='target',
            dataset_input_manifest_sha256=parent['membership']['sha256'],text='test',
            system_prompt=parent['system_prompt'],config={'seed':7}))
        hashes={s:sha256((dest/s/'response.wav').read_bytes()).hexdigest() for s in ('baseline','candidate')}
        public.append(dict(id=case['id'],audio_sha256={'A':hashes['baseline'],'B':hashes['candidate']}))
        keys.append(dict(id=case['id'],order=['baseline','candidate'],audio_sha256=hashes))
        answers.update({case['id']+':A_content':'聞き取れる',case['id']+':B_content':'聞き取れる',
            case['id']+':voice':'B',case['id']+':voice_match':'両方似ていない'})
    presentation=make_presentation(public,reference_sha256=parent['reference']['sha256'],include_voice_match=True)
    values=dict(presentation=presentation,responses=dict(version=1,presentation_sha256=presentation['presentation_sha256'],answers=answers),
        answer_key=dict(presentation_sha256=presentation['presentation_sha256'],cases=keys))
    paths={}
    for key,value in values.items():
        p=tmp_path/(key+'.json');p.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');paths[key+'_path']=p
    return parent_path,dict(paths,comparisons_root=tmp_path/'comparisons',adapter_path=artifact,arm_id='transition')


def test_execution_binds_actual_review_audio_adapter_and_prepared_teacher_before_gpu(tmp_path,monkeypatch):
    from haru_training.codec_content_comparison import prepare_execution
    parent,paths=execution_inputs(tmp_path,monkeypatch)
    result=prepare_execution(parent,**paths)
    assert result['arm']['id']=='transition' and result['arm']['rank']==32
    assert result['arm']['ctc']['revision']=='c'*40
    assert result['evidence']['responses_sha256']
    assert result['review']['candidate_target_match_count']==0
    assert result['review']['relative_improvement_count']==3
    assert not result['review']['voice_similarity_confirmed']


@pytest.mark.parametrize('mutation',['missing_review','negative_review','wrong_audio','wrong_reference','wrong_adapter'])
def test_execution_refuses_unbound_or_unsupportive_evidence_before_cuda(tmp_path,monkeypatch,mutation):
    import json
    from haru_training.codec_adapter_diagnostic import run_small_training
    parent,paths=execution_inputs(tmp_path,monkeypatch)
    def forbidden():raise AssertionError('CUDA reached before evidence validation')
    monkeypatch.setattr('haru_training.codec_adapter_trial.require_vast_cuda',forbidden)
    if mutation=='missing_review':paths['responses_path']=tmp_path/'absent.json'
    if mutation=='negative_review':
        p=paths['responses_path'];value=json.loads(p.read_text(encoding='utf-8'))
        value['answers']['dev-0:voice']='A';p.write_text(json.dumps(value),encoding='utf-8')
    if mutation=='wrong_audio':(paths['comparisons_root']/'dev-0/candidate/response.wav').write_bytes(b'other')
    if mutation=='wrong_reference':
        p=paths['presentation_path'];value=json.loads(p.read_text(encoding='utf-8'))
        value['reference_sha256']='f'*64;p.write_text(json.dumps(value),encoding='utf-8')
    if mutation=='wrong_adapter':paths['adapter_path']=tmp_path/'missing-adapter'
    with pytest.raises((ValueError,OSError)):
        run_small_training(parent,tmp_path/'output',gate_path=tmp_path/'unused',
            roundtrip_report=tmp_path/'unused',comparison=paths)
    assert not (tmp_path/'output').exists()


def test_supported_execution_reaches_provider_gate_without_loading_a_model(tmp_path,monkeypatch):
    import json
    from hashlib import sha256
    from haru_training.codec_adapter_diagnostic import run_small_training
    parent_path,paths=execution_inputs(tmp_path,monkeypatch)
    parent=json.loads(parent_path.read_text(encoding='utf-8'))
    report=tmp_path/'roundtrip.json';report.write_text('{}',encoding='utf-8')
    gate=tmp_path/'gate.json'
    gate.write_text(json.dumps(dict(small_training_gate_passed=True,
        spec_sha256=parent['parent']['sha256'],roundtrip_report_sha256=sha256(report.read_bytes()).hexdigest(),
        cases=[dict(id=name,side=side,status='supported_for_small_training',reasons=[])
            for name in parent['training_ids'] for side in ('source','target')])),encoding='utf-8')
    def provider():raise RuntimeError('verified boundary before paid model loading')
    monkeypatch.setattr('haru_training.codec_adapter_trial.require_vast_cuda',provider)
    with pytest.raises(RuntimeError,match='verified boundary'):
        run_small_training(parent_path,tmp_path/'out',gate_path=gate,roundtrip_report=report,comparison=paths)
    assert not (tmp_path/'out').exists()
