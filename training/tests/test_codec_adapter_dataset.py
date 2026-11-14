from hashlib import sha256
import json

import pytest

from haru_training.codec_adapter_data import save_pair
from haru_training.codec_adapter_training import _data_digest
from test_codec_adapter_data import metadata,pair


def digest(path):return sha256(path.read_bytes()).hexdigest()


def dataset(tmp_path):
    inputs=tmp_path/'inputs.json'
    splits={'train':['sentence-a'],'validation':['sentence-b'],'test':['sentence-c'],'reference':['sentence-d']}
    row=dict(id='sentence-a',group_id='sentence-a',split='train',source_sha256='a'*64,
             target_sha256='b'*64,target_speaker='target')
    inputs.write_text(json.dumps(dict(records=[row],splits=splits,target_accepted_pairs=1)))
    meta=metadata();meta['dataset_input_manifest_sha256']=digest(inputs)
    item=pair(meta);save_pair(tmp_path/'pairs/sentence-a',item)
    entry=dict(row,path='pairs/sentence-a',pair_manifest_sha256=digest(tmp_path/'pairs/sentence-a/manifest.json'))
    controls=[dict(kind=k,status='rejected') for k in
        ('mislabeled_other_utterance','reversed_observations','silence_with_speech_posteriors')]
    preparation=tmp_path/'preparation.json'
    preparation.write_text(json.dumps(dict(expected_ids=['sentence-a'],records=[
        dict(id='sentence-a',status='accepted',negative_controls=controls)])))
    timing=tmp_path/'timing.json';timing.write_text(json.dumps(dict(records=[dict(
        id='sentence-a',status='supported',median_seconds=.01,p95_seconds=.03)])))
    data=dict(kind='haru-codec-adapter-dataset',version=1,base_model='test/tiny-haru',base_revision='c'*40,
        target_speaker='target',splits=splits,records=[entry],accepted_count=1,candidate_count=1,held_count=0,
        dataset_sha256=_data_digest([item]),dataset_input_manifest='inputs.json',dataset_input_manifest_sha256=digest(inputs),
        preparation_complete=True,controls_passed=True,
        preparation_report=dict(path='preparation.json',sha256=digest(preparation)),
        timing_report=dict(path='timing.json',sha256=digest(timing)))
    path=tmp_path/'dataset.json';path.write_text(json.dumps(data))
    return path


def test_load_and_cli_preflight_validate_all_evidence_without_loading_a_model(tmp_path,capsys):
    from haru_training.codec_adapter_dataset import load_dataset,main
    path=dataset(tmp_path)
    spec,pairs=load_dataset(path)
    assert len(pairs)==1 and spec['dataset_sha256']==_data_digest(pairs)
    assert main([str(path)])==0
    report=json.loads(capsys.readouterr().out)
    assert report['accepted_count']==1 and report['model_loaded'] is False


@pytest.mark.parametrize('change',['input','split','tensors','controls','timing','fingerprint'])
def test_boolean_ready_flags_do_not_hide_invalid_data_or_reports(tmp_path,change):
    from haru_training.codec_adapter_dataset import load_dataset
    path=dataset(tmp_path);data=json.loads(path.read_text())
    if change=='input':(tmp_path/'inputs.json').write_text('{}')
    elif change=='split':data['splits']['train']=[]
    elif change=='tensors':(tmp_path/'pairs/sentence-a/tensors.safetensors').write_bytes(b'broken')
    elif change in ('controls','timing'):
        report=tmp_path/('preparation.json' if change=='controls' else 'timing.json')
        content=json.loads(report.read_text())
        if change=='controls':content['records'][0]['negative_controls'][0]['status']='unexpected_accept'
        else:content['records'][0]['p95_seconds']=.5
        report.write_text(json.dumps(content))
        data['preparation_report' if change=='controls' else 'timing_report']['sha256']=digest(report)
    else:data['dataset_sha256']='f'*64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_dataset(path)
