"""Read-only preflight for paired codec data; never loads or trains a model."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from .codec_adapter_data import load_pair
from .codec_adapter_training import _data_digest


def _digest(path):
    value=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def _json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_dataset(path):
    """Verify frozen membership, failed controls, independent timing, and tensors."""
    path=Path(path)
    try:
        data=_json(path)
        if (data['kind']!='haru-codec-adapter-dataset' or data['version']!=1
                or data['preparation_complete'] is not True or data['controls_passed'] is not True):
            raise ValueError('dataset preparation has not passed')
        source=path.parent/data['dataset_input_manifest']
        if _digest(source)!=data['dataset_input_manifest_sha256']:
            raise ValueError('input membership manifest differs')
        inputs=_json(source);records=data['records']
        if (not records or data['splits']!=inputs['splits']
                or data['accepted_count']!=len(records)
                or data['candidate_count']!=len(inputs['records'])
                or data['candidate_count']!=data['accepted_count']+data['held_count']
                or len(records)<inputs['target_accepted_pairs']):
            raise ValueError('membership, split, or retained denominator differs')
        reports={}
        for name in ('preparation_report','timing_report'):
            entry=data[name];file=path.parent/entry['path']
            if _digest(file)!=entry['sha256']:raise ValueError('QC report hash differs')
            reports[name]=_json(file)
        ids=[r['id'] for r in inputs['records']]
        accepted=[r['id'] for r in records]
        prepared=reports['preparation_report']
        if (len(set(ids))!=len(ids) or len(set(accepted))!=len(accepted)
                or prepared['expected_ids']!=ids or [r['id'] for r in prepared['records']]!=ids
                or [r['id'] for r in prepared['records'] if r['status']=='accepted']!=accepted
                or any(r['status'] not in ('accepted','held') for r in prepared['records'])):
            raise ValueError('prepared cases do not preserve the frozen cohort')
        expected_controls={'mislabeled_other_utterance','reversed_observations','silence_with_speech_posteriors'}
        for row in prepared['records']:
            controls=row['negative_controls']
            if (len(controls)!=3 or {c['kind'] for c in controls}!=expected_controls
                    or any(c['status']!='rejected' for c in controls)):
                raise ValueError('a negative control is missing, unknown, or incorrectly accepted')
        timing=reports['timing_report']['records']
        if [r['id'] for r in timing]!=accepted or any(r['status']!='supported'
                or not 0<=r['median_seconds']<=3200/24000
                or not r['median_seconds']<=r['p95_seconds']<=6400/24000 for r in timing):
            raise ValueError('independent timing agreement is absent or insufficient')
        membership={r['id']:r for r in inputs['records']};pairs=[]
        for row in records:
            relative=Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('pair path must stay within the dataset')
            folder=path.parent/relative
            if _digest(folder/'manifest.json')!=row['pair_manifest_sha256']:
                raise ValueError('pair manifest differs')
            item=load_pair(folder,splits=data['splits'],codec_revision=data['base_revision'],
                           expected_sources=(row['source_sha256'],row['target_sha256']))
            original=membership[row['id']]
            if (row['split']!='train' or item.metadata['id']!=row['id'] or row['group_id']!=row['id']
                    or item.metadata['target_speaker']!=data['target_speaker']
                    or item.metadata['dataset_input_manifest_sha256']!=data['dataset_input_manifest_sha256']
                    or any(row[k]!=original[k] for k in ('group_id','source_sha256','target_sha256'))):
                raise ValueError('pair identity differs from the frozen input')
            pairs.append(item)
        if _data_digest(pairs)!=data['dataset_sha256']:
            raise ValueError('dataset tensor fingerprint differs')
        return data,pairs
    except (KeyError,TypeError,OSError,ValueError) as error:
        raise ValueError('invalid codec dataset: '+str(error)) from error


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    args=parser.parse_args(argv)
    spec,pairs=load_dataset(args.dataset)
    print(json.dumps(dict(status='ready_for_real_weight_preflight',accepted_count=len(pairs),
        candidate_count=spec['candidate_count'],held_count=spec['held_count'],
        dataset_sha256=spec['dataset_sha256'],controls_verified=True,timing_verified=True,
        model_loaded=False,training_started=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
