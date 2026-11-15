"""A separately bounded, three-pair diagnostic; preserves the original pilot cap."""
from copy import deepcopy
import math

from .codec_adapter_trial import validate_trial


def validate_diagnostic(spec, *, root):
    try:
        if (spec['kind']!='haru-codec-diagnostic' or type(spec['max_steps']) is not int
                or not 1<=spec['max_steps']<=300 or not math.isfinite(spec['max_seconds'])
                or not 0<spec['max_seconds']<=300 or spec['rank']!=8
                or spec['milestones']!=[0,30,100,300] or spec['max_steps']!=300
                or len(spec['training_ids'])!=3 or len(set(spec['training_ids']))!=3
                or any(not isinstance(v,str) or not v for v in spec['training_ids'])):
            raise ValueError('invalid bounded three-pair diagnostic')
    except (KeyError,TypeError) as error:
        raise ValueError('incomplete diagnostic specification') from error
    pilot=deepcopy(spec)
    pilot.update(max_steps=30,max_seconds=min(spec['max_seconds'],600))
    validate_trial(pilot,root=root)
    return spec


def select_training_pairs(pairs, ids):
    selected=tuple(p for p in pairs if p.metadata['id'] in ids)
    if (len(ids)!=3 or len(set(ids))!=3 or len(selected)!=3
            or any(p.metadata['split']!='train' for p in selected)):
        raise ValueError('diagnostic requires exactly three existing train groups')
    return selected


def run_stages(trainer, milestones, observe):
    """Report initial/terminal states even if the finite time limit stops early."""
    if (trainer.step!=0 or milestones[0]!=0 or milestones[-1]!=trainer.spec.max_steps
            or list(milestones)!=sorted(set(milestones))):
        raise ValueError('diagnostic stages must cover the initial and final bounds')
    history=[]
    observe(0,'initial')
    for boundary in milestones[1:]:
        history.extend(trainer.fit(until_step=boundary))
        observe(trainer.step,'time_limit' if trainer.step<boundary else 'milestone')
        if trainer.step<boundary:break
    return history


def validate_roundtrip_inputs(spec, *, root):
    from pathlib import Path
    from .codec_adapter_trial import file_hash
    root=Path(root).resolve()
    rows=spec['roundtrip']
    if (len(rows)!=6 or {(r['id'],r['side']) for r in rows}!=
            {(name,side) for name in spec['training_ids'] for side in ('source','target')}):
        raise ValueError('roundtrip inputs differ from the three training groups')
    for row in rows:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=row['sha256']:
            raise ValueError('roundtrip waveform path or hash differs')
    return rows


def run_roundtrip(path, destination):
    """Fresh CUDA artifact readback and codec controls; deliberately no training."""
    from pathlib import Path
    import json
    import numpy as np
    import soundfile as sf
    import torch
    from .codec_adapter_trial import file_hash, require_vast_cuda, capture, compare_traces
    from .codec_adapter_dataset import load_dataset
    from .codec_adapter_inference import load_adapter, codec_digest
    from .codec_adapter import get_adapter, set_adapter_enabled
    from .checkpoint import read_checkpoint
    from .data import read_wave
    path=Path(path);root=path.parent;out=Path(destination)
    spec=validate_diagnostic(json.loads(path.read_text(encoding='utf-8')),root=root)
    rows=validate_roundtrip_inputs(spec,root=root)
    data,pairs=load_dataset(root/spec['dataset']['path'])
    selected=select_training_pairs(pairs,spec['training_ids'])
    if [p.metadata['id'] for p in selected]!=[p.metadata['id'] for p in pairs[:3]]:
        raise ValueError('three diagnostic groups must remain the dataset prefix')
    if data['dataset_sha256']!=spec['dataset_sha256']:
        raise ValueError('full original dataset differs')
    membership=json.loads((root/spec['membership']['path']).read_text(encoding='utf-8'))
    originals={r['id']:r for r in membership['records']}
    for row in rows:
        if row['sha256']!=originals[row['id']][row['side']+'_sha256']:
            raise ValueError('roundtrip input is not the original prepared waveform')
    artifact=(root/spec['previous_adapter']['path']).resolve()
    if not artifact.is_relative_to(root.resolve()) or file_hash(artifact/'manifest.json')!=spec['previous_adapter']['sha256']:
        raise ValueError('previous adapter identity differs')
    require_vast_cuda()
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=load_adapter(artifact,device='cuda')
    bundle=read_checkpoint(artifact)
    assert model.dtype==torch.bfloat16 and model.codec.dtype==torch.float32
    assert codec_digest(model.codec)==bundle['codec_sha256']
    assert all(torch.equal(v.cpu(),bundle['weights'][k]) for k,v in get_adapter(model.codec).state_dict().items())
    if bundle['spec']['base_revision']!=spec['base_revision']:
        raise ValueError('fresh loader base revision differs from the diagnostic')
    user=read_wave(root/spec['cases'][0]['path'])[:,0]
    reference=read_wave(root/spec['reference']['path'])[:,0]
    set_adapter_enabled(model.codec,False)
    base,base_z,base_trace=capture(model,user,reference,spec,max_frames=4)
    set_adapter_enabled(model.codec,True)
    adapted,other_z,other_trace=capture(model,user,reference,spec,max_frames=4)
    assert compare_traces(base_trace,other_trace) and torch.equal(base_z,other_z)
    fresh=dict(actual_weights=True,device='cuda',model_dtype=str(model.dtype),codec_dtype=str(model.codec.dtype),
        adapter_tensors_exact=True,frozen_codec_exact=True,conversation_trace_exact=True,frames=4,
        waveform_changed=not np.array_equal(base['audio'],adapted['audio']),
        scope='Fresh v2 load; four-frame conversation-path check, not full conversational quality')
    (out/'fresh-cuda.json').write_text(json.dumps(fresh,indent=2),encoding='utf-8')
    set_adapter_enabled(model.codec,False)
    results=[]
    with torch.no_grad():
        for row in rows:
            audio=read_wave(root/row['path'])[:,0]
            mean=model.codec.encode(torch.from_numpy(audio))
            generator=torch.Generator(device='cuda').manual_seed(spec['seed'])
            for representation,z in [('mean',mean),('sample',model.codec.sample(mean,generator))]:
                wave=model.codec.decode(z)[0,:len(audio)].float().cpu().numpy()
                assert len(wave)==len(audio) and np.isfinite(wave).all()
                name=row['id']+'.'+row['side']+'.'+representation+'.wav'
                sf.write(out/name,wave,24000,subtype='FLOAT')
                results.append(dict(id=row['id'],side=row['side'],representation=representation,
                    path=name,sha256=file_hash(out/name),input_sha256=row['sha256'],samples=len(wave),
                    rms=float(np.sqrt(np.mean(wave.astype('float64')**2))),
                    clipped_fraction=float(np.mean(np.abs(wave)>=.999)),
                    voice_similarity_confirmed=False))
    report=dict(spec_sha256=file_hash(path),records=results,execution_complete=True,training_started=False,
        base_revision=spec['base_revision'],codec_sha256=codec_digest(model.codec),quality_gate='pending_independent_evaluation')
    (out/'roundtrip.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(roundtrip_waveforms=len(results),training_started=False,fresh_cuda_verified=True)),flush=True)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['roundtrip'])
    parser.add_argument('spec');parser.add_argument('destination')
    args=parser.parse_args()
    run_roundtrip(args.spec,args.destination)
