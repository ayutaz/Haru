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


def validate_correction(spec, *, root):
    """The correction may change only rank, not its frozen data or finite bounds."""
    from pathlib import Path
    import json
    from .codec_adapter_trial import file_hash
    try:
        parent=spec['parent'];root=Path(root).resolve();path=(root/parent['path']).resolve()
        if not path.is_relative_to(root) or file_hash(path)!=parent['sha256']:
            raise ValueError('correction parent identity differs')
        original=validate_diagnostic(json.loads(path.read_text(encoding='utf-8')),root=path.parent)
        expected=deepcopy(original)
        expected.update(kind='haru-codec-rank-correction',rank=32,parent=parent)
        if spec!=expected:
            raise ValueError('rank correction must preserve every other diagnostic condition')
    except (KeyError,TypeError,OSError) as error:
        raise ValueError('incomplete correction parent') from error
    return spec


def validate_full_data_correction(spec, *, root):
    """Restore the entire original 15-pair cohort, keeping the rank32 parent fixed."""
    from pathlib import Path
    import json
    from .codec_adapter_trial import file_hash
    from .codec_adapter_dataset import load_dataset
    try:
        root=Path(root).resolve();parent=spec['parent'];path=(root/parent['path']).resolve()
        if not path.is_relative_to(root) or path.parent!=root or file_hash(path)!=parent['sha256']:
            raise ValueError('full-data correction parent identity differs')
        original=validate_correction(json.loads(path.read_text(encoding='utf-8')),root=root)
        data,pairs=load_dataset(root/original['dataset']['path'])
        if (len(pairs)!=15 or any(p.metadata['split']!='train' for p in pairs)
                or data['dataset_sha256']!=original['dataset_sha256']):
            raise ValueError('full-data correction requires the entire original fifteen train pairs')
        expected=deepcopy(original)
        expected.update(kind='haru-codec-full-data-correction',training_ids=[p.metadata['id'] for p in pairs],
            parent=parent,roundtrip_parent_sha256=original['parent']['sha256'])
        if spec!=expected:
            raise ValueError('full-data correction may change only the training cohort')
    except (KeyError,TypeError,OSError) as error:
        raise ValueError('incomplete full-data correction') from error
    return spec


def run_stages(trainer, milestones, observe):
    """Report initial/terminal states even if the finite time limit stops early."""
    if (trainer.step!=0 or milestones[0]!=0 or milestones[-1]!=trainer.spec.max_steps
            or list(milestones)!=sorted(set(milestones))):
        raise ValueError('diagnostic stages must cover the initial and final bounds')
    history=[]
    observe(0,'initial')
    for boundary in milestones[1:]:
        previous_step=trainer.step
        history.extend(trainer.fit(until_step=boundary))
        if trainer.step!=previous_step:
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


def validate_training_gate(spec, gate, *, spec_sha256, report_sha256):
    try:
        rows=gate['cases']
        if (gate['small_training_gate_passed'] is not True or gate['spec_sha256']!=spec_sha256
                or gate['roundtrip_report_sha256']!=report_sha256 or len(rows)!=6
                or {(r['id'],r['side']) for r in rows}!=
                    {(name,side) for name in spec['training_ids'] for side in ('source','target')}
                or any(r['status']!='supported_for_small_training' or r['reasons'] for r in rows)):
            raise ValueError('independent roundtrip screening does not support small training')
    except (KeyError,TypeError) as error:
        raise ValueError('incomplete independent roundtrip screening') from error
    return gate


def run_small_training(path, destination, *, gate_path, roundtrip_report):
    """Train only the registered groups, then decode both input distributions."""
    from pathlib import Path
    import json
    import numpy as np
    import torch
    import soundfile as sf
    from haru.models.haru import HaruModel
    from haru.models.loaders import resolve_checkpoint
    from safetensors import safe_open
    from .codec_adapter_trial import require_vast_cuda,file_hash
    from .codec_adapter_dataset import load_dataset
    from .codec_adapter_training import CodecTrainer,TrainSpec,paired_loss
    from .codec_adapter import get_adapter,decode_training
    from .codec_adapter_inference import codec_digest,export_adapter
    from .checkpoint import read_checkpoint
    from .codec_adapter_evaluation import save_decoder_comparison
    path=Path(path);root=path.parent;out=Path(destination)
    spec=json.loads(path.read_text(encoding='utf-8'))
    full=spec.get('kind')=='haru-codec-full-data-correction'
    correction=spec.get('kind')=='haru-codec-rank-correction'
    spec=(validate_full_data_correction(spec,root=root) if full else
          validate_correction(spec,root=root) if correction else validate_diagnostic(spec,root=root))
    gate=json.loads(Path(gate_path).read_text(encoding='utf-8'))
    gate_spec=dict(spec,training_ids=spec['training_ids'][:3]) if full else spec
    gate_hash=spec['roundtrip_parent_sha256'] if full else spec['parent']['sha256'] if correction else file_hash(path)
    validate_training_gate(gate_spec,gate,spec_sha256=gate_hash,
        report_sha256=file_hash(roundtrip_report))
    data,pairs=load_dataset(root/spec['dataset']['path'])
    if data['dataset_sha256']!=spec['dataset_sha256']:
        raise ValueError('full original dataset differs from the diagnostic')
    selected=tuple(pairs) if full else select_training_pairs(pairs,spec['training_ids'])
    if [p.metadata['id'] for p in selected]!=(spec['training_ids'] if full else [p.metadata['id'] for p in pairs[:3]]):
        raise ValueError('diagnostic membership differs from the original fixed prefix')
    require_vast_cuda()
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    directory=resolve_checkpoint(spec['base_model'],spec['base_revision'])
    if file_hash(directory/'model.safetensors')!=data['codec_weights_sha256']:
        raise ValueError('base weight file differs from prepared teachers')
    model=HaruModel.from_pretrained(str(directory),device='cuda',dtype=torch.bfloat16)
    model.codec.float()
    with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as saved:
        state={k.removeprefix('codec.'):saved.get_tensor(k).to('cuda',torch.float32) for k in saved.keys() if k.startswith('codec.')}
    model.codec.load_state_dict(state,strict=True);del state
    frozen=codec_digest(model.codec)
    trainer=CodecTrainer(model,selected,TrainSpec(base_model=spec['base_model'],base_revision=spec['base_revision'],
        target_speaker=spec['target_speaker'],max_steps=spec['max_steps'],learning_rate=spec['learning_rate'],
        seed=spec['seed'],max_seconds=spec['max_seconds']),rank=spec['rank'])
    observations=[]
    @torch.no_grad()
    def observe(step,reason):
        folder=out/('step-'+str(step));folder.mkdir()
        model.requires_grad_(False)
        export_adapter(model,folder/'adapter',**trainer._identity())
        measured=[]
        for pair in pairs:
            wave,_=decode_training(model.codec,pair.latents[None]);wave=wave[0]
            loss=paired_loss(wave,pair)
            audio=wave.float().cpu().numpy()
            valid=audio[:pair.metadata['valid_samples']]
            name=pair.metadata['id']+'.wav';sf.write(folder/name,valid,24000,subtype='FLOAT')
            measured.append(dict(id=pair.metadata['id'],
                trained_id=pair.metadata['id'] in spec['training_ids'],spectral_loss=float(loss['spectral']),
                rms=float(np.sqrt(np.mean(valid.astype('float64')**2))),
                samples=len(valid),wav=name,wav_sha256=file_hash(folder/name)))
        for case in spec['cases']:
            raw=read_checkpoint(root/'generated'/case['id']/'raw-latents')
            if (raw['metadata']['input_sha256']!=case['sha256']
                    or raw['metadata']['base_revision']!=spec['base_revision']
                    or raw['metadata']['codec_sha256']!=frozen):
                raise ValueError('saved generated latent provenance differs')
            metadata=dict(raw['metadata'],diagnostic_training_ids=spec['training_ids'],
                effective_dataset_sha256=trainer.data_sha256)
            save_decoder_comparison(folder/case['id'],model,raw['latents'],metadata)
        item=dict(step=step,reason=reason,records=measured,frozen_codec_exact=codec_digest(model.codec)==frozen,
                  effective_dataset_sha256=trainer.data_sha256,voice_similarity_confirmed=False)
        assert item['frozen_codec_exact']
        (folder/'measurements.json').write_text(json.dumps(item,indent=2),encoding='utf-8')
        observations.append(dict(step=step,reason=reason,path=folder.name))
        get_adapter(model.codec).requires_grad_(True)
        print(json.dumps(dict(step=step,reason=reason,mean_train_loss=float(np.mean([r['spectral_loss'] for r in measured if r['trained_id']])))),flush=True)
    history=run_stages(trainer,spec['milestones'],observe)
    trainer.save(out/'terminal-checkpoint')
    report=dict(training_steps=trainer.step,training_seconds=trainer.elapsed,history=history,observations=observations,
        stop_reason='update_cap' if trainer.step==spec['max_steps'] else 'training_time_limit',
        rank=spec['rank'],parent_diagnostic_sha256=gate_hash if (correction or full) else None,
        original_dataset_sha256=data['dataset_sha256'],effective_dataset_sha256=trainer.data_sha256,
        training_ids=spec['training_ids'],gate_sha256=file_hash(gate_path),spec_sha256=file_hash(path),
        execution_complete=True,frozen_codec_exact=codec_digest(model.codec)==frozen,
        voice_similarity_confirmed=False,scope=('Full original fifteen-pair correction; no new independent corpus test' if full else
            'Three-pair diagnostic; other original train pairs are not new independent test data'))
    (out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['roundtrip','train'])
    parser.add_argument('spec');parser.add_argument('destination')
    parser.add_argument('--gate');parser.add_argument('--roundtrip-report')
    args=parser.parse_args()
    if args.command=='roundtrip':run_roundtrip(args.spec,args.destination)
    else:
        if not args.gate or not args.roundtrip_report:parser.error('training requires the independent gate and roundtrip report')
        run_small_training(args.spec,args.destination,gate_path=args.gate,roundtrip_report=args.roundtrip_report)
