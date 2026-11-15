"""Same-latent decoding and anonymous listening; metrics never certify identity."""
import base64
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import re

import numpy as np
import torch

from .checkpoint import write_checkpoint,read_checkpoint
from .codec_adapter import get_adapter,set_adapter_enabled
from .codec_adapter_inference import codec_digest
from .evaluation import evaluate_manifest
from .replay import save_replay


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _tensor_hash(value):
    value=value.detach().cpu().contiguous()
    return sha256(str((tuple(value.shape),value.dtype)).encode()+
                  value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def save_decoder_comparison(destination,model,latents,metadata):
    """Input is raw generated latents, after Haru unscale_latents, batch=1.

    This compares decoder outputs only; it does not claim to have replayed the
    conversation. Preserve the generation input hash, seed and text separately.
    """
    destination=Path(destination)
    adapter=get_adapter(model.codec)
    if (model.training or any(p.requires_grad for p in model.parameters()) or not adapter.enabled
            or latents.ndim!=3 or latents.shape[0]!=1 or latents.shape[1]<1 or latents.shape[2]!=64
            or not torch.isfinite(latents).all()):
        raise ValueError('comparison requires a frozen model and finite raw latents')
    if (metadata.get('representation')!='raw_codec_sample'
            or not re.fullmatch('[a-f0-9]{64}',metadata.get('input_sha256',''))
            or not re.fullmatch('[a-f0-9]{40}',metadata.get('base_revision',''))
            or not isinstance(metadata.get('text'),str) or not isinstance(metadata.get('system_prompt'),str)
            or not isinstance(metadata.get('target_speaker'),str) or not metadata['target_speaker']
            or not re.fullmatch('[a-f0-9]{64}',metadata.get('dataset_input_manifest_sha256',''))
            or type(metadata.get('config',{}).get('seed')) is not int):
        raise ValueError('incomplete raw generation provenance')
    if destination.exists():raise FileExistsError('comparison already exists')
    # Decode both before writing, so runtime failures do not masquerade as a pair.
    try:
        set_adapter_enabled(model.codec,False)
        left=model.codec.decode(latents).float().clamp(-1,1)[0].cpu().numpy()
    finally:
        set_adapter_enabled(model.codec,True)
    right=model.codec.decode(latents).float().clamp(-1,1)[0].cpu().numpy()
    zhash=_tensor_hash(latents)
    weights=adapter.state_dict()
    ahash=sha256(''.join(key+_tensor_hash(value) for key,value in sorted(weights.items())).encode()).hexdigest()
    info=dict(metadata,raw_latent_sha256=zhash,codec_sha256=codec_digest(model.codec),adapter_sha256=ahash,
              voice_similarity_confirmed=False,scope='same raw latent decoder comparison; not conversation replay')
    destination.mkdir(parents=True)
    write_checkpoint(destination/'raw-latents',dict(kind='haru-decoder-evaluation-latents',version=1,
                                                    latents=latents,metadata=info))
    for side,wave in [('baseline',left),('candidate',right)]:
        save_replay(destination/side,dict(audio=wave,sample_rate=24000,text=metadata['text'],
            config=metadata['config'],system_prompt=metadata['system_prompt'],
            feedback_mode='unchanged_raw_latent_decoder_comparison'),provenance=info)
    files={str(p.relative_to(destination).as_posix()):_hash(p) for p in destination.rglob('*') if p.is_file()}
    (destination/'comparison.json').write_text(json.dumps(dict(info,files=files),indent=2),encoding='utf-8')
    return destination


def _verify(path):
    path=Path(path)
    info=json.loads((path/'comparison.json').read_text(encoding='utf-8'))
    for name,expected in info['files'].items():
        resolved=(path/name).resolve()
        if not resolved.is_relative_to(path.resolve()) or _hash(resolved)!=expected:
            raise ValueError('comparison artifact hash mismatch')
    saved=read_checkpoint(path/'raw-latents')
    if _tensor_hash(saved['latents'])!=info['raw_latent_sha256']:
        raise ValueError('raw latent identity differs')
    return info


def evaluate_comparison(path,manifest):
    _verify(path)
    # Force exactly this pair; unrelated or cherry-picked cases cannot replace it.
    if (manifest.get('expected_cases')!=['comparison'] or len(manifest.get('cases',[]))!=1
            or manifest['cases'][0].get('id')!='comparison'
            or manifest['cases'][0]['baseline']['directory']!='baseline'
            or manifest['cases'][0]['candidate']['directory']!='candidate'):
        raise ValueError('evaluation must refer to the frozen decoder pair')
    return evaluate_manifest(manifest,root=Path(path))


def write_blind_html(path,reference,*,reference_sha256,seed,reference_info,training_manifest):
    path=Path(path);info=_verify(path)
    if (path/'listen.html').exists() or (path/'answer-key.json').exists():
        raise FileExistsError('listening presentation already exists')
    if _hash(reference)!=reference_sha256:raise ValueError('reference audio identity differs')
    if _hash(training_manifest)!=info['dataset_input_manifest_sha256']:
        raise ValueError('training membership manifest differs')
    try:
        records=json.loads(Path(training_manifest).read_text(encoding='utf-8'))['records']
        groups=[r['group_id'] for r in records]
        hashes={r[k] for r in records for k in ('source_sha256','target_sha256',
            'source_original_sha256','target_original_sha256') if k in r}
        if (not records or len(set(groups))!=len(groups) or any(r['split']!='train' for r in records)
                or reference_info['group_id'] in groups or reference_sha256 in hashes
                or not reference_info['group_id'] or reference_info['split'] not in ('reference','validation')
                or reference_info['speaker']!=info['target_speaker']
                or reference_info['wav_sha256']!=reference_sha256
                or any(r['target_speaker']!=info['target_speaker'] for r in records)):
            raise ValueError('reference must be an untrained recording of the target speaker')
    except (KeyError,TypeError) as error:
        raise ValueError('reference or training membership evidence is incomplete') from error
    import soundfile as sf
    audio,rate=sf.read(reference)
    if audio.ndim!=1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('reference needs finite nonempty mono speech')
    order=['baseline','candidate'];random.Random(seed).shuffle(order)
    def player(file):
        encoded=base64.b64encode(Path(file).read_bytes()).decode('ascii')
        return '<audio controls src="data:audio/wav;base64,'+encoded+'"></audio>'
    html='<!doctype html><meta charset="utf-8"><title>音声比較</title>'
    html+='<style>body{max-width:700px;margin:2em auto;font-family:sans-serif}audio{width:100%}</style>'
    html+='<h1>音声比較</h1><p>本人の声に近い方と、発話内容を聞き取れるかを別々に確認してください</p>'
    html+='<h2>本人の参照音声</h2>'+player(reference)
    for label,side in zip(('A','B'),order):html+='<h2>'+label+'</h2>'+player(path/side/'response.wav')
    (path/'answer-key.json').write_text(json.dumps(dict(seed=seed,order=order,
        reference_sha256=reference_sha256,reference_info=reference_info,
        dataset_input_manifest_sha256=info['dataset_input_manifest_sha256'],
        voice_similarity_confirmed=False)),encoding='utf-8')
    (path/'listen.html').write_text(html,encoding='utf-8')
    return path/'listen.html'


def write_blind_collection(destination,cases,reference,**reference_args):
    """One self-contained three-question page; all condition keys remain separate."""
    from html import escape
    from io import BytesIO
    import soundfile as sf
    from .listening_review import make_presentation, review_controls
    destination=Path(destination)
    if destination.exists():raise FileExistsError('listening collection already exists')
    if (len(cases)!=3 or len({r['id'] for r in cases})!=3
            or len({Path(r['directory']).resolve() for r in cases})!=3):
        raise ValueError('listening collection needs three distinct frozen cases')
    infos=[_verify(r['directory']) for r in cases]
    for key in ('base_revision','target_speaker','codec_sha256','adapter_sha256','dataset_input_manifest_sha256'):
        if len({info[key] for info in infos})!=1:
            raise ValueError('collection conditions or training membership differ')
    for row in cases:
        folder=Path(row['directory'])
        if (folder/'listen.html').exists() or (folder/'answer-key.json').exists():
            raise FileExistsError('individual listening comparison already exists')
    keys=[]
    for index,row in enumerate(cases):
        folder=Path(row['directory'])
        write_blind_html(folder,reference,**dict(reference_args,seed=reference_args['seed']+index))
        keys.append(json.loads((folder/'answer-key.json').read_text(encoding='utf-8')))
    presentation=make_presentation([dict(id=row['id'],audio_sha256={label:_hash(Path(row['directory'])/side/'response.wav')
        for label,side in zip(('A','B'),key['order'])}) for row,key in zip(cases,keys)],
        reference_sha256=reference_args['reference_sha256'])
    def player(file):
        audio,rate=sf.read(file,dtype='float32')
        if audio.ndim!=1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError('listening needs finite mono audio')
        rms=float(np.sqrt(np.mean(audio.astype('float64')**2)))
        gain=min(.1/max(rms,1e-12),.99/max(float(np.abs(audio).max()),1e-12))
        buffer=BytesIO();sf.write(buffer,audio*gain,rate,format='WAV',subtype='PCM_16')
        return '<audio controls preload="metadata" src="data:audio/wav;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')+'"></audio>'
    options='<option>未確認</option><option>A</option><option>B</option><option>差が分からない</option><option>両方似ていない</option><option>判定できない</option>'
    html='<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>3問の音声比較</title>'
    html+='<style>body{max-width:800px;margin:2em auto;padding:0 1em;font-family:sans-serif;line-height:1.7;background:#f6f7fa;color:#182030}section{background:white;border:1px solid #dae0e9;border-radius:12px;margin:1.2em 0;padding:1.2em}audio{width:100%}label{display:block;margin:.7em 0}select{margin-left:1em;padding:.3em}summary{cursor:pointer}</style>'
    html+='<h1>3問の音声比較</h1><p>本人の参照を聴いてからAとBを比較してください。声の近さと聞き取りやすさを別々に確認します</p><section><h2>本人の参照音声</h2>'+player(reference)+'</section>'
    for index,(row,info,key) in enumerate(zip(cases,infos,keys),1):
        html+='<section><h2>'+str(index)+'. '+escape(row['question'])+'</h2>'
        for label,side in zip(('A','B'),key['order']):
            html+='<h3>'+label+'</h3>'+player(Path(row['directory'])/side/'response.wav')
            html+='<label>'+label+'の内容<select data-review-field="'+escape(row['id']+':'+label+'_content',quote=True)+'"><option>未確認</option><option>聞き取れる</option><option>一部不明</option><option>意味が分からない</option><option>判定できない</option></select></label>'
        html+='<label>本人の声に近い方<select data-review-field="'+escape(row['id']+':voice',quote=True)+'">'+options+'</select></label>'
        html+='<details><summary>聴いた後に生成textを確認</summary><p>'+escape(info['text'])+'</p></details></section>'
    html+='<p>再生用の音声には同じ方式で音量調整を行っています。元の録音と条件の対応表は別に保存しています</p>'+review_controls(presentation)+'</html>'
    destination.mkdir(parents=True)
    (destination/'listen.html').write_text(html,encoding='utf-8')
    (destination/'presentation.json').write_text(json.dumps(presentation,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'answer-key.json').write_text(json.dumps(dict(cases=[dict(id=r['id'],**key) for r,key in zip(cases,keys)],
        presentation_sha256=presentation['presentation_sha256'],
        normalization=dict(rms_target=.1,peak_limit=.99,subtype='PCM_16',pitch_or_speed_changed=False),
        voice_similarity_confirmed=False),ensure_ascii=False,indent=2),encoding='utf-8')
    return destination/'listen.html'
