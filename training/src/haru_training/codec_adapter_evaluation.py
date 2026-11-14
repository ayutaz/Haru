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


def write_blind_html(path,reference,*,reference_sha256,seed):
    path=Path(path);_verify(path)
    if (path/'listen.html').exists() or (path/'answer-key.json').exists():
        raise FileExistsError('listening presentation already exists')
    if _hash(reference)!=reference_sha256:raise ValueError('reference audio identity differs')
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
        reference_sha256=reference_sha256,voice_similarity_confirmed=False)),encoding='utf-8')
    (path/'listen.html').write_text(html,encoding='utf-8')
    return path/'listen.html'
