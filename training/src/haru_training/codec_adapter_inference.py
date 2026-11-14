"""Separate, immutable codec-adapter artifacts with validation before mutation."""
from dataclasses import asdict
from hashlib import sha256
import json
import re

import torch

from .checkpoint import read_checkpoint,write_checkpoint
from .codec_adapter import attach_adapter,get_adapter


DTYPES={'float32':torch.float32,'bfloat16':torch.bfloat16}


def config_digest(config):
    return sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def codec_digest(codec):
    """Hash the frozen codec using base names, before or after wrapping the stem."""
    digest=sha256()
    for name,value in codec.state_dict().items():
        if name.startswith('decoder.stem.adapter.'):continue
        name=name.replace('decoder.stem.base.','decoder.stem.')
        digest.update(name.encode());digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate(bundle):
    try:
        if bundle['kind']!='haru-codec-adapter' or bundle['version']!=1:
            raise ValueError('unsupported codec adapter artifact')
        spec=bundle['spec'];config=spec['codec_config'];rank=spec['rank']
        if (not re.fullmatch(r'[\w.-]+/[\w.-]+',spec['base_model'])
                or not re.fullmatch('[a-f0-9]{40}',spec['base_revision'])
                or not re.fullmatch('[a-f0-9]{64}',spec['dataset_sha256'])
                or not re.fullmatch('[a-f0-9]{64}',bundle['codec_sha256'])
                or not isinstance(spec['target_speaker'],str) or not spec['target_speaker']
                or spec['backend']!='torch' or spec['dtype'] not in DTYPES
                or config_digest(config)!=bundle['config_sha256']):
            raise ValueError('invalid codec adapter identity')
        channels=config['base_channels']<<len(config['ratios'])
        if type(rank) is not int or not 1<=rank<=channels:
            raise ValueError('invalid adapter structure')
        expected={'down.weight':(rank,channels,1),'down.bias':(rank,),
                  'up.weight':(channels,rank,1),'up.bias':(channels,)}
        weights=bundle['weights']
        if set(weights)!=set(expected) or any(
                not torch.is_tensor(weights[key]) or tuple(weights[key].shape)!=shape
                or weights[key].dtype!=DTYPES[spec['dtype']] or not torch.isfinite(weights[key]).all()
                for key,shape in expected.items()):
            raise ValueError('invalid adapter tensor names, shape, dtype, or values')
    except (KeyError,TypeError) as error:
        raise ValueError('incomplete codec adapter artifact') from error
    return spec


def adapter_bundle(model,*,base_model,base_revision,target_speaker,dataset_sha256):
    adapter=get_adapter(model.codec)
    if not adapter.enabled:raise ValueError('cannot export a disabled training adapter')
    if any(p.requires_grad for name,p in model.named_parameters()
           if not name.startswith('codec.decoder.stem.adapter.')):
        raise ValueError('parameters outside the codec adapter are trainable')
    config=asdict(model.codec.config)
    bundle=dict(kind='haru-codec-adapter',version=1,
                spec=dict(base_model=base_model,base_revision=base_revision,target_speaker=target_speaker,
                          dataset_sha256=dataset_sha256,rank=adapter.rank,codec_config=config,
                          backend='torch',dtype=str(model.codec.dtype).removeprefix('torch.')),
                config_sha256=config_digest(config),codec_sha256=codec_digest(model.codec),
                weights={key:value.detach().cpu().clone() for key,value in adapter.state_dict().items()})
    _validate(bundle)
    return bundle


def export_adapter(model,path,**identity):
    return write_checkpoint(path,adapter_bundle(model,**identity))


def _apply(model,bundle,*,base_model,base_revision):
    spec=_validate(bundle)
    if (spec['base_model']!=base_model or spec['base_revision']!=base_revision
            or config_digest(asdict(model.codec.config))!=bundle['config_sha256']
            or model.codec.dtype!=DTYPES[spec['dtype']]
            or codec_digest(model.codec)!=bundle['codec_sha256']):
        raise ValueError('adapter base, codec configuration, precision, or frozen weights differ')
    # Stage all device copies before mutating the existing model.
    weights={key:value.to(device=model.codec.device,dtype=model.codec.dtype).clone()
             for key,value in bundle['weights'].items()}
    adapter=attach_adapter(model,rank=spec['rank'])
    adapter.load_state_dict(weights,strict=True)
    return model.eval().requires_grad_(False)


def apply_adapter(model,path,*,base_model,base_revision):
    return _apply(model,read_checkpoint(path),base_model=base_model,base_revision=base_revision)


def load_adapter(path,*,device='cpu'):
    bundle=read_checkpoint(path);spec=_validate(bundle)
    from haru.models.haru import HaruModel
    model=HaruModel.from_pretrained(spec['base_model'],revision=spec['base_revision'],
                                   device=device,dtype=DTYPES[spec['dtype']])
    return _apply(model,bundle,base_model=spec['base_model'],base_revision=spec['base_revision'])
