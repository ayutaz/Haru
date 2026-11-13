"""Immutable, hashed checkpoints: JSON structure and safetensors, never pickle."""

import hashlib
import json
import os
from pathlib import Path
import uuid

import torch
from safetensors.torch import load_file, save_file


def _encode(value, tensors):
    if torch.is_tensor(value):
        name = f't{len(tensors):08d}'
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {'type': 'tensor', 'name': name}
    if isinstance(value, dict):
        return {'type': 'dict', 'items': [[_encode(k, tensors), _encode(v, tensors)] for k,v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'type': 'tuple' if isinstance(value, tuple) else 'list',
                'items': [_encode(v, tensors) for v in value]}
    if value is None or type(value) in (str, int, float, bool):
        return {'type': 'scalar', 'value': value}
    raise ValueError('unsupported checkpoint state type')


def _decode(value, tensors):
    kind = value['type']
    if kind == 'tensor': return tensors[value['name']]
    if kind == 'scalar':
        result = value['value']
        if result is not None and type(result) not in (str,int,float,bool):
            raise ValueError('invalid scalar state')
        return result
    if kind == 'dict': return {_decode(k,tensors):_decode(v,tensors) for k,v in value['items']}
    if kind in ('tuple','list'):
        items = [_decode(v,tensors) for v in value['items']]
        return tuple(items) if kind == 'tuple' else items
    raise ValueError('unknown checkpoint state encoding')


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def write_checkpoint(destination: Path, bundle: dict):
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('checkpoint already exists; never overwrite it')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.parent/f'.pending-{uuid.uuid4().hex}'
    pending.mkdir()
    tensors = {}
    structure = _encode(bundle, tensors)
    save_file(tensors, str(pending/'tensors.safetensors'))
    (pending/'metadata.json').write_text(json.dumps(structure, allow_nan=False), encoding='utf-8')
    manifest = {'version':1, 'files':{name:_hash(pending/name)
                for name in ('metadata.json','tensors.safetensors')}}
    (pending/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    for name in ('tensors.safetensors','metadata.json','manifest.json'):
        with (pending/name).open('r+b') as stream: os.fsync(stream.fileno())
    os.replace(pending, destination)
    return destination


def read_checkpoint(path: Path):
    path = Path(path)
    try:
        if path.name.startswith('.pending-'): raise ValueError('checkpoint write was not committed')
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        if manifest['version'] != 1 or set(manifest['files']) != {'metadata.json','tensors.safetensors'}:
            raise ValueError('invalid checkpoint manifest')
        if any(_hash(path/name) != expected for name,expected in manifest['files'].items()):
            raise ValueError('checkpoint hash mismatch')
        tensors = load_file(str(path/'tensors.safetensors'))
        structure = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        return _decode(structure, tensors)
    except Exception as error:
        raise ValueError('checkpoint is missing, incomplete, corrupt, or unsupported') from error
