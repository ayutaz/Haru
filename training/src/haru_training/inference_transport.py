"""Reconstruct an original head artifact before lossless delta recovery.

Only exact original weights are accepted. This avoids retransferring the full
base head when it is already present in a freshly loaded pinned model. Normal
apply_inference still validates the model's body, configuration and buffers.
"""
import json
import os
from pathlib import Path
import shutil
import uuid

from safetensors.torch import save_file

from .checkpoint import _hash,_decode,read_checkpoint


def rebuild_original_head(model,template,destination):
    template,destination=Path(template),Path(destination)
    if destination.exists():raise FileExistsError('never overwrite a recovered artifact')
    manifest=json.loads((template/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('version')!=1 or set(manifest.get('files',{}))!={'metadata.json','tensors.safetensors'}
            or _hash(template/'metadata.json')!=manifest['files']['metadata.json']):
        raise ValueError('original head template hash mismatch')
    structure=json.loads((template/'metadata.json').read_text(encoding='utf-8'))
    parameters=dict(model.named_parameters());tensors={}
    try:
        weights=[v for k,v in structure['items'] if k=={'type':'scalar','value':'weights'}]
        if len(weights)!=1 or weights[0]['type']!='dict':raise ValueError('missing head mapping')
        for key,value in weights[0]['items']:
            name=key['value']
            if not name.startswith('head.') or value['type']!='tensor':raise ValueError('not an original whole head')
            tensors[value['name']]=parameters[name].detach().cpu().contiguous().clone()
        bundle=_decode(structure,tensors)
        if bundle['kind']!='haru-inference' or bundle['version']!=1 or bundle['step']!=0 or bundle['spec']['mode']!='head':
            raise ValueError('not an original head artifact')
    except (KeyError,TypeError,IndexError) as error:
        raise ValueError('invalid original head template or model') from error
    destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.parent/('.pending-'+uuid.uuid4().hex);pending.mkdir()
    try:
        save_file(tensors,str(pending/'tensors.safetensors'))
        if _hash(pending/'tensors.safetensors')!=manifest['files']['tensors.safetensors']:
            raise ValueError('fresh model head differs from the immutable original')
        for name in ['metadata.json','manifest.json']:shutil.copyfile(template/name,pending/name)
        os.replace(pending,destination)
    finally:
        if pending.exists():
            assert pending.resolve().parent==destination.parent.resolve() and pending.name.startswith('.pending-')
            shutil.rmtree(pending)
    read_checkpoint(destination)
    return destination
