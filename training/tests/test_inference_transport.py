from copy import deepcopy
from hashlib import sha256
import shutil

import pytest
import torch

from haru_training.checkpoint import write_checkpoint,read_checkpoint


def template(tmp_path):
    torch.manual_seed(4);model=torch.nn.Module();model.head=torch.nn.Linear(3,2)
    original=write_checkpoint(tmp_path/'original',dict(kind='haru-inference',version=1,step=0,spec=dict(mode='head'),
        weights={n:p for n,p in model.named_parameters()}))
    partial=tmp_path/'template';partial.mkdir()
    for name in ['metadata.json','manifest.json']:shutil.copyfile(original/name,partial/name)
    return model,original,partial


def test_original_head_rebuilt_from_fresh_parameters_has_the_exact_tensor_file_hash(tmp_path):
    from haru_training.inference_transport import rebuild_original_head
    model,original,partial=template(tmp_path)
    with pytest.raises(ValueError):read_checkpoint(partial)
    rebuilt=rebuild_original_head(model,partial,tmp_path/'rebuilt')
    assert sha256((rebuilt/'tensors.safetensors').read_bytes()).hexdigest()==sha256((original/'tensors.safetensors').read_bytes()).hexdigest()
    actual=read_checkpoint(rebuilt);expected=read_checkpoint(original)
    assert actual['step']==0 and actual['spec']==expected['spec']
    for name,tensor in expected['weights'].items():torch.testing.assert_close(actual['weights'][name],tensor,atol=0,rtol=0)


def test_modified_head_is_not_accepted_as_original_and_never_commits_an_artifact(tmp_path):
    from haru_training.inference_transport import rebuild_original_head
    model,original,partial=template(tmp_path)
    with torch.no_grad():model.head.weight.add_(.1)
    with pytest.raises(ValueError):rebuild_original_head(model,partial,tmp_path/'rebuilt')
    assert not (tmp_path/'rebuilt').exists()


def test_metadata_template_is_hash_verified_before_rebuilding(tmp_path):
    from haru_training.inference_transport import rebuild_original_head
    model,original,partial=template(tmp_path)
    with (partial/'metadata.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError):rebuild_original_head(model,partial,tmp_path/'rebuilt')
    assert not (tmp_path/'rebuilt').exists()
