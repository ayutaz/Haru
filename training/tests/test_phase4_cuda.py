"""Real Haru AdamW continuation across fresh processes, only on vast.ai.

Three short optimizer steps verify machinery, not speaker adaptation quality.
Exact comparison is declared before running, under deterministic CUDA settings.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from haru import HaruModel
from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.data import make_sequence, read_cache
from haru_training.prepare import encode_user_audio
from haru_training.trainer import RunSpec, Trainer, capture_rng, dataset_fingerprint
from test_phase4_haru import tensor_digest

pytestmark=pytest.mark.cuda
REVISION='8d6e83bcd3701a76fc27a764965665da1df7da78'


def actual_trainer(output,mode):
    assert os.environ.get('HARU_COMPUTE_PROVIDER')=='vast.ai'
    assert torch.cuda.is_available(), 'CUDA is mandatory'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(73);random.seed(73);np.random.seed(73)
    model=HaruModel.from_pretrained(os.environ['HARU_MODEL'],device='cuda',dtype=torch.bfloat16,
                                    revision=REVISION)
    configure_trainable(model,TrainingConfig(mode=mode,train_projectors=mode=='lm_lora'))
    cache=Path(os.environ['HARU_PHASE4_CACHE'])
    manifest=json.loads((cache/'manifest.json').read_text())
    splits=json.loads((cache.parent/'splits.json').read_text())['splits']
    rows={r['id']:r for r in manifest['records']}
    reference_row=rows[splits['reference'][0]]
    reference=read_cache(cache/reference_row['reference_cache'],reference_row['reference_key'])[0,:4]
    def sequences(ids):
        result=[]
        for i,record_id in enumerate(ids):
            row=rows[record_id]
            latent=read_cache(cache/row['cache'],row['key'])[0,:[4,5,3][i]]
            count=len(latent)
            user=encode_user_audio(model.codec,np.zeros((count*3200,1),np.float32),seed=17)[0]
            text=torch.tensor([10,11]+[model.config.pad_token_id]*(count-2))
            # Short graph with explicit probe tokens, not a complete utterance training recipe.
            result.append(make_sequence(record_id,[3,4],reference,latent,user,text,
                                         model.config.pad_token_id,model.config.eop_token_id))
        return result
    train,val=sequences(splits['train'][:3]),sequences(splits['validation'][:2])
    spec=RunSpec(base_model='higurashi-ai/Haru-4B',base_revision=REVISION,
        dataset_sha256=dataset_fingerprint(train),validation_sha256=dataset_fingerprint(val),
        split_sha256=hashlib.sha256((cache.parent/'splits.json').read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest(),
        max_steps=3,accumulation_steps=2,learning_rate=.001,warmup_steps=2,
        mode=mode,train_projectors=mode=='lm_lora')
    return Trainer(model,train,val,spec,output)


def worker(mode,phase,root):
    output=root/phase
    start=time.monotonic()
    trainer=actual_trainer(output,mode)
    selected=lambda:((n,p) for n,p in trainer.model.named_parameters() if p.requires_grad)
    before=tensor_digest(selected())
    torch.cuda.reset_peak_memory_stats()
    if phase=='resume': trainer.resume(root/'split/checkpoints/step-00000001')
    trainer.fit(until_step=1 if phase=='split' else None)
    checkpoint=trainer.save() if phase in ('split','resume') else None
    moments=[(f'{n}:{k}',value) for n,p in trainer.model.named_parameters() if p.requires_grad
             for k,value in trainer.optimizer.state[p].items()]
    rng=capture_rng(trainer.model)
    payload={'phase':phase,'mode':mode,'state':asdict(trainer.state),
        'selected_before':before,'selected_after':tensor_digest(selected()),
        'frozen_before':trainer.frozen_digest,'frozen_after':trainer._frozen_digest(),
        'optimizer_digest':tensor_digest(moments),'scheduler':trainer.scheduler.state_dict(),
        'torch_rng_digest':tensor_digest([('cpu',rng['torch'])]+[(f'cuda-{i}',v) for i,v in enumerate(rng['cuda'])]),
        'future_random':[random.random(),float(np.random.rand()),*torch.rand(3).tolist(),*torch.rand(3,device='cuda').cpu().tolist()],
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-start,
        'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
        'checkpoint':str(checkpoint.relative_to(root)) if checkpoint else None}
    (root/f'{phase}.json').write_text(json.dumps(payload,indent=2))


@pytest.mark.parametrize('mode',['head','lm_lora'])
def test_actual_cuda_adam_continuation_matches_uninterrupted_training_in_fresh_processes(mode):
    assert os.environ.get('HARU_COMPUTE_PROVIDER')=='vast.ai'
    root=Path(os.environ['HARU_EVIDENCE_DIR'])/mode
    root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['PYTHONPATH']=os.pathsep.join([str(Path(__file__).parent.resolve()),
                                     str(Path(__file__).parents[2]/'haru'),env.get('PYTHONPATH','')])
    for phase in ('baseline','split','resume'):
        with (root/f'{phase}.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),mode,phase,str(root)],
                            env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    baseline,split,resumed=[json.loads((root/f'{phase}.json').read_text())
                            for phase in ('baseline','split','resume')]
    for key in ('state','selected_after','optimizer_digest','scheduler','torch_rng_digest','future_random'):
        assert baseline[key]==resumed[key],key
    assert split['state']['step']==1 and resumed['state']['step']==3
    assert baseline['selected_before']!=baseline['selected_after']
    assert baseline['frozen_before']==baseline['frozen_after']==resumed['frozen_before']==resumed['frozen_after']
    def training_ids(phase):
        return [e['record_ids'] for line in (root/phase/'events.jsonl').read_text().splitlines()
                if (e:=json.loads(line))['event']=='train']
    assert training_ids('baseline')==training_ids('split')+training_ids('resume')
    (root/'comparison.json').write_text(json.dumps({'exact_match':True,'steps':3,
        'comparison_keys':['weights','AdamW moments','scheduler','data order','RNG and next draws'],
        'tolerance':0,'scope':'short optimizer/resume probe, no voice quality evaluation'},indent=2))


if __name__=='__main__': worker(sys.argv[1],sys.argv[2],Path(sys.argv[3]))
