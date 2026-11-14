"""A durable, run-bound operator decision before advancing training."""

from hashlib import sha256
import json
import math
from pathlib import Path
import time

from .trials import _json


class FileReviewGate:
    def __init__(self,directory,*,run_id,timeout_seconds=300,wait=time.sleep,clock=time.monotonic):
        if (not isinstance(run_id,str) or not run_id.strip()
                or type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds)
                or not 0<timeout_seconds<=3600):
            raise ValueError('invalid review identity or timeout')
        self.directory=Path(directory)
        self.run_id=run_id
        self.timeout=timeout_seconds
        self.wait,self.clock=wait,clock

    def __call__(self,observation):
        step=observation.get('step')
        if type(step) is not int or step<0: raise ValueError('invalid review step')
        digest=sha256(json.dumps(observation,sort_keys=True,ensure_ascii=False,
                                allow_nan=False,separators=(',',':')).encode()).hexdigest()
        root=self.directory/f'step-{step:08d}'
        root.mkdir(parents=True,exist_ok=False)
        request={'run_id':self.run_id,'step':step,'observation_sha256':digest,
                 'observation':observation,'scope':'operator text review; not human voice approval'}
        _json(root/'request.json',request)
        deadline=self.clock()+self.timeout
        while not (root/'decision.json').exists():
            if self.clock()>=deadline: raise TimeoutError('review decision did not arrive; no further updates')
            self.wait(min(.5,max(0,deadline-self.clock())))
        decision=json.loads((root/'decision.json').read_text(encoding='utf-8'))
        if (not isinstance(decision,dict) or decision.get('run_id')!=self.run_id
                or type(decision.get('step')) is not int or decision['step']!=step
                or decision.get('observation_sha256')!=digest or type(decision.get('accepted')) is not bool
                or not isinstance(decision.get('reason'),str) or not decision['reason'].strip()):
            raise ValueError('review decision does not match this run, observation or decision contract')
        return {'accepted':decision['accepted'],'reason':decision['reason'].strip()}
