"""A durable, run-bound operator decision before advancing training."""

from hashlib import sha256
import json
import math
from pathlib import Path
import time

from .trials import _json


class FileReviewGate:
    def __init__(self,directory,*,run_id,timeout_seconds=300,wait=time.sleep,clock=time.monotonic,
                 score_baseline=None):
        if (not isinstance(run_id,str) or not run_id.strip()
                or type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds)
                or not 0<timeout_seconds<=3600):
            raise ValueError('invalid review identity or timeout')
        self.directory=Path(directory)
        self.run_id=run_id
        self.timeout=timeout_seconds
        self.wait,self.clock=wait,clock
        if score_baseline is not None and (not isinstance(score_baseline,dict) or not score_baseline
                or any(not isinstance(key,str) or not key or type(value) is not int or not 0<=value<=2
                       for key,value in score_baseline.items())):
            raise ValueError('rating baseline needs named cases with integer scores from zero to two')
        self.score_baseline=dict(score_baseline) if score_baseline is not None else None

    def __call__(self,observation):
        step=observation.get('step')
        if type(step) is not int or step<0: raise ValueError('invalid review step')
        digest=sha256(json.dumps(observation,sort_keys=True,ensure_ascii=False,
                                allow_nan=False,separators=(',',':')).encode()).hexdigest()
        root=self.directory/f'step-{step:08d}'
        root.mkdir(parents=True,exist_ok=False)
        request={'run_id':self.run_id,'step':step,'observation_sha256':digest,
                 'observation':observation,'scope':'operator text review; not human voice approval'}
        if self.score_baseline is not None:
            request['score_baseline']=self.score_baseline
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
        if self.score_baseline is not None:
            ratings=decision.get('ratings')
            if (not isinstance(ratings,dict) or ratings.keys()!=self.score_baseline.keys()
                    or any(type(value) is not int or not 0<=value<=2 for value in ratings.values())
                    or type(decision.get('new_severe_failure')) is not bool):
                raise ValueError('review ratings must cover every baseline case with explicit severity')
            if decision['accepted'] and (sum(ratings.values())<sum(self.score_baseline.values())
                    or sum(value==0 for value in ratings.values())>sum(value==0 for value in self.score_baseline.values())
                    or decision['new_severe_failure']):
                raise ValueError('review acceptance violates the fixed score or severity policy')
        return {'accepted':decision['accepted'],'reason':decision['reason'].strip()}
