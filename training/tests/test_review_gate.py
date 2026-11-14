import json
import pytest


def test_gate_waits_for_bound_decision_and_returns_explicit_reason(tmp_path):
    from haru_training.review import FileReviewGate
    waits=[]
    def wait(seconds):
        waits.append(seconds)
        request=json.loads((tmp_path/'step-00000001/request.json').read_text())
        decision={'run_id':request['run_id'],'step':1,
                  'observation_sha256':request['observation_sha256'],
                  'accepted':False,'reason':'reply ignored food input'}
        (tmp_path/'step-00000001/decision.json').write_text(json.dumps(decision))
    gate=FileReviewGate(tmp_path,run_id='fresh-run-2',timeout_seconds=10,wait=wait)
    answer=gate({'step':1,'texts':{'dialogue':'birthday'}})
    assert waits and answer=={'accepted':False,'reason':'reply ignored food input'}


@pytest.mark.parametrize('change',[{'run_id':'old-run'},{'step':True},
    {'observation_sha256':'0'*64},{'accepted':'true'},{'reason':''}])
def test_stale_or_malformed_review_fails_closed(tmp_path,change):
    from haru_training.review import FileReviewGate
    def wait(seconds):
        request=json.loads((tmp_path/'step-00000001/request.json').read_text())
        decision={key:request[key] for key in ('run_id','observation_sha256')}
        decision.update(step=1,accepted=True,reason='checked input and output')
        decision.update(change)
        (tmp_path/'step-00000001/decision.json').write_text(json.dumps(decision))
    gate=FileReviewGate(tmp_path,run_id='fresh-run-2',timeout_seconds=10,wait=wait)
    with pytest.raises(ValueError,match='review'): gate({'step':1,'texts':{'dialogue':'birthday'}})


def test_review_timeout_keeps_request_and_does_not_fabricate_acceptance(tmp_path):
    from haru_training.review import FileReviewGate
    now=[0.]
    def wait(seconds): now[0]+=seconds
    gate=FileReviewGate(tmp_path,run_id='fresh-run-2',timeout_seconds=2,
                       wait=wait,clock=lambda:now[0])
    with pytest.raises(TimeoutError): gate({'step':1,'texts':{'dialogue':'birthday'}})
    assert (tmp_path/'step-00000001/request.json').is_file()
    assert not (tmp_path/'step-00000001/decision.json').exists()
