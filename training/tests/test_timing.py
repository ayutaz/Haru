import numpy as np
import pytest


def test_overlap_and_acoustic_response_delays_do_not_invent_turn_attribution():
    from haru_training.timing import activity_timing
    user=np.zeros(20,dtype=bool); user[2:5]=True; user[10:13]=True
    model=np.zeros(20,dtype=bool); model[4:7]=True; model[15:18]=True
    result=activity_timing(model,user,hop_seconds=.1)
    assert result['model_speech_seconds']==pytest.approx(.6)
    assert result['user_speech_seconds']==pytest.approx(.6)
    assert result['overlap_seconds']==pytest.approx(.1)
    assert result['fraction_model_speech_overlapping_user']==pytest.approx(1/6)
    assert result['first_model_speech_seconds']==pytest.approx(.4)
    first,second=result['user_offsets']
    assert first['status']=='already_speaking' and first['next_model_onset_delay_seconds'] is None
    assert second['status']=='new_model_onset' and second['next_model_onset_delay_seconds']==pytest.approx(.2)


def test_silence_and_missing_response_are_unknown_instead_of_zero_latency():
    from haru_training.timing import activity_timing
    silence=np.zeros(10,dtype=bool)
    result=activity_timing(silence,silence,hop_seconds=.032)
    assert result['first_model_speech_seconds'] is None
    assert result['fraction_model_speech_overlapping_user'] is None
    assert result['user_offsets']==[]
    user=silence.copy(); user[1:3]=True
    result=activity_timing(silence,user,hop_seconds=.032)
    assert result['user_offsets'][0]['status']=='no_new_onset_before_next_user'
    assert result['user_offsets'][0]['next_model_onset_delay_seconds'] is None


def test_onset_at_user_offset_and_last_frame_boundary():
    from haru_training.timing import activity_timing
    user=np.array([True,True,False,False,True,True])
    model=np.array([False,False,True,True,False,False])
    result=activity_timing(model,user,hop_seconds=.1)
    assert result['user_offsets'][0]['next_model_onset_delay_seconds']==0
    assert result['user_offsets'][1]['status']=='no_following_audio'


@pytest.mark.parametrize('model,user,hop',[
    ([True],[True,False],.1),([],[],.1),([.5],[True],.1),([True],[True],0),([True],[True],float('nan'))])
def test_timing_rejects_ambiguous_masks_and_rates(model,user,hop):
    from haru_training.timing import activity_timing
    with pytest.raises(ValueError): activity_timing(np.array(model),np.array(user),hop_seconds=hop)
