import numpy as np
import pytest

from haru_training.saslaw import convert_pair, read_annotations


class Tokenizer:
    def encode(self, text, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(ids=[3]*len(text))


def test_saslaw_preserves_sync_and_quantizes_timed_text_without_inventing_words():
    left, right = np.zeros(44100*2, np.float32), np.zeros(44100*2, np.float32)
    left[4410:35280] = .1
    right[26460:70560] = .2
    turns = {'spk05': [{'start': .1, 'end': .8, 'text': 'はい'}],
             'spk06': [{'start': .6, 'end': 1.6, 'text': 'そうです'}]}
    audio, row = convert_pair(left, right, 44100, turns, Tokenizer(),
        dialogue_id='dialogue007', model_speaker='spk05', user_speaker='spk06',
        split='train', revision='a'*40, pad_id=0, eop_id=1)
    assert audio.shape == (48000, 2)
    assert row['source_ids'] == ['SaSLaW/spk05-spk06/dialogue007']
    assert row['model_speaker_id'] == 'SaSLaW/spk05'
    assert row['model_chunks'][0]['tokens'] == [3, 3]
    assert row['model_chunks'][0]['text'] == 'はい'
    assert row['turns'][0]['source_start_sec'] == .1
    assert row['turns'][0]['start_sec'] == 0
    assert audio[16000, 0] > .09 and audio[16000, 1] > .19


def test_missing_source_transcript_is_rejected_not_silently_filled(tmp_path):
    (tmp_path/'seg_info').mkdir()
    (tmp_path/'transcript_segment').mkdir()
    (tmp_path/'seg_info/dialogue001.txt').write_text('0.1 0.5\n0.7 1.2', encoding='utf-8')
    (tmp_path/'transcript_segment/dialogue001-seg00.txt').write_text('はい', encoding='utf-8')
    with pytest.raises(ValueError, match='transcript'):
        read_annotations(tmp_path, 'dialogue001')


def test_adjacent_subframe_segments_merge_without_same_speaker_overlap():
    turns = {'spk05': [{'start': .1, 'end': .3, 'text': 'あ'},
                       {'start': .31, 'end': .8, 'text': 'はい'}],
             'spk06': [{'start': 1.1, 'end': 1.8, 'text': 'そう'}]}
    _, row = convert_pair(np.ones(48000)*.1, np.ones(48000)*.2, 24000, turns, Tokenizer(),
        dialogue_id='dialogue001', model_speaker='spk05', user_speaker='spk06',
        split='validation', revision='a'*40, pad_id=0, eop_id=1)
    own = [t for t in row['turns'] if t['speaker'] == 'model']
    assert len(own) == 1 and own[0]['text'] == 'あはい'
    assert len(own[0]['source_segments']) == 2


def test_misaligned_recordings_or_overlong_text_are_not_trimmed_to_pass():
    turns = {'spk05': [{'start': .1, 'end': .3, 'text': '長すぎるテキスト'}],
             'spk06': [{'start': 1., 'end': 1.8, 'text': 'はい'}]}
    kwargs = dict(dialogue_id='dialogue001', model_speaker='spk05', user_speaker='spk06',
                  split='train', revision='a'*40, pad_id=0, eop_id=1)
    with pytest.raises(ValueError, match='synchron'):
        convert_pair(np.ones(48000), np.ones(47999), 24000, turns, Tokenizer(), **kwargs)
    with pytest.raises(ValueError, match='turn|chunk'):
        convert_pair(np.ones(48000)*.1, np.ones(48000)*.2, 24000, turns, Tokenizer(), **kwargs)


def test_partial_final_frame_has_explicit_zero_tail_for_rasterized_end_time():
    turns = {'spk05': [{'start': .1, 'end': .8, 'text': 'はい'}],
             'spk06': [{'start': 1.9, 'end': 48001/24000, 'text': 'そう'}]}
    audio, row = convert_pair(np.ones(48001)*.1, np.ones(48001)*.2, 24000, turns, Tokenizer(),
        dialogue_id='dialogue007', model_speaker='spk05', user_speaker='spk06',
        split='train', revision='a'*40, pad_id=0, eop_id=1)
    assert len(audio) == 51200 and row['tail_padding_samples'] == 3199
    assert not np.any(audio[-3199:])
    assert max(turn['end_sec'] for turn in row['turns']) <= len(audio)/24000
