import hashlib
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from haru_training.corpus import Clip, inspect_corpus, split_corpus, validate_splits


def corpus(tmp_path, count=4):
    audio = tmp_path / '02 WAV（+12dB増幅）'
    scripts = tmp_path / '04 台本と補足資料' / '★台本テキスト'
    audio.mkdir()
    scripts.mkdir(parents=True)
    lines = []
    for i in range(1, count + 1):
        name = f'VOICEACTRESS100_{i:03}'
        wave = np.random.default_rng(i).normal(0, .03, 4800 * i).astype('float32')
        sf.write(audio / f'{name}.wav', wave, 48000, subtype='FLOAT')
        lines.append(f'{name}:発話{i}')
    (scripts / '01 補足なし台本（JSUTコーパス・JVSコーパス版）.txt').write_text(
        '\n'.join(lines), encoding='utf-8-sig')
    return tmp_path


def test_inventory_matches_audio_and_script_and_records_signal_quality(tmp_path):
    clips = inspect_corpus(corpus(tmp_path), expected_count=4)
    assert len(clips) == 4
    assert clips[0].text == '発話1'
    assert clips[0].sample_rate == 48000
    assert clips[0].seconds == pytest.approx(.1)
    assert 0 < clips[0].rms < clips[0].peak < 1
    assert clips[0].sha256 == hashlib.sha256(clips[0].path.read_bytes()).hexdigest()


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'extra'])
def test_bad_transcript_is_rejected(tmp_path, problem):
    root = corpus(tmp_path)
    script = next(root.rglob('01 補足なし*.txt'))
    lines = script.read_text(encoding='utf-8-sig').splitlines()
    if problem == 'missing':
        lines.pop()
    elif problem == 'duplicate':
        lines.append(lines[0])
    else:
        lines.append('VOICEACTRESS100_999:余分')
    script.write_text('\n'.join(lines), encoding='utf-8')
    with pytest.raises(ValueError):
        inspect_corpus(root, expected_count=4)


def clips(count=100):
    from pathlib import Path
    return [Clip(f'VOICEACTRESS100_{i:03}', Path(f'{i}.wav'), '声', 24000,
                 2.5 if i == 0 else 4 + i / 100, str(i), .3, .1, .01)
            for i in range(count)]


def test_split_is_deterministic_order_independent_and_reference_is_separate():
    rows = clips()
    first = split_corpus(rows, seed=20261114)
    assert first == split_corpus(list(reversed(rows)), seed=20261114)
    assert {k: len(v) for k, v in first.items()} == {'train': 80, 'validation': 10, 'test': 8, 'reference': 2}
    assert all(next(c for c in rows if c.id == i).seconds >= 3 for i in first['reference'])
    validate_splits(first, rows)
    assert first != split_corpus(rows, seed=8)


def test_leakage_including_processed_variant_id_is_rejected():
    rows = clips()
    splits = split_corpus(rows)
    splits['test'][0] = splits['train'][0]
    with pytest.raises(ValueError, match='overlap|duplicate'):
        validate_splits(splits, rows)


def test_duplicated_source_and_too_short_references_are_rejected():
    rows = clips()
    with pytest.raises(ValueError):
        split_corpus(rows + [rows[0]])
    with pytest.raises(ValueError):
        split_corpus([replace(c, seconds=2) for c in rows])


def test_nonfinite_audio_is_rejected(tmp_path):
    root = corpus(tmp_path)
    sf.write(next(root.rglob('*.wav')), np.full(4800, np.nan), 48000, subtype='FLOAT')
    with pytest.raises(ValueError, match='finite'):
        inspect_corpus(root, expected_count=4)
