"""Hash-bound, offline evaluation of saved replay pairs; never loads a model.

Decisions select/stop experiments, not certify perceived identity or semantic
correctness. Thresholds are explicit provisional screening rules. Cached ASR
and embeddings must have been computed independently of training losses.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .speech_quality import transcript_comparison
from .voice_metrics import speaker_cosine


@dataclass(frozen=True)
class EvaluationPolicy:
    minimum_reference_characters: int = 12
    maximum_baseline_cer: float = .35
    maximum_cer_increase: float = .15
    minimum_added_character_errors: int = 3
    minimum_rms: float = 1e-5
    maximum_clipped_fraction: float = .01
    minimum_speaker_gain: float = .02
    minimum_calibrated_gain_fraction: float = .02
    minimum_references_per_group: int = 3
    minimum_independent_encoders: int = 2
    minimum_repeats: int = 3
    maximum_level_only_residual: float = 1e-6
    calibration_status: str = 'provisional_not_perceptually_validated'


class EvidenceError(ValueError):
    """An unusable observation must remain unknown, not score zero."""


def _json(path):
    def invalid_constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid_constant)


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _wave(path, policy):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    reasons = []
    if not len(audio):
        return {'sample_rate': rate, 'samples': 0}, ['empty_audio']
    if audio.shape[1] != 1:
        reasons.append('nonmono_audio')
    if not np.isfinite(audio).all():
        return {'sample_rate': rate, 'samples': len(audio)}, reasons + ['nonfinite_audio']
    values = audio.astype(np.float64)
    rms = float(np.sqrt(np.mean(values ** 2)))
    clipped = float(np.mean(np.abs(values) >= .999))
    if rms < policy.minimum_rms:
        reasons.append('silent_audio')
    if clipped > policy.maximum_clipped_fraction or np.max(np.abs(values)) > 1.:
        reasons.append('clipped_audio')
    return dict(sample_rate=rate, samples=len(audio), duration_seconds=len(audio) / rate,
                rms=rms, clipped_fraction=clipped), reasons


def _replay(entry, root, policy):
    folder = root / entry['directory']
    files = _json(folder / 'manifest.json')['files']
    hashes = {}
    for name in ('response.wav', 'response.json'):
        actual = _digest(folder / name)
        if not _hash(files.get(name)) or files[name] != actual:
            raise EvidenceError('artifact_hash_mismatch')
        hashes[name] = actual
    metadata = _json(folder / 'response.json')
    metrics, reasons = _wave(folder / 'response.wav', policy)
    if metadata.get('sample_rate') != metrics['sample_rate']:
        raise EvidenceError('sample_rate_metadata_mismatch')
    return metadata, hashes, metrics, reasons


def _conditions(metadata):
    provenance = metadata.get('provenance', {})
    input_sha = provenance.get('input_sha256', provenance.get('question', {}).get('wav_sha256'))
    config = metadata.get('config', {})
    if (not _hash(input_sha) or type(config.get('seed')) is not int
            or not isinstance(metadata.get('system_prompt'), str)):
        raise EvidenceError('unpaired_conditions')
    return input_sha, config, metadata['system_prompt']


def _level_relation(base_path, candidate_path, policy):
    """Detect a sample-aligned scalar gain, not a general voice-equivalence test."""
    left,lr=sf.read(base_path,dtype='float64',always_2d=True)
    right,rr=sf.read(candidate_path,dtype='float64',always_2d=True)
    if lr!=rr or left.shape!=right.shape:
        return dict(status='not_sample_aligned',only_level_change=False)
    energy=float(np.sum(left*left))
    if energy<=0:
        return dict(status='unknown',only_level_change=False)
    gain=float(np.sum(left*right)/energy)
    residual=float(np.sqrt(np.sum((right-gain*left)**2)/max(float(np.sum(right*right)),1e-30)))
    return dict(status='measured',gain=gain,relative_residual=residual,
        only_level_change=gain>0 and residual<=policy.maximum_level_only_residual,
        scope='Aligned scalar-gain diagnostic; not perceived similarity')


def _content(base, candidate, base_meta, candidate_meta, hashes, policy):
    reasons = []
    for cache, meta, wav_hash in zip((base, candidate), (base_meta, candidate_meta), hashes):
        if not isinstance(cache, dict):
            raise EvidenceError('invalid_asr_evidence')
        if cache.get('wav_sha256') != wav_hash:
            raise EvidenceError('asr_audio_mismatch')
        if cache.get('generated_text') != meta.get('text'):
            raise EvidenceError('asr_reference_mismatch')
        if not isinstance(cache.get('recognized_text'), str) or not isinstance(meta.get('text'), str):
            raise EvidenceError('invalid_asr_text')
    if base_meta['text'] != candidate_meta['text']:
        raise EvidenceError('different_generated_text')
    specs = [tuple(cache.get(k) for k in ('model', 'revision', 'settings')) for cache in (base, candidate)]
    comparable = specs[0] == specs[1] and all(isinstance(x, str) and x for x in specs[0][:2])
    if not comparable:
        reasons.append('asr_spec_mismatch')
    if not isinstance(specs[0][2], dict) or not specs[0][2]:
        reasons.append('asr_settings_unrecorded')
    settings = specs[0][2] if isinstance(specs[0][2], dict) else {}
    if any(settings.get(key) for key in ('initial_prompt', 'prefix', 'hotwords')):
        reasons.append('asr_prompt_leakage')
        comparable = False
    comparisons = [transcript_comparison(base_meta['text'], cache['recognized_text'])
                   for cache in (base, candidate)]
    left, right = [row['character_error_rate'] for row in comparisons]
    count = comparisons[0]['reference_characters']
    result = dict(baseline_cer=left, candidate_cer=right, reference_characters=count,
                  asr_model=base.get('model'), asr_revision=base.get('revision'),
                  scope='ASR agreement only; no semantic or perceived intelligibility guarantee')
    if count < policy.minimum_reference_characters:
        reasons.append('asr_reference_too_short')
    elif left is not None and right is not None:
        # CER originates from integer edit counts. Subtract those counts so a
        # boundary such as 7/20 - 4/20 does not round below the 0.15 threshold.
        added_errors = round(right * count) - round(left * count)
        increase = added_errors / count
        result.update(cer_increase=increase, added_character_errors=added_errors)
        if left > policy.maximum_baseline_cer:
            reasons.append('baseline_asr_risk')
        if (comparable and increase >= policy.maximum_cer_increase
                and added_errors >= policy.minimum_added_character_errors):
            reasons.append('asr_regression')
    result['status'] = 'asr_regression' if 'asr_regression' in reasons else (
        'unknown' if reasons else 'no_detected_regression')
    return result, reasons


def _panel(panel, hashes, root, policy):
    if not all(isinstance(panel.get(k), str) and panel[k] for k in ('encoder', 'revision', 'target_speaker')):
        raise EvidenceError('speaker_spec_missing')
    if not _hash(panel.get('preprocessing_sha256')):
        raise EvidenceError('speaker_spec_missing')
    references = panel['references']
    source_ids, file_hashes = set(), set()
    groups = [[], []]

    def check_reference(ref):
        actual = _digest(root / ref['path'])
        if ref.get('wav_sha256') != actual:
            raise EvidenceError('speaker_reference_hash_mismatch')
        if actual in hashes or actual in file_hashes or ref['source_id'] in source_ids:
            raise EvidenceError('speaker_reference_leakage')
        if not ref['source_id'] or not ref['speaker_id']:
            raise EvidenceError('speaker_reference_identity_missing')
        file_hashes.add(actual)
        source_ids.add(ref['source_id'])

    for ref in references:
        check_reference(ref)
        groups[ref['speaker_id'] != panel['target_speaker']].append(ref['embedding'])
    target, negatives = groups
    if min(map(len, groups)) < policy.minimum_references_per_group:
        raise EvidenceError('speaker_references_insufficient')
    # Self matches are excluded. Genuine/negative ranges define this encoder's
    # reference domain; scores are never interpreted as probabilities.
    score = lambda vector: float(np.mean([speaker_cosine(vector, ref) for ref in target]))
    positive = [float(np.mean([speaker_cosine(ref, other) for j, other in enumerate(target) if i != j]))
                for i, ref in enumerate(target)]
    calibrated = 'genuine_controls' in panel
    if calibrated:
        controls = panel['genuine_controls']
        if not isinstance(controls, list) or len(controls) < policy.minimum_references_per_group:
            raise EvidenceError('speaker_genuine_controls_insufficient')
        for ref in controls:
            check_reference(ref)
            if ref['speaker_id'] != panel['target_speaker']:
                raise EvidenceError('speaker_genuine_control_identity_mismatch')
            positive.append(score(ref['embedding']))
    negative = [score(vector) for vector in negatives]
    lower, upper, negative_upper = min(positive), max(positive), max(negative)
    if negative_upper >= lower:
        raise EvidenceError('speaker_calibration_overlap')
    for side, expected in zip(('baseline', 'candidate'), hashes):
        if panel[side].get('wav_sha256') != expected:
            raise EvidenceError('speaker_audio_mismatch')
    baseline, candidate = [score(panel[side]['embedding']) for side in ('baseline', 'candidate')]
    repeats = panel.get('baseline_repeats', [])
    if len(repeats) < policy.minimum_repeats:
        raise EvidenceError('speaker_repeatability_missing')
    if any(row.get('wav_sha256') != hashes[0] for row in repeats):
        raise EvidenceError('speaker_repeatability_audio_mismatch')
    noise = max(abs(score(row['embedding']) - baseline) for row in repeats)
    gain = candidate - baseline
    calibration_gap = lower - negative_upper
    minimum_gain = (policy.minimum_calibrated_gain_fraction * calibration_gap
                    if calibrated else policy.minimum_speaker_gain)
    required = max(minimum_gain, 2 * noise)
    boundary = (lower + negative_upper) / 2 if calibrated else lower
    reasons = []
    # A maximum of a few reference scores is not a valid upper bound for an
    # unseen genuine recording. With cross-utterance controls, retain that
    # observation as a diagnostic rather than automatically rejecting it.
    if not calibrated and candidate > upper + 1e-6:
        reasons.append('speaker_outside_calibration')
    if gain > minimum_gain and gain <= required:
        reasons.append('speaker_gain_within_noise')
    if candidate < boundary and gain > required:
        reasons.append('speaker_below_target_range')
    return dict(encoder=panel['encoder'], revision=panel['revision'],
                preprocessing_sha256=panel['preprocessing_sha256'],
                baseline_cosine=baseline, candidate_cosine=candidate, gain=gain,
                required_gain=required, repeatability_noise=noise,
                genuine_range=[lower, upper], negative_upper=negative_upper,
                gain_in_calibration_units=gain / calibration_gap,
                target_boundary=boundary,
                calibration_method='cross_utterance_controls' if calibrated else 'reference_only',
                outside_observed_genuine_range=not lower <= candidate <= upper,
                control_audit=panel.get('control_audit'),
                metric_gain=not reasons and gain > required), reasons


def _speaker(panels, hashes, root, policy):
    if not isinstance(panels, list):
        return {'status': 'unknown', 'encoders': []}, ['invalid_speaker_evidence']
    if not panels:
        return {'status': 'unknown', 'encoders': []}, ['speaker_evidence_missing']
    rows, reasons, names = [], [], set()
    for panel in panels:
        if not isinstance(panel, dict) or not isinstance(panel.get('encoder'), str):
            reasons.append('invalid_speaker_evidence')
            continue
        if panel.get('encoder') in names:
            reasons.append('independent_encoders_missing')
        names.add(panel.get('encoder'))
        try:
            row, issues = _panel(panel, hashes, root, policy)
            rows.append(row)
            reasons.extend(issues)
        except EvidenceError as error:
            reasons.append(str(error))
        except (KeyError, TypeError, ValueError, OSError, sf.LibsndfileError):
            reasons.append('invalid_speaker_evidence')
    if len(names) < policy.minimum_independent_encoders:
        reasons.append('independent_encoders_missing')
    # Disagreement is retained, never averaged into a single high score.
    if len(rows) > 1 and any(row['metric_gain'] for row in rows) and not all(row['metric_gain'] for row in rows):
        reasons.append('speaker_encoder_disagreement')
    if reasons:
        status = 'unknown'
    else:
        status = 'metric_gain' if all(row['metric_gain'] for row in rows) else 'no_metric_gain'
    return {'status': status, 'encoders': rows}, reasons


def _case(case, root, policy, *, native_teacher_rates=False):
    result = dict(id=case['id'], decision='hold', reasons=[], voice_similarity_confirmed=False)
    try:
        base, candidate = [_replay(case[side], root, policy) for side in ('baseline', 'candidate')]
        result['artifacts'] = {'baseline': base[1], 'candidate': candidate[1]}
        result['waveform'] = {'baseline': base[2], 'candidate': candidate[2]}
        if candidate[3]:
            result.update(decision='reject', reasons=candidate[3])
            return result
        if base[3]:
            raise EvidenceError('baseline_waveform_invalid')
        if not native_teacher_rates and base[2]['sample_rate'] != candidate[2]['sample_rate']:
            raise EvidenceError('unpaired_audio_format')
        if _conditions(base[0]) != _conditions(candidate[0]):
            raise EvidenceError('unpaired_conditions')
        hashes = [base[1]['response.wav'], candidate[1]['response.wav']]
        result['level_relation']=_level_relation(root/case['baseline']['directory']/'response.wav',
            root/case['candidate']['directory']/'response.wav',policy)
        content, reasons = _content(case['baseline']['asr'], case['candidate']['asr'], base[0], candidate[0], hashes, policy)
        if result['level_relation']['only_level_change']:
            reasons.append('waveform_only_level_change')
        result['content'] = content
        speaker, issues = _speaker(case.get('speaker_panels', []), hashes, root, policy)
        result['speaker'] = speaker
        result['reasons'] = sorted(set(reasons + issues))
        if 'asr_regression' in reasons:
            result['decision'] = 'reject'
        elif result['reasons']:
            result['decision'] = 'hold'
        elif speaker['status'] == 'metric_gain':
            result['decision'] = 'candidate_for_listening'
        else:
            result['decision'] = 'no_metric_gain'
    except EvidenceError as error:
        result['reasons'].append(str(error))
    except (KeyError, TypeError, ValueError, OSError, sf.LibsndfileError):
        result['reasons'].append('invalid_or_missing_evidence')
    return result


def evaluate_manifest(manifest, *, root, policy=None):
    """Keep the frozen expected case set, including missing/invalid observations."""
    policy = policy or EvaluationPolicy()
    if manifest.get('version') != 1:
        raise ValueError('unsupported evaluation manifest version')
    expected, cases = manifest['expected_cases'], manifest['cases']
    if not expected or any(not isinstance(item, str) or not item for item in expected):
        raise ValueError('expected_cases must contain nonempty case IDs')
    ids = [case['id'] for case in cases]
    if len(set(expected)) != len(expected) or len(set(ids)) != len(ids):
        raise ValueError('duplicate case IDs')
    if set(ids) - set(expected):
        raise ValueError('unplanned case IDs')
    lookup = {case['id']: case for case in cases}
    results = [_case(lookup[key], Path(root), policy) if key in lookup else dict(
        id=key, decision='hold', reasons=['missing_case'], voice_similarity_confirmed=False)
        for key in expected]
    priority = ('reject', 'hold', 'no_metric_gain', 'candidate_for_listening')
    counts = {status: sum(row['decision'] == status for row in results) for status in priority}
    return dict(version=1, decision=next(status for status in priority if counts[status]),
                counts=counts, expected_count=len(expected), evaluated_count=len(cases), cases=results,
                policy=asdict(policy), voice_similarity_confirmed=False,
                limitations=['CER can reflect ASR errors and Japanese spelling differences.',
                             'Cached model/preprocessing provenance is declared by the producer.',
                             'Reference calibration and encoder agreement do not establish perceived identity.',
                             'Semantic coherence is not evaluated. Human labels are not yet calibrated.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.resolve() == args.output.resolve():
        parser.error('output must differ from input')
    report = evaluate_manifest(_json(args.manifest), root=args.manifest.parent)
    report['manifest_sha256'] = _digest(args.manifest)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(report['decision'])
    return {'candidate_for_listening': 0, 'no_metric_gain': 1, 'reject': 2, 'hold': 3}[report['decision']]


if __name__ == '__main__':
    raise SystemExit(main())
