"""Independent output checks for normal-feedback head adaptation.

The two generated texts may differ. Each WAV is checked against its own text;
this never claims that different sentences should have matching waveforms or
that speaker embeddings certify perceived voice identity.
"""
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from .evaluation import (EvaluationPolicy, EvidenceError, _conditions, _content,
                         _hash, _level_relation, _replay, _speaker)
from .speech_quality import transcript_comparison


def _direct_case(case, root, policy, negatives):
    row = dict(id=case['id'], decision='hold', reasons=[], voice_similarity_confirmed=False)
    try:
        base, candidate = [_replay(case[s], root, policy) for s in ('baseline', 'candidate')]
        row['artifacts'] = dict(baseline=base[1], candidate=candidate[1])
        row['waveform'] = dict(baseline=base[2], candidate=candidate[2])
        if candidate[3]:
            row.update(decision='reject', reasons=candidate[3])
            return row
        if base[3]:
            raise EvidenceError('baseline_waveform_invalid')
        if _conditions(base[0]) != _conditions(candidate[0]):
            raise EvidenceError('unpaired_direct_conditions')
        for metadata in (base[0], candidate[0]):
            if (metadata.get('feedback_mode') != 'generated_audio'
                    or not _hash(metadata.get('provenance', {}).get('reference_sha256'))):
                raise EvidenceError('unpaired_direct_conditions')
        for key in ('reference_sha256', 'base_revision'):
            if (not base[0]['provenance'].get(key)
                    or base[0]['provenance'][key] != candidate[0]['provenance'].get(key)):
                raise EvidenceError('unpaired_direct_conditions')
        if base[2]['sample_rate'] != candidate[2]['sample_rate']:
            raise EvidenceError('unpaired_audio_format')
        hashes = [base[1]['response.wav'], candidate[1]['response.wav']]
        content = dict(same_generated_text=base[0]['text'] == candidate[0]['text'],
                       scope='Each output against its own text; not a same-sentence conversion comparison')
        reasons = []
        for side, entry in zip(('baseline', 'candidate'), (base, candidate)):
            cache = case[side]['asr']
            # Reuse the strict cache/text/settings checks without weakening the
            # legacy two-WAV same-text contract used by codec comparisons.
            _, issues = _content(cache, cache, entry[0], entry[0],
                                 [entry[1]['response.wav']] * 2, policy)
            reasons += [side+'_asr_risk' if reason == 'baseline_asr_risk' else reason for reason in issues]
            content[side] = transcript_comparison(entry[0]['text'], cache['recognized_text'])
        specs = [tuple(case[side]['asr'].get(k) for k in ('model', 'revision', 'settings'))
                 for side in ('baseline', 'candidate')]
        if specs[0] != specs[1]:
            reasons.append('asr_spec_mismatch')
        row['content'] = content
        row['level_relation'] = _level_relation(root/case['baseline']['directory']/'response.wav',
                                               root/case['candidate']['directory']/'response.wav', policy)
        if row['level_relation']['only_level_change']:
            reasons.append('waveform_only_level_change')
        if hashes[1] in negatives:
            reasons.append('known_human_voice_negative')
        speaker, issues = _speaker(case.get('speaker_panels', []), hashes, root, policy)
        row['speaker'] = speaker
        row['reasons'] = sorted(set(reasons + issues))
        if not row['reasons']:
            row['decision'] = 'candidate_for_listening' if speaker['status'] == 'metric_gain' else 'no_metric_gain'
    except EvidenceError as error:
        row['reasons'].append(str(error))
    except (KeyError, TypeError, ValueError, OSError, sf.LibsndfileError):
        row['reasons'].append('invalid_or_missing_evidence')
    return row


def evaluate_direct_manifest(manifest, *, root, policy=None, known_negative_wav_sha256=()):
    policy = policy or EvaluationPolicy()
    if manifest.get('version') != 1 or manifest.get('kind') != 'haru-direct-head-evaluation':
        raise ValueError('not a direct-head evaluation manifest')
    expected, cases = manifest.get('expected_cases', []), manifest.get('cases', [])
    ids = [case['id'] for case in cases]
    if (not expected or len(expected) != len(set(expected)) or len(ids) != len(set(ids))
            or set(ids) - set(expected) or any(not _hash(h) for h in known_negative_wav_sha256)):
        raise ValueError('invalid fixed case set or negative audio identities')
    lookup = {case['id']: case for case in cases}
    rows = [_direct_case(lookup[key], Path(root), policy, set(known_negative_wav_sha256)) if key in lookup else
            dict(id=key, decision='hold', reasons=['missing_case'], voice_similarity_confirmed=False)
            for key in expected]
    priority = ('reject', 'hold', 'no_metric_gain', 'candidate_for_listening')
    counts = {status: sum(row['decision'] == status for row in rows) for status in priority}
    return dict(version=1, kind='haru-direct-head-evaluation',
                decision=next(status for status in priority if counts[status]), counts=counts,
                expected_count=len(expected), evaluated_count=len(cases), cases=rows, policy=asdict(policy),
                voice_similarity_confirmed=False, semantic_quality='unknown',
                limitations=['Different generated texts are checked independently, not treated as aligned speech.',
                             'ASR errors and speaker scores do not establish human intelligibility or voice identity.',
                             'Exact known-negative WAV matches are vetoes, not an unseen-voice classifier.'])
