"""Screen hash-bound external VC outputs without certifying a student model.

VCs may emit a native rate different from Haru's. ASR and speaker evidence
still refer to the actual WAV bytes and their pinned inference frontends.
The existing same-decoder evaluator retains its identical-rate requirement.
"""
from pathlib import Path

from .evaluation import (EvidenceError, EvaluationPolicy, _case, _conditions,
                         _hash, _replay, evaluate_manifest)
from .speech_quality import transcript_comparison


def screen_vc_teachers(manifest, conversion, *, root, conversion_sha256, policy=None):
    policy=policy or EvaluationPolicy()
    root=Path(root)
    # Validate the frozen case denominator with the original evaluator first.
    report=evaluate_manifest(manifest,root=root,policy=policy)
    expected=manifest['expected_cases']
    rows=conversion.get('cases',[])
    ids=[row.get('id') for row in rows]
    if (not _hash(conversion_sha256) or conversion.get('expected_cases')!=expected
            or conversion.get('training_started') is not False
            or len(set(ids))!=len(ids) or set(ids)-set(expected)):
        raise ValueError('teacher conversion membership or delivery identity differs')
    converted={row['id']:row for row in rows}
    cases={row['id']:row for row in manifest['cases']}
    results=[]
    for original in report['cases']:
        name=original['id']
        if name not in cases:
            results.append(original)
            continue
        result=dict(id=name,decision='hold',reasons=[],voice_similarity_confirmed=False)
        try:
            if name not in converted:
                raise EvidenceError('teacher_conversion_missing')
            case=cases[name];row=converted[name]
            base,candidate=[_replay(case[side],root,policy) for side in ('baseline','candidate')]
            provenance=candidate[0].get('provenance',{})
            if (not _hash(row.get('output_sha256')) or base[1]['response.wav']!=row.get('source_sha256')
                    or provenance.get('teacher_conversion_sha256')!=conversion_sha256
                    or provenance.get('teacher_original_wav_sha256')!=row['output_sha256']
                    or candidate[0].get('feedback_mode')!='external_vc_teacher_only'):
                raise EvidenceError('teacher_conversion_identity_mismatch')
            if _conditions(base[0])!=_conditions(candidate[0]):
                raise EvidenceError('unpaired_conditions')
            result=_case(case,root,policy,native_teacher_rates=True)
            if 'content' in result and not {'asr_spec_mismatch','asr_settings_unrecorded','asr_prompt_leakage'}&set(result['reasons']):
                source=transcript_comparison(case['baseline']['asr']['recognized_text'],
                                             case['candidate']['asr']['recognized_text'])
                result['source_content']=source
                count=source['reference_characters']
                if count<policy.minimum_reference_characters:
                    result['reasons'].append('source_asr_reference_too_short')
                    if result['decision']!='reject':result['decision']='hold'
                elif (source['character_error_rate']>=policy.maximum_cer_increase
                      and round(source['character_error_rate']*count)>=policy.minimum_added_character_errors):
                    result['reasons'].append('source_asr_regression')
                    result['decision']='reject'
        except EvidenceError as error:
            result['reasons'].append(str(error))
        except (KeyError,TypeError,ValueError,OSError):
            result['reasons'].append('invalid_or_missing_teacher_evidence')
        result.update(timing_alignment_confirmed=False,student_training_authorized=False)
        results.append(result)
    priority=('reject','hold','no_metric_gain','candidate_for_listening')
    counts={status:sum(row['decision']==status for row in results) for status in priority}
    report.update(cases=results,counts=counts,decision=next(status for status in priority if counts[status]),
        scope='Native-rate external VC teacher screening only; no student adoption or training permission',
        student_training_authorized=False,timing_alignment_confirmed=False,
        teacher_conversion_sha256=conversion_sha256)
    report['limitations'] += ['Source ASR agreement is another provisional screen, not a human transcript.',
        'Accepted fine temporal correspondence and human voice/content confirmation are still required.']
    return report
