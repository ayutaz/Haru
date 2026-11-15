"""Combine the two teacher listening judgments with audio-bound screening.

This is a listening gate only. Fine timing, codec reconstruction and the
student training path require their own evidence, even after positive answers.
"""
from .listening_review import decode_review


def assess_teacher_listening(presentation, responses, condition_key, automatic):
    result=decode_review(presentation,responses,condition_key)
    ids=[case['id'] for case in result['cases']]
    try:
        rows=automatic['cases'];auto_ids=[row['id'] for row in rows]
        if (len(ids)!=2 or not presentation.get('include_voice_match')
                or automatic['expected_count']!=2 or automatic['evaluated_count']!=2
                or len(auto_ids)!=2 or len(set(auto_ids))!=2 or set(auto_ids)!=set(ids)):
            raise ValueError('teacher gate requires both frozen cases and complete screening')
        keyed={row['id']:row for row in condition_key['cases']}
        screened={row['id']:row for row in rows}
        for row in result['cases']:
            auto=screened[row['id']]
            if (auto['decision'] not in ('candidate_for_listening','no_metric_gain','hold','reject')
                    or any(auto['artifacts'][side]['response.wav']!=keyed[row['id']]['audio_sha256'][side]
                           for side in ('baseline','candidate'))):
                raise ValueError('teacher screening belongs to different audio or conditions')
            reasons=[];status='supported'
            voice=row['voice_preference']
            if voice=='差が分からない' or row['preferred_condition']=='baseline':
                reasons.append('no_perceived_voice_gain');status='failed'
            elif row['preferred_condition']!='candidate':
                reasons.append('voice_judgment_unknown');status='hold'
            content=row['content_by_condition']['candidate']
            if content in ('一部不明','意味が分からない'):
                reasons.append('teacher_not_intelligible');status='failed'
            elif content!='聞き取れる':
                reasons.append('teacher_intelligibility_unknown')
                if status!='failed':status='hold'
            if row['content_by_condition']['baseline']!='聞き取れる':
                reasons.append('source_intelligibility_unconfirmed')
                if status!='failed':status='hold'
            if auto['decision']!='candidate_for_listening':
                reasons.append('automatic_screening_not_supported')
                if auto['decision']=='reject':status='failed'
                elif status!='failed':status='hold'
            row.update(status=status,reasons=reasons,automatic_decision=auto['decision'])
    except (KeyError,TypeError) as error:
        raise ValueError('incomplete teacher listening or screening evidence') from error
    statuses=[row['status'] for row in result['cases']]
    decision=('failed' if 'failed' in statuses else 'hold' if 'hold' in statuses
              else 'supported_for_teacher_preflight')
    result.update(decision=decision,teacher_listening_supported=decision=='supported_for_teacher_preflight',
        candidate_intelligible_count=sum(row['content_by_condition']['candidate']=='聞き取れる' for row in result['cases']),
        fine_timing_supported=False,codec_roundtrip_supported=False,training_authorized=False,
        scope='Two-case human teacher gate; not student adoption, timing validation, or training approval')
    return result
