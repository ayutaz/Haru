"""Offline, presentation-bound listening responses; no identity certification."""
from copy import deepcopy
from hashlib import sha256
import json
import re

CONTENT = ('未確認','聞き取れる','一部不明','意味が分からない','判定できない')
VOICE = ('未確認','A','B','差が分からない','両方似ていない','判定できない')
RELATIVE = ('未確認','A','B','差が分からない','判定できない')
MATCH = ('未確認','A','B','両方','両方似ていない','判定できない')


def _hash(value):
    return sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_presentation(cases, *, reference_sha256, include_voice_match=False):
    cases=deepcopy(cases)
    if (type(include_voice_match) is not bool or not cases or len({c['id'] for c in cases})!=len(cases)
            or any(not isinstance(c['id'],str) or not c['id'] for c in cases)
            or not re.fullmatch('[a-f0-9]{64}',reference_sha256)):
        raise ValueError('invalid listening presentation')
    fields=[]
    for case in cases:
        if set(case['audio_sha256'])!={'A','B'} or any(
                not re.fullmatch('[a-f0-9]{64}',v) for v in case['audio_sha256'].values()):
            raise ValueError('presentation needs both original audio identities')
        for suffix in ('A_content','B_content','voice'):
            kind='voice' if suffix=='voice' else 'content'
            options=(RELATIVE if include_voice_match else VOICE) if kind=='voice' else CONTENT
            fields.append(dict(id=case['id']+':'+suffix,kind=kind,options=list(options)))
        if include_voice_match:
            fields.append(dict(id=case['id']+':voice_match',kind='voice_match',options=list(MATCH)))
    value=dict(version=1,reference_sha256=reference_sha256,cases=cases,fields=fields)
    if include_voice_match:value['include_voice_match']=True
    value['presentation_sha256']=_hash(value)
    return value


def assess_review(presentation, responses):
    """Validate every choice and audio identity before interpreting the review."""
    expected=make_presentation(presentation['cases'],reference_sha256=presentation['reference_sha256'],
        include_voice_match=presentation.get('include_voice_match',False))
    if (expected!=presentation or type(responses.get('version')) is not int
            or responses['version']!=1 or responses.get('presentation_sha256')!=expected['presentation_sha256']):
        raise ValueError('listening presentation identity differs')
    fields={f['id']:f for f in expected['fields']}
    answers=responses.get('answers')
    if not isinstance(answers,dict) or set(answers)!=set(fields) or any(
            value=='未確認' or value not in fields[key]['options'] for key,value in answers.items()):
        raise ValueError('incomplete or invalid listening responses')
    rows=[dict(id=c['id'],A_content=answers[c['id']+':A_content'],
               B_content=answers[c['id']+':B_content'],voice_preference=answers[c['id']+':voice']) for c in expected['cases']]
    if presentation.get('include_voice_match'):
        for row in rows:row['voice_match']=answers[row['id']+':voice_match']
    return dict(presentation_sha256=expected['presentation_sha256'],cases=rows,
        candidate_adopted=False,voice_similarity_confirmed=False,semantic_quality='unknown',
        scope='Raw anonymous preferences only; decode the separate condition key for relative candidate comparison.')


def decode_review(presentation, responses, condition_key):
    """Decode independently recorded relative and absolute judgments, never adopt."""
    result=assess_review(presentation,responses)
    try:
        public={c['id']:c for c in presentation['cases']}
        rows=condition_key['cases']
        if (condition_key['presentation_sha256']!=presentation['presentation_sha256']
                or len(rows)!=len(public) or {r['id'] for r in rows}!=set(public)):
            raise ValueError('condition key belongs to a different presentation')
        keys={r['id']:r for r in rows}
        for row in result['cases']:
            key=keys[row['id']];order=key['order']
            if len(order)!=2 or set(order)!={'baseline','candidate'}:
                raise ValueError('invalid anonymous condition order')
            mapping=dict(zip(('A','B'),order))
            if any(key['audio_sha256'][side]!=public[row['id']]['audio_sha256'][label]
                   for label,side in mapping.items()):
                raise ValueError('condition order and original audio identities differ')
            row['preferred_condition']=mapping.get(row['voice_preference'])
            row['content_by_condition']={side:row[label+'_content'] for label,side in mapping.items()}
            candidate_label=next(label for label,side in mapping.items() if side=='candidate')
            match=row.get('voice_match')
            row['candidate_voice_match']=(match in (candidate_label,'両方')) if match in ('A','B','両方','両方似ていない') else None
        result.update(relative_improvement_count=sum(r['preferred_condition']=='candidate' for r in result['cases']),
            baseline_preferred_count=sum(r['preferred_condition']=='baseline' for r in result['cases']),
            candidate_target_match_count=sum(r['candidate_voice_match'] is True for r in result['cases']),
            absolute_match_unknown_count=sum(r['candidate_voice_match'] is None for r in result['cases']))
    except (KeyError,TypeError) as error:
        raise ValueError('incomplete anonymous condition key') from error
    return result


def review_controls(presentation):
    # The spec is public and contains no A/B condition key or automatic scores.
    encoded=json.dumps(presentation,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    return ('<section><button type="button" id="review-export">回答JSONを保存</button>'
            '<p id="review-status" role="status" aria-live="polite"></p>'
            '<p>回答はこのブラウザーに保存します。完了後は回答JSONも保存してください。外部への送信はありません</p></section>'
            '<script type="application/json" id="review-spec">'+encoded+'</script>'
            '<script id="review-runtime">'+REVIEW_RUNTIME+'</script>')


REVIEW_RUNTIME = r'''
(() => {
  'use strict';
  const spec = JSON.parse(document.getElementById('review-spec').textContent);
  const controls = [...document.querySelectorAll('[data-review-field]')];
  const status = document.getElementById('review-status');
  const key = 'haru-listening:' + spec.presentation_sha256;
  const fields = new Map(spec.fields.map(f => [f.id, f]));
  function valid(record, complete=false) {
    return record?.version === 1 && record.presentation_sha256 === spec.presentation_sha256 &&
      record.answers && Object.keys(record.answers).length === fields.size &&
      Object.entries(record.answers).every(([id, v]) => fields.has(id) && fields.get(id).options.includes(v) && (!complete || v !== '未確認'));
  }
  function collect() {
    return {version:1, presentation_sha256:spec.presentation_sha256,
      answers:Object.fromEntries(controls.map(c => [c.dataset.reviewField,c.value])),
      recorded_at:new Date().toISOString()};
  }
  function save() {
    try {
      localStorage.setItem(key, JSON.stringify(collect()));
      status.textContent = 'このブラウザーに保存しました。完了後は回答JSONを保存してください';
    } catch (_) {
      status.textContent = 'ブラウザーに保存できません。ページを閉じずに回答JSONを保存してください';
    }
  }
  try {
    const cached = localStorage.getItem(key);
    if (cached) {
      const record = JSON.parse(cached);
      if (!valid(record)) throw Error('identity');
      controls.forEach(c => {c.value=record.answers[c.dataset.reviewField]});
      status.textContent = '保存済みの回答を復元しました';
    } else status.textContent = '未回答です。選択すると自動保存します';
  } catch (_) {
    status.textContent = '保存できません、または保存済み回答がこのページと一致しません。現在の選択から回答JSONを保存してください';
  }
  controls.forEach(c => c.addEventListener('change', save));
  document.getElementById('review-export').addEventListener('click', () => {
    const record = collect();
    if (!valid(record, true)) {status.textContent='未回答の項目があります'; return;}
    const blob = new Blob([JSON.stringify(record,null,2)], {type:'application/json;charset=utf-8'});
    const url=URL.createObjectURL(blob), link=document.createElement('a');
    link.href=url; link.download='haru-listening-'+spec.presentation_sha256.slice(0,12)+'.json';
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent='回答JSONの保存を開始しました。保存先を確認してください';
  });
})();
'''


def main():
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Validate and decode locally saved listening answers; success does not mean voice quality passed.')
    parser.add_argument('presentation',type=Path)
    parser.add_argument('responses',type=Path)
    parser.add_argument('--answer-key',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('do not overwrite a recorded listening result')
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    result=decode_review(read(args.presentation),read(args.responses),read(args.answer_key))
    result['input_file_sha256']={key:sha256(path.read_bytes()).hexdigest() for key,path in (
        ('presentation',args.presentation),('responses',args.responses),('answer_key',args.answer_key))}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('relative_improvement_count','candidate_target_match_count','voice_similarity_confirmed')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
