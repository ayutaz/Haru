"""Human decisions remain bound to the actual anonymous presentation."""
from copy import deepcopy
import json

import pytest


def protocol():
    from haru_training.listening_review import make_presentation
    return make_presentation([{'id':f'dev-{i}', 'audio_sha256':{'A':str(i+1)*64,'B':str(i+4)*64}}
                              for i in range(3)], reference_sha256='a'*64)


def answers(presentation):
    return dict(version=1,presentation_sha256=presentation['presentation_sha256'],
                answers={field['id']:('両方似ていない' if field['kind']=='voice' else '一部不明')
                         for field in presentation['fields']})


def test_recovered_negative_answers_do_not_become_voice_success():
    from haru_training.listening_review import assess_review
    p=protocol()
    result=assess_review(p,answers(p))
    assert len(result['cases'])==3
    assert result['candidate_adopted'] is False
    assert result['voice_similarity_confirmed'] is False
    assert all(row['voice_preference']=='両方似ていない' for row in result['cases'])
    assert result['semantic_quality']=='unknown'


@pytest.mark.parametrize('mutation',['wrong_page','changed_audio','missing','extra','unanswered','invalid','version'])
def test_stale_incomplete_or_wrong_audio_answers_are_not_accepted(mutation):
    from haru_training.listening_review import assess_review
    p=protocol(); a=answers(p)
    first=p['fields'][0]['id']
    if mutation=='wrong_page': a['presentation_sha256']='f'*64
    if mutation=='changed_audio': p['cases'][0]['audio_sha256']['A']='f'*64
    if mutation=='missing': del a['answers'][first]
    if mutation=='extra': a['answers']['unexpected']='聞き取れる'
    if mutation=='unanswered': a['answers'][first]='未確認'
    if mutation=='invalid': a['answers'][first]='perfect'
    if mutation=='version': a['version']=True
    with pytest.raises(ValueError): assess_review(p,a)


def test_relative_preference_alone_never_confirms_perceived_identity():
    from haru_training.listening_review import assess_review
    p=protocol(); a=answers(p)
    for field in p['fields']:
        a['answers'][field['id']]='A' if field['kind']=='voice' else '聞き取れる'
    result=assess_review(p,a)
    assert result['candidate_adopted'] is False
    assert result['voice_similarity_confirmed'] is False


def test_slight_relative_gain_and_neither_matching_target_are_recorded_separately():
    from haru_training.listening_review import make_presentation,assess_review
    original=protocol()
    p=make_presentation(original['cases'],reference_sha256=original['reference_sha256'],include_voice_match=True)
    assert len(p['fields'])==12
    a=dict(version=1,presentation_sha256=p['presentation_sha256'],answers={})
    for field in p['fields']:
        a['answers'][field['id']]={'content':'聞き取れる','voice':'B','voice_match':'両方似ていない'}[field['kind']]
    result=assess_review(p,a)
    assert all(c['voice_preference']=='B' and c['voice_match']=='両方似ていない' for c in result['cases'])
    assert result['voice_similarity_confirmed'] is False
    assert p['presentation_sha256']!=original['presentation_sha256']


def keyed_review():
    from haru_training.listening_review import make_presentation
    original=protocol();p=make_presentation(original['cases'],reference_sha256=original['reference_sha256'],include_voice_match=True)
    a=dict(version=1,presentation_sha256=p['presentation_sha256'],answers={})
    key=dict(presentation_sha256=p['presentation_sha256'],cases=[])
    for index,case in enumerate(p['cases']):
        order=['baseline','candidate'] if index!=1 else ['candidate','baseline']
        key['cases'].append(dict(id=case['id'],order=order,
            audio_sha256={side:case['audio_sha256'][label] for label,side in zip(('A','B'),order)}))
        for label in ('A','B'):a['answers'][case['id']+':'+label+'_content']='聞き取れる'
        a['answers'][case['id']+':voice']=('B' if index==0 else 'A' if index==1 else '差が分からない')
        a['answers'][case['id']+':voice_match']='両方似ていない'
    return p,a,key


def test_decode_reports_relative_gain_without_turning_it_into_absolute_identity():
    from haru_training.listening_review import decode_review
    p,a,key=keyed_review();report=decode_review(p,a,key)
    assert report['relative_improvement_count']==2
    assert report['baseline_preferred_count']==0
    assert report['candidate_target_match_count']==0
    assert report['voice_similarity_confirmed'] is False
    assert report['candidate_adopted'] is False


@pytest.mark.parametrize('mutation',['order','page','case','audio'])
def test_response_decoding_rejects_a_mixed_or_swapped_condition_key(mutation):
    from haru_training.listening_review import decode_review
    p,a,key=keyed_review()
    if mutation=='order':key['cases'][0]['order'].reverse()
    if mutation=='page':key['presentation_sha256']='f'*64
    if mutation=='case':key['cases'][0]['id']='other'
    if mutation=='audio':key['cases'][0]['audio_sha256']['candidate']='f'*64
    with pytest.raises(ValueError):decode_review(p,a,key)


def test_presentation_script_cannot_be_closed_by_case_identifier():
    from haru_training.listening_review import make_presentation, review_controls
    p=make_presentation([{'id':'</script><script>alert(1)</script>', 'audio_sha256':{'A':'b'*64,'B':'c'*64}}],reference_sha256='a'*64)
    html=review_controls(p)
    assert '</script><script>alert' not in html
    assert '回答JSONを保存' in html


def test_browser_save_restore_export_and_storage_failures(tmp_path):
    """Execute the generated script, including events, reload, and export bytes."""
    import shutil
    import subprocess
    from haru_training.listening_review import review_controls
    node=shutil.which('node')
    if not node: pytest.skip('Node.js required for the browser-state contract')
    p=protocol();html=review_controls(p)
    script=html.split('<script id="review-runtime">',1)[1].split('</script>',1)[0]
    spec=json.dumps(p,ensure_ascii=False)
    harness=r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const spec=SPEC, runtime=RUNTIME;
let saved={},blob;
function page(fail=false){
 const controls=spec.fields.map(f=>({dataset:{reviewField:f.id},value:'未確認',listeners:{},addEventListener(k,f){this.listeners[k]=f}}));
 const status={textContent:''},button={addEventListener(k,f){this[k]=f}},events={};
 const ctx={JSON,Date,console,Set,Blob:class {constructor(parts){blob=parts.join('')}},
  URL:{createObjectURL(){return 'blob:test'},revokeObjectURL(){}},
  localStorage:{getItem(k){if(fail)throw Error('denied');return saved[k]??null},setItem(k,v){if(fail)throw Error('quota');saved[k]=v}},
  document:{getElementById(id){return id==='review-spec'?{textContent:JSON.stringify(spec)}:id==='review-status'?status:button},
    querySelectorAll(){return controls},createElement(){return {click(){},remove(){}}},body:{appendChild(){}}},
  addEventListener(k,f){events[k]=f},setTimeout(f){f()}};
 vm.runInNewContext(runtime,ctx); return {controls,status,button,events};
}
let one=page();
one.button.click();assert.match(one.status.textContent,/未回答/);assert.equal(blob,undefined);
one.controls.forEach((c,i)=>{c.value=spec.fields[i].kind==='voice'?'両方似ていない':'一部不明';c.listeners.change()});
assert.ok(Object.keys(saved).length===1);
let two=page();assert.ok(two.controls.every(c=>c.value!=='未確認'));
two.button.click();let result=JSON.parse(blob);assert.equal(result.presentation_sha256,spec.presentation_sha256);assert.equal(Object.keys(result.answers).length,9);
let failed=page(true);assert.match(failed.status.textContent,/保存できません/);
failed.controls.forEach((c,i)=>c.value=spec.fields[i].kind==='voice'?'A':'聞き取れる');failed.button.click();assert.equal(Object.keys(JSON.parse(blob).answers).length,9);
// Stale/tampered saved state must never populate the controls.
saved[Object.keys(saved)[0]]=JSON.stringify({...result,presentation_sha256:'f'.repeat(64)});
let stale=page();assert.ok(stale.controls.every(c=>c.value==='未確認'));assert.match(stale.status.textContent,/一致しません/);
console.log('save/restore/export/failure/stale guards passed');
'''.replace('SPEC',spec).replace('RUNTIME',json.dumps(script))
    file=tmp_path/'review.cjs';file.write_text(harness,encoding='utf-8')
    result=subprocess.run([node,str(file)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
