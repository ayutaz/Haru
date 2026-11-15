import pytest


def test_unavailable_labiovelar_phone_is_expanded_with_explicit_provenance():
    from haru_training.alignment_phones import approximate_julius_phones
    native=['kw','i','i','N','z','u','a','b','e','ny','u','u']
    expanded,substitutions=approximate_julius_phones(native)
    assert expanded==['k','w','i','i','N','z','u','a','b','e','ny','u','u']
    assert substitutions=={'kw':['k','w']} and native[0]=='kw'


def test_existing_v_approximation_is_reported_and_other_phones_are_not_guessed():
    from haru_training.alignment_phones import approximate_julius_phones
    expanded,substitutions=approximate_julius_phones(['v','a','ky','o','unknown'])
    assert expanded==['b','a','ky','o','unknown'] and substitutions=={'v':['b']}


def test_alignment_phone_approximation_rejects_missing_or_malformed_labels():
    from haru_training.alignment_phones import approximate_julius_phones
    for native in [[],['k',None],['k','invalid label']]:
        with pytest.raises(ValueError):approximate_julius_phones(native)
