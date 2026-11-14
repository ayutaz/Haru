def test_labels_match_ctc_training_orthography_instead_of_phonetic_g2p():
    from haru_training.ctc_labels import ctc_labels
    assert ctc_labels('一方通行の話です。') == 'いっぽう つうこう の はなし です'
    assert ctc_labels('私は元気です。') == 'わたし は げんき です'
    assert not ctc_labels('こんにちは。').endswith(' ')
