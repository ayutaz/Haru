"""Orthographic labels for the pinned hiragana CTC model, not phonetic truth.

Match the model author's MeCab wakati + pykakasi J/K->H preparation:
https://huggingface.co/vumichien/wav2vec2-large-xlsr-japanese-hiragana
"""
import re
import warnings


def ctc_labels(text):
    import MeCab
    import pykakasi
    if not isinstance(text,str) or not text.strip():
        raise ValueError('CTC labels need nonempty Japanese text')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        kana=pykakasi.kakasi()
        kana.setMode('J','H');kana.setMode('K','H');kana.setMode('r','Hepburn')
        converted=kana.getConverter().do(MeCab.Tagger('-Owakati').parse(text).strip())
    # Remove punctuation before final strip; a trailing delimiter otherwise has
    # no acoustic evidence and invalidates even a correctly aligned sentence.
    return re.sub('[,、。．「」…？・]','',converted).strip()
