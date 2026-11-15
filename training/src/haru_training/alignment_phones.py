"""Explicit approximations for the pinned Julius monophone alignment model.

These labels estimate timing only. Neither the authentic WAV nor its transcript
is rewritten. Unsupported phones outside these two documented cases still fail
in the aligner rather than being silently replaced.
"""
import re


def approximate_julius_phones(phones):
    native=list(phones)
    if not native or any(not isinstance(p,str) or not re.fullmatch('[a-zA-Z]+',p) for p in native):
        raise ValueError('alignment needs nonempty explicit phone labels')
    aliases={'v':['b'],'kw':['k','w']}
    used={p:aliases[p].copy() for p in aliases if p in native}
    return [unit for phone in native for unit in used.get(phone,[phone])],used
