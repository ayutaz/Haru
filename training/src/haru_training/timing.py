"""Acoustic activity timing; no semantic reply or speaker-attribution verdict."""

import math

import numpy as np


def _runs(mask):
    edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def activity_timing(model, user, *, hop_seconds):
    """Measure aligned binary VAD masks, without turning silence into latency 0.

    User offsets are acoustic run boundaries, not annotated conversational turns.
    An already active model is kept distinct from a new onset after an offset.
    No onset is called a semantic answer, and overlap is not called interruption.
    """
    model, user = np.asarray(model), np.asarray(user)
    if (model.ndim != 1 or model.shape != user.shape or not model.size
            or model.dtype != np.bool_ or user.dtype != np.bool_
            or isinstance(hop_seconds, bool) or not math.isfinite(hop_seconds)
            or hop_seconds <= 0):
        raise ValueError('timing requires aligned nonempty boolean masks and a positive finite hop')
    model_runs, user_runs = _runs(model), _runs(user)
    starts = [int(start) for start, _ in model_runs]
    model_count, user_count = int(model.sum()), int(user.sum())
    overlap = int((model & user).sum())
    offsets = []
    for index, (_, end) in enumerate(user_runs):
        next_user = int(user_runs[index + 1][0]) if index + 1 < len(user_runs) else len(user)
        onset = next((start for start in starts if end <= start < next_user), None)
        if end == len(user):
            status = 'no_following_audio'
        elif model[end - 1] and model[end]:
            status = 'already_speaking'
        elif onset is not None:
            status = 'new_model_onset'
        else:
            status = 'no_new_onset_before_next_user'
        offsets.append({'user_offset_seconds': float(end * hop_seconds), 'status': status,
                        'next_model_onset_delay_seconds': float((onset - end) * hop_seconds)
                        if status == 'new_model_onset' else None})
    return {'duration_seconds': float(len(model) * hop_seconds),
            'model_speech_seconds': float(model_count * hop_seconds),
            'user_speech_seconds': float(user_count * hop_seconds),
            'overlap_seconds': float(overlap * hop_seconds),
            'fraction_model_speech_overlapping_user': overlap / model_count if model_count else None,
            'fraction_user_speech_overlapped': overlap / user_count if user_count else None,
            'first_model_speech_seconds': float(starts[0] * hop_seconds) if starts else None,
            'model_onsets': len(model_runs), 'user_offsets': offsets,
            'scope': 'VAD activity only; no semantic response or interruption judgment'}
