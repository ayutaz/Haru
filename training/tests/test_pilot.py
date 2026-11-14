import pytest

from haru_training.pilot import select_records


def test_pilot_selects_full_length_range_only_from_train_and_validation():
    splits = {'train': ['b', 'a', 'c', 'd'], 'validation': ['v1', 'v2'],
              'test': ['heldout'], 'reference': ['ref']}
    rows = {k: {'frames': n} for k, n in zip(['a', 'b', 'c', 'd', 'v1', 'v2', 'heldout', 'ref'],
                                           [20, 40, 60, 100, 30, 50, 200, 60])}
    result = select_records(splits, rows)
    assert result == {'train': ['a', 'c', 'd'], 'validation': ['v1', 'v2'], 'reference': 'ref'}
    assert set(result['train']).isdisjoint(splits['test']+splits['reference'])


def test_pilot_refuses_split_overlap_or_missing_cache_record():
    splits = {'train': ['a', 'b', 'c'], 'validation': ['v1', 'v2'],
              'test': ['a'], 'reference': ['ref']}
    rows = {k: {'frames': 30} for k in ['a', 'b', 'c', 'v1', 'v2', 'ref']}
    with pytest.raises(ValueError): select_records(splits, rows)
    splits['test'] = ['missing']
    with pytest.raises(ValueError): select_records(splits, rows)
