import importlib
import io
import json
from urllib.error import HTTPError

import pytest

from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.vast import VastClient, read_api_key


@pytest.mark.parametrize('kwargs', [dict(mode='all'), dict(lora_rank=0), dict(max_seconds=0),
                                   dict(max_dollars=101), dict(guidance_max=3)])
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_configuration_and_api_client_have_no_import_or_init_side_effects(monkeypatch):
    def fail(*a, **k):
        raise AssertionError('unexpected network request')
    monkeypatch.setattr('urllib.request.urlopen', fail)
    module = importlib.import_module('haru_training.vast')
    importlib.reload(module)
    client = module.VastClient('synthetic-secret')
    assert 'synthetic-secret' not in repr(client)
    assert TrainingConfig().mode == 'head'


def test_key_is_read_only_when_requested_and_never_exposed(tmp_path):
    path = tmp_path / '.env'
    path.write_text('VASTAI_API="synthetic-secret"\n')
    assert read_api_key(path) == 'synthetic-secret'
    with pytest.raises(ValueError):
        read_api_key(tmp_path / 'missing.env')


def test_api_error_is_sanitized_without_echoing_body_or_key(monkeypatch):
    def fail(request, **kwargs):
        assert request.get_header('Authorization') == 'Bearer synthetic-secret'
        raise HTTPError(request.full_url, 401, 'synthetic-secret', {}, io.BytesIO(b'synthetic-secret'))
    monkeypatch.setattr('urllib.request.urlopen', fail)
    with pytest.raises(RuntimeError) as error:
        VastClient('synthetic-secret').instances()
    assert '401' in str(error.value)
    assert 'synthetic-secret' not in str(error.value)


def test_rental_is_explicit_and_budget_bound(monkeypatch):
    calls = []
    class Response(io.BytesIO):
        pass
    def request(req, **kwargs):
        calls.append(req)
        return Response(b'{"success":true,"new_contract":42}')
    monkeypatch.setattr('urllib.request.urlopen', request)
    client = VastClient('synthetic-secret')
    with pytest.raises(ValueError):
        client.create(11, image='test/image:tag', hourly_price=4, hours=2, max_dollars=3, label='test')
    assert not calls
    assert client.create(11, image='test/image:tag', hourly_price=.3, hours=1,
                         max_dollars=3, label='test') == 42
    assert calls[0].method == 'PUT'
    assert calls[0].full_url.endswith('/asks/11/')
    assert json.loads(calls[0].data)['label'] == 'test'
    assert json.loads(calls[0].data)['runtype'] == 'ssh_direc ssh_proxy'


def test_invalid_module_name_cannot_silently_train_nothing():
    import torch
    with pytest.raises(ValueError, match='head'):
        configure_trainable(torch.nn.Linear(2, 2), TrainingConfig())


def test_deadline_guard_stops_only_owned_instance_after_limit(monkeypatch):
    from haru_training.vast import enforce_deadline
    calls = []
    client = VastClient('synthetic-secret')
    monkeypatch.setattr(client, 'request', lambda method, path, payload=None: calls.append((method,path,payload)))
    run = {'instance_id': 42, 'requested_at': 10, 'max_hours': 2}
    assert not enforce_deadline(client, run, now=100)
    assert not calls
    assert enforce_deadline(client, run, now=7211)
    assert calls == [('PUT', '/instances/42/', {'state':'stopped'})]
    with pytest.raises(ValueError):
        client.stop(43, expected_id=42)
