"""Explicit Vast API operations. Importing or constructing a client is inert."""

from pathlib import Path
import json
import math
import urllib.error
import urllib.request


def read_api_key(path: Path) -> str:
    try:
        lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    except OSError:
        raise ValueError('API key file is unavailable') from None
    values = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, separator, value = line.partition('=')
        if separator:
            values[key.strip()] = value.strip().strip('\"\'')
    for name in ('VASTAI_API', 'VASTAI_API_KEY', 'VAST_API_KEY'):
        if values.get(name):
            return values[name]
    raise ValueError('Vast API key is not configured')


class VastClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('API key is empty')
        self._api_key = api_key

    def __repr__(self):
        return 'VastClient(api_key=<redacted>)'

    def request(self, method: str, endpoint: str, payload=None):
        if not endpoint.startswith('/') or '://' in endpoint or '?' in endpoint:
            raise ValueError('invalid API endpoint')
        request = urllib.request.Request('https://console.vast.ai/api/v0' + endpoint,
            data=None if payload is None else json.dumps(payload).encode(), method=method,
            headers={'Authorization': 'Bearer ' + self._api_key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Vast API HTTP {error.code}') from None
        except (OSError, ValueError):
            raise RuntimeError('Vast API request failed; response omitted') from None
        if isinstance(data, dict) and data.get('success') is False:
            raise RuntimeError('Vast API operation was rejected; response omitted')
        return data

    def instances(self):
        return self.request('GET', '/instances/')

    def offers(self, *, minimum_vram=24000, maximum_hourly=1.0):
        return self.request('POST', '/bundles/', {
            'verified': {'eq': True}, 'rentable': {'eq': True}, 'rented': {'eq': False},
            'num_gpus': {'eq': 1}, 'gpu_ram': {'gte': minimum_vram},
            'cuda_max_good': {'gte': 12.8}, 'dph_total': {'lte': maximum_hourly},
            'reliability': {'gte': .98}, 'disk_space': {'gte': 60},
            'type': 'on-demand', 'limit': 20, 'order': [['dph_total', 'asc']]})

    def create(self, offer_id: int, *, image: str, hourly_price: float, hours: float,
               max_dollars: float, label: str, disk: int = 60) -> int:
        values = (hourly_price, hours, max_dollars)
        if (not all(math.isfinite(v) and v > 0 for v in values) or max_dollars > 100
                or hours > 2 or hourly_price * hours > max_dollars or offer_id <= 0 or not label):
            raise ValueError('rental does not fit explicit execution limits')
        data = self.request('PUT', f'/asks/{int(offer_id)}/', {
            'client_id': 'me', 'image': image, 'disk': disk, 'label': label,
            'runtype': 'ssh', 'target_state': 'running', 'cancel_unavail': True})
        if not data.get('success') or not isinstance(data.get('new_contract'), int):
            raise RuntimeError('Vast did not confirm an instance ID; reconcile before retrying')
        return data['new_contract']

    def attach_ssh(self, instance_id: int, public_key: str):
        return self.request('POST', f'/instances/{int(instance_id)}/ssh/', {'ssh_key': public_key})

    def destroy(self, instance_id: int, *, expected_id: int):
        if instance_id <= 0 or instance_id != expected_id:
            raise ValueError('instance ID does not match this run')
        return self.request('DELETE', f'/instances/{int(instance_id)}/')

    def stop(self, instance_id: int, *, expected_id: int):
        if instance_id <= 0 or instance_id != expected_id:
            raise ValueError('instance ID does not match this run')
        return self.request('PUT', f'/instances/{int(instance_id)}/', {'state': 'stopped'})


def enforce_deadline(client: VastClient, run: dict, *, now: float) -> bool:
    if not 0 < run['max_hours'] <= 2 or run['instance_id'] <= 0:
        raise ValueError('invalid owned run or deadline')
    if now < run['requested_at'] + run['max_hours'] * 3600:
        return False
    client.stop(run['instance_id'], expected_id=run['instance_id'])
    return True
