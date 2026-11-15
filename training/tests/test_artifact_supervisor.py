import json
from pathlib import Path
import sys
import tarfile

import pytest

from haru_training.artifact_supervisor import run_and_package


@pytest.mark.parametrize('returncode', [0, 7])
def test_success_and_failure_preserve_actual_outputs_before_ready_marker(tmp_path, returncode):
    script = (
        "from pathlib import Path; import sys; "
        "Path('evidence').mkdir(); Path('evidence/result.txt').write_text('actual-output'); "
        f"sys.exit({returncode})"
    )
    result = run_and_package([sys.executable, '-c', script], root=tmp_path, timeout_seconds=10)
    assert result['returncode'] == returncode
    assert not result['timed_out']
    assert (tmp_path / 'exit.code').read_text().strip() == str(returncode)
    assert json.loads((tmp_path / 'recovery-ready.json').read_text()) == result
    with tarfile.open(tmp_path / 'delivery.tar.gz') as archive:
        assert archive.extractfile('evidence/result.txt').read() == b'actual-output'
        status = json.load(archive.extractfile('execution-status.json'))
        assert status['returncode'] == returncode
    assert (tmp_path / 'exit.code').stat().st_mtime_ns >= (tmp_path / 'delivery.tar.gz').stat().st_mtime_ns
    assert not list(tmp_path.glob('*.tmp'))


def test_timeout_preserves_partial_evidence_and_is_not_success(tmp_path):
    script = "from pathlib import Path; import time; Path('evidence').mkdir(); Path('evidence/partial').write_text('saved'); time.sleep(30)"
    result = run_and_package([sys.executable, '-c', script], root=tmp_path, timeout_seconds=.5)
    assert result['timed_out']
    assert result['returncode'] != 0
    with tarfile.open(tmp_path / 'delivery.tar.gz') as archive:
        assert archive.extractfile('evidence/partial').read() == b'saved'
        assert json.load(archive.extractfile('execution-status.json'))['timed_out']


def test_archive_failure_never_exposes_ready_or_exit_marker(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        assert not (tmp_path / 'exit.code').exists()
        assert not (tmp_path / 'recovery-ready.json').exists()
        raise OSError('injected archive failure')
    monkeypatch.setattr(tarfile, 'open', fail)
    with pytest.raises(OSError, match='injected'):
        run_and_package([sys.executable, '-c', 'pass'], root=tmp_path, timeout_seconds=10)
    assert not (tmp_path / 'exit.code').exists()
    assert not (tmp_path / 'recovery-ready.json').exists()


def test_never_overwrites_an_existing_delivery(tmp_path):
    (tmp_path / 'delivery.tar.gz').write_bytes(b'previous')
    with pytest.raises(FileExistsError):
        run_and_package([sys.executable, '-c', "raise AssertionError('must not launch')"], root=tmp_path, timeout_seconds=10)
    assert (tmp_path / 'delivery.tar.gz').read_bytes() == b'previous'
