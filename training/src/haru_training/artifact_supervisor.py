"""Bound a child job, then publish recoverable artifacts atomically.

The job has its own process group. Timing it out cannot kill the supervisor
while it archives evidence. The exit marker is published last, so collectors
never mistake an unfinished archive for a completed delivery.
"""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tarfile


def _atomic_json(path, value):
    pending = path.with_name(path.name + '.tmp')
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def run_and_package(command, *, root, timeout_seconds):
    root = Path(root).resolve()
    if (not isinstance(command, (list, tuple)) or not command
            or not all(isinstance(arg, str) and arg for arg in command)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('an explicit command and positive finite timeout are required')
    root.mkdir(parents=True, exist_ok=True)
    for name in ['delivery.tar.gz', 'delivery.tar.gz.tmp', 'execution-status.json',
                 'recovery-ready.json', 'exit.code']:
        if (root / name).exists():
            raise FileExistsError('never overwrite existing job artifacts')
    process = subprocess.Popen(command, cwd=root, start_new_session=os.name != 'nt')
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == 'nt':
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    status = dict(returncode=process.returncode, timed_out=timed_out,
                  timeout_seconds=timeout_seconds, command_finished=True)
    _atomic_json(root / 'execution-status.json', status)
    pending = root / 'delivery.tar.gz.tmp'
    with tarfile.open(pending, 'w:gz') as archive:
        for name in ['evidence', 'execution-status.json', 'run.log']:
            if (root / name).exists():
                archive.add(root / name, arcname=name)
    with pending.open('r+b') as stream:
        os.fsync(stream.fileno())
    os.replace(pending, root / 'delivery.tar.gz')
    digest = sha256()
    with (root / 'delivery.tar.gz').open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    ready = dict(status, archive_sha256=digest.hexdigest())
    _atomic_json(root / 'recovery-ready.json', ready)
    marker = root / 'exit.code.tmp'
    with marker.open('x', encoding='ascii') as stream:
        stream.write(str(process.returncode) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(marker, root / 'exit.code')
    return ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    result = run_and_package(command, root=args.root, timeout_seconds=args.timeout)
    print(json.dumps(result), flush=True)
    return 124 if result['timed_out'] else (0 if result['returncode'] == 0 else 1)


if __name__ == '__main__':
    raise SystemExit(main())
