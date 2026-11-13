"""Read-only corpus inventory and recording-level splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import random

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Clip:
    id: str
    path: Path
    text: str
    sample_rate: int
    seconds: float
    sha256: str
    peak: float
    rms: float
    silence_fraction: float

    def record(self, root: Path) -> dict:
        return {**asdict(self), 'path': self.path.relative_to(root).as_posix()}


def inspect_corpus(root: Path, expected_count: int = 100) -> list[Clip]:
    root = Path(root)
    script = root / '04 台本と補足資料' / '★台本テキスト' / '01 補足なし台本（JSUTコーパス・JVSコーパス版）.txt'
    texts = {}
    for line in script.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(':')
        if not separator or not value.strip() or key in texts:
            raise ValueError('missing text or duplicate transcript ID')
        texts[key] = value.strip()
    paths = sorted((root / '02 WAV（+12dB増幅）').glob('VOICEACTRESS100_*.wav'))
    if len(paths) != expected_count or set(texts) != {p.stem for p in paths}:
        raise ValueError('audio and transcript IDs/counts do not match')
    clips = []
    for path in paths:
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        if len(audio) == 0 or audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError(f'{path.stem}: expected finite, nonempty mono speech')
        signal = audio[:, 0].astype(np.float64)
        clips.append(Clip(path.stem, path, texts[path.stem], rate, len(audio) / rate,
                          sha256(path.read_bytes()).hexdigest(), float(np.abs(signal).max()),
                          float(np.sqrt(np.mean(signal ** 2))), float(np.mean(np.abs(signal) < 1e-4))))
    return clips


def validate_splits(splits: dict[str, list[str]], clips: list[Clip]) -> None:
    by_id = {c.id: c for c in clips}
    ids = [i for group in splits.values() for i in group]
    if len(by_id) != len(clips) or len(ids) != len(set(ids)):
        raise ValueError('duplicate source ID or split overlap')
    if set(splits) != {'train', 'validation', 'test', 'reference'} or set(ids) != set(by_id):
        raise ValueError('split names or coverage do not match corpus')
    if not splits['reference'] or any(by_id[i].seconds < 3 for i in splits['reference']):
        raise ValueError('reference speech must be at least 3 seconds')
    seen_hashes = {}
    for group, members in splits.items():
        for i in members:
            previous = seen_hashes.setdefault(by_id[i].sha256, group)
            if previous != group:
                raise ValueError('duplicate audio content across splits')


def split_corpus(clips: list[Clip], seed: int = 20261114,
                 sizes: tuple[int, int, int, int] = (80, 10, 8, 2)) -> dict[str, list[str]]:
    if len({c.id for c in clips}) != len(clips) or sum(sizes) != len(clips) or any(n <= 0 for n in sizes):
        raise ValueError('unique sources and positive split sizes covering the corpus are required')
    ordered = sorted(clips, key=lambda c: c.id)
    random.Random(seed).shuffle(ordered)
    references = [c.id for c in ordered if c.seconds >= 3][:sizes[3]]
    if len(references) != sizes[3]:
        raise ValueError('not enough references of at least 3 seconds')
    remaining = [c.id for c in ordered if c.id not in references]
    n, m = sizes[0], sizes[1]
    splits = {'train': remaining[:n], 'validation': remaining[n:n+m],
              'test': remaining[n+m:], 'reference': references}
    validate_splits(splits, clips)
    return splits
