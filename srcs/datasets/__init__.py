"""
datasets
========
Every dataset behind one interface (base.Dataset), described by configs/datasets/<name>.json:

    from srcs import datasets
    ds = datasets.get('mohtat')
    df, cfg = ds.load('C07', 'C20')

    mohtat   Mohtat NMC532 / graphite pouch cells (UofM), cells C01..C21
    lgm50    LG M50T 21700 NMC811 / graphite-SiOx cells (Kirkaldy et al. 2024), cells E4A..E5H
"""
from __future__ import annotations

import json
import os

from srcs.paths import CONFIGS

from .base import Dataset
from .lgm50 import LGM50
from .mohtat import Mohtat

CLASSES = {'mohtat': Mohtat, 'lgm50': LGM50}


def spec_path(name: str) -> str:
    return os.path.join(CONFIGS, 'datasets', '%s.json' % name)


def get(name: str) -> Dataset:
    spec = json.load(open(spec_path(name), encoding='utf-8'))
    return CLASSES[spec.get('kind', name)](spec)


def names() -> list:
    """Every dataset described in configs/datasets/."""
    return sorted(f[:-5] for f in os.listdir(os.path.join(CONFIGS, 'datasets')) if f.endswith('.json'))

