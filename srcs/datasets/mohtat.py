"""
datasets/mohtat.py
==================
University of Michigan Mohtat NMC532 / graphite 5 Ah pouch cells, beginning of life. Preparation from
the raw CSVs is srcs/real_data.py (unchanged: HPPC, C20, VAL, CRATE1C per cell); this class only adds
the common interface. Cells are 'C01'..'C21'.
"""
from __future__ import annotations

import sys

from srcs.datasets.base import Dataset


class Mohtat(Dataset):

    def prepare(self, cells=None, force=False):
        if force:
            raise SystemExit('mohtat: --force is not supported; delete the data/DB_REAL_* folders to rebuild')
        from srcs import real_data
        real_data.CELLS = tuple(int(c[1:]) for c in (cells or self.cells))
        real_data.USE_Q = bool(self.spec.get('use_Q', False))
        real_data.FOLDER = self.spec['folder']
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            real_data.main()
        finally:
            sys.argv = argv
