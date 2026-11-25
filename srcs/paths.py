"""Folders of the repository. Everything is relative to its root."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DATA = os.path.join(ROOT, 'data')                   # prepared datasets: DB_REAL_* (Mohtat), DB_LGM50_* (LG M50T)
CONFIGS = os.path.join(ROOT, 'configs')             # study configs; configs/datasets/ describes each dataset
RUNS = os.path.join(ROOT, 'runs')                   # identification and evaluation outputs
STUDIES = os.path.join(RUNS, 'studies')             # one folder per study: runs/studies/<name>/
FIGURES = os.path.join(ROOT, 'figures')    # the paper's figures
# the MIONet surrogate particles (mionet-clean): NOT needed for the paper, which uses the exact finite-volume particles
MIONET = os.environ.get('MIONET_CLEAN', os.path.join(ROOT, 'mionet'))


def next_run(prefix: str) -> str:
    """runs/<prefix>_001, _002, ... - never overwrites."""
    os.makedirs(RUNS, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(RUNS, '%s_%03d' % (prefix, i))):
        i += 1
    d = os.path.join(RUNS, '%s_%03d' % (prefix, i))
    os.makedirs(d)
    return d
