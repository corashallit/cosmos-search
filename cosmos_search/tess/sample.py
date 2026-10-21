"""Random samples of QLP light curves, drawn by walking the S3 key hierarchy.

QLP keys nest the 16-digit TIC as four 4-digit directories. TIC IDs were assigned in sky-ordered blocks, so
one third-level directory (10⁴ consecutive IDs) is a patch of sky. Sampling a few stars from each of many
random patches gives a sample that is spread over the sector's footprint, and still has enough stars per
camera/CCD for the common-mode test in dimming.population.
"""

import re

import numpy as np

from .http import STPUBDATA, s3_list
from .qlp import PREFIX

_TIC = re.compile(r"s(\d{4})-(\d{16})_tess_v\d+_llc\.fits$")


def _prefixes(prefix):
    return [e.key for e in s3_list(STPUBDATA, prefix, delimiter="/") if e.is_prefix]


def qlp_sector_sample(sector, n, rng, per_patch=25, max_patches=None):
    """List of (sector, tic) pairs, about n of them, from random sky patches of one sector."""
    root = f"{PREFIX}/s{int(sector):04d}/0000/"
    level2 = _prefixes(root)
    patches = []
    for p in level2:
        patches.extend(_prefixes(p))
    rng.shuffle(patches)
    want = max_patches or int(np.ceil(n / per_patch * 1.3))
    out = []
    for p in patches[:want * 3]:
        files = [e.key for e in s3_list(STPUBDATA, p, delimiter=None) if not e.is_prefix]
        tics = [int(m.group(2)) for m in (_TIC.search(k) for k in files) if m]
        if not tics:
            continue
        take = rng.choice(len(tics), size=min(per_patch, len(tics)), replace=False)
        out.extend((int(sector), tics[i]) for i in take)
        if len(out) >= n:
            break
    return out[:n]
