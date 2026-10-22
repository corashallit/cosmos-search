"""Visits and the split into halves (docs/population.md §2, §5).

A visit is a run of a star's G epochs with no gap longer than VISIT_GAP days: one transit pair (the two fields of
view, 106.5 min apart), sometimes two or three pairs 6 h apart. Each visit belongs to half A or B by a hash of the
day of its first epoch. Every star seen on one day is in the same half, so instrument trouble on that day lands in
one half, and a day hash cannot resonate with the scanning law's 63-day precession as alternating blocks could.
"""

import numpy as np

VISIT_GAP = 1.0                      # days
SALT = 0x706F70756C617465            # "populate"; public, since half B is kept blind by sealing, not by the split
HALVES = ("A", "B")


def splitmix64(x):
    with np.errstate(over="ignore"):
        z = np.asarray(x).astype(np.int64).view(np.uint64) ^ np.uint64(SALT)
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def half_of_day(day):
    """0 for half A, 1 for half B, for integer days (floor of BTJD)."""
    return (splitmix64(day) & np.uint64(1)).astype(np.int8)


def half_of_time(t):
    return half_of_day(np.floor(np.asarray(t, float)).astype(np.int64))


def quarter_of_day(day):
    """2 * half + sub-half, the sub-half from the hash's next bit: 0, 1 are A1, A2 and 2, 3 are B1, B2. Splitting
    half A into sub-halves lets DR4 separate static from scan-tied overdispersion without opening half B (§9)."""
    z = splitmix64(day)
    return (2 * (z & np.uint64(1)) + ((z >> np.uint64(1)) & np.uint64(1))).astype(np.int8)


def visits(gid, t):
    """For epochs sorted by (group, t): visit index per epoch (global, increasing), and the half of each visit,
    taken from the day of the visit's first epoch."""
    gid, t = np.asarray(gid), np.asarray(t, float)
    new = np.ones(len(t), bool)
    new[1:] = (gid[1:] != gid[:-1]) | (np.diff(t) > VISIT_GAP)
    vid = np.cumsum(new) - 1
    first_t = t[new]
    return vid, half_of_time(first_t)
