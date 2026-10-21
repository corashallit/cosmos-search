"""Shared statistics for every arm: empirical nulls, tail p-values, multiple testing, audits.

See docs/statistics.md. Nothing here knows about light curves or astrometry; detectors hand in
statistics, and get back calibrated p-values, false-alarm counts and estimates.
"""

from .audit import audit_inclusion, horvitz_thompson, poisson_sample
from .multiple import bh, cauchy_combine, expected_false_alarms, storey_pi0, storey_q
from .nulls import empirical_p, gpd_fit, gpd_sf, stratified_p, tail_p

__all__ = [
    "audit_inclusion", "horvitz_thompson", "poisson_sample",
    "bh", "cauchy_combine", "expected_false_alarms", "storey_pi0", "storey_q",
    "empirical_p", "gpd_fit", "gpd_sf", "stratified_p", "tail_p",
]
