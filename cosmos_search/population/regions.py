"""Known star-forming regions and young associations for attribution (docs/population.md §6).

Approximate centres (ICRS deg), angular radii and distances from the standard literature (Zucker et al. 2020 cloud
distances; de Zeeuw et al. 1999 for Sco-Cen). Attribution only: a replicated window overlapping one of these is
called "star-forming region" before anything is called unexplained. Not used in the null model.
"""

import numpy as np

# name: (ra, dec, radius_deg, distance_pc)
REGIONS = {
    "Taurus": (68.0, 25.0, 8.0, 140),
    "Ophiuchus": (246.8, -24.5, 4.0, 140),
    "Lupus": (237.0, -37.0, 6.0, 160),
    "Chamaeleon": (167.0, -77.0, 4.0, 190),
    "Corona Australis": (285.5, -37.0, 3.0, 150),
    "Orion": (84.0, -3.0, 8.0, 400),
    "Perseus": (54.0, 31.5, 3.0, 300),
    "Serpens/Aquila": (277.5, 0.5, 4.0, 440),
    "Upper Sco": (243.0, -23.0, 10.0, 145),
    "UCL/LCC": (210.0, -50.0, 18.0, 125),
    "Cepheus Flare": (315.0, 72.0, 6.0, 350),
    "NGC 2264": (100.25, 9.9, 1.5, 720),
    "Mon R2": (92.0, -6.4, 2.0, 830),
    "Cygnus X": (308.0, 41.0, 3.0, 1500),
    "Carina": (161.0, -59.7, 2.0, 2300),
    "Lagoon": (271.0, -24.4, 1.0, 1300),
}


def overlapping(ra, dec, radius_deg, d_lo, d_hi):
    """Names of regions whose disc overlaps the window's disc and whose distance lies in [d_lo, d_hi)."""
    from .analysis import _angdist
    out = []
    for name, (r, d, rad, dist) in REGIONS.items():
        if _angdist(ra, dec, r, d) < radius_deg + rad and d_lo <= dist < d_hi:
            out.append(name)
    return out


def window_distance_range(cand):
    from .scan import N_SHELLS, SHELL_EDGES_PC
    lo, hi = cand["shells"]
    return (0.0, np.inf) if hi > N_SHELLS else (SHELL_EDGES_PC[lo], SHELL_EDGES_PC[hi])
