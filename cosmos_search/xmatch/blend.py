"""How much of a survey's flux at a Gaia source's position could come from other Gaia sources.

A WISE source is 6 arcsec across and TESS's pixel is 21 arcsec, so a "match" can be a blend. For an
energy-budget test (an infrared excess or its absence) the question is not only "which survey source is
this star" but "how much of that source's flux is this star". Gaia resolves pairs down to ~0.5 arcsec, so it
can answer the second question from its own catalogue: every Gaia neighbour within the beam, weighted by
how much its PSF overlaps the target's.

For Gaussian PSFs of width sigma, the flux a PSF fit at the target's position picks up from a neighbour at
separation r scales as exp(-r^2 / (4 sigma^2)). Fluxes come from a Gaia band (RP by default, the reddest);
for mid-infrared surveys this underestimates red neighbours, so treat the ratio as a floor on contamination
and look at the neighbour list for anything very red.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import MAS_PER_ARCSEC, MAS_PER_DEG, chord, healpix_groups, healpix_order_for_side, separation_mas, unit_vectors
from .surveys import GAIA_DR3


def beam_contamination(targets, fwhm_arcsec, source, *, band="phot_rp_mean_mag", radius_fwhm=2.0,
                       gaia=GAIA_DR3, group_side_deg=0.5, id_col="source_id"):
    """Per target: Gaia neighbours within radius_fwhm * FWHM and the overlap-weighted flux ratio.

    Returns a DataFrame: id, n_beam (neighbours within 1 FWHM), n_near (within radius_fwhm FWHM),
    nearest_arcsec, brightest_dmag (neighbour minus target, in band), contamination (sum of weighted
    neighbour flux / target flux).
    """
    df = targets.to_pandas() if hasattr(targets, "to_pandas") else pd.DataFrame(targets)
    ra, dec = np.asarray(df["ra"], float), np.asarray(df["dec"], float)
    mag = np.asarray(df[band], float) if band in df else np.full(len(df), np.nan)
    ids = np.asarray(df[id_col])
    fwhm = fwhm_arcsec * MAS_PER_ARCSEC
    sigma = fwhm / 2.3548
    r_near = radius_fwhm * fwhm
    out = {k: np.full(len(df), np.nan) for k in ("nearest_arcsec", "brightest_dmag", "contamination")}
    out["n_beam"] = np.zeros(len(df), np.int64)
    out["n_near"] = np.zeros(len(df), np.int64)
    pix = healpix_groups(ra, dec, healpix_order_for_side(group_side_deg))
    for p in np.unique(pix):
        idx = np.nonzero(pix == p)[0]
        v = unit_vectors(ra[idx], dec[idx]).sum(axis=0)
        v /= np.linalg.norm(v)
        cra, cdec = np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))
        radius = (np.max(separation_mas(cra, cdec, ra[idx], dec[idx])) + r_near) / MAS_PER_DEG
        nb = source.fetch(gaia, cra, cdec, radius)
        if len(nb) == 0:
            continue
        nra, ndec = np.asarray(nb["ra"], float), np.asarray(nb["dec"], float)
        nmag = np.asarray(nb[band], float)
        nid = np.asarray(nb["source_id"])
        tree = cKDTree(unit_vectors(nra, ndec))
        hits = tree.query_ball_point(unit_vectors(ra[idx], dec[idx]), chord(r_near))
        for a, h in zip(idx, hits):
            h = np.asarray([j for j in h if nid[j] != ids[a]], dtype=np.int64)
            if not len(h):
                out["contamination"][a] = 0.0
                continue
            r = separation_mas(ra[a], dec[a], nra[h], ndec[h])
            w = np.exp(-(r**2) / (4 * sigma**2))
            flux_ratio = 10 ** (-0.4 * (nmag[h] - mag[a]))
            out["n_near"][a] = len(h)
            out["n_beam"][a] = int((r <= fwhm).sum())
            out["nearest_arcsec"][a] = r.min() / MAS_PER_ARCSEC
            out["brightest_dmag"][a] = np.nanmin(nmag[h]) - mag[a] if np.isfinite(nmag[h]).any() else np.nan
            out["contamination"][a] = float(np.nansum(w * flux_ratio))
    res = pd.DataFrame({id_col: ids, **out})
    return res
