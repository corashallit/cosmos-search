"""One wide row per Gaia target with everything the other surveys say about it.

This is what the dimmer arms and vetting consume: infrared photometry for the energy-budget test (2MASS,
AllWISE W1-W4, CatWISE/unWISE W1-W2), the TIC id for TESS, X-ray, known variability, literature, and how
much of each beam is other Gaia stars. Every survey contributes its best match (if P >= p_min), the match
probability, and a coverage count, because a missing match outside a footprint is not a non-detection.

    wide, results = context(gaia_table)
"""

import numpy as np
import pandas as pd

from .blend import beam_contamination
from .run import crossmatch
from .sources import AutoSource

DEFAULT_SURVEYS = ("2mass", "allwise", "catwise2020", "unwise", "tic", "erosita_dr2", "ztf_dr22", "vsx",
                   "asassn_vars", "simbad")
# Beams for blending: W1/W2 (6.1"), W3 (6.5"), W4 (12"), TESS pixel (21").
DEFAULT_BEAMS = {"wise_w1": 6.1, "wise_w4": 12.0, "tess": 21.0}


def context(gaia, surveys=DEFAULT_SURVEYS, *, source=None, p_min=0.5, beams=DEFAULT_BEAMS, id_col="source_id",
            **kw):
    """Run crossmatch for each survey and join the best matches into one row per target.

    Columns are prefixed with the survey name: <s>_id, <s>_p, <s>_sep_arcsec, <s>_multiplicity, <s>_n_local
    and the survey's kept columns. Grouped surveys (ZTF) give a count and the list of matched object ids
    instead. Returns (wide DataFrame, {survey: Result}).
    """
    source = source or AutoSource()
    df = gaia.to_pandas() if hasattr(gaia, "to_pandas") else pd.DataFrame(gaia)
    wide = df[[id_col]].copy()
    results = {}
    for s in surveys:
        res = crossmatch(df, s, source=source, id_col=id_col, **kw)
        results[s] = res
        wide = wide.merge(_columns(res, s, p_min, id_col), on=id_col, how="left")
    for name, fwhm in (beams or {}).items():
        b = beam_contamination(df, fwhm, getattr(source, "hats", source), id_col=id_col)
        b = b.rename(columns={c: f"blend_{name}_{c}" for c in b.columns if c != id_col})
        wide = wide.merge(b, on=id_col, how="left")
    return wide, results


def _columns(res, s, p_min, id_col):
    m = res.matches
    from .surveys import SURVEYS

    survey = SURVEYS[s]
    cover = res.best.groupby(id_col)["n_local"].max().rename(f"{s}_n_local")
    if survey.group_by or survey.group_fn is not None:
        ok = m[m.p_match >= p_min]
        g = ok.groupby(id_col)
        out = pd.DataFrame({f"{s}_n_matched": g.size(), f"{s}_ids": g[f"{s}_id"].agg(list)})
        return out.join(cover, how="outer").reset_index()
    top = m[(m["rank"] == 1) & (m.p_match >= p_min)].set_index(id_col)
    cols = {f"{s}_id": top[f"{s}_id"], f"{s}_p": top.p_match, f"{s}_p_cal": top.p_cal,
            f"{s}_sep_arcsec": top.sep_arcsec,
            f"{s}_multiplicity": top.survey_multiplicity, f"{s}_p_chance": top["p_chance_0.9"]}
    for c in survey.keep:
        if c in top:
            cols[f"{s}_{c}"] = top[c]
    out = pd.DataFrame(cols)
    # Targets with nothing above p_min still report P(no counterpart) and coverage.
    p_none = res.best.groupby(id_col)["p_none"].min().rename(f"{s}_p_none")
    return out.join([p_none, cover], how="outer").reset_index().rename(columns={"index": id_col})
