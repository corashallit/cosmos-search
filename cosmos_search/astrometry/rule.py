"""The adopted decision rule on cascade output, and how a blind batch is scored against it.

Labels, in order of precedence:
    instrument   fov_pair_z > FOV_Z: the two fields of view disagree, so the signal is not on the sky
    neighbour    frac_multi > FRAC_MULTI: Gaia's image parameters see a second peak in more than this share of
                 transits; the source's centroid motion belongs to a known blend, so no non-Keplerian claim is made
    unexplained  S = max(dbic_cp, dbic_implausible) > S_MIN and S > REL x (chi2_5p - (n - 5)): the data prefer a
                 change point or a physically impossible model over every possible natural one, by a margin that is
                 also a non-negligible share of the source's whole signal (a 0.1% misfit of a huge orbit is not an
                 anomaly)
    natural      everything else
Thresholds were set on calibration run 7 (docs/arms/astrometry.md) before the blind batch was drawn.
"""

import numpy as np
import pandas as pd
from scipy.stats import beta

ADOPTED_NAME = "astrometry-v1"
# set by scripts/astrometry_thresholds.py on calibration run 7 (docs/arms/astrometry.md), before the blind batch
S_MIN = 50.0
REL = 0.0
FOV_Z = 4.0
FRAC_MULTI = 0.10


def statistic(res):
    imp = res["dbic_implausible"].replace(-np.inf, np.nan).fillna(-1e9)
    return np.maximum(res["dbic_cp"].fillna(-1e9), imp)


def apply(res, s_min=None, rel=None, fov_z=None, frac_multi=None):
    s_min = S_MIN if s_min is None else s_min
    rel = REL if rel is None else rel
    fov_z = FOV_Z if fov_z is None else fov_z
    frac_multi = FRAC_MULTI if frac_multi is None else frac_multi
    s = statistic(res)
    signal = np.maximum(res["chi2_5p"] - (res["n"] - 5), 1.0)
    label = np.where(res["fov_pair_z"].fillna(0) > fov_z, "instrument",
                     np.where(res["frac_multi"].fillna(0) > frac_multi, "neighbour",
                              np.where((s > s_min) & (s > rel * signal), "unexplained", "natural")))
    if "error" in res:
        label = np.where(res["error"].notna(), "error", label)
    return pd.DataFrame({"id": res["id"], "S": s, "signal": signal, "label": label,
                         "flagged": label == "unexplained"})


def cp_upper(k, n, conf=0.95):
    return float(beta.ppf(conf, k + 1, n - k)) if k < n else 1.0


def cp_lower(k, n, conf=0.95):
    return float(beta.ppf(1 - conf, k, n - k + 1)) if k > 0 else 0.0


def score(m):
    """m: flags merged with the revealed truth (truth column 'kind', 'amp' for injections)."""
    from .simulate import INSTRUMENT, NONGRAV

    out = {}
    nat = m[~m["kind"].isin(NONGRAV + INSTRUMENT)]
    k, n = int(nat["flagged"].sum()), len(nat)
    out["natural"] = {"n": n, "flagged": k, "rate": k / max(n, 1), "cp95_upper": cp_upper(k, n),
                      "by_kind": nat.groupby("kind")["flagged"].agg(["sum", "size"]).astype(int).to_dict("index"),
                      "labels": nat["label"].value_counts().to_dict()}
    ng = m[m["kind"].isin(NONGRAV)]
    bins = [0, 5, 10, 20, 50, np.inf]
    ng = ng.assign(abin=pd.cut(ng["amp"], bins).astype(str))
    out["nongrav"] = {"n": len(ng), "flagged": int(ng["flagged"].sum()),
                      "by_amp": ng.groupby("abin", observed=True)["flagged"].agg(["sum", "size"]).astype(int)
                      .to_dict("index"),
                      "by_kind": ng.groupby("kind")["flagged"].agg(["sum", "size"]).astype(int).to_dict("index")}
    hi = ng[ng["amp"] >= 20]
    out["nongrav"]["amp_ge_20"] = {"n": len(hi), "flagged": int(hi["flagged"].sum()),
                                   "cp95_lower": cp_lower(int(hi["flagged"].sum()), len(hi))}
    ins = m[m["kind"].isin(INSTRUMENT)]
    out["instrument"] = {"n": len(ins), "labelled_instrument": int((ins["label"] == "instrument").sum()),
                         "flagged_unexplained": int(ins["flagged"].sum()),
                         "amp_ge_10": int(((ins["label"] == "instrument") & (ins["amp"] >= 10)).sum()),
                         "n_amp_ge_10": int((ins["amp"] >= 10).sum())}
    return out
