"""Identifier routes, where a catalogue already carries Gaia ids.

Positional matching is the general tool, but when another catalogue was built on Gaia (TIC v8.2 on DR2,
and DR4 will ship a DR3 neighbourhood table) the identifier is independent evidence. We use both and
report where they disagree, because a disagreement is either a matcher failure or a catalogue quirk
(merged or split sources between releases) worth knowing about before trusting either.
"""

import numpy as np
import pandas as pd


def previous_release_ids(source_ids, table="gaiadr3.dr2_neighbourhood", new="dr3_source_id",
                         old="dr2_source_id"):
    """Map Gaia source ids to the previous release through DPAC's neighbourhood table.

    Returns one row per new id: old id of the closest neighbour, its angular distance and magnitude
    difference, and n_old (how many old sources sit near this new one) and n_new (how many new sources
    claim that old one). n_old > 1 or n_new > 1 means the source was split or merged between releases.
    """
    from astropy.table import Table

    from cosmos_search.gaia.archive import tap

    ids = Table({"source_id": np.asarray(source_ids, dtype=np.int64)})
    adql = (f"SELECT n.{new}, n.{old}, n.angular_distance, n.magnitude_difference "
            f"FROM TAP_UPLOAD.ids AS u JOIN {table} AS n ON n.{new} = u.source_id")
    d = tap().query(adql, uploads={"ids": ids}).to_pandas()
    return pick_closest(d, new, old)


def pick_closest(d, new="dr3_source_id", old="dr2_source_id"):
    if d.empty:
        return pd.DataFrame(columns=[new, old, "angular_distance", "magnitude_difference", "n_old", "n_new"])
    d = d.copy()
    d["n_old"] = d.groupby(new)[old].transform("size")
    d["n_new"] = d.groupby(old)[new].transform("size")
    d = d.sort_values([new, "angular_distance"]).drop_duplicates(new, keep="first")
    return d.reset_index(drop=True)


def check_ids(matches, column, expected, id_col="source_id", p_min=0.5):
    """Compare a positional match with an id the survey row itself carries.

    matches: Result.matches (has the survey's kept columns, e.g. TIC "GAIA" or ASAS-SN "GaiaDR3").
    column: the survey column holding a Gaia id. expected: Series mapping our Gaia source_id to the id that
    column should hold (the source_id itself for DR3 ids; previous_release_ids() for DR2 ids).
    Returns the rank-1 matches with P >= p_min plus columns expected_id and id_agrees (NA when either is
    missing), and a summary dict.
    """
    m = matches[(matches["rank"] == 1) & (matches.p_match >= p_min)].copy()
    got = pd.to_numeric(m[column], errors="coerce")
    exp = pd.to_numeric(m[id_col].map(expected), errors="coerce")
    agrees = pd.Series(pd.NA, index=m.index, dtype="boolean")
    known = got.notna() & exp.notna()
    agrees[known] = (got[known].astype("int64") == exp[known].astype("int64")).to_numpy()
    m["expected_id"] = exp
    m["id_agrees"] = agrees
    summary = {"matches": int(len(m)), "comparable": int(known.sum()), "agree": int(agrees.sum()),
               "disagree": int((agrees == False).sum())}  # noqa: E712 (nullable boolean)
    return m, summary


def tic_expected_ids(dr2):
    """Expected TIC "GAIA" values (Gaia DR2 ids) per DR3 source_id, from previous_release_ids()."""
    return dr2.set_index("dr3_source_id")["dr2_source_id"]
