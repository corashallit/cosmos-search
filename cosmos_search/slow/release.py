"""Release-day thresholds: set the slow-change rule's thresholds from DR4's own quiet stars.

The frozen rules' thresholds come from quiet DR3 stars, and DR3 only published light curves for stars it classed
as variable, so its quiet stars are not typical (docs/arms/slow-changes.md, S3). On release day the same
false-alarm level, 0.1% of quiet stars passing when brightening, is set again from a random sample of DR4 light
curves:

1. Draw a random sample of DR4 sources with light curves (`sample_adql`) and fetch them (bulk files, a partner
   centre's TAP table, or DataLink: `gaia.epoch`). `stars_needed` says how many quiet stars a given precision
   takes.
2. `quiet_ids`: quiet stars by the same definition as on DR3 (`drift.reference_mask`), computed from the light
   curves themselves.
3. `cross_corrected`: the drift table fitted on each half of the quiet stars corrects the other half, so no star
   is corrected with a table it helped fit, and both halves count towards the thresholds (on DR3 half B alone
   left ~6 stars beyond the 0.1% point, too few: thresholds moved by +-1 between runs). The table fitted on
   all quiet stars is the one the search itself uses.
4. `thresholds`: per magnitude range, the 0.1% points of the brightening mirror of each score, with the number
   of stars beyond them and a bootstrap interval.
5. `write_rule`: the files a rule module loads (thresholds.json, period_thresholds.json, drift_table.parquet)
   plus a provenance record.

`scripts/slow_release_thresholds.py` runs the steps on a directory of light curves; run on DR3 it reproduces the
frozen thresholds (docs/arms/slow-changes.md, "Release-day thresholds").
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_search.slow import drift
from cosmos_search.slow.rule import mag_group

SHARE = 0.001
# Mirror (false-alarm) statistic for each score of a rule: the threshold on the score is the SHARE point of the
# mirror among quiet stars.
# s_rise_ns is s_rise with change dates shared by neighbours set aside (rule_v3).
MIRRORS = {"s": "s_rise_ns", "z": "slope_z", "rep": "s_rep_bright"}


def quiet_ids(epochs, max_ratio=1.5, max_extreme=8.0, min_n=20):
    """Quiet stars from normalised epochs (drift.normalise), as drift.reference_mask defines them on the DR3
    features: G robust scatter within max_ratio of the median error, no epoch beyond max_extreme robust sigma
    on either side, at least min_n G epochs."""
    g = epochs[epochs["band"] == 0]
    grp = g.groupby("source_id")
    med = grp["frac"].transform("median")
    dev = g["frac"] - med
    mad = dev.abs().groupby(g["source_id"]).median() * 1.4826
    s = pd.DataFrame({"n": grp.size(), "sigma": mad, "err": grp["err_frac"].median(),
                      "lo": dev.groupby(g["source_id"]).min(), "hi": dev.groupby(g["source_id"]).max()})
    ok = ((s["n"] >= min_n) & (s["sigma"] < max_ratio * s["err"]) & (s["lo"] > -max_extreme * s["sigma"])
          & (s["hi"] < max_extreme * s["sigma"]))
    return s.index[ok].to_numpy()


def cross_corrected(epochs, cells, quiet):
    """Drift-corrected epochs (frac_corr) for every star in `epochs`, each corrected with the table fitted on the
    quiet stars of the other half, and the table fitted on all quiet stars."""
    sid = epochs["source_id"].to_numpy()
    half = drift.half(sid)
    is_q = np.isin(sid, quiet)
    tables = {h: drift.fit(epochs[is_q & (half == h)], cells) for h in (0, 1)}
    corr = np.zeros(len(epochs))
    for h in (0, 1):
        m = half == h
        corr[m] = drift.drift_at(epochs[m], cells, tables[1 - h])
    return epochs.assign(frac_corr=epochs["frac"].to_numpy() - corr), drift.fit(epochs[is_q], cells)


def stars_needed(share=SHARE, rel_err=0.2):
    """Quiet stars per magnitude range for the false-alarm rate at a threshold set from them to be known to
    about rel_err (one standard deviation): the stars beyond the threshold are roughly Poisson, so 1 / rel_err^2
    of them are needed."""
    return int(math.ceil(1.0 / (share * rel_err ** 2)))


def thresholds(stats, mirrors=None, share=SHARE, n_boot=200, seed=0):
    """Per magnitude range ("G<19", "G>=19"): the 1 - share quantile of each mirror statistic among the given
    (quiet) stars, the stars beyond it, and a bootstrap 16-84% interval.

    Returns {group: {key: {"value", "n", "beyond", "lo", "hi"}}}."""
    mirrors = mirrors or MIRRORS
    rng = np.random.default_rng(seed)
    out = {}
    for grp, d in stats.groupby(mag_group(stats["gbin"])):
        res = {}
        for key, col in mirrors.items():
            if col not in d:
                continue
            # A missing score (too few visits, or set aside as shared) counts as not passing.
            x = np.nan_to_num(d[col].to_numpy(float), nan=-np.inf)
            if not np.isfinite(x).any():
                continue
            v = float(np.quantile(x, 1 - share))
            boot = [np.quantile(rng.choice(x, len(x)), 1 - share) for _ in range(n_boot)]
            res[key] = {"value": v, "n": int(len(x)), "beyond": int(np.sum(x > v)),
                        "lo": float(np.quantile(boot, 0.16)), "hi": float(np.quantile(boot, 0.84))}
        out[grp] = res
    return out


def write_rule(rule_dir, thr, table, meta, share=SHARE):
    """Rule files in the frozen rules' format: thresholds.json {group: {share: {"s", "z"}}},
    period_thresholds.json {group: {share: value}} when a "rep" threshold exists, drift_table.parquet, and
    release.json with the full threshold record and `meta` (what sample, which statistics)."""
    d = Path(rule_dir)
    d.mkdir(parents=True, exist_ok=True)
    key = str(share)
    det = {g: {key: {"s": v["s"]["value"], "z": v["z"]["value"]}} for g, v in thr.items()}
    (d / "thresholds.json").write_text(json.dumps(det, indent=1))
    if all("rep" in v for v in thr.values()):
        (d / "period_thresholds.json").write_text(
            json.dumps({g: {key: v["rep"]["value"]} for g, v in thr.items()}, indent=1))
    table.to_parquet(d / "drift_table.parquet", index=False)
    (d / "release.json").write_text(json.dumps({"share": share, "thresholds": thr, **meta}, indent=1))
    return d


def sample_adql(n, table="gaiadr4.gaia_source", min_g=None, max_g=None):
    """A random sample of DR4 sources with epoch photometry among the first n of random_index (a random
    permutation of 0..N-1), optionally within [min_g, max_g): most sources are fainter than G = 19, so the
    bright range is drawn on its own to reach stars_needed() there. has_epoch_photometry is in gaia_source as in
    DR3 and in all_source_flags in the draft data model: check the released schema. Magnitude and colour come
    along for the drift cells."""
    g = (f" AND phot_g_mean_mag >= {min_g}" if min_g is not None else "") + \
        (f" AND phot_g_mean_mag < {max_g}" if max_g is not None else "")
    return (f"SELECT source_id, phot_g_mean_mag, bp_rp FROM {table}\n"
            f"WHERE random_index < {int(n)} AND has_epoch_photometry = 'True'{g}")


def patch_ranges(n_patches, level=9, seed=None):
    """source_id ranges of n_patches random HEALPix pixels at `level` (nested): every star in them is the
    calibration sample. Each star is in it with the same probability whatever the sky density, and its
    neighbours come with it for the sky offsets (slow.sky). Level 9 pixels are ~0.013 deg^2."""
    rng = np.random.default_rng(seed)
    pix = np.sort(rng.choice(12 * 4 ** level, size=n_patches, replace=False))
    width = 2 ** 35 * 4 ** (12 - level)
    return [(int(p) * width, (int(p) + 1) * width - 1) for p in pix]
