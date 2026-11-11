"""Injection test of the release-day slow-change run: fake fades, dips and decoys put into real light curves and
scored the way the run scores every star, beside the earlier archive-side selection.

    python scripts/slow_search.py pass --out R --lightcurves LC --rule RULE          # the real stars
    python scripts/slow_search.py flag --run R --rule RULE --out R/lists
    python scripts/slow_inject_test.py --run R --lists R/lists --lightcurves LC --rule RULE --out T [--gaia G.parquet]

Hosts: real stars with scores, the same number from each magnitude bin (--per-bin), in a repeatable pseudo-random
order. Each host carries every kind of fake, one at a time, in G, BP and RP:

- steady fades of FADE_RATES percent a year about the middle of the light curve (the 2026-11-10 review's test),
  and steady brightenings of RISE_RATES as decoys: the same change with the sign turned;
- one draw from each of FAMILIES (`inject.lightcurve.sample`: long dips, ramps, steady fades of any size and
  colour, repeated long dips, slow cycles), and the brightening mirror of the long dip as a decoy;
- "none": the host untouched, which gives the share of hosts on a list with nothing injected.

Each fake is scored by `search.score_all` (summary, then detector and period search where the run would run them),
its trend is measured against the real stars' reference with the host's magnitude (`summary.apply_reference`: the
fake never enters the reference), the shared-date test compares it with the real stars of its sky pixel, and
`search.flag` applies the rule. Recorded for each: on the fading list or not and by which score, strength, the
rising and early lists, and for the earlier route `picked_archive`: whether the archive-side cut would have
chosen the star at all (`select.slow_z` of its G epochs above the 0.1% point of all stars in its magnitude bin),
and `picked_abbe`: the same for the Abbe value alone at a 1% share (the DataLink fallback list's cut).

With --gaia (gaia_source rows of the field: parallax, proper motion, colour) each host gets its kind of star
(`kinds.star_kind`) and each fake its place among the real stars of that kind: `n_stronger`, the real stars of
the kind fading at least as strongly, and `q_kind`, the estimated share of chance among them (`kinds.rank`).

recall.json holds the tables; fakes.parquet one row per fake.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_search.inject import lightcurve as inj
from cosmos_search.slow import drift, kinds, rule_v3, search, select, summary
from cosmos_search.stats import mirror

FADE_RATES = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
RISE_RATES = (1.0, 3.0, 10.0)
FAMILIES = ("long_dip", "ramp_fade", "secular_fade", "repeat_long_dip", "slow_cycle")
MIRRORED = ("long_dip",)
BAND_NAMES = ("G", "BP", "RP")
_STATE = {}


def fake_kinds():
    return (["none"] + [f"fade_{r:g}" for r in FADE_RATES] + [f"rise_{r:g}" for r in RISE_RATES] + list(FAMILIES)
            + [f"{f}_mirror" for f in MIRRORED])


def factors(kind, sid, band, t, seed):
    """Flux factor of every row for one kind of fake, and each star's size (percent a year for steady changes,
    the largest G dimming in percent for drawn families). Rows are grouped by star."""
    f = np.ones(len(t))
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    ends = np.r_[starts[1:], len(sid)]
    size = np.zeros(len(starts))
    if kind == "none":
        return f, size
    for k, (a, b) in enumerate(zip(starts, ends)):
        g = band[a:b] == 0
        tg = t[a:b][g] if g.any() else t[a:b]
        if kind.startswith(("fade_", "rise_")):
            rate = float(kind.split("_")[1]) * (-1 if kind.startswith("fade") else 1)
            f[a:b] = 1 + rate / 100 * (t[a:b] - tg.mean()) / 365.25
            size[k] = abs(rate)
            continue
        family = kind.removesuffix("_mirror")
        rng = np.random.default_rng([seed, int(sid[a] % 2 ** 32), FAMILIES.index(family)])
        sig = inj.sample(rng, family, float(tg.min()), float(tg.max()))
        for code, name in enumerate(BAND_NAMES):
            m = band[a:b] == code
            if m.any():
                f[a:b][m] = sig.factor(t[a:b][m], name)
        size[k] = 100 * (1 - f[a:b][g].min()) if g.any() else np.nan
        if kind.endswith("_mirror"):
            f[a:b] = 2 - f[a:b]
    return np.maximum(f, 1e-3), size


def _file(path):
    """One light-curve file: the archive-style statistics of every real star in it, and every fake of its hosts."""
    s = _STATE
    lc = pq.read_table(path)
    sid = np.asarray(lc["source_id"])
    if len(sid) and not np.all(sid[1:] >= sid[:-1]):
        lc = lc.take(pa.array(np.argsort(sid, kind="stable")))
        sid = np.asarray(lc["source_id"])
    real = summary.star_summary(lc, s["table"], marks=s["marks"], own=s["own"])
    real = real.loc[np.isfinite(real["sigma_visit"].to_numpy()), ["source_id", "gbin", "n_g", "uwv_g", "abbe_g"]]
    real["slow_z"] = select.slow_z(real["n_g"], real["uwv_g"], real["abbe_g"])
    hosts = s["hosts"]
    mine = np.isin(sid, hosts.index.to_numpy())
    if not mine.any():
        return real, pd.DataFrame()
    lc = lc.filter(pa.array(mine))
    sid = np.asarray(lc["source_id"])
    band = np.asarray(lc["band"].combine_chunks().indices).astype(np.int64)
    t, flux, err = np.asarray(lc["t"]), np.asarray(lc["flux"]), np.asarray(lc["err"])
    i_flux, i_err = lc.schema.get_field_index("flux"), lc.schema.get_field_index("err")
    star_ids = sid[np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])]
    out = []
    for kind in fake_kinds():
        f, size = factors(kind, sid, band, t, s["seed"])
        fake = lc.set_column(i_flux, "flux", pa.array(flux * f)).set_column(i_err, "err", pa.array(err * f))
        summ, stats, _ = search.score_all(fake, s["table"], s["thr"], s["thr_p"], marks=s["marks"], own=s["own"])
        m = search.merge(summ, stats if len(stats) else None)
        m = m[np.isfinite(m["sigma_visit"].to_numpy())].reset_index(drop=True)
        if not len(m):
            continue
        ids = m["source_id"].to_numpy()
        m = summary.apply_reference(m, s["ref"], g_mag=hosts["g_mag"].reindex(ids).to_numpy(),
                                    bp_rp=hosts["bp_rp"].reindex(ids).to_numpy() if "bp_rp" in hosts else None)
        pix = drift.sky_pixel(ids, rule_v3.SHARED_LEVEL)
        near = s["real"][np.isin(s["real_pix"], np.unique(pix))]
        sh = rule_v3.shared(near, targets=m, target_patch=pix)
        fl = search.flag(m, s["rule"], shared_flags=sh)
        row = fl[["id", "gbin", "flagged", "early", "rising", "rising_rep", "by", "strength", "rise_strength",
                  "strength_by", "s_fade", "fade_kind", "slope_z", "s_rep", "fade_set_aside"]].copy()
        row["slope_z_raw"], row["slope"] = m["slope_z_raw"].to_numpy(), m["slope"].to_numpy()
        row["slow_z"] = np.asarray(select.slow_z(m["n_g"], m["uwv_g"], m["abbe_g"]))
        row["abbe_g"] = m["abbe_g"].to_numpy()
        row["fake"], row["size"] = kind, pd.Series(size, index=star_ids).reindex(ids).to_numpy()
        out.append(row)
    return real, pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def choose_hosts(real, per_bin):
    """Up to per_bin scored stars from each magnitude bin, in the pseudo-random order of `summary._hash`."""
    d = real.iloc[np.argsort(summary._hash(real["source_id"].to_numpy()), kind="stable")]
    return d[d.groupby("gbin").cumcount() < per_bin].set_index("source_id")


def place_in_kind(fakes, listed):
    """Each fake's place among the real stars of its host's kind: n_stronger (real stars of the kind whose fade
    wins at the fake's strength or above), n_brightening (the same for brightenings) and q_kind, the estimated
    share of chance among the kind's fades at least as strong, the fake included: (1 + n_brightening) /
    (1 + n_stronger), at most 1. Fakes whose brightening wins or whose strength is below 1 get NaN."""
    w_real = mirror.competition(np.nan_to_num(listed["strength"].to_numpy(float), nan=-np.inf, neginf=-np.inf),
                                np.nan_to_num(listed["rise_strength"].to_numpy(float), nan=-np.inf, neginf=-np.inf))
    n_f, n_r = np.full(len(fakes), np.nan), np.full(len(fakes), np.nan)
    w = np.where(fakes["strength"].to_numpy() > fakes["rise_strength"].to_numpy(), fakes["strength"].to_numpy(), np.nan)
    w = np.where(w > 1, w, np.nan)
    for name, idx in fakes.groupby("kind").indices.items():
        k = listed["kind"].to_numpy() == name
        pos, neg = np.sort(w_real[k & (w_real > 0)]), np.sort(-w_real[k & (w_real < 0)])
        ok = np.isfinite(w[idx])
        n_f[idx[ok]] = len(pos) - np.searchsorted(pos, w[idx][ok], "left")
        n_r[idx[ok]] = len(neg) - np.searchsorted(neg, w[idx][ok], "left")
    return pd.DataFrame({"n_stronger": n_f, "n_brightening": n_r, "q_kind": np.minimum(1.0, (1 + n_r) / (1 + n_f))},
                        index=fakes.index)


def table(fakes, by, cols=("flagged", "picked_archive", "picked_abbe")):
    g = fakes.groupby(by, sort=False, observed=True)
    out = g[list(cols)].mean().round(3)
    out.insert(0, "n", g.size())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="the real stars' pass output (summary/)")
    ap.add_argument("--lists", required=True, help="its flag output (reference.parquet, flags.parquet)")
    ap.add_argument("--lightcurves", required=True, help="the light-curve files the pass read (epoch.SCHEMA)")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gaia", help="gaia_source rows of the field, for the kind of each star")
    ap.add_argument("--per-bin", type=int, default=600, help="hosts per magnitude bin")
    ap.add_argument("--archive-share", type=float, default=0.001, help="share of stars the archive-side cut picked")
    ap.add_argument("--abbe-share", type=float, default=0.01, help="share picked by the Abbe value alone")
    ap.add_argument("--no-marks", action="store_true", help="as the pass's --no-marks")
    ap.add_argument("--no-own-test", action="store_true", help="as the pass's --no-own-test")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["source_id", "g_mag", "bp_rp", "n_visits", "sigma_visit", "slope", "slope_err", "s_fade", "s_rise",
            "dim_t0", "rise_t0", "quiet"]
    real = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(Path(a.run, "summary").glob("*.parquet"))],
                     ignore_index=True)
    real = summary.from_store(real[np.isfinite(real["sigma_visit"].to_numpy())].reset_index(drop=True))
    flags = pd.read_parquet(Path(a.lists) / "flags.parquet")
    hosts = choose_hosts(real, a.per_bin)
    hosts["listed"] = hosts.index.isin(flags.loc[flags["flagged"] | flags["rising"] | flags["early"], "id"])
    thr, thr_p, tab = rule_v3.load(a.rule)
    _STATE.update(hosts=hosts, table=tab, thr=thr, thr_p=thr_p, rule=a.rule, seed=a.seed, marks=not a.no_marks,
                  own=not a.no_own_test, ref=pd.read_parquet(Path(a.lists) / "reference.parquet"),
                  real=real[["source_id", "s_fade", "s_rise", "dim_t0", "rise_t0"]],
                  real_pix=drift.sky_pixel(real["source_id"].to_numpy(), rule_v3.SHARED_LEVEL))
    files = [f for f in sorted(Path(a.lightcurves).glob("*.parquet")) if pq.read_metadata(f).num_rows]
    with Pool(a.workers) as pool:
        parts = pool.map(_file, files, chunksize=1)
    arch = pd.concat([p[0] for p in parts], ignore_index=True)
    fakes = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    cut_z = arch.groupby("gbin")["slow_z"].quantile(1 - a.archive_share)
    cut_abbe = arch.groupby("gbin")["abbe_g"].quantile(a.abbe_share)
    fakes["picked_archive"] = fakes["slow_z"].to_numpy() > fakes["gbin"].map(cut_z).to_numpy()
    fakes["picked_abbe"] = fakes["abbe_g"].to_numpy() < fakes["gbin"].map(cut_abbe).to_numpy()
    h = hosts.reindex(fakes["id"].to_numpy())
    fakes["host_g"], fakes["host_quiet"], fakes["host_listed"] = (h["g_mag"].to_numpy(), h["quiet"].to_numpy(),
                                                                  h["listed"].to_numpy())
    fakes["mag"] = pd.cut(fakes["host_g"], [0, 13, 16, 19, 99], right=False, labels=["G<13", "G 13-16", "G 16-19", "G>=19"])
    fakes["kind"] = ""
    if a.gaia:
        g = pd.read_parquet(a.gaia).drop_duplicates("source_id").set_index("source_id")
        still = search.motion(g)["still"].to_numpy()
        kind = kinds.star_kind(g, still)
        fakes["kind"] = kind.reindex(fakes["id"].to_numpy()).fillna("no astrometry").to_numpy()
        listed = flags[["id", "strength", "rise_strength"]].assign(
            kind=kind.reindex(flags["id"].to_numpy()).fillna("no astrometry").to_numpy())
        fakes = pd.concat([fakes, place_in_kind(fakes, listed)], axis=1)
    fakes.to_parquet(out / "fakes.parquet", index=False)

    pd.options.display.width = 200
    order = fake_kinds()
    fakes["fake"] = pd.Categorical(fakes["fake"], categories=order)
    bright = fakes[fakes["host_g"] < 19]
    clean = bright[bright["host_quiet"] & ~bright["host_listed"]]
    res = {"hosts": len(hosts), "fakes": len(fakes), "rule": str(a.rule),
           "archive_cut_slow_z": {int(k): float(v) for k, v in cut_z.items()},
           "marks": not a.no_marks, "own_test": not a.no_own_test}
    for title, d in (("all hosts with G < 19", bright), ("quiet hosts on no list, G < 19", clean),
                     ("hosts with G >= 19", fakes[fakes["host_g"] >= 19])):
        t = table(d.sort_values("fake"), "fake", ("flagged", "rising", "early", "picked_archive", "picked_abbe"))
        print(f"\n{title}: share on the fading list, the rising list, the early list; picked by the archive-side "
              f"cut; picked by the Abbe value alone\n{t.to_string()}")
        res[title] = t.reset_index().to_dict("records")
    t = table(bright.sort_values("fake"), ["fake", "mag"], ("flagged", "picked_archive"))
    print(f"\nby magnitude, all hosts with G < 19\n{t.unstack('mag').to_string()}")
    res["by magnitude"] = t.reset_index().astype({"mag": str}).to_dict("records")
    drawn = bright[bright["fake"].isin(FAMILIES)].copy()
    drawn["depth"] = pd.cut(drawn["size"], [0, 0.5, 1, 2, 5, 10, 100], right=False)
    t = table(drawn.sort_values(["fake", "depth"]), ["fake", "depth"], ("flagged", "picked_archive"))
    print(f"\ndrawn families by largest G dimming in percent, all hosts with G < 19\n{t.to_string()}")
    res["by depth"] = t.reset_index().astype({"depth": str}).to_dict("records")
    if a.gaia:
        t = table(bright.sort_values("fake"), ["kind", "fake"], ("flagged", "picked_archive"))
        steady = [f"fade_{r:g}" for r in FADE_RATES]
        print("\nsteady fades by kind of star, all hosts with G < 19: share on the fading list\n"
              + t["flagged"].unstack("fake")[steady].assign(hosts=t["n"].groupby("kind").max()).to_string())
        res["by kind"] = t.reset_index().to_dict("records")
        on = bright[bright["flagged"] & bright["fake"].isin(steady) & np.isfinite(bright["q_kind"])]
        t = on.groupby(["kind", "fake"], observed=True).agg(
            n=("id", "size"), median_strength=("strength", "median"), median_n_stronger=("n_stronger", "median"),
            median_q_kind=("q_kind", "median")).round(2)
        print(f"\nflagged steady fades among the real stars of their kind (place by strength, estimated share of "
              f"chance)\n{t.to_string()}")
        res["place in kind"] = t.reset_index().to_dict("records")
    (out / "recall.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
