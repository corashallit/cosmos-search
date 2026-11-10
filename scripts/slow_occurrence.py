"""Upper limit on how common steady fades are among FGK dwarfs (cosmos_search.slow.occurrence; docs/arms/occurrence.md).

    python scripts/slow_occurrence.py aux --lightcurves S/lightcurves --schema gaiadr4 --out S/aux_plx.parquet
    python scripts/slow_occurrence.py inject --lightcurves S/lightcurves --aux S/aux_plx.parquet --rule RULE --out O
    python scripts/slow_occurrence.py limit --stars R/stats --aux AUX --injections O/injections --out O/limit

`aux`: G, BP-RP, parallax and its error for every star of a directory of light-curve files, from gaia_source: ESA's
archive with the project's login (uploaded id lists), or with `--mirror` a partner centre by source_id range. The
release-day threshold sample's own aux.parquet has no parallaxes.

`inject`: steady fades at each rate of the frozen config, injected into the light curves of FGK dwarfs and scored
as the search scores any star (the rule's drift table, no period search). Light curves are a directory of Parquet
files in epoch.SCHEMA: the release-day threshold sample (`slow_search.py sample`), a `dr4_bulk pull`, or the
field below. `--aux` needs phot_g_mean_mag, bp_rp, parallax and parallax_error. Done-markers let a second run
resume.

`limit`: the per-star scores of the stars searched (`slow_search.py score` output, or the per-star summary of the
bulk-file pass: source_id, slope, slope_z or slope_err, n_visits, t_first, t_last), the same `--aux` columns for
them or `--kind`, and the injections. A table with a neighbour reference (`slope_ref`) is counted one file at a
time, whatever its size. Writes limit.json (config, inputs, the limit per rate, the symmetry tables), curve.csv,
regions.parquet (counts and recall of every region per rate), past_cut.parquet (every searched star past the first
cut either way, strongest fades first) and, with matplotlib, the figure.

`planted` checks the limit on the real stars: a random share of them is given its own injected scores, so that
share really fades, and the limit is computed again; over many draws it should fall below the planted share no
more than 5% of the time. The subtracted figure is counted the same way.

`field` fetches the DR3 demonstration sample: every star of Gaia's Andromeda survey (the only part of DR3 with
light curves for stars not flagged variable) brighter than --g-max, from the AIP mirror, one file per level-6
HEALPix pixel. Score it with `slow_search.py score --lightcurves F/lightcurves --aux F/aux.parquet --no-period`.
"""

import argparse
import datetime
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_search.gaia import features
from cosmos_search.slow import drift, occurrence, recall, rule_v3

AUX_COLUMNS = ["source_id", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error"]
# What the limit reads of a per-star table, and what is kept of an injected star's scores.
STAR_COLUMNS = ["source_id", "slope", "slope_z", "slope_err", "slope_ref", "slope_ref_err", "slope_z_raw", "n_visits",
                "t_first", "t_last"]
INJECTION_COLUMNS = ["slope", "slope_z", "slope_err", "n_visits", "sigma_visit", "abbe", "s_fade", "s_rise",
                     "fade_kind"]
_STATE = {}


def _aux(path):
    if path not in _STATE:
        _STATE[path] = pd.read_parquet(path, columns=AUX_COLUMNS).drop_duplicates("source_id")
    return _STATE[path]


# -- inject -------------------------------------------------------------------------------------------------------

def inject_one(item, a):
    """Injected fades in the FGK dwarfs of one light-curve file, scored; writes its table and a done-marker."""
    from cosmos_search.slow import search

    t0 = time.time()
    cfg = occurrence.Config.load(a["config"])
    aux = _aux(a["aux"])
    out = Path(a["out"])
    lc = pq.read_table(item["path"])
    ids = np.unique(np.asarray(lc["source_id"]))
    mine = aux[aux["source_id"].isin(ids)]
    hosts = mine.loc[occurrence.fgk_dwarf(mine, cfg) & (occurrence.stratum(mine["phot_g_mean_mag"], cfg) >= 0),
                     "source_id"].to_numpy()
    if a["host_share"] < 1:
        hosts = hosts[features.control_mask(hosts, a["host_share"])]
    lc = lc.filter(pc.is_in(lc["source_id"], value_set=pa.array(hosts, pa.int64())))
    cells = drift.star_cells(mine[mine["source_id"].isin(hosts)])
    table = rule_v3.load(a["rule"])[2]
    res = recall.injected_scores(lc, cfg.rates, lambda t: search.score(t, cells, table, period_search=False),
                                 columns=INJECTION_COLUMNS) if lc.num_rows else pd.DataFrame()
    (out / "injections").mkdir(parents=True, exist_ok=True)
    tmp = out / "injections" / f"{item['name']}.parquet.tmp"
    res.to_parquet(tmp, index=False)
    tmp.replace(out / "injections" / f"{item['name']}.parquet")
    rec = {"name": item["name"], "hosts": int(len(hosts)), "rows": int(len(res)), "seconds": round(time.time() - t0, 1)}
    (out / "done").mkdir(exist_ok=True)
    (out / "done" / f"{item['name']}.json").write_text(json.dumps(rec) + "\n")
    return rec


def inject(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [{"name": p.stem, "path": str(p)} for p in sorted(Path(a.lightcurves).glob("*.parquet"))]
    todo = [it for it in items if not (out / "done" / f"{it['name']}.json").exists()][: a.limit or None]
    print(f"{len(items)} light-curve files, {len(todo)} to inject into", flush=True)
    opts = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "func"}
    t0, n, failures = time.time(), 0, 0
    with (out / "manifest.jsonl").open("a") as fh:
        for k, (it, rec) in enumerate(_each(inject_one, todo, opts, a.workers), 1):
            if "error" in rec:
                failures += 1
            n += rec.get("hosts", 0)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 10 == 0 or "error" in rec or k == len(todo):
                print(f"[{k}/{len(todo)}] {n} hosts in {(time.time() - t0) / 60:.1f} min {rec}", flush=True)
    return failures


def _each(fn, items, opts, workers):
    """(item, fn(item, opts)) as each finishes, in `workers` processes (in this one when workers is 1); a failure
    comes back as {"name", "error"} and the item is retried on the next run."""
    def failed(it, exc):
        return {"name": it["name"], "error": repr(exc)[:500]}

    if workers <= 1:
        for it in items:
            try:
                yield it, fn(it, opts)
            except Exception as exc:  # noqa: BLE001
                yield it, failed(it, exc)
        return
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(fn, it, opts): it for it in items}
        for fut in as_completed(futures):
            try:
                yield futures[fut], fut.result()
            except Exception as exc:  # noqa: BLE001
                yield futures[fut], failed(futures[fut], exc)


# -- aux ----------------------------------------------------------------------------------------------------------

def aux_cmd(a):
    from astropy.table import Table

    from cosmos_search.gaia import epoch
    from cosmos_search.gaia.archive import tap
    from cosmos_search.net.tap import Tap

    files = sorted(Path(a.lightcurves).glob("*.parquet"))
    ids = [np.unique(np.asarray(pq.read_table(f, columns=["source_id"])["source_id"])) for f in files]
    cols = ", ".join(AUX_COLUMNS)
    t0 = time.time()
    if a.mirror:
        t = Tap(epoch.MIRRORS.get(a.mirror, a.mirror))

        def fetch(x):
            parts = _split(lambda lo, hi: t.query(
                f"SELECT {cols} FROM {a.schema}.gaia_source WHERE source_id BETWEEN {lo} AND {hi}", sync=True,
                maxrec=10 ** 7).to_pandas(), int(x.min()), int(x.max()))
            d = pd.concat(parts, ignore_index=True)
            d.columns = [c.lower() for c in d.columns]
            return d[d["source_id"].isin(x)]
        jobs = [x for x in ids if len(x)]
    else:
        t = tap()
        every = np.unique(np.concatenate(ids)) if ids else np.empty(0, np.int64)
        jobs = [every[i:i + a.chunk] for i in range(0, len(every), a.chunk)]

        def fetch(x):
            q = (f"SELECT {', '.join('g.' + c for c in AUX_COLUMNS)} FROM TAP_UPLOAD.ids AS u "
                 f"JOIN {a.schema}.gaia_source AS g ON g.source_id = u.source_id")
            for attempt in (0, 1):
                try:
                    d = t.query(q, uploads={"ids": Table({"source_id": x})}).to_pandas()
                    d.columns = [c.lower() for c in d.columns]
                    return d
                except Exception:  # noqa: BLE001 (one retry, then the step fails loudly)
                    if attempt:
                        raise
    with ThreadPoolExecutor(a.workers) as ex:
        aux = pd.concat(list(ex.map(fetch, jobs)), ignore_index=True).drop_duplicates("source_id")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    aux.to_parquet(a.out, index=False)
    n = sum(len(x) for x in ids)
    print(f"{len(aux)} of {n} stars' gaia_source rows in {time.time() - t0:.0f} s ({len(jobs)} queries)")
    return int(len(aux) < 0.99 * n)


# -- limit --------------------------------------------------------------------------------------------------------

def _files(path):
    p = Path(path)
    return sorted(f for f in p.glob("*.parquet") if pq.read_metadata(f).num_rows) if p.is_dir() else [p]


def _population(f, aux, cfg, kind):
    """The searched population among one per-star score file's rows, and how many rows it held."""
    have = set(pq.read_schema(f).names)
    d = pd.read_parquet(f, columns=[c for c in STAR_COLUMNS + ["kind", "phot_g_mean_mag", "g_mag"] if c in have])
    scored = len(d)
    if "g_mag" in d and "phot_g_mean_mag" not in d:
        d = d.rename(columns={"g_mag": "phot_g_mean_mag"})
    if kind is not None:
        if "kind" not in d:
            raise SystemExit(f"--kind given and {f} has no kind column")
        d = d[d["kind"] == kind].drop(columns="kind")
    else:
        d = d.drop(columns=["phot_g_mean_mag", "g_mag", "kind"], errors="ignore").join(aux, on="source_id", how="inner")
        d = d[occurrence.fgk_dwarf(d, cfg)]
    ok = (d["n_visits"] >= cfg.min_visits) & np.isfinite(d["slope_z"]) \
        & (occurrence.stratum(d["phot_g_mean_mag"], cfg) >= 0)
    return d[ok], scored


def _aux_index(aux_path, kind):
    if kind is None and not aux_path:
        raise SystemExit("--aux (parallaxes) or --kind (a label of the per-star table's kind column) is needed")
    return _aux(aux_path).set_index("source_id") if kind is None else None


def read_stars(path, aux_path, cfg, kind=None):
    """The searched population from per-star score files, in memory: FGK dwarfs in the config's G range with
    enough visits and a measured trend. With `kind`, the table's own `kind` column (slow.kinds) says which stars
    are FGK dwarfs and its `g_mag` their brightness; otherwise the stars are placed from `--aux`."""
    aux = _aux_index(aux_path, kind)
    parts = [_population(f, aux, cfg, kind) for f in _files(path)]
    return pd.concat([d for d, _ in parts], ignore_index=True).drop_duplicates("source_id"), sum(n for _, n in parts)


def gather(path, aux_path, cfg, kind, host_ids):
    """What the limit needs of the searched stars: their counts, the stars past the first cut, the hosts' rows,
    and the reference the injected stars are measured against.

    Per-star tables that carry a neighbour reference (`slope_ref`) are counted one file at a time, so a
    whole-sky pass never has to fit in memory. Others (`slow_search.py score` output) are read whole, because
    their reference is the running median of every star's trend."""
    files = _files(path)
    referenced = all("slope_ref" in pq.read_schema(f).names for f in files)
    if referenced:
        aux = _aux_index(aux_path, kind)
        batches = (_population(f, aux, cfg, kind) for f in files)
        ref = None
    else:
        stars, scored = read_stars(path, aux_path, cfg, kind)
        batches = [(stars, scored)]
        ref = occurrence.brightness_reference(stars["phot_g_mean_mag"].to_numpy(float), stars["slope"].to_numpy(float),
                                              cfg.reference_window)
    total, past, hosts, scored, spans, visits = None, [], [], 0, [], []
    for d, n in batches:
        scored += n
        if not len(d):
            continue
        z, z_raw, strat = occurrence.star_significances(d, cfg, ref)
        t = occurrence.tally(strat, z, z_raw, cfg)
        total = t if total is None else total + t
        past.append(occurrence.past_cut(d, z, strat, cfg))
        hosts.append(d.loc[d["source_id"].isin(host_ids), [c for c in ("source_id", "phot_g_mean_mag", "slope_ref",
                                                                      "slope_ref_err") if c in d]])
        spans.append((float(np.nanmedian(d["t_last"] - d["t_first"])) / 365.25, len(d)))
        visits.append((float(d["n_visits"].median()), len(d)))
    if total is None:
        raise SystemExit(f"no searched FGK dwarfs in {path}")
    return {"tally": total, "past": pd.concat(past, ignore_index=True), "hosts": pd.concat(hosts, ignore_index=True),
            "ref": ref, "scored": scored, "span": _weighted_median(spans), "visits": _weighted_median(visits),
            "referenced": referenced}


def _weighted_median(pairs):
    """Median of per-file medians, each weighted by its stars (exact for one file)."""
    v, w = np.array(pairs, float).T
    order = np.argsort(v)
    return float(v[order][np.searchsorted(np.cumsum(w[order]), w.sum() / 2)])


def figure(curve, cfg, title_note):
    """The limit against the fade rate, and what subtracting chance fades would give if fading and brightening
    were equally common by chance."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from cosmos_search.viz import style

    fig = style.figure(height=440)
    ax = fig.add_axes([0.17, 0.13, 0.79, 0.66])
    c = curve[np.isfinite(curve["limit"]) & (curve["limit"] < 1)]
    x = 100 * c["rate"]
    line, = ax.plot(x, c["limit"], color=style.c("accent"), marker="o", zorder=3, label=f"upper limit ({cfg.cl:.0%})")
    style.hover(line, [f"{100 * r.rate:g}% per year: fewer than 1 in {style.number(1 / r.limit)} "
                       f"({r.fading} fading at G < {r.g_max:g} beyond {r.z_min:g} errors; recall {r.recall:.0%})"
                       for r in c.itertuples()])
    if "mirror_limit" in c:
        ax.plot(x, c["mirror_limit"], color=style.c("muted"), ls=(0, (3, 3)), lw=style.px(1.5), marker="o",
                ms=style.px(4), zorder=2, label="if stars faded and brightened by chance equally often (not assumed)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style.plain_log(ax.xaxis)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"1 in {style.number(1 / v)}" if v < 1 else "all"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("fade rate (% of the flux at the first visit, per year)")
    ax.set_ylabel("share of FGK dwarfs fading")
    ax.legend(loc="upper right")
    style.title(fig, "Upper limit on the share of FGK dwarfs fading steadily", title_note)
    return fig


def limit(a):
    cfg = occurrence.Config.load(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inj = pd.concat([pd.read_parquet(f) for f in _files(a.injections)], ignore_index=True)
    t0 = time.time()
    got = gather(a.stars, a.aux, cfg, a.kind, pd.unique(inj["source_id"]))
    t, past, span = got["tally"], got["past"], got["span"]
    inj_z, inj_strat = occurrence.injection_significances(got["hosts"], inj, cfg, got["ref"])
    curve, cells = occurrence.curve_from_tally(t, past, inj_z, inj_strat, inj["rate"], cfg, span)
    sym = occurrence.symmetry(t, cfg)
    n_stars = int(t.stars.sum())
    # Fading against brightening counts over the whole G range, by significance: how far the symmetry holds.
    tails = pd.DataFrame({"z_min": cfg.z_cuts, "fading": occurrence.regions(t.fading)[-1].astype(int),
                          "brightening": occurrence.regions(t.brightening)[-1].astype(int)})
    rec = {"config": json.loads(cfg.to_json()), "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "inputs": {"stars": str(a.stars), "aux": str(a.aux), "injections": str(a.injections),
                      "stars_scored": int(got["scored"]), "fgk_dwarfs_searched": n_stars,
                      "injection_hosts": int((inj_strat[np.isclose(inj["rate"], inj["rate"].iloc[0])] >= 0).sum()),
                      "median_span_years": span, "median_visits": got["visits"],
                      "neighbour_reference": got["referenced"]},
           "curve": json.loads(curve.to_json(orient="records")), "symmetry": json.loads(sym.to_json(orient="records")),
           "tails": json.loads(tails.to_json(orient="records"))}
    (out / "limit.json").write_text(json.dumps(rec, indent=1))
    curve.to_csv(out / "curve.csv", index=False)
    sym.to_csv(out / "symmetry.csv", index=False)
    cells.to_parquet(out / "regions.parquet", index=False)
    # Every searched star past the first cut either way, strongest fades first: the stars behind the counts.
    past.sort_values("z").to_parquet(out / "past_cut.parquet", index=False)
    pd.options.display.width = 250
    print(f"{n_stars} FGK dwarfs searched of {got['scored']} stars scored; median span {span:.2f} yr; "
          f"{rec['inputs']['injection_hosts']} hosts; {time.time() - t0:.0f} s")
    show = curve.assign(rate_pct=100 * curve["rate"], recall=curve["recall"].round(3), one_in=curve["one_in"].round(0))
    cols = ["rate_pct", "g_max", "z_min", "stars", "fading", "brightening", "recall", "one_in"]
    if "mirror_limit" in curve:
        show = show.assign(mirror_one_in=(1 / curve["mirror_limit"]).round(0), excess_p=curve["excess_p"].round(4),
                           sky_dispersion=curve["sky_dispersion"].round(2))
        cols += ["mirror_g_max", "mirror_z_min", "mirror_fading", "mirror_brightening", "mirror_one_in", "excess_p",
                 "sky_dispersion"]
    print(show[cols].to_string(index=False))
    print(sym.round(3).to_string(index=False))
    print(tails.set_index("z_min").T.to_string())
    try:
        from cosmos_search.viz import style
    except ImportError:
        print("no figure: matplotlib is not installed (pip install -e '.[plots]')")
        return
    note = f"{n_stars:,} FGK dwarfs, G < {cfg.g_edges[-1]:g}, {span:.1f} years of Gaia photometry"
    style.render(figure, out / "limit", curve, cfg, note)


# -- planted ------------------------------------------------------------------------------------------------------

def planted(a):
    """Coverage on the real stars: plant fades in a random share of them (their own injected scores), compute the
    limit, and count how often it falls below the share planted."""
    cfg = occurrence.Config.load(a.config)
    stars, _ = read_stars(a.stars, a.aux, cfg, a.kind)
    inj = pd.concat([pd.read_parquet(f) for f in _files(a.injections)], ignore_index=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for rate in a.rates:
        for share in a.shares:
            below, mirror_below, p05, lims = 0, 0, 0, []
            for _ in range(a.trials):
                s, k = occurrence.plant(stars, inj, rate, share, rng)
                r = occurrence.limit_curve(s, inj, cfg, rates=[rate])[0].iloc[0]
                below += r["limit"] < k / len(stars)
                mirror_below += r.get("mirror_limit", np.inf) < k / len(stars)
                p05 += r.get("excess_p", 1.0) < 0.05
                lims.append(r["limit"])
            rows.append({"rate": rate, "share": share, "trials": a.trials, "limit_below_planted": below,
                         "mirror_limit_below_planted": mirror_below, "excess_p_below_0.05": p05,
                         "median_limit": float(np.median(lims))})
            print(rows[-1], flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "planted.csv", index=False)


# -- field --------------------------------------------------------------------------------------------------------

FIELD_COLUMNS = ("source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, parallax_error, pmra, pmra_error, pmdec, "
                 "pmdec_error, ruwe, phot_variable_flag, astrometric_params_solved, phot_g_n_obs")
M31 = (10.6847, 41.2687)   # Gaia's Andromeda photometric survey: 5.5 degrees around M31
FIELD_LEVEL = 6


def _split(fetch, lo, hi, depth=4):
    """fetch(lo, hi), or the same range in quarters when the service refuses (a synchronous query over ~10^4 stars)."""
    try:
        return [fetch(lo, hi)]
    except Exception:  # noqa: BLE001
        if depth == 0:
            raise
    q = (hi - lo + 1) // 4
    return [x for k in range(4) for x in _split(fetch, lo + k * q, hi if k == 3 else lo + (k + 1) * q - 1, depth - 1)]


def _field_pixel(p, a):
    from cosmos_search.gaia import epoch
    from cosmos_search.net.tap import Tap

    t0 = time.time()
    out = Path(a["out"])
    width = 2 ** 35 * 4 ** (12 - FIELD_LEVEL)
    lo, hi = int(p) * width, (int(p) + 1) * width - 1
    tap = Tap(epoch.MIRRORS.get(a["mirror"], a["mirror"]))
    aux = pd.concat(_split(lambda x, y: tap.query(
        f"SELECT {FIELD_COLUMNS} FROM gaiadr3.gaia_source WHERE source_id BETWEEN {x} AND {y} "
        f"AND in_andromeda_survey = 'true' AND phot_g_mean_mag < {a['g_max']}", sync=True, maxrec=10 ** 7).to_pandas(),
        lo, hi), ignore_index=True)
    aux.columns = [c.lower() for c in aux.columns]
    if not len(aux):
        return {"pix": int(p), "stars": 0}
    step = width // 16
    lcs = [t for k in range(16) for t in _split(
        lambda x, y: epoch.mirror_epoch_photometry(source_id_range=(x, y), mirror=a["mirror"]),
        lo + k * step, lo + (k + 1) * step - 1)]
    lc = pa.concat_tables(lcs)
    lc = lc.filter(pc.is_in(lc["source_id"], value_set=pa.array(aux["source_id"].to_numpy(), pa.int64())))
    aux.to_parquet(out / "aux" / f"pix_{p:06d}.parquet", index=False)
    tmp = out / "lightcurves" / f"pix_{p:06d}.parquet.tmp"
    pq.write_table(lc, tmp, compression="zstd")
    tmp.replace(out / "lightcurves" / f"pix_{p:06d}.parquet")
    return {"pix": int(p), "stars": int(len(aux)), "rows": lc.num_rows, "seconds": round(time.time() - t0, 1)}


def field(a):
    import healpy as hp

    out = Path(a.out)
    for d in ("lightcurves", "aux"):
        (out / d).mkdir(parents=True, exist_ok=True)
    pix = np.sort(hp.query_disc(2 ** FIELD_LEVEL, hp.ang2vec(*M31, lonlat=True), np.radians(5.7), inclusive=True,
                                nest=True))
    done = {int(f.stem.split("_")[1]) for f in (out / "lightcurves").glob("pix_*.parquet")}
    todo = [int(p) for p in pix if int(p) not in done]
    print(f"{len(pix)} level-{FIELD_LEVEL} pixels, {len(todo)} to fetch", flush=True)
    opts = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "func"}
    t0, n, failures = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as pool, (out / "manifest.jsonl").open("a") as fh:
        futures = {pool.submit(_field_pixel, p, opts): p for p in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
                n += rec["stars"]
            except Exception as exc:  # noqa: BLE001 (the pixel is retried on the next run)
                failures += 1
                rec = {"pix": futures[fut], "error": repr(exc)[:300]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 10 == 0 or "error" in rec or k == len(todo):
                print(f"[{k}/{len(todo)}] {n} stars in {(time.time() - t0) / 60:.1f} min", flush=True)
    parts = sorted((out / "aux").glob("pix_*.parquet"))
    aux = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True).drop_duplicates("source_id")
    aux.to_parquet(out / "aux.parquet", index=False)
    print(f"aux.parquet: {len(aux)} stars")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inject", help="steady fades injected into FGK dwarfs' light curves and scored")
    i.add_argument("--lightcurves", required=True, help="directory of light-curve Parquet files (epoch.SCHEMA)")
    i.add_argument("--aux", required=True, help="Parquet with " + ", ".join(AUX_COLUMNS))
    i.add_argument("--rule", default=str(rule_v3.RULE_DIR), help="rule directory (drift table)")
    i.add_argument("--out", required=True)
    i.add_argument("--config", default=occurrence.Config.name, help="frozen config: a name in prereg/occurrence or a path")
    i.add_argument("--host-share", type=float, default=1.0, help="share of the FGK dwarfs used as hosts (by hash)")
    i.add_argument("--workers", type=int, default=os.cpu_count())
    i.add_argument("--limit", type=int, default=0, help="first N files only")
    i.set_defaults(func=inject)
    x = sub.add_parser("aux", help="G, BP-RP and parallax of the stars of a light-curve directory, from gaia_source")
    x.add_argument("--lightcurves", required=True)
    x.add_argument("--out", required=True, help="Parquet file to write")
    x.add_argument("--schema", default="gaiadr4")
    x.add_argument("--mirror", help="a partner centre's TAP service (a name in epoch.MIRRORS, or a URL) instead of ESA")
    x.add_argument("--chunk", type=int, default=50_000, help="source_ids per uploaded query (ESA)")
    x.add_argument("--workers", type=int, default=4)
    x.set_defaults(func=aux_cmd)
    m = sub.add_parser("limit", help="the limit per rate from per-star scores and injections")
    m.add_argument("--stars", required=True, help="per-star score files (a directory or one Parquet file)")
    m.add_argument("--aux", help="Parquet with " + ", ".join(AUX_COLUMNS) + " (not needed with --kind)")
    m.add_argument("--kind", help="take the FGK dwarfs from the per-star table's `kind` column: the label to keep")
    m.add_argument("--injections", required=True, help="the inject step's injections directory")
    m.add_argument("--out", required=True)
    m.add_argument("--config", default=occurrence.Config.name)
    m.set_defaults(func=limit)
    c = sub.add_parser("planted", help="check on real stars: how often the limit falls below a share of fades planted")
    c.add_argument("--stars", required=True)
    c.add_argument("--aux")
    c.add_argument("--kind")
    c.add_argument("--injections", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--config", default=occurrence.Config.name)
    c.add_argument("--rates", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    c.add_argument("--shares", type=float, nargs="+", default=[0.0, 0.0005, 0.002])
    c.add_argument("--trials", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=planted)
    f = sub.add_parser("field", help="DR3 demonstration sample: Andromeda-survey stars and light curves from AIP")
    f.add_argument("--out", required=True)
    f.add_argument("--g-max", type=float, default=19.0)
    f.add_argument("--mirror", default="aip")
    f.add_argument("--workers", type=int, default=4)
    f.set_defaults(func=field)
    a = ap.parse_args(argv)
    r = a.func(a)
    return 1 if isinstance(r, int) and r else 0


if __name__ == "__main__":
    sys.exit(main())
