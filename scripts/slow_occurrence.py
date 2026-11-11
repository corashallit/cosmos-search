"""Upper limit on how common steady fades are among FGK dwarfs (cosmos_search.slow.occurrence; docs/arms/occurrence.md).

    python scripts/slow_occurrence.py aux --lightcurves S/lightcurves --schema gaiadr4 --out S/aux_plx.parquet
    python scripts/slow_occurrence.py inject --lightcurves S/lightcurves --aux S/aux_plx.parquet --rule RULE --out O
    python scripts/slow_occurrence.py limit --stars R/summary --reference L/reference.parquet --aux AUX \\
        --injections O/injections --out O/limit

    python scripts/slow_occurrence.py aux --sky 4 --schema gaiadr4 --out A          # every star: the catalogue in slices
    python scripts/slow_occurrence.py inject --lightcurves R/lightcurves --random-share 0.001 --aux A --rule RULE --out O

`aux`: G, BP-RP, parallax and its error from gaia_source, and the ratio (BP + RP) / G of the mean fluxes. The
light curves carry no parallax, and only the parallax tells a dwarf from a giant; the ratio tells which dwarfs
have a neighbour's light mixed with theirs. For a sample (`--lightcurves`): every star of a directory of light-curve
files, from ESA's archive with the project's login (uploaded id lists), or with `--mirror` from a partner centre
by source_id range, into one file. For the whole sky (`--sky LEVEL --out DIR`): one query and one file per HEALPix
pixel, with the catalogue-side part of the FGK-dwarf selection in the query, so a few 10^8 rows come back
instead of 2 x 10^9; `inject` and `limit` read only the slices a file of stars needs. A second run resumes.

`inject`: steady fades at each rate of the frozen config, injected into the light curves of FGK dwarfs and scored
by the function the search's pass runs on every star (`summary.star_summary` with the rule's drift table, and the
pass's bad-epoch options). Light curves are a directory of Parquet files in epoch.SCHEMA: the light curves a
`slow_search.py pass --keep-lightcurves all` kept, a `dr4_bulk pull`, or the field below. `--aux` needs
phot_g_mean_mag, bp_rp, parallax and parallax_error. Done-markers let a second run resume. Every FGK dwarf is a
host here; `limit` uses the ones it searches.

`limit`: the per-star table of the stars searched, the same `--aux` columns for them, and the injections. The
table is the pass's summary/ directory (`slow_search.py pass`: one row per star with source_id, g_mag, slope,
slope_err, n_visits, t_first, t_last, and `ref`, which marks the rows that are a fair sample of their magnitude;
only those are counted) with `--reference`, the reference.parquet of `slow_search.py flag`: each star's trend, and
each injected star's, is then measured against the median trend of the stars near it on the sky and in brightness,
and the searched dwarfs' own median at each brightness and flux excess is taken out. Dwarfs whose flux ratio is
more than the config's `max_flux_excess` from the dwarfs' median at their colour are not searched (their G
photometry drifts), so `--aux` must carry the ratio: `flux_ratio` as `aux` writes it, DR3's
`phot_bp_rp_excess_factor`, or the three mean fluxes. `--config steady-fades-v1` searches every FGK dwarf and
needs none. The files are read one at a time, twice, so a whole-sky table never has to fit in memory. Without
`--reference` a table that carries no `slope_ref` is read whole and measured against the running median of the
searched stars of similar brightness. Writes
limit.json (config, inputs, the limit per rate, the symmetry tables, the trend by flux excess), curve.csv,
regions.parquet (counts and recall of every region per rate), past_cut.parquet (every searched star past the first cut either way, strongest fades
first) and, with matplotlib, the figure.

`planted` checks the limit on the real stars: a random share of them is given its own injected scores, so that
share really fades, and the limit is computed again; over many draws it should fall below the planted share no
more than 5% of the time. The subtracted figure is counted the same way.

`field` fetches the DR3 demonstration sample: every star of Gaia's Andromeda survey (the only part of DR3 with
light curves for stars not flagged variable) brighter than --g-max, from the AIP mirror, one file per level-6
HEALPix pixel. Score it with `slow_search.py pass --lightcurves F/lightcurves --rule RULE --no-period`.
"""

import argparse
import datetime
import json
import os
import re
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
from cosmos_search.slow import drift, occurrence, recall, rule_v3, summary

AUX_COLUMNS = ["source_id", "phot_g_mean_mag", "bp_rp", "parallax", "parallax_error"]
# (BP + RP) / G mean flux, for the stars whose light is mixed with a neighbour's (`occurrence.flux_excess`). The
# catalogue step asks the archive for the ratio; a file from elsewhere can carry DR3's own column or the fluxes.
RATIO_COLUMNS = [*occurrence.RATIO_COLUMNS, "phot_g_mean_flux", "phot_bp_mean_flux", "phot_rp_mean_flux"]


def aux_select(prefix=""):
    """The columns the catalogue step asks for, as the SELECT list of a query on gaia_source."""
    bp, rp, g = (f"{prefix}phot_{band}_mean_flux" for band in ("bp", "rp", "g"))
    return ", ".join([*(prefix + c for c in AUX_COLUMNS), f"({bp} + {rp}) / {g} AS flux_ratio"])


def _read_aux(f):
    """AUX_COLUMNS of one catalogue file, with whichever flux-ratio columns it has."""
    have = set(pq.read_schema(f).names)
    return pd.read_parquet(f, columns=[*AUX_COLUMNS, *(c for c in RATIO_COLUMNS if c in have)])
# What the limit reads of a per-star table, and what is kept of an injected star's scores.
STAR_COLUMNS = ["source_id", "g_mag", "slope", "slope_z", "slope_err", "slope_ref", "slope_ref_err", "slope_z_raw",
                "n_visits", "t_first", "t_last"]
INJECTION_COLUMNS = ["g_mag", "bp_rp", "slope", "slope_z", "slope_err", "n_visits", "t_first", "t_last", "sigma_visit",
                     "abbe", "s_fade", "s_rise", "fade_kind"]
_STATE = {}


def _aux_slices(path):
    """(first source_id, last source_id, file) of each sky slice in a directory written by `aux --sky`."""
    key = ("slices", str(path))
    if key not in _STATE:
        rows = []
        for f in sorted(Path(path).glob("aux_L*_*.parquet")):
            level, pix = (int(x) for x in re.match(r"aux_L(\d+)_(\d+)", f.name).groups())
            width = 2 ** 35 * 4 ** (12 - level)
            rows.append((pix * width, (pix + 1) * width - 1, f))
        if not rows:
            raise SystemExit(f"no aux_L*_*.parquet sky slices in {path}")
        _STATE[key] = rows
    return _STATE[key]


def flux_locus(path, cfg):
    """The FGK dwarfs' median flux ratio per colour step (`occurrence.locus`), from every catalogue row of
    `path` (a file, or a directory of sky slices read one at a time), computed once."""
    key = ("locus", str(path), cfg.name)
    if key not in _STATE:
        files = [f for _a, _b, f in _aux_slices(path)] if Path(path).is_dir() else [path]
        hist = sum(occurrence.locus_hist(_read_aux(f), cfg) for f in files)
        if not hist.sum():
            raise SystemExit(f"{path} has no flux ratio for its FGK dwarfs ({', '.join(RATIO_COLUMNS[:2])}, or the "
                             f"mean fluxes), which {cfg.name} needs to leave out stars mixed with a neighbour: "
                             "fetch the catalogue rows again with `aux`, or use --config steady-fades-v1")
        _STATE[key] = occurrence.locus(hist)
    return _STATE[key]


def aux_rows(path, lo, hi):
    """Catalogue rows (AUX_COLUMNS and the flux ratio, where the file has one) indexed by source_id. `path` is
    one Parquet file, read whole and once (a sample), or a directory of sky slices from `aux --sky`, of which
    only those that overlap the source_ids lo to hi are read, so the catalogue of the whole sky never has to fit
    in memory."""
    path = str(path)
    if not Path(path).is_dir():
        if path not in _STATE:
            _STATE[path] = _read_aux(path).drop_duplicates("source_id").set_index("source_id")
        return _STATE[path]
    parts = [_read_aux(f) for a, b, f in _aux_slices(path) if a <= hi and b >= lo]
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=AUX_COLUMNS)
    return d[(d["source_id"] >= lo) & (d["source_id"] <= hi)].drop_duplicates("source_id").set_index("source_id")


# -- inject -------------------------------------------------------------------------------------------------------

def inject_one(item, a):
    """Injected fades in the FGK dwarfs of one light-curve file, scored; writes its table and a done-marker.

    Each injected star is scored as the pass scores a real one: placed in its magnitude and colour cell by its
    own (faded) median fluxes, or by the catalogue when the pass was given one (--catalogue-cells)."""
    t0 = time.time()
    cfg = occurrence.Config.load(a["config"])
    out = Path(a["out"])
    lc = pq.read_table(item["path"])
    ids = np.unique(np.asarray(lc["source_id"]))
    aux = aux_rows(a["aux"], ids.min(), ids.max()) if len(ids) else pd.DataFrame(columns=AUX_COLUMNS[1:])
    mine = aux.loc[aux.index.intersection(ids)].reset_index(names="source_id")
    hosts = mine.loc[occurrence.fgk_dwarf(mine, cfg) & (occurrence.stratum(mine["phot_g_mean_mag"], cfg) >= 0),
                     "source_id"].to_numpy()
    if a["random_share"] > 0:    # the pass's random share: the kept light curves that are a fair sample
        hosts = hosts[summary.random_share(hosts, a["random_share"])]
    if a["host_share"] < 1:
        hosts = hosts[features.control_mask(hosts, a["host_share"])]
    lc = lc.filter(pc.is_in(lc["source_id"], value_set=pa.array(hosts, pa.int64())))
    cells = drift.star_cells(mine[mine["source_id"].isin(hosts)]) if a["catalogue_cells"] else None
    table = rule_v3.load(a["rule"])[2]
    res = recall.injected_scores(lc, cfg.rates, lambda t: summary.star_summary(
        t, table, cells=cells, marks=not a["no_marks"], own=not a["no_own_test"]),
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

def sky_where(cfg):
    """The catalogue-side part of the FGK-dwarf selection (`occurrence.fgk_dwarf` applies all of it again): what
    keeps the rows sent back to a few 10^8 for the whole sky."""
    return (f"phot_g_mean_mag < {cfg.g_edges[-1]:g} AND parallax >= {cfg.min_parallax_snr:g} * parallax_error "
            f"AND bp_rp >= {cfg.colour_lo:g} AND bp_rp < {cfg.colour_hi:g}")


def sky_rows(query, lo, hi, depth=6):
    """`query(lo, hi)` (a frame of AUX_COLUMNS and the flux ratio for a source_id range), in quarters when the
    service refuses the whole range, as one frame sorted by source_id."""
    parts = [d for d in _split(query, lo, hi, depth) if len(d)]
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*AUX_COLUMNS, "flux_ratio"])
    d.columns = [c.lower() for c in d.columns]
    cols = [*AUX_COLUMNS, *(c for c in ("flux_ratio",) if c in d)]
    return d[cols].drop_duplicates("source_id").sort_values("source_id").reset_index(drop=True)


def _sky_slice(pix, a):
    """Catalogue rows of the FGK-dwarf candidates of one HEALPix pixel, as one file of the --out directory."""
    from cosmos_search.gaia import epoch
    from cosmos_search.gaia.archive import tap
    from cosmos_search.net.tap import Tap

    t0 = time.time()
    width = 2 ** 35 * 4 ** (12 - a["sky"])
    if not a["mirror"]:
        t = tap()
    else:   # one handle for every slice, so its limit on queries at once is shared
        t = _STATE.setdefault(("tap", a["mirror"]), Tap(epoch.MIRRORS.get(a["mirror"], a["mirror"]),
                                                        job_slots=a["workers"]))
    where = sky_where(occurrence.Config.load(a["config"]))

    def query(lo, hi):
        return t.query(f"SELECT {aux_select()} FROM {a['schema']}.gaia_source WHERE source_id BETWEEN {lo} "
                       f"AND {hi} AND {where}", sync=bool(a["mirror"]), maxrec=10 ** 7, cache=False).to_pandas()

    d = sky_rows(query, pix * width, (pix + 1) * width - 1)
    path = Path(a["out"]) / f"aux_L{a['sky']}_{pix:07d}.parquet"
    d.to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)
    return {"pix": int(pix), "rows": int(len(d)), "seconds": round(time.time() - t0, 1)}


def aux_sky(a):
    """The whole sky (or --pixels of it), one HEALPix pixel per query and per file; a second run resumes."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    first, last = (int(x) for x in a.pixels.split("-")) if a.pixels else (0, 12 * 4 ** a.sky - 1)
    done = {int(re.match(r"aux_L\d+_(\d+)", f.name).group(1)) for f in out.glob(f"aux_L{a.sky}_*.parquet")}
    todo = [p for p in range(first, last + 1) if p not in done]
    print(f"{last - first + 1} level-{a.sky} pixels, {len(todo)} to query", flush=True)
    opts = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "func"}
    t0, n, failures = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as pool, (out / "manifest.jsonl").open("a") as fh:
        futures = {pool.submit(_sky_slice, p, opts): p for p in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
                n += rec["rows"]
            except Exception as exc:  # noqa: BLE001 (the pixel is retried on the next run)
                failures += 1
                rec = {"pix": futures[fut], "error": repr(exc)[:300]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 50 == 0 or "error" in rec or k == len(todo):
                print(f"[{k}/{len(todo)}] {n} rows in {(time.time() - t0) / 60:.1f} min {rec}", flush=True)
    return failures


def aux_cmd(a):
    if a.sky is not None:
        return aux_sky(a)
    if not a.lightcurves:
        raise SystemExit("aux needs --lightcurves (a sample) or --sky (the whole sky in slices)")
    from astropy.table import Table

    from cosmos_search.gaia import epoch
    from cosmos_search.gaia.archive import tap
    from cosmos_search.net.tap import Tap

    files = sorted(Path(a.lightcurves).glob("*.parquet"))
    ids = [np.unique(np.asarray(pq.read_table(f, columns=["source_id"])["source_id"])) for f in files]
    cols = aux_select()
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
            q = (f"SELECT {aux_select('g.')} FROM TAP_UPLOAD.ids AS u "
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


def _reference(path):
    """The neighbour reference of `slow_search.py flag` (reference.parquet), when one is given."""
    if path and path not in _STATE:
        _STATE[path] = pd.read_parquet(path)
    return _STATE.get(path)


def _referenced(d, reference):
    """Rows measured against the neighbour reference: `summary.apply_reference` by each row's own g_mag, which
    adds slope_ref, slope_ref_err, slope_z_raw and replaces slope_z. Rows that already carry `slope_ref`, and
    every row when no reference is given, are returned as they are (with slope_z = slope / slope_err when they
    had none). The reference's part by flux excess is left out: stars carry the catalogue's flux ratio here and
    fakes none, and the limit centres both by excess itself (`occurrence.centre`)."""
    for c in d.columns:
        if d[c].dtype == np.float32:
            d[c] = d[c].astype(np.float64)
    if "slope_z" not in d:
        with np.errstate(invalid="ignore", divide="ignore"):
            d["slope_z"] = d["slope"] / d["slope_err"]
    if reference is None or "slope_ref" in d or not len(d):
        return d
    if "g_mag" not in d:
        raise SystemExit("--reference needs g_mag in the per-star table and the injections (the pass's own magnitude)")
    return summary.apply_reference(d, reference, flux_ratio=False)


def with_host_excess(inj, hosts):
    """Injected stars with their hosts' flux excess (`hosts`: searched stars' rows), which a fade in all three
    bands leaves as it was; NaN for a host that is not searched."""
    if "flux_excess" not in hosts:
        return inj
    excess = hosts.drop_duplicates("source_id").set_index("source_id")["flux_excess"]
    return inj.assign(flux_excess=excess.reindex(inj["source_id"].to_numpy()).to_numpy())


def read_injections(path, reference=None, offset=None, hosts=None):
    """Every injected star's scores, measured as the searched stars are: against the neighbour reference when
    one is given, then with the searched stars' own median taken out (`offset`, by each host's flux excess when
    the offsets have steps of it: `hosts`, the searched stars' rows)."""
    inj = _referenced(pd.concat([pd.read_parquet(f) for f in _files(path)], ignore_index=True), reference)
    if hosts is not None:
        inj = with_host_excess(inj, hosts)
    return inj if offset is None else occurrence.centre(inj, offset)


def _dwarfs(f, aux, cfg, reference=None):
    """The FGK dwarfs among one per-star score file's rows, and how many rows it held: by the catalogue rows of
    `aux` (a file or a directory of sky slices, `aux_rows`), in the config's G range, with enough visits and a
    measured trend, measured against `reference`. Rows of a bulk-file summary that are not a fair sample of
    their magnitude (`ref` false) are left out. With a cut on the flux excess in the config, each star's excess
    is in `flux_excess`."""
    have = set(pq.read_schema(f).names)
    d = pd.read_parquet(f, columns=[c for c in [*STAR_COLUMNS, "ref"] if c in have])
    scored = len(d)
    if "ref" in d:
        d = d[d["ref"].astype(bool)].drop(columns="ref")
    if len(d):
        d = d.join(aux_rows(aux, d["source_id"].min(), d["source_id"].max()), on="source_id", how="inner")
    else:
        d = d.assign(**{c: np.nan for c in AUX_COLUMNS[1:]})
    d = _referenced(d[occurrence.fgk_dwarf(d, cfg)].reset_index(drop=True), reference)
    ok = (d["n_visits"] >= cfg.min_visits) & np.isfinite(d["slope_z"]) \
        & (occurrence.stratum(d["phot_g_mean_mag"], cfg) >= 0)
    d = d[ok].reset_index(drop=True)
    if cfg.max_flux_excess is not None:
        ratio = occurrence.flux_ratio(d) if len(d) else np.empty(0)
        if ratio is None or not np.isfinite(ratio).all():
            raise SystemExit(f"FGK dwarfs of {f} have no flux ratio in {aux}, which {cfg.name} needs: fetch the "
                             "catalogue rows again with `aux`, or use --config steady-fades-v1")
        d = d.assign(flux_excess=occurrence.flux_excess(d, flux_locus(aux, cfg), cfg) if len(d) else ratio)
    return d, scored


def _population(f, aux, cfg, reference=None, offset=None):
    """The searched population among one per-star score file's rows (`_dwarfs` less the stars whose light is
    mixed with a neighbour's, with the searched stars' own median taken out when `offset` is given), how many
    rows the file held, and how many dwarfs were left out as mixed."""
    d, scored = _dwarfs(f, aux, cfg, reference)
    n = len(d)
    if "flux_excess" in d:
        d = d[occurrence.unblended(d["flux_excess"], cfg)].reset_index(drop=True)
    if offset is not None:
        d = occurrence.centre(d, offset)
        d = d[np.isfinite(d["slope_z"])]
    return d, scored, n - len(d)


def population_offset(path, aux, cfg, reference):
    """One pass over the per-star files for what the searched stars are then measured against: their median
    trend against the neighbour reference per G range (`occurrence.offsets`) with the stars behind each, and, as
    a check on the cut, the median trend of every FGK dwarf by class of flux excess (`occurrence.excess_table`;
    None when the config has no cut)."""
    hist, by_excess = 0, 0
    rows = len(occurrence.EXCESS_EDGES) + 2
    for f in _files(path):
        d = _dwarfs(f, aux, cfg, reference)[0]
        if "flux_excess" in d:
            by_excess = by_excess + occurrence.offset_hist(d, occurrence.excess_class(d["flux_excess"]), rows)
            d = d[occurrence.unblended(d["flux_excess"], cfg)]
        hist = hist + occurrence.offset_hist(d)
    table = occurrence.excess_table(by_excess, cfg) if cfg.max_flux_excess is not None else None
    return occurrence.offsets(hist), hist.sum(axis=1), table


def read_stars(path, aux_path, cfg, reference=None):
    """The searched population from per-star score files, in memory (`_population` of every file), and with a
    neighbour reference the offset that centred it (None without one)."""
    offset = population_offset(path, aux_path, cfg, reference)[0] if reference is not None else None
    parts = [_population(f, aux_path, cfg, reference, offset) for f in _files(path)]
    stars = pd.concat([p[0] for p in parts], ignore_index=True).drop_duplicates("source_id")
    return stars, sum(p[1] for p in parts), offset


def gather(path, aux_path, cfg, host_ids, reference=None):
    """What the limit needs of the searched stars: their counts, the stars past the first cut, the hosts' rows,
    and the reference the injected stars are measured against.

    Per-star tables measured against a neighbour reference (`reference`, or their own `slope_ref`) are counted
    one file at a time, so the scores of a whole-sky pass never have to fit in memory; with `reference` the
    files are read twice, first for the searched stars' own median per G range. Others are read whole, because
    their reference is the running median of every star's trend."""
    files = _files(path)
    referenced = reference is not None or all("slope_ref" in pq.read_schema(f).names for f in files)
    offset = counts = by_excess = None
    if referenced:
        if reference is not None:
            offset, counts, by_excess = population_offset(path, aux_path, cfg, reference)
        batches = (_population(f, aux_path, cfg, reference, offset) for f in files)
        ref = None
    else:
        parts = [_population(f, aux_path, cfg) for f in files]
        stars = pd.concat([p[0] for p in parts], ignore_index=True).drop_duplicates("source_id")
        batches = [(stars, sum(p[1] for p in parts), sum(p[2] for p in parts))]
        ref = occurrence.brightness_reference(stars["phot_g_mean_mag"].to_numpy(float), stars["slope"].to_numpy(float),
                                              cfg.reference_window)
    total, past, hosts, scored, mixed, spans, visits = None, [], [], 0, 0, [], []
    for d, n, left_out in batches:
        scored += n
        mixed += left_out
        if not len(d):
            continue
        z, z_raw, strat = occurrence.star_significances(d, cfg, ref)
        t = occurrence.tally(strat, z, z_raw, cfg)
        total = t if total is None else total + t
        past.append(occurrence.past_cut(d, z, strat, cfg))
        hosts.append(d.loc[d["source_id"].isin(host_ids), [c for c in ("source_id", "phot_g_mean_mag", "slope_ref",
                                                                      "slope_ref_err", "flux_excess") if c in d]])
        spans.append((float(np.nanmedian(d["t_last"] - d["t_first"])) / 365.25, len(d)))
        visits.append((float(d["n_visits"].median()), len(d)))
    if total is None:
        raise SystemExit(f"no searched FGK dwarfs in {path}")
    return {"tally": total, "past": pd.concat(past, ignore_index=True), "hosts": pd.concat(hosts, ignore_index=True),
            "ref": ref, "scored": scored, "span": _weighted_median(spans), "visits": _weighted_median(visits),
            "referenced": referenced, "offset": offset, "offset_stars": counts, "mixed": mixed,
            "by_excess": by_excess}


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
    reference = _reference(a.reference)
    inj = read_injections(a.injections, reference)
    t0 = time.time()
    got = gather(a.stars, a.aux, cfg, pd.unique(inj["source_id"]), reference)
    centred = pd.DataFrame()
    if got["offset"] is not None:
        inj = occurrence.centre(with_host_excess(inj, got["hosts"]), got["offset"])
        centred = occurrence.offset_table(got["offset"], got["offset_stars"])
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
                      "reference": str(a.reference) if a.reference else None,
                      "stars_scored": int(got["scored"]), "fgk_dwarfs_searched": n_stars,
                      # FGK dwarfs not searched because their light is mixed with a neighbour's (flux excess).
                      "fgk_dwarfs_mixed_with_a_neighbour": int(got["mixed"]),
                      "injection_hosts": int((inj_strat[np.isclose(inj["rate"], inj["rate"].iloc[0])] >= 0).sum()),
                      "median_span_years": span, "median_visits": got["visits"],
                      "neighbour_reference": got["referenced"]},
           "curve": json.loads(curve.to_json(orient="records")), "symmetry": json.loads(sym.to_json(orient="records")),
           "tails": json.loads(tails.to_json(orient="records")),
           # The searched stars' median trend against the neighbour reference, taken out per G range (and, with
           # a cut on the flux excess, per step of it).
           "population_offset": json.loads(centred.to_json(orient="records"))}
    if cfg.max_flux_excess is not None:
        edges = occurrence.locus_colours(cfg)
        # The dwarfs' median (BP + RP) / G flux ratio by colour, and their median trend by excess over it.
        rec["flux_ratio_by_colour"] = [{"bp_rp_lo": float(a), "bp_rp_hi": float(b), "median_ratio": float(r)}
                                       for a, b, r in zip(edges[:-1], edges[1:], flux_locus(a.aux, cfg))]
        if got["by_excess"] is not None:
            rec["trend_by_flux_excess"] = json.loads(got["by_excess"].to_json(orient="records"))
    (out / "limit.json").write_text(json.dumps(rec, indent=1))
    curve.to_csv(out / "curve.csv", index=False)
    sym.to_csv(out / "symmetry.csv", index=False)
    cells.to_parquet(out / "regions.parquet", index=False)
    # Every searched star past the first cut either way, strongest fades first: the stars behind the counts.
    past.sort_values("z").to_parquet(out / "past_cut.parquet", index=False)
    pd.options.display.width = 250
    print(f"{n_stars} FGK dwarfs searched of {got['scored']} stars scored ({got['mixed']} more left out as mixed "
          f"with a neighbour); median span {span:.2f} yr; {rec['inputs']['injection_hosts']} hosts; "
          f"{time.time() - t0:.0f} s")
    show = curve.assign(rate_pct=100 * curve["rate"], recall=curve["recall"].round(3), one_in=curve["one_in"].round(0))
    cols = ["rate_pct", "g_max", "z_min", "stars", "fading", "brightening", "recall", "one_in"]
    if "mirror_limit" in curve:
        show = show.assign(mirror_one_in=(1 / curve["mirror_limit"]).round(0), excess_p=curve["excess_p"].round(4),
                           sky_dispersion=curve["sky_dispersion"].round(2))
        cols += ["mirror_g_max", "mirror_z_min", "mirror_fading", "mirror_brightening", "mirror_one_in", "excess_p",
                 "sky_dispersion"]
    print(show[cols].to_string(index=False))
    print(sym.round(3).to_string(index=False))
    if len(centred) and "excess_lo" in centred:
        # % a year taken out, by G range (rows) and step of flux excess (columns, by their lower edge).
        print((100 * centred[centred["g_lo"] >= 12].pivot(index="g_lo", columns="excess_lo", values="median_trend_per_year"))
              .round(3).to_string())
    elif len(centred):
        print(centred[centred["stars"] > 0].assign(pct_per_year=lambda d: (100 * d["median_trend_per_year"]).round(4))
              .drop(columns="median_trend_per_year").to_string(index=False))
    print(tails.set_index("z_min").T.to_string())
    if got["by_excess"] is not None:
        print(got["by_excess"].assign(pct_per_year=lambda d: (100 * d["median_trend_per_year"]).round(4))
              .drop(columns="median_trend_per_year").to_string(index=False))
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
    reference = _reference(a.reference)
    stars, _, offset = read_stars(a.stars, a.aux, cfg, reference)
    inj = read_injections(a.injections, reference, offset, stars)
    rng = np.random.default_rng(a.seed)
    rows = []
    for rate in a.rates:
        for share in a.shares:
            below, mirror_below, p05, lims = 0, 0, 0, []
            for _ in range(a.trials):
                s, k = occurrence.plant(stars, inj, rate, share, rng)
                if offset is not None:
                    # As the limit step would see such a sky: the dwarfs' own median, moved by the planted fades,
                    # is taken out again, while recall still comes from fades injected into one star at a time.
                    s = occurrence.centre(s, occurrence.offsets(occurrence.offset_hist(s)))
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
                 "pmdec_error, ruwe, phot_variable_flag, astrometric_params_solved, phot_g_n_obs, "
                 "phot_bp_rp_excess_factor")
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
    i.add_argument("--aux", required=True, help="Parquet with " + ", ".join(AUX_COLUMNS) + ", or an `aux --sky` directory")
    i.add_argument("--rule", default=str(rule_v3.RULE_DIR), help="rule directory (drift table)")
    i.add_argument("--out", required=True)
    i.add_argument("--config", default=occurrence.Config.name, help="frozen config: a name in prereg/occurrence or a path")
    i.add_argument("--host-share", type=float, default=1.0, help="share of the FGK dwarfs used as hosts (by hash)")
    i.add_argument("--random-share", type=float, default=0.0,
                   help="hosts are only the stars of the pass's --random-share: needed when --lightcurves is a pass's "
                        "lightcurves/ directory, which also holds the stars near a threshold")
    i.add_argument("--no-marks", action="store_true", help="as the pass's --no-marks (DR3 read as DR4 arrives)")
    i.add_argument("--no-own-test", action="store_true", help="as the pass's --no-own-test")
    i.add_argument("--catalogue-cells", action="store_true",
                   help="place stars in magnitude and colour cells by --aux, when the pass was given --aux")
    i.add_argument("--workers", type=int, default=os.cpu_count())
    i.add_argument("--limit", type=int, default=0, help="first N files only")
    i.set_defaults(func=inject)
    x = sub.add_parser("aux", help="G, BP-RP and parallax of the stars of a light-curve directory, from gaia_source")
    x.add_argument("--lightcurves", help="a sample: the stars of this directory of light-curve files")
    x.add_argument("--sky", type=int, help="the whole sky: FGK-dwarf candidates of every HEALPix pixel at this level, "
                                           "one query and one file each (4: 3,072 pixels)")
    x.add_argument("--pixels", help="with --sky: only pixels FIRST-LAST (to split the work, or to try it)")
    x.add_argument("--config", default=occurrence.Config.name)
    x.add_argument("--out", required=True, help="Parquet file to write; with --sky a directory")
    x.add_argument("--schema", default="gaiadr4")
    x.add_argument("--mirror", help="a partner centre's TAP service (a name in epoch.MIRRORS, or a URL) instead of ESA")
    x.add_argument("--chunk", type=int, default=50_000, help="source_ids per uploaded query (ESA)")
    x.add_argument("--workers", type=int, default=4)
    x.set_defaults(func=aux_cmd)
    m = sub.add_parser("limit", help="the limit per rate from per-star scores and injections")
    m.add_argument("--stars", required=True, help="per-star table: the pass's summary/ directory, or one Parquet file")
    m.add_argument("--reference", help="reference.parquet of `slow_search.py flag`: trends measured against neighbours")
    m.add_argument("--aux", required=True, help="Parquet with " + ", ".join(AUX_COLUMNS) + ", or an `aux --sky` directory")
    m.add_argument("--injections", required=True, help="the inject step's injections directory")
    m.add_argument("--out", required=True)
    m.add_argument("--config", default=occurrence.Config.name)
    m.set_defaults(func=limit)
    c = sub.add_parser("planted", help="check on real stars: how often the limit falls below a share of fades planted")
    c.add_argument("--stars", required=True)
    c.add_argument("--reference")
    c.add_argument("--aux", required=True)
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
