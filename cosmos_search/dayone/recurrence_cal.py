"""Calibration of the recurrence test (vetting.recurrence) before rule v3 is pre-registered (docs/dayone.md).

    python -m cosmos_search.dayone.recurrence_cal ebs --n 300 --out cal/ebs.parquet --ext cal/ebs_ext.parquet
    python -m cosmos_search.dayone.recurrence_cal run --ebs cal/ebs.parquet --ebs-ext cal/ebs_ext.parquet \\
        --hosts-ext ext/external.parquet --hosts ext/hosts.parquet --draws 5 --out cal/recurrence.parquet

Two populations, measured on ZTF and TESS epochs only (the bands the test exists for):

- Gaia DR3 eclipsing binaries (vari_eclipsing_binary) in the real candidates' G range and ZTF's sky, with their
  DPAC period, so the test's false "aperiodic" rate on eclipses and its period recovery can be read off.
- Dip trains from inject.lightcurve injected into the external epochs of batch 3's decoy hosts (raw, before any
  plan), drawn as in dayone.decoys: over DR3's span and continued to the end of the external data. Poisson, prime
  and geometric trains are the aperiodic families; periodic trains are a check that injected eclipses read as
  periodic.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cosmos_search.vetting import recurrence
from cosmos_search.vetting.candidate import LightCurve

FAMILIES = ("dip_train_poisson", "dip_train_primes", "dip_train_geometric", "dip_train_periodic")
EXT_BANDS = ("ztf_g", "ztf_r", "ztf_i", "TESS")


def eb_sample(tap, n, g_min=11.0, g_max=15.5, dec_min=-25.0, random_index_max=4_500_000):
    """A random set of Gaia DR3 EBs ZTF can see, with DPAC's model frequency (1/d). A cut on random_index (an
    indexed column) draws the random subset; ORDER BY random_index over the join does not finish."""
    adql = (f"SELECT TOP {int(n)} e.source_id, s.ra, s.dec, s.phot_g_mean_mag, e.frequency, "
            "e.derived_primary_ecl_depth, e.derived_primary_ecl_duration, e.model_type "
            "FROM gaiadr3.vari_eclipsing_binary AS e JOIN gaiadr3.gaia_source AS s ON s.source_id = e.source_id "
            f"WHERE s.random_index < {int(random_index_max)} AND s.dec > {dec_min} "
            f"AND s.phot_g_mean_mag BETWEEN {g_min} AND {g_max}")
    return tap.query(adql).to_pandas()


def _lcs(df):
    return [LightCurve(b, g["t"].to_numpy(), g["flux"].to_numpy(), g["err"].to_numpy())
            for b, g in df.groupby("band") if b in EXT_BANDS]


def _one(job):
    kind, sid, family, truth, df = job
    r = recurrence.search(*recurrence.epochs(_lcs(df)))
    return {"kind": kind, "source_id": sid, "family": family, **r}


def injected_jobs(ext, host_ids, draws, rng):
    from cosmos_search.inject import lightcurve as lcinj
    from cosmos_search.inject.classes import gaia_stream

    from .external import apply

    t_ext = float(ext["t"].max())
    for sid in host_ids:
        raw = ext[ext.source_id == sid]
        if raw.empty:
            continue
        for fam in FAMILIES:
            for _ in range(draws):
                sig = lcinj.sample(rng, fam, *gaia_stream.DR3_SPAN, t_ext=t_ext)
                rec = {"signal": lcinj.truth(sig), "bands": None}
                yield ("injected", int(sid), fam, None, apply(raw, {int(sid): rec}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cosmos_search.dayone.recurrence_cal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("ebs")
    e.add_argument("--n", type=int, default=300)
    e.add_argument("--out", required=True)
    e.add_argument("--ext", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ebs", required=True)
    r.add_argument("--ebs-ext", required=True)
    r.add_argument("--hosts-ext", required=True)
    r.add_argument("--hosts", required=True)
    r.add_argument("--draws", type=int, default=5)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "ebs":
        from cosmos_search.gaia.archive import tap

        from .external import ztf

        ebs = eb_sample(tap(), a.n)
        ebs.to_parquet(a.out)
        from concurrent.futures import ThreadPoolExecutor

        def one(row):
            sid, ra, dec = row
            try:
                return ztf(int(sid), float(ra), float(dec))
            except Exception as exc:  # one failed star is missing data
                print(f"{sid}: ZTF failed ({type(exc).__name__})", flush=True)
                return None

        parts = []
        with ThreadPoolExecutor(4) as pool:
            for i, df in enumerate(pool.map(one, zip(ebs.source_id.astype("int64"), ebs.ra, ebs.dec))):
                parts.append(df)
                if i % 25 == 0:
                    print(f"{i + 1}/{len(ebs)}", flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(a.ext)
        print(f"{len(ebs)} EBs -> {a.out}, ZTF -> {a.ext}")
        return 0
    rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))
    eb_ext = pd.read_parquet(a.ebs_ext)
    jobs = [("eb", int(sid), "eclipsing_binary", None, g) for sid, g in eb_ext.groupby("source_id")]
    hosts = pd.read_parquet(a.hosts, columns=["source_id"])["source_id"].astype("int64")
    jobs += list(injected_jobs(pd.read_parquet(a.hosts_ext), hosts, a.draws, rng))
    print(f"{len(jobs)} light curves", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(_one, jobs, chunksize=4))
    out = pd.DataFrame(rows)
    ebs = pd.read_parquet(a.ebs)[["source_id", "frequency", "derived_primary_ecl_duration", "model_type"]]
    out = out.merge(ebs, on="source_id", how="left")
    out.to_parquet(a.out)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
