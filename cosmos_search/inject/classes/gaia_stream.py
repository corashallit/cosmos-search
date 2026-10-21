"""Sealed injections into the Gaia epoch-photometry stream (bulk files and DataLink alike).

A plan maps host source_ids to signals from `inject.lightcurve`. It is drawn with an os.urandom seed and
written only as age ciphertext, to Cora's key (the sealed truth) and to the reveal key (so the stream can
decrypt it in memory, and so recovery can be scored after the shortlist is committed). The stream applies it
to every light-curve table before Stage 0 and features, so recovery is measured after cleaning, and the
DataLink route applies the same plan, so a host looks the same whichever route fetched it.

    python -m cosmos_search.inject.classes.gaia_stream plan --hosts aux_gaia_source.parquet --n 6000 \\
        --out plan.json.age --recipient-file keys/cora.age.pub --recipient-file keys/reveal.age.pub

Choices:

- Hosts are drawn half uniformly over the sample (so injected sources look like the sample) and half
  uniformly in G (so bright bins, where the search has power, get enough injections for a recall map).
  The two halves carry their inclusion weights in the truth.
- Signals are drawn over DR3's full span, not over each host's epochs, so recall includes the dips that
  fall between epochs, as real ones do.
- Photon noise follows sqrt(k) (`vetting_decoys._photon_scaled`); the extra noise per epoch is drawn from a
  hash of (plan seed, transit_id, band), so bulk and DataLink copies of a host get identical injected fluxes.
- BP and RP see a change diluted by r ~ U(0.2, 1) on hosts whose BP/RP transits are blended, as a real
  change would be (Stage 0 removed real eclipses before this was modelled; docs/vetting.md §12).
- `g_only` puts a dip in G alone: an instrument fault, which Stage 0 and vetting should stop.
"""

import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.inject.sealed import write_sealed

# DR3 epoch photometry spans 2014-07-25 to 2017-05-28 (BTJD = BJD - 2457000).
DR3_SPAN = (-140.0, 905.0)

# family -> share of injections. Flares are left out: they only feed the bright side of the mirror.
MIX = {
    "grey_dip": 0.2, "dust_dip": 0.1, "blue_dip": 0.05,
    "dip_train_poisson": 0.1, "dip_train_periodic": 0.1, "dip_train_primes": 0.075, "dip_train_geometric": 0.075,
    "secular_fade": 0.075, "step": 0.05, "chromatic_flip": 0.075, "g_only": 0.1,
}
BLEND_MAX = 0.05  # as vetting.stage0.BLEND_MAX
G_FLAT = (8.0, 20.5)


def signal_from_truth(d):
    """Inverse of lightcurve.truth()."""
    kind = d["kind"]
    if kind == "composite":
        return lcinj.Composite([signal_from_truth(p) for p in d["parts"]])
    cls = {"dip": lcinj.Dip, "trend": lcinj.SecularTrend, "step": lcinj.Step, "flare": lcinj.Flare}[kind]
    kw = {k: v for k, v in d.items() if k != "kind"}
    if cls is lcinj.Dip and kw.get("tau_ref") is None:
        kw["tau_ref"] = np.inf
    return cls(**kw)


def draw_hosts(source_id, g_mag, n, rng):
    """n distinct hosts: half uniform over the sample, half uniform in G over G_FLAT. Returns (ids, half)."""
    n_uni = n // 2
    uni = rng.choice(len(source_id), n_uni, replace=False)
    g = np.asarray(g_mag, float)
    ok = np.flatnonzero(np.isfinite(g) & (g >= G_FLAT[0]) & (g < G_FLAT[1]))
    bins = np.floor((g[ok] - G_FLAT[0]) * 2).astype(int)
    taken = set(uni.tolist())
    flat = []
    by_bin = {b: ok[bins == b] for b in np.unique(bins)}
    keys = sorted(by_bin)
    while len(flat) < n - n_uni:
        members = by_bin[keys[rng.integers(len(keys))]]
        j = int(members[rng.integers(len(members))])
        if j not in taken:
            taken.add(j)
            flat.append(j)
    idx = np.concatenate([uni, np.array(flat, dtype=np.int64)])
    half = np.array(["uniform"] * n_uni + ["g_flat"] * len(flat))
    return np.asarray(source_id)[idx], half


def plan(hosts_parquet, n, rng, blend_parquet=None, mix=MIX, span=DR3_SPAN):
    """The injection plan: one record per host."""
    t = pq.read_table(hosts_parquet, columns=["source_id", "phot_g_mean_mag"])
    ids, half = draw_hosts(t["source_id"].to_numpy(), t["phot_g_mean_mag"].to_numpy(zero_copy_only=False), n, rng)
    blended = {}
    if blend_parquet:
        from cosmos_search.gaia.dr3_pass import blend_for
        blended = blend_for(ids, blend_parquet)
    fams = list(mix)
    p = np.array([mix[f] for f in fams], float)
    out = []
    for sid, h in zip(ids.tolist(), half.tolist()):
        fam = fams[rng.choice(len(fams), p=p / p.sum())]
        sig = lcinj.sample(rng, "grey_dip" if fam == "g_only" else fam, *span)
        frac = blended.get(sid)
        r = float(rng.uniform(0.2, 1.0)) if frac is not None and frac >= BLEND_MAX else 1.0
        out.append({"source_id": int(sid), "family": fam, "half": h, "bands": ["G"] if fam == "g_only" else None,
                    "bp_rp_dilution": r, "seed": int(rng.integers(2**62)), "signal": lcinj.truth(sig)})
    return out


def load_plan(path, secret=None):
    """Decrypt a sealed plan in memory with the reveal key; returns {source_id: record}. Never logs it."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(b"age-encryption.org/"):
        from pyrage import decrypt, x25519
        secret = (secret or os.environ.get("COSMOS_REVEAL_KEY", "")).strip()
        if not secret:
            raise RuntimeError("a sealed injection plan needs COSMOS_REVEAL_KEY")
        raw = decrypt(raw, [x25519.Identity.from_str(secret)])
    return {int(r["source_id"]): r for r in json.loads(raw)}


def _hash_normal(seed, transit_id, band):
    """Standard normals that depend only on (seed, transit_id, band): splitmix64 and Box-Muller."""
    def mix(z):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))
    with np.errstate(over="ignore"):
        z = np.uint64(seed) ^ (np.asarray(transit_id).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
        z = z + np.asarray(band).astype(np.uint64) * np.uint64(0xD1B54A32D192ED03)
        a, b = mix(z), mix(z + np.uint64(0x632BE59BD9B4E019))
    u1 = ((a >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0**53
    u2 = (b >> np.uint64(11)).astype(np.float64) / 2.0**53
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)


_SIGNALS = {}


def apply(table, plan_map):
    """Inject the plan's signals into a light-curve table (gaia.epoch.SCHEMA). Other sources are untouched."""
    if not plan_map or table.num_rows == 0:
        return table
    sid = table["source_id"].to_numpy()
    hit = np.isin(sid, np.fromiter(plan_map, dtype=np.int64, count=len(plan_map)))
    if not hit.any():
        return table
    band = np.asarray(table["band"].combine_chunks().indices) if isinstance(table["band"].type, pa.DictionaryType) \
        else np.array([("G", "BP", "RP").index(b) for b in table["band"].to_pylist()])
    names = ("G", "BP", "RP")
    t = table["t"].to_numpy()
    flux = table["flux"].to_numpy().copy()
    err = table["err"].to_numpy().copy()
    tid = table["transit_id"].to_numpy()
    for s in np.unique(sid[hit]):
        rec = plan_map[int(s)]
        key = id(rec)
        if key not in _SIGNALS:
            _SIGNALS[key] = signal_from_truth(rec["signal"])
        sig = _SIGNALS[key]
        rows = np.flatnonzero(sid == s)
        for code, name in enumerate(names):
            r = rows[band[rows] == code]
            if not len(r) or (rec.get("bands") and name not in rec["bands"]):
                continue
            k = sig.factor(t[r], name)
            if name != "G":
                k = 1 - rec.get("bp_rp_dilution", 1.0) * (1 - k)
            noise = _hash_normal(rec["seed"], tid[r], code)
            flux[r] = flux[r] * k + noise * err[r] * np.sqrt(np.clip(k - k**2, 0, None))
            err[r] = err[r] * np.sqrt(np.clip(k, 0, None))
    table = table.set_column(table.schema.get_field_index("flux"), "flux", pa.array(flux))
    return table.set_column(table.schema.get_field_index("err"), "err", pa.array(err))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cosmos_search.inject.classes.gaia_stream")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--hosts", required=True, help="Parquet with source_id, phot_g_mean_mag for the whole sample")
    s.add_argument("--blend", help="Parquet of source_id + stage0.BLEND_COLS")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--recipient-file", action="append", required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(int.from_bytes(os.urandom(32), "little"))
    records = plan(a.hosts, a.n, rng, a.blend)
    digest = write_sealed(records, a.out, a.recipient_file)
    # Counts only. Never print anything derived from the plan beyond its size.
    print(f"sealed {len(records)} injections to {a.out} (sha256 {digest})")


if __name__ == "__main__":
    sys.exit(main())
