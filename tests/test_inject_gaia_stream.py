import json

import numpy as np
import pyarrow.parquet as pq
from pyrage import x25519
from test_gaia_epoch import synth_csv
from test_gaia_pass import FakeClient

from cosmos_search.gaia import dr3_pass
from cosmos_search.gaia.epoch import read_epoch_csv
from cosmos_search.inject import lightcurve as lcinj
from cosmos_search.inject.classes import gaia_stream


def _lc(tmp_path):
    rng = np.random.default_rng(5)
    srcs = [(s, 1000 + rng.normal(0, 3, 40), {}) for s in (11, 22, 33)]
    return read_epoch_csv(synth_csv(srcs))


def _plan(sid, signal, **kw):
    return {sid: {"source_id": sid, "family": "x", "half": "uniform", "bands": None, "bp_rp_dilution": 1.0,
                  "seed": 12345, "signal": lcinj.truth(signal), **kw}}


def test_signal_round_trip():
    sig = lcinj.Composite([lcinj.Dip(t0=1.0, covering=0.3), lcinj.SecularTrend(rate_per_year=-0.01, t_ref=0.0)])
    back = gaia_stream.signal_from_truth(json.loads(json.dumps(lcinj.truth(sig))))
    t = np.linspace(0, 2, 50)
    assert np.allclose(back.factor(t, "G"), sig.factor(t, "G"))


def test_apply_touches_only_planned_sources_and_is_deterministic(tmp_path):
    lc = _lc(tmp_path)
    t = lc["t"].to_numpy()
    sig = lcinj.Dip(t0=float(np.median(t)), covering=0.5, ingress=1e-3, egress=1e-3, flat=1e3)
    plan = _plan(22, sig)
    a, b = gaia_stream.apply(lc, plan), gaia_stream.apply(lc.slice(0), plan)
    assert np.array_equal(a["flux"].to_numpy(), b["flux"].to_numpy())
    sid = lc["source_id"].to_numpy()
    f0, f1 = lc["flux"].to_numpy(), a["flux"].to_numpy()
    assert np.array_equal(f0[sid != 22], f1[sid != 22])
    assert np.median(f1[sid == 22] / f0[sid == 22]) < 0.8


def test_g_only_and_dilution(tmp_path):
    lc = _lc(tmp_path)
    sig = lcinj.Dip(t0=-1e4, covering=0.5, ingress=1e-3, egress=1e-3, flat=1e6)  # covers everything
    band = np.asarray(lc["band"].combine_chunks().indices)
    sid = lc["source_id"].to_numpy()
    f0 = lc["flux"].to_numpy()
    g_only = gaia_stream.apply(lc, _plan(11, sig, bands=["G"]))["flux"].to_numpy()
    m = sid == 11
    assert np.array_equal(f0[m & (band != 0)], g_only[m & (band != 0)])
    diluted = gaia_stream.apply(lc, _plan(11, sig, bp_rp_dilution=0.5))["flux"].to_numpy()
    assert 0.65 < np.median(diluted[m & (band == 1)] / f0[m & (band == 1)]) < 0.85


def test_sealed_plan_through_the_stream(tmp_path, monkeypatch):
    ident = x25519.Identity.generate()
    pub = tmp_path / "k.pub"
    pub.write_text(str(ident.to_public()))
    hosts = tmp_path / "hosts.parquet"
    import pyarrow as pa
    pq.write_table(pa.table({"source_id": np.arange(1, 1001, dtype=np.int64),
                             "phot_g_mean_mag": np.linspace(8, 21, 1000)}), hosts)
    out = tmp_path / "plan.age"
    gaia_stream.main(["plan", "--hosts", str(hosts), "--n", "20", "--out", str(out), "--recipient-file", str(pub)])
    assert out.read_bytes().startswith(b"age-encryption.org/")
    plan = gaia_stream.load_plan(out, secret=str(ident))
    assert len(plan) == 20 and {r["half"] for r in plan.values()} == {"uniform", "g_flat"}

    # The stream applies it: a plaintext plan injecting a deep dip everywhere into source 123.
    src = tmp_path / "EpochPhotometry_000001-000002.csv.gz"
    rng = np.random.default_rng(1)
    src.write_bytes(synth_csv([(s, 1000 + rng.normal(0, 3, 30), {}) for s in (7, 123)]))
    monkeypatch.setattr(dr3_pass, "get_client", lambda: FakeClient(src))
    sig = lcinj.Dip(t0=-1e4, covering=0.6, ingress=1e-3, egress=1e-3, flat=1e6)
    plain = tmp_path / "plan.json"
    plain.write_text(json.dumps(list(_plan(123, sig).values())))
    rec = dr3_pass.process_file({"name": src.name, "url": "x"}, tmp_path / "w", inject=str(plain), keep=(8.0, 0.0))
    feat = pq.read_table(tmp_path / "w" / "features" / f"{src.stem.replace('.csv', '')}.parquet").to_pandas()
    med = feat.set_index("source_id")["med_flux_g"]
    assert rec["sources"] == 2 and med[123] < 0.5 * med[7]
