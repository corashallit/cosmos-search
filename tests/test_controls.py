import numpy as np
import pandas as pd
from pyrage import x25519

from cosmos_search.dayone import controls


def test_draw_seal_reserve_and_load(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    classes = list(controls.STRATA)
    ranked = pd.DataFrame({"source_id": np.arange(1, 2001), "best_class_name": [classes[i % 8] for i in range(2000)],
                           "phot_g_mean_mag": rng.uniform(9, 16, 2000), "tier": 1})
    extra = pd.DataFrame({"source_id": np.arange(5001, 5101), "best_class_name": "AGN",
                          "phot_g_mean_mag": 15.0})
    exclude = set(range(1, 400))
    rows = controls.draw(ranked, exclude, rng, extra, n=10)
    assert len(rows) == 10 * len(controls.STRATA)
    assert not exclude & {r["source_id"] for r in rows}
    assert {r["selected"] for r in rows if r["stratum"] == "AGN"} == {False}
    g = ranked.set_index("source_id").phot_g_mean_mag
    rng_of = lambda r: controls.G_RANGE_BY_STRATUM.get(r["stratum"], controls.G_RANGE)
    assert all(rng_of(r)[0] <= g[r["source_id"]] <= rng_of(r)[1] for r in rows if r["selected"])
    ident = x25519.Identity.generate()
    pub = tmp_path / "k.pub"
    pub.write_text(str(ident.to_public()) + "\n")
    info = controls.seal(rows, "pool", pool_dir=tmp_path, recipients=[pub])
    assert info["n"] == len(rows) and b"source_id" not in (tmp_path / "pool.json.age").read_bytes()
    ids = [r["source_id"] for r in rows]
    mask = controls.reserved(ids + [999_999], pool_dir=tmp_path)
    assert mask[:-1].all() and not mask[-1]
    monkeypatch.setenv("COSMOS_REVEAL_KEY", str(ident))
    assert controls.load("pool", pool_dir=tmp_path) == rows
    # A second draw leaves the reserved pool out.
    monkeypatch.setattr(controls, "PREREG", tmp_path)
    again = controls.draw(ranked, exclude, rng, extra, n=10)
    assert not set(ids) & {r["source_id"] for r in again}


def test_queue_sample_skips_reserved_and_used(tmp_path, monkeypatch):
    ranked = pd.DataFrame({"source_id": np.arange(1, 301), "tier": [1, 2, 3] * 100})
    monkeypatch.setattr(controls, "PREREG", tmp_path)
    (tmp_path / "p.sha256").write_text("\n".join(controls._hash(s) for s in range(1, 31)))
    q = controls.queue_sample(ranked, 500, np.random.default_rng(0), exclude=range(31, 61))
    assert set(q.source_id) == set(range(61, 301, 3)) and (q.role == "queue_sample").all()
