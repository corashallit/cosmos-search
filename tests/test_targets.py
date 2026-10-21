import json

import numpy as np
import pytest
from astropy.table import Table

from cosmos_search.targets import prefilter, queries, releases


def test_templates_render_for_both_releases_without_unsupported_adql():
    for release in ("dr3", "dr4"):
        for side in ("faint", "bright"):
            q = queries.render_histogram(release, side)
            assert releases.tables(release)["vari_summary"] in q
        q = queries.render_prefilter(release, {28: 12.0, 29: 13.5})
        # The Gaia archive's ADQL parser rejects these.
        for word in ("CASE", "LEAST", "UNION"):
            assert word not in q
        assert "median_mag_g_fov >= 14 AND median_mag_g_fov < 14.5" in q
        assert f"MOD(source_id, {queries.CONTROL_MODULUS}) = 0" in q


def test_no_thresholds_selects_only_controls():
    assert queries.threshold_terms({}) == "1 = 0"


def test_unknown_release_is_rejected():
    with pytest.raises(ValueError):
        queries.render_histogram("dr5", "faint")


def hist(rows):
    return Table(rows=rows, names=("mag_bin", "d_bin", "n"))


def test_threshold_keeps_alpha_of_the_mirror_null():
    # 1000 sources: d = 0..9.75 in steps of 0.25, 25 each (d_bin = 0..39).
    h = hist([(30, b, 25) for b in range(40)])
    t = prefilter.threshold(h, alpha=0.05, mag_bin=30)
    edges, s = prefilter.survival(h, 30)
    assert s[edges.tolist().index(t)] <= 0.05
    assert s[edges.tolist().index(t) - 1] > 0.05


def test_sparse_bins_fall_back_to_pooled_threshold():
    rows = [(30, b, 100) for b in range(40)] + [(10, 0, 5), (10, 39, 1)]
    thr, pooled = prefilter.thresholds(hist(rows), prefilter.Rule(alpha=0.01, min_count=100))
    assert thr[10] == pooled
    assert thr[30] == prefilter.threshold(hist(rows), 0.01, 30)


def test_freeze_then_verify(tmp_path, monkeypatch):
    path = tmp_path / "frozen.json"
    prefilter.freeze("test", path)
    assert prefilter.verify(path) == []
    m = json.loads(path.read_text())
    m["templates_sha256"]["dimmer_prefilter"] = "0" * 64
    path.write_text(json.dumps(m))
    assert prefilter.verify(path) == ["template dimmer_prefilter changed"]


def test_run_records_every_query(tmp_path):
    class FakeTap:
        def __init__(self):
            self.adql = []

        def query(self, adql, **kw):
            self.adql.append(adql)
            if "COUNT(*)" in adql:
                return hist([(30, b, 50) for b in range(80)])
            rng = np.random.default_rng(0)
            n = 20
            return Table({"source_id": np.arange(n, dtype=np.int64) * 4093, "mag_bin": np.full(n, 30),
                          "d_faint": rng.uniform(0, 30, n), "d_bright": rng.uniform(0, 30, n),
                          "control_mod": np.zeros(n, dtype=np.int64)})

    tap = FakeTap()
    sel, m = prefilter.run("dr3", tap, tmp_path, prefilter.Rule(alpha=0.05, min_count=10))
    assert len(tap.adql) == 3
    assert set(m["queries"]) == {"histogram_faint", "histogram_bright", "prefilter"}
    assert m["queries"]["prefilter"]["adql"] == tap.adql[-1]
    assert (tmp_path / "manifest.json").exists() and (tmp_path / "prefilter.parquet").exists()
    assert m["selected"]["controls"] == 20
