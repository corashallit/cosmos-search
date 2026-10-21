import json
from contextlib import contextmanager

import numpy as np
import pyarrow.parquet as pq

from cosmos_search.gaia import dr3_pass
from test_gaia_epoch import synth_csv


class FakeClient:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def streamed(self, url, **kw):
        yield self.path


def make_file(tmp_path):
    rng = np.random.default_rng(3)
    sources = [(i * 4093 if i % 2 else i * 7 + 1, 1000 + rng.uniform(-10, 10, 30), {}) for i in range(1, 9)]
    dip = 1000 + rng.uniform(-10, 10, 30)
    dip[4:6] *= 0.5
    sources.append((123, dip, {}))
    p = tmp_path / "EpochPhotometry_000001-000002.csv.gz"
    p.write_bytes(synth_csv(sources))
    return p


def test_process_file_publishes_to_durable_store_and_resumes(tmp_path, monkeypatch):
    src = make_file(tmp_path)
    monkeypatch.setattr(dr3_pass, "get_client", lambda: FakeClient(src))
    info = {"name": src.name, "url": "x", "md5": None}
    out, durable = tmp_path / "work", tmp_path / "durable"
    assert not dr3_pass.is_done(src.name, out, durable)
    rec = dr3_pass.process_file(info, out, durable, sketch=lambda f: {"n": len(f)})
    assert rec["sources"] == 9 and dr3_pass.is_done(src.name, out, durable)
    feat = pq.read_table(durable / "features" / "EpochPhotometry_000001-000002.parquet")
    lc = pq.read_table(durable / "lightcurves" / "EpochPhotometry_000001-000002.parquet")
    assert feat.num_rows == 9 and 123 in lc["source_id"].to_pylist()
    assert feat.schema.metadata[b"control_fraction"] == b"0.01"
    assert json.loads((durable / "sketches" / "EpochPhotometry_000001-000002.json").read_text()) == [{"n": 9}]
    assert not list((out / "features").glob("*.parquet"))  # local copies removed after publishing


def test_feature_cut_keeps_matching_rows(tmp_path, monkeypatch):
    src = make_file(tmp_path)
    monkeypatch.setattr(dr3_pass, "get_client", lambda: FakeClient(src))
    rec = dr3_pass.process_file({"name": src.name, "url": "x"}, tmp_path / "w", feature_cut="min_z_rob <= -5")
    feat = pq.read_table(tmp_path / "w" / "features" / "EpochPhotometry_000001-000002.parquet")
    assert rec["sources"] == 9 and feat["source_id"].to_pylist() == [123]


def test_shards_partition_the_file_list():
    names = [f"EpochPhotometry_{i:06d}-{i + 9:06d}.csv.gz" for i in range(0, 5000, 10)]
    owned = [sum(dr3_pass.in_shard(n, (i, 3)) for i in range(3)) for n in names]
    assert owned == [1] * len(names)
    sizes = [sum(dr3_pass.in_shard(n, (i, 3)) for n in names) for i in range(3)]
    assert min(sizes) > 100
