from contextlib import contextmanager

import numpy as np
import pyarrow.parquet as pq

from cosmos_search.gaia import dr3_pass, sketches
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
    rec = dr3_pass.process_file(info, out, durable, sketch=sketches.MirrorSketches)
    assert rec["sources"] == 9 and dr3_pass.is_done(src.name, out, durable)
    feat = pq.read_table(durable / "features" / "EpochPhotometry_000001-000002.parquet")
    lc = pq.read_table(durable / "lightcurves" / "EpochPhotometry_000001-000002.parquet")
    assert feat.num_rows == 9 and 123 in lc["source_id"].to_pylist()
    assert feat.schema.metadata[b"control_fraction"] == b"0.01"
    sk = sketches.read(durable / "sketches" / "EpochPhotometry_000001-000002.json.gz")
    assert set(sk) == {"dim|G=18.0|n=20-40", "bright|G=18.0|n=20-40"} and sk["dim|G=18.0|n=20-40"].n == 9
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


def test_mirror_sketches_merge_like_one_pass(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(5)

    def feats(n):
        return pd.DataFrame({"n_g": 40, "med_flux_g": 10 ** ((25.6874 - 15.2) / 2.5),
                             "min_z_rob": -np.abs(rng.standard_t(4, n)), "max_z_rob": np.abs(rng.standard_t(4, n))})

    parts = [feats(3000), feats(5000)]
    paths = []
    for i, f in enumerate(parts):
        paths.append(tmp_path / f"{i}.json.gz")
        sketches.write(paths[-1], sketches.MirrorSketches().add(f).to_dict())
    merged = sketches.merge(paths)["dim|G=15.0|n=40-80"]
    whole = -pd.concat(parts)["min_z_rob"].to_numpy()
    assert merged.n == whole.size
    s = np.quantile(whole, [0.99, 0.995, 0.999])
    p, _ = merged.p(s, min_exceed=1)  # exact counts, no GPD extrapolation
    np.testing.assert_allclose(p, [(1 + (whole >= x).sum()) / (1 + whole.size) for x in s])


def test_mirror_sketches_stratify_by_transit_count():
    import pandas as pd

    f = pd.DataFrame({"n_g": [9, 10, 19, 20, 79, 80, 300], "med_flux_g": 1000.0, "min_z_rob": -1.0, "max_z_rob": 1.0})
    keys = {k: sk.n for k, sk in sketches.MirrorSketches().add(f).sketches().items() if k.startswith("dim")}
    assert keys == {"dim|G=18.0|n=10-20": 2, "dim|G=18.0|n=20-40": 1, "dim|G=18.0|n=40-80": 1, "dim|G=18.0|n=80+": 2}


def test_blend_for_matches_stage0_blend_fraction(tmp_path):
    import pandas as pd
    import pyarrow as pa

    from cosmos_search.vetting import stage0

    rows = pd.DataFrame({"source_id": [5, 3, 9, 7], "phot_bp_n_obs": [20, 10, 0, 30],
                         "phot_bp_n_blended_transits": [2, 5, 0, None],
                         "phot_rp_n_obs": [20, 10, 0, 30], "phot_rp_n_blended_transits": [4, 1, 0, None]})
    path = tmp_path / "blend.parquet"
    pq.write_table(pa.Table.from_pandas(rows.astype({"phot_bp_n_blended_transits": "Int64",
                                                     "phot_rp_n_blended_transits": "Int64"})), path)
    got = dr3_pass.blend_for(np.array([3, 3, 5, 7, 9, 11]), str(path))
    want = {r.source_id: stage0.blend_fraction({k: (None if pd.isna(v) else v) for k, v in r._asdict().items()})
            for r in rows.itertuples(index=False)}
    assert got == {k: v for k, v in want.items() if v is not None}
