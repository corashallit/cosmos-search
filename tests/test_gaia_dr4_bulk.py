import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_search.gaia import dr4, dr4_bulk, features
from cosmos_search.gaia.dr3_pass import in_shard
from cosmos_search.net import S3Entry
from test_gaia_dr4 import dr4_rows


def bulk_file(path, ids=(99, 7, 12345)):
    """A DR4-shaped bulk Parquet holding the test light curve once per source id."""
    parts = []
    for sid in ids:
        t = dr4_rows()
        t["source_id"] = sid
        parts.append(pa.table({c: (pa.array([list(v) for v in t[c]], pa.list_(pa.int16())) if c == "phot_ccd_proc_flags"
                                   else pa.array(np.asarray(t[c]))) for c in t.colnames}))
    pq.write_table(pa.concat_tables(parts), path, row_group_size=50)
    return path


def test_product_paths():
    assert dr4_bulk.is_epoch_photometry("Gaia/gdr4/epoch_photometry/Norder=3/Dir=0/Npix=12.parquet")
    assert dr4_bulk.is_epoch_photometry("Gaia/gdr4/Photometry/EpochPhotometry/part-001.parquet")
    assert not dr4_bulk.is_epoch_photometry("Gaia/gdr4/epoch_photometry_ccd/Norder=3/Dir=0/Npix=12.parquet")
    assert not dr4_bulk.is_epoch_photometry("Gaia/gdr4/crowded_field_epoch_photometry/Npix=1.parquet")
    assert not dr4_bulk.is_epoch_photometry("Gaia/gdr4/epoch_photometry/_metadata")
    assert dr4_bulk.stem("Gaia/gdr4/epoch_photometry/Norder=3/Dir=0/Npix=12.parquet") == \
        "epoch_photometry__Norder=3__Dir=0__Npix=12"


def test_pyarrow_rows_convert_like_astropy_rows(tmp_path):
    p = bulk_file(tmp_path / "f.parquet", ids=(99,))
    a = dr4.epoch_photometry_to_lc(dr4_rows())
    b = dr4.epoch_photometry_to_lc(pq.read_table(p))
    assert a.equals(b)


def test_process_file_keeps_selected_and_random_stars(tmp_path):
    p = bulk_file(tmp_path / "Npix=1.parquet")
    sel = tmp_path / "sel.parquet"
    pq.write_table(pa.table({"source_id": pa.array([7, 555], pa.int64())}), sel)
    rec = dr4_bulk.process_file({"name": "a"}, tmp_path / "out", select_path=str(sel), local=p)
    lc = pq.read_table(tmp_path / "out" / "lightcurves" / "a.parquet")
    assert set(lc["source_id"].to_pylist()) == {7}
    assert rec["stars_kept"] == 1 and rec["transits_kept"] == 40
    assert json.loads((tmp_path / "out" / "done" / "a.json").read_text())["rows"] == lc.num_rows
    share = 0.5
    rec = dr4_bulk.process_file({"name": "b"}, tmp_path / "out", random_share=share, local=p)
    want = {s for s in (99, 7, 12345) if features.control_mask(np.array([s]), share)[0]}
    got = set(pq.read_table(tmp_path / "out" / "lightcurves" / "b.parquet")["source_id"].to_pylist())
    assert got == want


def test_pull_resumes_and_shards(tmp_path):
    files = []
    for i in range(8):
        p = bulk_file(tmp_path / f"Npix={i}.parquet")
        files.append({"name": f"Npix={i}", "size": p.stat().st_size, "local": str(p)})
    out = tmp_path / "out"
    assert dr4_bulk.pull(out, files, keep_all=True, workers=2, shard=(0, 2), log=lambda m: None) == 0
    want = sum(in_shard(f["name"], (0, 2)) for f in files)
    assert len(list((out / "done").glob("*.json"))) == want
    assert dr4_bulk.pull(out, files, keep_all=True, workers=2, log=lambda m: None) == 0
    assert len(list((out / "done").iterdir())) == 8
    lines = (out / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 8  # the second run did only the files the first left


class FakeClient:
    def __init__(self, sums):
        self.sums = sums

    def get_bytes(self, url, **kw):
        return self.sums.encode()


def test_find_files_reads_md5_sums(monkeypatch):
    keys = ["Gaia/gdr4/epoch_photometry/Norder=0/Dir=0/Npix=1.parquet", "Gaia/gdr4/epoch_photometry/_MD5SUM.txt",
            "Gaia/gdr4/epoch_photometry_ccd/Norder=0/Dir=0/Npix=1.parquet", "Gaia/gdr4/gaia_source/Npix=1.parquet"]
    monkeypatch.setattr(dr4_bulk, "s3_list", lambda *a, **k: [S3Entry(k_, 10) for k_ in keys])
    sums = "abc123  Norder=0/Dir=0/Npix=1.parquet\n"
    files = dr4_bulk.find_files(client=FakeClient(sums))
    assert [f["key"] for f in files] == [keys[0]]
    assert files[0]["md5"] == "abc123" and files[0]["url"].endswith(keys[0])


def test_route_recommendation_prefers_bulk_then_mirror_then_datalink():
    from cosmos_search.gaia import dr4_routes

    none = {"cdn": {"epoch_photometry_files": 0}, "tables": {"esa": {"tables": []}, "aip": {"tables": []}},
            "datalink": {"per_second": 0.5}}
    assert "DataLink" in dr4_routes.recommend(none) and "43,200" in dr4_routes.recommend(none)
    mirror = {**none, "tables": {"esa": {"tables": []}, "aip": {"tables": ["gaiadr4.epoch_photometry"]}}}
    assert dr4_routes.recommend(mirror).startswith("No bulk files yet; aip")
    bulk = {**mirror, "cdn": {"epoch_photometry_files": 3000, "epoch_photometry_bytes": 7.2e12}}
    assert dr4_routes.recommend(bulk).startswith("Bulk files are up (3000 files, 7.20 TB)")


def test_datalink_pull_runs_small_calls_in_parallel_and_resumes(tmp_path, monkeypatch):
    from cosmos_search.dayone import datalink

    calls = []

    def fake(ids, cache=True):
        calls.append(list(ids))
        parts = []
        for sid in ids:
            t = dr4_rows()
            t["source_id"] = sid
            parts.append(dr4.epoch_photometry_to_lc(t))
        return pa.concat_tables(parts)

    monkeypatch.setattr(datalink.epoch, "datalink_epoch_photometry", fake)
    ids = list(range(1, 13))
    datalink.pull(ids, tmp_path, workers=3, chunk=5, log=lambda r: None)
    assert sorted(len(c) for c in calls) == [2, 5, 5]
    assert sorted(x.name for x in (tmp_path / "features").iterdir()) == [
        "chunk-00000000-5.parquet", "chunk-00000005-5.parquet", "chunk-00000010-2.parquet"]
    datalink.pull(ids, tmp_path, workers=3, chunk=5, log=lambda r: None)
    assert len(calls) == 3  # resumed: nothing pulled twice
    got = pa.concat_tables([pq.read_table(p) for p in sorted((tmp_path / "lightcurves").iterdir())])
    assert set(got["source_id"].to_pylist()) == set(ids)


def test_raw_file_goes_to_work_tmp_and_is_deleted(tmp_path, monkeypatch):
    import contextlib
    import shutil

    src = bulk_file(tmp_path / "src.parquet")
    seen = []

    class Streaming:
        @contextlib.contextmanager
        def streamed(self, url, dest=None, md5=None):
            shutil.copyfile(src, dest)
            seen.append(dest)
            try:
                yield dest
            finally:
                dest.unlink()

    monkeypatch.setattr(dr4_bulk, "get_client", lambda: Streaming())
    out = tmp_path / "out"
    dr4_bulk.process_file({"name": "x", "url": "https://cdn/x.parquet", "size": src.stat().st_size}, out,
                          random_share=0.5)
    assert seen[0].parent == out / "tmp" and not any((out / "tmp").iterdir())


def test_check_space_refuses_a_file_that_does_not_fit(tmp_path):
    import pytest

    dr4_bulk.check_space(tmp_path, 1)
    with pytest.raises(OSError, match="free"):
        dr4_bulk.check_space(tmp_path, 10**18)
