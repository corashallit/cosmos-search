"""The practice path end to end on synthetic DR3 files: fake DR4 files, the stand-in server (loopback only), the
bulk reader in bounded memory, and the DR4 DataLink reader."""

import gzip
import io
import threading
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from test_gaia_epoch import synth_csv

from cosmos_search.gaia import archive, dr4, dr4_bulk, dr4_datalink, dr4_fake, epoch
from cosmos_search.net import get_client, s3_list

ORDER = 5
KEY = ["source_id", "band", "transit_id"]


def sid(pixel, k):
    return (pixel << (35 + 2 * (12 - ORDER))) + k


def dr3_files(tmp_path):
    """Two DR3-shaped files, three order-5 pixels (pixel 7 spans both files), with missing BP and rejected G."""
    rng = np.random.default_rng(5)
    d = tmp_path / "dr3"
    d.mkdir()

    def star(s, n, **kw):
        g = 1000 + rng.uniform(-20, 20, n)
        return (s, g, kw)

    bp = 500 + rng.uniform(-5, 5, 30)
    bp[[3, 4]] = np.nan
    reject = np.zeros(40, bool)
    reject[[5, 6]] = True
    a = [star(sid(4, 10), 30, bp=bp), star(sid(4, 11), 36), star(sid(7, 1), 40, g_reject=reject)]
    b = [star(sid(7, 2), 34), star(sid(9, 5), 30), star(sid(9, 6), 38), star(sid(9, 7), 32)]
    for name, rows, seed in (("EpochPhotometry_000001-000002.csv.gz", a, 1), ("EpochPhotometry_000003-000004.csv.gz", b, 2)):
        (d / name).write_bytes(gzip.compress(synth_csv(rows, seed)))
    return d


def build(tmp_path, rejected="keep", row_group=37):
    d = dr3_files(tmp_path)
    root = tmp_path / f"fake-{rejected}"
    for p in sorted(d.glob("*.csv.gz")):
        dr4_fake._build_one(str(p), str(root), ORDER, rejected, row_group)
    dr4_fake.finish(root, ORDER, row_group)
    (root / "build.json").write_text('{"order": %d}' % ORDER)
    dr3 = pa.concat_tables([epoch.read_epoch_csv(p) for p in sorted(d.glob("*.csv.gz"))]).to_pandas()
    return root, dr3


@pytest.fixture
def served(tmp_path, monkeypatch):
    root, dr3 = build(tmp_path)
    srv = dr4_fake.make_server(root)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    for name, value in (("CDN_URL", url), ("CDN_LIST_URL", url), ("DATALINK_URL", f"{url}/data-server/data"),
                        ("LOGIN_URL", f"{url}/tap-server/login")):
        monkeypatch.setattr(archive, name, value)
    yield root, dr3, url, srv
    srv.shutdown()
    srv.server_close()


def read_all(root):
    files = sorted((root / dr4_fake.PRODUCT).rglob("*.parquet"))
    return pa.concat_tables([dr4.epoch_photometry_to_lc(pq.read_table(p)) for p in files]).to_pandas()


def same(a, b):
    a, b = a.sort_values(KEY).reset_index(drop=True), b.sort_values(KEY).reset_index(drop=True)
    assert len(a) == len(b) and (a[KEY].to_numpy() == b[KEY].to_numpy()).all()
    assert np.allclose(a["t"], b["t"], rtol=0, atol=1e-9) and np.allclose(a["flux"], b["flux"], rtol=1e-6)
    for c in ("fov", "ccd_row", "n_obs"):
        assert (a[c] == b[c]).all()
    return a, b


def test_fake_files_hold_dr3s_numbers_in_the_draft_models_layout(tmp_path):
    root, dr3 = build(tmp_path)
    files = sorted((root / dr4_fake.PRODUCT).rglob("*.parquet"))
    assert [p.name for p in files] == ["Npix=4.parquet", "Npix=7.parquet", "Npix=9.parquet"]
    t = pq.read_table(files[0])
    assert t.schema.field("g_flux").type == pa.float32() and t.num_rows == 66  # one row per transit
    assert set(dr4.READ_COLUMNS) <= set(t.column_names) and "variability_flag_g_reject" not in t.column_names
    assert pq.read_table(files[0])["bp_flux"].null_count == 2  # even pixel: missing fluxes are nulls
    a, b = same(dr3, read_all(root))
    # The draft table has no variability-rejection column, so DR3's rejected transits arrive unmarked.
    assert (a["flags"] & epoch.FLAG_VARI_REJECT).sum() == 2 and (b["flags"] & epoch.FLAG_VARI_REJECT).sum() == 0
    sums = (root / dr4_fake.PRODUCT / "_MD5SUM.txt").read_text().split()
    assert len(sums) == 6 and sums[1] == "Norder=5/Dir=0/Npix=4.parquet"


def test_blank_removes_the_transits_dr3_rejected(tmp_path):
    root, dr3 = build(tmp_path, rejected="blank")
    same(dr3[(dr3["flags"] & epoch.FLAG_VARI_REJECT) == 0], read_all(root))


def test_listing_download_and_reader_through_the_stand_in_server(served, tmp_path):
    root, dr3, url, srv = served
    keys = [e.key for e in s3_list(url, "Gaia/gdr4/", page_size=2)]  # paginated
    assert len(keys) == 7 and keys == sorted(keys)
    summ = dr4_bulk.summary()
    assert summ["epoch_photometry_files"] == 3 and summ["with_md5"] == 3
    assert summ["top_level"] == ["Gaia/gdr4/epoch_photometry/", "Gaia/gdr4/epoch_photometry_ccd/"]
    out = tmp_path / "out"
    files = dr4_bulk.find_files()
    assert dr4_bulk.pull(out, files, keep_all=True, workers=2, log=lambda m: None) == 0
    got = pa.concat_tables([pq.read_table(p) for p in sorted((out / "lightcurves").glob("*.parquet"))]).to_pandas()
    same(dr3, got)
    assert not any((out / "tmp").iterdir()) and len(list((out / "done").iterdir())) == 3


def test_ranged_download_from_the_stand_in_server(served, tmp_path, monkeypatch):
    root, dr3, url, srv = served
    monkeypatch.setattr(dr4_bulk, "RANGE_BYTES", 2000)
    monkeypatch.setattr(dr4_bulk, "RANGE_MIN", 4000)
    f = dr4_bulk.find_files()[0]
    with dr4_bulk.fetched(f["url"], tmp_path / "x.parquet", f["size"], f["md5"], streams=3) as p:
        assert pq.read_table(p).num_rows == 66
    assert srv.RequestHandlerClass.counts["range"] >= 2


def chunks(path, **kw):
    return [t.to_pandas() for t in dr4_bulk.iter_lightcurves(path, keep_all=True, **kw)]


def test_iter_lightcurves_never_splits_a_star(tmp_path):
    root, dr3 = build(tmp_path, row_group=7)
    path = root / dr4_fake.PRODUCT / "Norder=5/Dir=0/Npix=9.parquet"
    assert dr4_bulk.stars_contiguous(path)
    parts = chunks(path, stars=1, batch_rows=16)  # batches and row groups both cut through stars
    assert [p["source_id"].nunique() for p in parts] == [1, 1, 1]
    whole = dr4.epoch_photometry_to_lc(pq.read_table(path)).to_pandas()
    same(whole, pa.concat_tables([pa.Table.from_pandas(p) for p in parts]).to_pandas())
    assert len(chunks(path, stars=30_000, batch_rows=16)) == 1


def test_iter_lightcurves_reads_files_not_grouped_by_star_and_nested_files(tmp_path):
    root, dr3 = build(tmp_path)
    path = root / dr4_fake.PRODUCT / "Norder=5/Dir=0/Npix=9.parquet"
    t = pq.read_table(path)
    whole = dr4.epoch_photometry_to_lc(t).to_pandas()
    shuffled = tmp_path / "shuffled.parquet"
    pq.write_table(t.take(pa.array(np.random.default_rng(1).permutation(t.num_rows))), shuffled, row_group_size=20)
    assert not dr4_bulk.stars_contiguous(shuffled)
    parts = chunks(shuffled, stars=2, batch_rows=16)
    assert [p["source_id"].nunique() for p in parts] == [2, 1]
    same(whole, pa.concat_tables([pa.Table.from_pandas(p) for p in parts]).to_pandas())
    # One row per source with arrays (DR3's layout, and how a mirror served DR3).
    ids = pc.unique(t["source_id"])
    cols = {"source_id": ids}
    for name in ("transit_id", "g_n_obs", "g_obs_time", "g_flux", "g_flux_error", "bp_obs_time", "bp_flux",
                 "bp_flux_error", "rp_obs_time", "rp_flux", "rp_flux_error"):
        cols[name] = pa.array([t.filter(pc.equal(t["source_id"], i))[name].to_pylist() for i in ids])
    nested = tmp_path / "nested.parquet"
    pq.write_table(pa.table(cols), nested)
    same(whole, pa.concat_tables([pa.Table.from_pandas(p) for p in chunks(nested)]).to_pandas())


def test_reader_accepts_other_spellings_and_names_what_is_missing(tmp_path):
    root, dr3 = build(tmp_path)
    t = pq.read_table(root / dr4_fake.PRODUCT / "Norder=5/Dir=0/Npix=4.parquet")
    renamed = t.rename_columns([{"g_obs_time": "G_TRANSIT_TIME", "g_flux": "g_transit_flux"}.get(c, c)
                                for c in t.column_names])
    assert dr4.epoch_photometry_to_lc(renamed).equals(dr4.epoch_photometry_to_lc(t))
    bad = tmp_path / "bad.parquet"
    pq.write_table(t.drop_columns(["g_flux"]), bad)
    with pytest.raises(KeyError, match="g_flux"):
        list(dr4_bulk.iter_lightcurves(bad, keep_all=True))


def test_datalink_reader_takes_the_layouts_the_archive_may_send(served, tmp_path):
    root, dr3, url, srv = served
    ids = [sid(9, 6), sid(4, 10), 12345]  # the last one has no light curve
    lc = dr4_datalink.fetch(ids).to_pandas()
    want = dr3[dr3["source_id"].isin(ids)]
    same(want, lc)
    raw = get_client().post(archive.DATALINK_URL, data={"RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY", "ID": str(ids[0])}).content
    one = dr4_datalink.read_response(raw)
    assert dr4_datalink.read_response(gzip.compress(raw)).equals(one)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("EPOCH_PHOTOMETRY-Gaia DR4 1.csv", raw)
        z.writestr("EPOCH_PHOTOMETRY-Gaia DR4 2.csv", raw)
    assert dr4_datalink.read_response(buf.getvalue()).num_rows == 2 * one.num_rows
    dr3_layout = synth_csv([(77, np.full(6, 100.0), {})])
    assert dr4_datalink.read_response(dr3_layout).num_rows == 18  # DR3's arrays per source still read
    assert dr4_datalink.read_response(b"").num_rows == 0
    with pytest.raises(ValueError, match="at most"):
        dr4_datalink.fetch(range(51))
    with pytest.raises(KeyError, match="g_flux"):
        dr4_datalink.read_response(b"source_id,transit_id\n1,2\n")


def test_datalink_pull_resumes(served, tmp_path):
    root, dr3, url, srv = served
    ids = sorted(dr3["source_id"].unique().tolist())
    out = tmp_path / "dl"
    assert dr4_datalink.pull(ids, out, workers=2, chunk=3, log=lambda m: None) == 0
    calls = srv.RequestHandlerClass.counts["datalink"]
    assert calls == 3
    assert dr4_datalink.pull(ids, out, workers=2, chunk=3, log=lambda m: None) == 0
    assert srv.RequestHandlerClass.counts["datalink"] == calls  # nothing pulled twice
    got = pa.concat_tables([pq.read_table(p) for p in sorted((out / "lightcurves").glob("*.parquet"))]).to_pandas()
    same(dr3, got)
