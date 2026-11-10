"""Readers for other surveys (cosmos_search.surveys) and the comparison with Gaia's change (slow.independent).
Offline: the ATLAS and ASAS-SN protocols run against a local scripted server."""

import base64
import io
import json
import socket

import numpy as np
import pandas as pd
import pytest
import requests
from astropy.io import fits
from test_net import FakeArchive, make_client

from cosmos_search import surveys
from cosmos_search.slow import independent as ind
from cosmos_search.surveys import asassn, atlas, noirlab, skymapper, tess_activity


@pytest.fixture
def archive():
    a = FakeArchive()
    yield a
    a.close()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("COSMOS_BLIND", raising=False)
    monkeypatch.delenv(atlas.TOKEN_ENV, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def epochs(rng, t, flux, survey="skymapper", band="sm_r", group="g1", err=0.005, source_id=1):
    """A COLUMNS table with Gaussian noise of `err`, not renormalised (levels as given)."""
    t = np.asarray(t, float)
    f = np.broadcast_to(np.asarray(flux, float), t.shape) + rng.normal(0, err, len(t))
    return pd.DataFrame({"source_id": source_id, "survey": survey, "band": band, "group": group, "t": t, "flux": f,
                         "err": err, "mag": np.nan})[surveys.COLUMNS]


def nightly(lo=-100.0, hi=1900.0, step=4.0):
    return np.arange(lo, hi, step)


TREND = {"kind": "trend", "t_first": 0.0, "t_last": 1800.0, "slope": -0.03, "d_bp": np.nan, "e_bp": np.nan,
         "d_rp": np.nan, "e_rp": np.nan}
EPISODE = {"kind": "episode", "t_first": 0.0, "t_last": 1800.0, "depth": 0.08, "t0": 600.0, "t1": 1000.0,
           "d_bp": np.nan, "e_bp": np.nan, "d_rp": np.nan, "e_rp": np.nan, "to_end": False}
FLOORS = {"skymapper": 0.005, "ztf": 0.005, "asassn": 0.005}


# ---- the common table ----------------------------------------------------------------------------------------

def test_table_normalises_each_group_and_drops_bad_rows():
    t = surveys.table(7, "x", ["a", "a", "a", "b", "b", "b"], ["1", "1", "1", "2", "2", "2"], [3, 1, 2, 1, 2, np.nan],
                      [2.0, 4.0, 6.0, 10.0, 30.0, 20.0], [0.2, 0.4, 0.0, 1.0, 3.0, 1.0])
    assert list(t.columns) == surveys.COLUMNS
    a, b = t[t["group"] == "1"], t[t["group"] == "2"]
    assert list(a["t"]) == [1.0, 3.0]                       # err 0 dropped, sorted by time
    assert np.allclose(a["flux"], [4 / 3, 2 / 3]) and np.allclose(a["err"], [0.4 / 3, 0.2 / 3])
    assert np.allclose(b["flux"], [0.5, 1.5])               # NaN time dropped; median of 10 and 30
    assert len(surveys.table(7, "x", [], [], [], [], [])) == 0


def test_at_year_moves_by_the_proper_motion():
    ra, dec = surveys.at_year(10.0, 60.0, 3600.0, -1800.0, 2016.0, 2020.0)     # mas per year
    assert dec == pytest.approx(60.0 - 4 * 1.8 / 3600)
    assert ra == pytest.approx(10.0 + 4 * 3.6 / 3600 / np.cos(np.radians(60.0)))
    assert surveys.at_year(10.0, 60.0, None, np.nan, 2016.0, 2020.0) == (10.0, 60.0)


def test_fetch_all_skips_uncovered_sky_and_reports_failures(monkeypatch):
    seen = {}

    def reader(name):
        def f(source_id, ra, dec, **kw):
            seen[name] = (ra, dec)
            if name == "noirlab":
                raise RuntimeError("down")
            return surveys.table(source_id, name, ["b"] * 3, ["g"] * 3, [1, 2, 3], [1, 1, 1], [0.1] * 3)
        return f

    monkeypatch.setattr(surveys, "reader", reader)
    star = {"source_id": 5, "ra": 100.0, "dec": 40.0, "pmra": 0.0, "pmdec": 3600.0, "ref_epoch": 2016.0}
    lc, errors = surveys.fetch_all(star, ("ztf", "skymapper", "noirlab"))
    assert "skymapper" not in seen                          # dec +40 is north of SkyMapper's sky
    assert set(lc["survey"]) == {"ztf"} and list(errors) == ["noirlab"] and "down" in errors["noirlab"]
    assert seen["ztf"][1] == pytest.approx(40.0 + 5.5 * 3.6 / 3600)    # at ZTF's mean year, 2021.5
    # A star ZTF saturates on is not asked for either; without a magnitude only the sky decides.
    assert not surveys.covers("ztf", 40.0, 12.0) and surveys.covers("ztf", 40.0, 13.5) and surveys.covers("ztf", 40.0)
    assert not surveys.covers("noirlab", -40.0, 12.0) and surveys.covers("skymapper", -40.0, 12.0)
    lc, errors = surveys.fetch_all({**star, "phot_g_mean_mag": 11.5}, ("ztf", "noirlab"))
    assert len(lc) == 0 and errors == {}


# ---- SkyMapper and NOIRLab ---------------------------------------------------------------------------------------

def test_skymapper_picks_the_object_matched_to_the_gaia_source():
    objects = pd.DataFrame({"object_id": [11, 12], "raj2000": [10.0, 10.0 + 3 / 3600], "dej2000": [0.0, 0.0],
                            "gaia_dr3_id1": [999, 42]})
    assert skymapper.pick_object(objects, 42, 10.0, 0.0) == 12        # the id match wins over the nearer object
    assert skymapper.pick_object(objects, 7, 10.0, 0.0) == 11         # no id match: the nearest within 1.5"
    assert skymapper.pick_object(objects, 7, 10.0, 10 / 3600) is None
    assert skymapper.pick_object(objects.iloc[:0], 7, 10.0, 0.0) is None


def test_skymapper_clean_keeps_good_detections_grouped_by_filter_and_image_type():
    rows = pd.DataFrame({
        "filter": ["g", "g", "g", "g", "r", "r", "r", "g"],
        "mag_psf": [15.0, 15.1, 15.0, 15.0, 14.0, 14.0, np.nan, 15.0],
        "e_mag_psf": [0.01, 0.01, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01],
        "flags": [0, 2, 0, 4, 0, 0, 0, 0], "nimaflags": [0, 0, 0, 0, 0, 1, 0, 0],
        "image_type": ["fs", "ms", "fs", "fs", "ms", "ms", "ms", "bad"],
        "date": [57000.0, 57100.0, 57200.0, 57300.0, 57400.0, 57500.0, 57600.0, 57700.0]})
    t = skymapper.clean(rows, 3)
    assert sorted(t["group"]) == ["sm_g_fs", "sm_g_ms", "sm_r_ms"]    # large error, flag 4, flagged pixels, NaN, bad out
    assert set(t["survey"]) == {"skymapper"} and np.allclose(t["flux"], 1.0)
    assert t.loc[t["group"] == "sm_g_fs", "t"].iloc[0] == pytest.approx(57000.0 + surveys.MJD_TO_BTJD)
    assert "CONTAINS" in skymapper.object_adql(10.0, -5.0) and "object_id = 12" in skymapper.photometry_adql(12)


def test_noirlab_clean_and_queries():
    rows = pd.DataFrame({"exposure": ["c4d_150101_010101_ooi_g_v1", "c4d_160101_010101_ooi_g_v1",
                                      "ksb_170101_010101_ooi_g_v1", "c4d_180101_010101_ooi_r_v1"],
                         "filter": ["g ", "g", "g", "r"], "mjd": [57023.0, 57388.0, 57754.0, 58119.0],
                         "mag_auto": [17.0, 17.2, 17.1, 99.0], "magerr_auto": [0.01, 0.01, 0.01, 0.01],
                         "flags": [0, 0, 0, 0]})
    t = noirlab.clean(rows, 3)
    assert sorted(set(t["group"])) == ["c4d_g", "ksb_g"]              # one group per instrument and filter
    c4d = t[t["group"] == "c4d_g"].sort_values("t")
    assert c4d["flux"].iloc[0] / c4d["flux"].iloc[1] == pytest.approx(10 ** (0.4 * 0.2))
    assert "q3c_radial_query" in noirlab.object_adql(10.0, -5.0)
    with pytest.raises(ValueError):
        noirlab.meas_adql("1'; DROP TABLE x; --")


# ---- ATLAS ---------------------------------------------------------------------------------------------------

ATLAS_TEXT = """###MJD          m      dm   uJy   duJy F err chi/N     RA        Dec        x        y     maj  min   phi  apfit mag5sig Sky   Obs
58000.5   16.000  0.010  1445    14 o  0  1.00  10.0  -5.0  5000.0  5000.0  2.50 2.40  10.0 -0.400 19.50 20.50 02a58000o0100o
58001.5   16.010  0.010  1432    14 o  0  1.00  10.0  -5.0  5000.0  5000.0  2.50 2.40  10.0 -0.400 19.50 20.50 02a58001o0100o
58002.5   16.020  0.010  1419    14 c  0  1.00  10.0  -5.0  5000.0  5000.0  2.50 2.40  10.0 -0.400 19.50 20.50 01a58002o0100c
58003.5   16.000  0.010  1445    14 o  1  1.00  10.0  -5.0  5000.0  5000.0  2.50 2.40  10.0 -0.400 19.50 20.50 02a58003o0100o
58004.5   16.000  0.010  1445    14 o  0  1.00  10.0  -5.0    50.0  5000.0  2.50 2.40  10.0 -0.400 19.50 20.50 02a58004o0100o
58005.5   16.000  0.010  1445    14 o  0  1.00  10.0  -5.0  5000.0  5000.0  6.50 2.40  10.0 -0.400 19.50 20.50 02a58005o0100o
58006.5   16.000  0.010  1445    14 o  0  1.00  10.0  -5.0  5000.0  5000.0  2.50 2.40  10.0 -0.400 16.50 20.50 02a58006o0100o
58007.5   16.000  0.010  1445    14 o  0  1.00  10.0  -5.0  5000.0  5000.0  2.50 2.40  10.0 -0.400 19.50 20.50 04a58007o0100o
"""


def test_atlas_parse_and_clean_apply_the_cuts():
    t = atlas.clean(atlas.parse(ATLAS_TEXT), 9)
    # Dropped: error flag 1, the detector edge, a 6.5-pixel image, a bright sky limit.
    assert sorted(t["group"]) == ["01a_c", "02a_o", "02a_o", "04a_o"]
    assert set(t["band"]) == {"atlas_o", "atlas_c"}
    o = t[t["group"] == "02a_o"].sort_values("t")
    assert o["flux"].iloc[0] / o["flux"].iloc[1] == pytest.approx(1445 / 1432)
    assert o["t"].iloc[0] == pytest.approx(58000.5 + surveys.MJD_TO_BTJD)
    assert len(atlas.clean(atlas.parse(""), 9)) == 0


def test_atlas_difference_fluxes_need_the_reference_flux():
    with pytest.raises(ValueError):
        atlas.fetch(9, 10.0, -5.0, use_reduced=False)
    diff = atlas.parse(ATLAS_TEXT.replace(" 1445 ", "  -72 ").replace(" 1432 ", "  -85 ").replace(" 1419 ", "  -98 "))
    t = atlas.clean(diff, 9, ref_ujy={"o": 1517.0, "c": 1517.0})
    assert sorted(t.loc[t["group"] == "02a_o", "mag"].round(3)) == [16.0, 16.01]   # reference + change, as a level


def _atlas_script(archive, result=ATLAS_TEXT, error=None):
    js = {"Content-Type": "application/json"}
    done = {"finishtimestamp": "2026-11-10T00:00:00Z", "result_url": f"{archive.url}/static/results/job1.txt"}
    if error:
        done = {"finishtimestamp": "2026-11-10T00:00:00Z", "error_msg": error, "result_url": None}
    archive.script["/queue/"] = [(201, js, json.dumps({"url": f"{archive.url}/queue/1/"}).encode())]
    archive.script["/queue/1/"] = [(200, js, json.dumps({"finishtimestamp": None}).encode()),
                                   (200, js, json.dumps(done).encode())]
    archive.script["/static/results/job1.txt"] = [(200, {}, result.encode())]


def test_atlas_job_runs_polls_reads_and_deletes(archive, tmp_path, monkeypatch):
    monkeypatch.setenv(atlas.TOKEN_ENV, "abc123")
    _atlas_script(archive)
    waits = []
    c = make_client(tmp_path)
    t = atlas.fetch(9, 10.0, -5.0, client=c, cache=True, base_url=archive.url, poll=7.0, sleep=waits.append)
    assert len(t) == 4 and waits == [7.0]
    assert archive.hits == {"/queue/": 1, "/queue/1/": 3, "/static/results/job1.txt": 1}   # two polls and the delete
    form = archive.bodies[0].decode()
    assert "use_reduced=True" in form and "ra=10.0000000" in form and "send_email=False" in form
    # A second call answers from the cache: no new job.
    assert len(atlas.fetch(9, 10.0, -5.0, client=c, cache=True, base_url=archive.url)) == 4
    assert archive.hits["/queue/"] == 1


def test_atlas_token_goes_in_the_header_only_when_set(monkeypatch):
    assert "Authorization" not in atlas.headers()            # a network secret can supply it instead
    monkeypatch.setenv(atlas.TOKEN_ENV, " abc123 ")
    assert atlas.headers()["Authorization"] == "Token abc123"


def test_atlas_without_a_token_says_what_to_set(archive, tmp_path):
    archive.script["/queue/"] = [(401, {}, b'{"detail": "Authentication credentials were not provided."}')]
    c = make_client(tmp_path)
    with pytest.raises(surveys.MissingCredential, match="ATLAS_TOKEN"):
        atlas.request(10.0, -5.0, client=c, base_url=archive.url)
    ok, why = surveys.usable(["skymapper", "atlas"], atlas={"client": c, "base_url": archive.url})
    assert ok == ["skymapper"] and "ATLAS_TOKEN" in why["atlas"]


def test_atlas_no_data_is_an_empty_table_and_other_failures_raise(archive, tmp_path):
    _atlas_script(archive, error="No data returned")
    c = make_client(tmp_path)
    assert len(atlas.fetch(9, 10.0, -5.0, client=c, cache=False, base_url=archive.url, sleep=lambda s: None)) == 0
    archive.hits.clear()
    _atlas_script(archive, error="disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        atlas.request(10.0, -5.0, client=c, base_url=archive.url, sleep=lambda s: None)
    assert archive.hits["/queue/1/"] == 3                    # the failed task is still removed from the queue


# ---- ASAS-SN -------------------------------------------------------------------------------------------------

def _parquet(df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _asassn_epochs():
    n = 12
    return pd.DataFrame({
        "asas_sn_id": [77] * n + [88] * 2, "jd": 2457000.0 + np.arange(n + 2) * 30.0,
        "flux": [10.0, 10.2, 9.8, 10.0, 20.0, 20.4, 19.6, 20.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0],
        "flux_err": [0.1] * (n + 2), "mag": [13.0] * 11 + [99.99, 14.0, 14.0], "mag_err": [0.01] * (n + 2),
        "camera": ["ba"] * 4 + ["bi"] * 4 + ["ba"] * 3 + ["zz"] + ["ba"] * 2,
        "quality": ["G"] * 8 + ["B", "B", "G", "G", "G", "G"]})


def test_asassn_query_hash_is_the_clients():
    radius, ra, dec = asassn.cone(10.5, -5.25)
    assert (radius, ra, dec) == ("0.00222222", "10.5", "-5.25")
    text = base64.decodebytes(asassn.query_hash(radius, ra, dec).encode()).decode()
    assert text == ("conectr-0.00222222_conera-10.5_conedec--5.25|catalog-master_list|cols-"
                    "asas_sn_id/ra_deg/dec_deg/catalog_sources")


def test_asassn_clean_keeps_good_images_one_group_per_camera():
    t = asassn.clean(_asassn_epochs()[lambda d: d["asas_sn_id"] == 77], 5)
    assert t.groupby("group").size().to_dict() == {"ba": 5, "bi": 4}   # two "B" images and an unlisted camera out
    assert t.groupby("group")["band"].first().to_dict() == {"ba": "asassn_V", "bi": "asassn_g"}
    assert np.allclose(t.groupby("group")["flux"].median(), 1.0)                  # each camera on its own level
    assert t["t"].min() == 0.0 and t["mag"].notna().all()                         # the 99.99 row was the unlisted one


def test_asassn_request_follows_the_protocol(archive, tmp_path):
    port = archive.server.server_port
    index = pd.DataFrame({"asas_sn_id": [77, 88], "ra_deg": [10.5, 10.5 + 6 / 3600], "dec_deg": [-5.25, -5.25],
                          "catalog_sources": ["gaia", "gaia"]})
    radius, ra, dec = asassn.cone(10.5, -5.25)
    block = requests.utils.requote_uri(
        f"/get_block/query_hash-{asassn.query_hash(radius, ra, dec)}-block_idx-0-catalog-master_list")
    archive.script[f"/lookup_cone/radius{radius}_ra{ra}_dec{dec}"] = [(200, {}, _parquet(index))]
    archive.script["/get_current_message"] = [(200, {}, b"ok")]         # the reachability check: one real request
    archive.script["/get_block_servers"] = [(200, {"Content-Type": "application/json"}, b'["127.0.0.1"]')]
    archive.script[block] = [(200, {}, _parquet(_asassn_epochs()))]
    c = make_client(tmp_path)
    t = asassn.fetch(5, 10.5, -5.25, client=c, cache=True, host="127.0.0.1", port=port)
    assert len(t) == 9 and set(t["group"]) == {"ba", "bi"}         # only target 77, the one at the position
    assert json.loads(next(b for b in archive.bodies if b)) == {"catalog": "master_list", "cols": asassn.COLS,
                                                                "format": "arrow", "download": True}
    asassn.fetch(5, 10.5, -5.25, client=c, cache=True, host="127.0.0.1", port=port)
    assert archive.hits[block] == 1                                 # the second call answers from the cache
    assert archive.hits["/get_current_message"] == 1                # asked once per process, never retried


def test_asassn_closed_port_is_unreachable_not_an_empty_light_curve(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(surveys.Unreachable, match="9006|port"):
        asassn.fetch(5, 10.5, -5.25, client=make_client(tmp_path), cache=False, host="127.0.0.1", port=port)
    ok, why = surveys.usable(["asassn", "noirlab"], asassn={"host": "127.0.0.1", "port": port})
    assert ok == ["noirlab"] and "asassn" in why
    assert "asassn" not in surveys.DEFAULT and "asassn" in surveys.ALL     # read only when asked for by name


# ---- TESS activity ---------------------------------------------------------------------------------------------

def test_variability_measures_a_rotation_and_calls_a_constant_quiet():
    rng = np.random.default_rng(1)
    t = np.arange(0, 26.0, 1 / 144)                                   # 10-minute cadence
    spot = tess_activity.variability(t, 1000 * (1 + 0.02 * np.sin(2 * np.pi * t / 3.3) + rng.normal(0, 0.002, len(t))))
    assert spot["amp"] == pytest.approx(2 * 0.02 * np.sin(np.radians(81)), rel=0.05)   # 5th to 95th percentile
    assert spot["period"] == pytest.approx(3.3, rel=0.03) and spot["power"] > 0.9
    quiet = tess_activity.variability(t, 1000 * (1 + rng.normal(0, 0.002, len(t))))
    assert quiet["amp"] < 0.0006 and quiet["noise"] == pytest.approx(3.29 * 0.002 / np.sqrt(18), rel=0.15)
    assert tess_activity.variability(t[:100], np.ones(100)) is None    # under MIN_BINS three-hour bins


def _tglc_file(t, aper, psf, sector=5, bad=()):
    flags = np.zeros(len(t), np.int32)
    flags[list(bad)] = 1
    cols = [fits.Column(name="time", format="D", array=t), fits.Column(name="aperture_flux", format="D", array=aper),
            fits.Column(name="psf_flux", format="D", array=psf), fits.Column(name="TESS_flags", format="J", array=flags),
            fits.Column(name="TGLC_flags", format="J", array=np.zeros(len(t), np.int32))]
    primary = fits.PrimaryHDU()
    primary.header["SECTOR"], primary.header["TESSMAG"] = sector, 12.5
    buf = io.BytesIO()
    fits.HDUList([primary, fits.BinTableHDU.from_columns(cols)]).writeto(buf)
    return buf.getvalue()


def _two_orbits(step=1 / 48):
    t = np.arange(0, 27.0, step)
    return t[(t < 13.0) | (t > 14.5)]


def test_flatten_removes_each_orbits_drift_and_its_first_day():
    rng = np.random.default_rng(12)
    t = _two_orbits()
    star = 1 + 0.01 * np.sin(2 * np.pi * t / 3.0)
    second = t > 14
    drift = np.where(second, 0.05 * (t - 14.5) / 12.5, -0.04 * t / 13)            # a different line in each orbit
    ramp = 0.2 * np.exp(-t / 0.2) + 0.2 * np.exp(-np.clip(t - 14.5, 0, None) / 0.2) * second   # orbit starts
    flux = 800 * (star + drift + ramp + rng.normal(0, 0.001, len(t)))
    raw = tess_activity.variability(t, flux)
    tf, ff = tess_activity.flatten(t, flux)
    flat = tess_activity.variability(tf, ff)
    assert raw["amp"] > 0.04
    assert flat["amp"] == pytest.approx(2 * 0.01 * np.sin(np.radians(81)), rel=0.1)
    assert flat["period"] == pytest.approx(3.0, rel=0.05)
    assert tf.min() >= 1.0 and not ((tf > 13) & (tf < 15.5)).any()                # first day of each orbit gone
    assert len(tess_activity.flatten(t[:30], flux[:30])[0]) == 0                  # too short to fit


def test_sector_row_takes_the_smaller_of_the_two_fluxes():
    rng = np.random.default_rng(2)
    t = _two_orbits()
    star = 1 + 0.01 * np.sin(2 * np.pi * t / 5.0)
    aper = 500 * (star + 0.02 * np.sin(2 * np.pi * t / 1.7) + rng.normal(0, 0.001, len(t)))   # pointing jitter
    psf = 400 * (star + rng.normal(0, 0.001, len(t)))
    psf[100] = 1e6                                                              # a flagged cadence
    row = tess_activity.sector_row(42, _tglc_file(t, aper, psf, bad=[100]))
    assert row["sector"] == 5 and row["tmag"] == 12.5 and row["source_id"] == 42
    assert row["amp"] == row["amp_psf"] < row["amp_aper"]
    assert row["amp"] == pytest.approx(2 * 0.01 * np.sin(np.radians(81)), rel=0.1)
    assert row["period"] == pytest.approx(5.0, rel=0.05)


def test_spread_and_summarise():
    assert tess_activity.spread(list(range(1, 21)), 4) == [1, 7, 14, 20]
    assert tess_activity.spread([3, 1, 2], 6) == [1, 2, 3]
    rows = pd.DataFrame({"source_id": 1, "sector": [1, 30, 55], "t": [10.0, 800.0, 1500.0], "n_bins": 200,
                         "range": 0.02, "noise": 0.001, "amp": [0.010, 0.020, 0.040], "amp_aper": 0.02,
                         "amp_psf": 0.02, "period": [3.0, 3.1, np.nan], "power": [0.5, 0.8, np.nan], "tmag": 12.0})
    s = tess_activity.summarise(rows)
    assert (s["tess_sectors"], s["tess_amp"], s["tess_period"]) == (3, 0.020, 3.1)
    assert s["tess_amp_change"] == pytest.approx(0.030) and (s["tess_amp_min"], s["tess_amp_max"]) == (0.010, 0.040)
    assert tess_activity.summarise(rows.iloc[:0]) == {"tess_sectors": 0}


# ---- the comparison --------------------------------------------------------------------------------------------

def test_band_factor_follows_the_colour_of_the_change():
    assert ind.band_factor("sm_u") == 1.0                                         # no BP and RP
    assert ind.band_factor("unlisted", -0.06, 0.004, -0.02, 0.004) == 1.0
    assert ind.band_factor("sm_u", -0.03, 0.01, -0.02, 0.01) == 1.0               # they agree within errors
    blue = ind.band_factor("sm_u", -0.06, 0.004, -0.02, 0.004)
    red = ind.band_factor("sm_z", -0.06, 0.004, -0.02, 0.004)
    assert blue > 1.3 and red < 0.6                                               # reddening: more in the blue
    assert ind.band_factor("sm_u", -0.5, 0.001, 0.001, 0.001) <= ind.BAND_FACTOR_RANGE[1]


def test_dilution_counts_neighbours_inside_the_beam():
    assert ind.dilution([0.0], [0.0], 16.0) == pytest.approx(1.0)                 # an equal star on top
    assert ind.dilution([0.0], [2.5], 16.0) == pytest.approx(0.1)
    assert ind.dilution([60.0], [0.0], 16.0) < 1e-3 and ind.dilution([], [], 16.0) == 0.0
    nb = pd.DataFrame({"source_id": [1, 2, 3], "ra": [10.0, 10.0, 10.0 + 4 / 3600], "dec": [0.0, 0.0, 0.0],
                       "phot_g_mean_mag": [14.0, np.nan, 14.0]})
    c = ind.dilutions(nb, 1, 10.0, 0.0, 14.0, ["asassn", "noirlab"])               # itself and the NaN left out
    assert c["asassn"] == pytest.approx(np.exp(-16 / (4 * (16 / 2.3548) ** 2))) and c["noirlab"] < 1e-6
    assert "gaiadr4.gaia_source" in ind.neighbours_adql(10.0, 0.0, "gaiadr4")


def test_bins_never_cross_an_edge_and_carry_the_floor():
    rng = np.random.default_rng(3)
    v = ind.visits(epochs(rng, nightly(), 1.0))
    b = ind.bins(v, edges=[600.0, 1000.0], floors=FLOORS)
    assert b["n"].sum() == len(v) and b["n"].max() <= 45               # 180 days of 4-day visits at most
    for e in (600.0, 1000.0, 777.0):
        # A bin holding visits from both sides of the edge would make the whole give fewer bins than its halves.
        whole = ind.bins(v, edges=[e], floors=FLOORS)
        halves = [ind.bins(v[v["t"] < e], edges=[e], floors=FLOORS), ind.bins(v[v["t"] >= e], edges=[e], floors=FLOORS)]
        assert len(whole) == len(halves[0]) + len(halves[1])
        assert np.allclose(np.sort(whole["t"]), np.sort(pd.concat(halves)["t"]))
    assert (b["err"] >= FLOORS["skymapper"]).all() and b["err"].max() < 0.006


def test_a_survey_sees_gaias_trend_and_a_constant_star_rules_it_out():
    rng = np.random.default_rng(4)
    t = nightly()
    fade = 1 + TREND["slope"] * (t - TREND["t_first"]) / 365.25
    lc = pd.concat([epochs(rng, t[::2], fade[::2], group="a"), epochs(rng, t[1::2], 1.07 * fade[1::2], group="b")])
    r = ind.same_years(lc, TREND, floors=FLOORS)
    assert r["skymapper"]["verdict"] == "seen" and r["all"]["verdict"] == "seen"
    assert abs(r["skymapper"]["k"] - 1) < 3 * r["skymapper"]["k_err"] < 0.3
    assert r["skymapper"]["n_groups"] == 2
    flat = ind.same_years(epochs(rng, t, 1.0), TREND, floors=FLOORS)["skymapper"]
    assert flat["verdict"] == "not_seen" and abs(flat["k"]) < 3 * flat["k_err"]
    up = ind.same_years(epochs(rng, t, 2 - fade), TREND, floors=FLOORS)["skymapper"]
    assert up["verdict"] == "opposite"
    few = ind.same_years(epochs(rng, [100.0, 400.0, 900.0], 1.0), TREND, floors=FLOORS)
    assert few["skymapper"]["verdict"] == "no_data" and few["all"]["verdict"] == "no_data"


def test_verdicts_ask_first_what_the_survey_rules_out():
    assert ind.verdict(1.0, 0.2) == "seen" and ind.verdict(0.6, 0.3) == "undecided"
    assert ind.verdict(0.0, 0.1) == "not_seen" and ind.verdict(-0.5, 0.1) == "opposite"
    assert ind.verdict(0.03, 0.005) == "not_seen"          # six errors above zero, and a thirtieth of Gaia's change
    assert ind.verdict(np.nan, 0.1) == "no_data" and ind.verdict(1.0, 0.0) == "no_data"


def test_a_trends_error_never_falls_below_the_surveys_own_drift():
    rng = np.random.default_rng(13)
    t = np.arange(-100.0, 1900.0, 1.0)
    small = {**TREND, "slope": -0.002}                                  # Gaia: 0.2% per year
    lc = pd.concat([epochs(rng, t, 1.0, group=f"g{i}", err=0.001) for i in range(4)])
    r = ind.same_years(lc, small, floors={"skymapper": 0.0005})["skymapper"]
    assert r["k_err"] >= ind.TREND_FLOOR / 0.002                        # 0.1% per year over a 0.2% per year trend
    assert r["verdict"] == "undecided"                                  # too close to the drift to rule out
    big = ind.same_years(lc, TREND, floors={"skymapper": 0.0005})["skymapper"]
    assert big["k_err"] < 0.05 and big["verdict"] == "not_seen"


def test_a_step_between_cameras_is_not_a_trend():
    rng = np.random.default_rng(5)
    t = nightly()
    lc = pd.concat([epochs(rng, t[t < 900], 1.0, survey="asassn", band="asassn_V", group="ba"),
                    epochs(rng, t[t >= 900], 0.93, survey="asassn", band="asassn_g", group="bi")])
    r = ind.same_years(lc, TREND, floors=FLOORS)["asassn"]
    assert r["verdict"] == "not_seen" and abs(r["k"]) < 3 * r["k_err"]


def test_the_stars_own_scatter_widens_the_error():
    rng = np.random.default_rng(6)
    t = nightly()
    quiet = ind.same_years(epochs(rng, t, 1.0), TREND, floors=FLOORS)["skymapper"]
    season = 1 + 0.04 * np.sin(2 * np.pi * t / 700.0)                              # a slow variable, no trend
    noisy = ind.same_years(epochs(rng, t, season), TREND, floors=FLOORS)["skymapper"]
    assert noisy["chi2_dof"] > 5 and noisy["k_err"] > 3 * quiet["k_err"]
    assert noisy["verdict"] != "seen"


def test_dilution_and_band_factor_scale_what_is_expected():
    rng = np.random.default_rng(7)
    t = nightly()
    fade = 1 + 0.5 * TREND["slope"] * (t - TREND["t_first"]) / 365.25              # half Gaia's rate: a blend
    lc = epochs(rng, t, fade, survey="asassn", band="asassn_g", group="bi")
    assert ind.same_years(lc, TREND, floors=FLOORS)["asassn"]["k"] == pytest.approx(0.5, abs=0.1)
    assert ind.same_years(lc, TREND, c={"asassn": 1.0}, floors=FLOORS)["asassn"]["k"] == pytest.approx(1.0, abs=0.2)


def test_an_episode_is_compared_inside_against_outside_its_window():
    rng = np.random.default_rng(8)
    t = nightly()
    dip = np.where((t >= EPISODE["t0"]) & (t <= EPISODE["t1"]), 1 - EPISODE["depth"], 1.0)
    r = ind.same_years(epochs(rng, t, dip), EPISODE, floors=FLOORS)["skymapper"]
    assert r["verdict"] == "seen" and r["k"] == pytest.approx(1.0, abs=0.15)
    assert ind.same_years(epochs(rng, t, 1.0), EPISODE, floors=FLOORS)["skymapper"]["verdict"] == "not_seen"
    outside = epochs(rng, t[(t < 500) | (t > 1100)], 1.0)                          # no bins inside the window
    assert ind.same_years(outside, EPISODE, floors=FLOORS)["skymapper"]["verdict"] == "no_data"


def test_after_gaia_a_trend_goes_on_stops_or_reverses():
    rng = np.random.default_rng(9)
    t = np.arange(1850.0, 3700.0, 4.0)
    years = (t - TREND["t_last"]) / 365.25
    on = ind.after(epochs(rng, t, 1 + TREND["slope"] * years, survey="ztf", band="ztf_r"), TREND, floors=FLOORS)
    assert on["ztf"]["verdict"] == "went_on" == on["all"]["verdict"]
    assert on["all"]["ratio"] == pytest.approx(1.0, abs=0.2) and on["all"]["slope"] == pytest.approx(-0.03, abs=0.006)
    assert ind.after(epochs(rng, t, 1.0, survey="ztf", band="ztf_r"), TREND, floors=FLOORS)["all"]["verdict"] == "stopped"
    back = ind.after(epochs(rng, t, 1 - TREND["slope"] * years, survey="ztf", band="ztf_r"), TREND, floors=FLOORS)
    assert back["all"]["verdict"] == "reversed"
    assert ind.after(epochs(rng, nightly(), 1.0), TREND, floors=FLOORS)["all"]["verdict"] == "no_data"   # nothing later


def test_after_gaia_an_episode_still_under_way_stays_faint_or_recovers():
    rng = np.random.default_rng(10)
    change = {**EPISODE, "t0": 1200.0, "t1": 1800.0, "to_end": True}
    t = np.arange(1000.0, 3700.0, 4.0)
    faint = np.where(t >= 1200, 1 - change["depth"], 1.0)
    r = ind.after(epochs(rng, t, faint, survey="ztf", band="ztf_r"), change, floors=FLOORS)
    assert r["all"]["verdict"] == "still_faint" and r["all"]["recovered"] == pytest.approx(0.0, abs=0.2)
    back = np.where((t >= 1200) & (t <= 1830), 1 - change["depth"], 1.0)
    r = ind.after(epochs(rng, t, back, survey="ztf", band="ztf_r"), change, floors=FLOORS)
    assert r["all"]["verdict"] == "recovered" and r["all"]["recovered"] == pytest.approx(1.0, abs=0.2)
    deeper = np.where(t >= 1200, 1 - change["depth"], 1.0) - np.where(t > 1830, 0.1, 0.0)
    r = ind.after(epochs(rng, t, deeper, survey="ztf", band="ztf_r"), change, floors=FLOORS)
    assert r["all"]["verdict"] == "fainter" and r["all"]["recovered"] == pytest.approx(-0.1 / 0.08, abs=0.3)
    assert ind.after(epochs(rng, t, faint), EPISODE, floors=FLOORS)["all"]["verdict"] == "no_data"   # it had ended


def test_gaia_change_reads_a_fading_list_row():
    row = {"strength_by": "trend", "by": "episode+trend", "slope": -0.02, "slope_bp": -0.03, "slope_bp_err": 0.002,
           "slope_rp": -0.01, "slope_rp_err": 0.002, "t_first": -100.0, "t_last": 1800.0}
    c = ind.gaia_change(row)
    assert c["kind"] == "trend" and c["slope"] == -0.02 and c["d_bp"] == -0.03
    assert ind.total_change(c) == pytest.approx(0.02 * 1900 / 365.25)
    e = ind.gaia_change(pd.Series({"by": "episode+trend", "dim_depth": 0.1, "dim_t0": 5.0, "dim_t1": 900.0,
                                   "fade_kind": "drop", "t_first": 0.0, "t_last": 900.0}))
    assert e["kind"] == "episode" and e["to_end"] and np.isnan(e["d_bp"]) and ind.total_change(e) == 0.1
    assert ind.gaia_change({"by": "repeat"})["kind"] == "repeat"
    assert ind.same_years(epochs(np.random.default_rng(0), nightly(), 1.0), ind.gaia_change({"by": "repeat"}))["all"][
        "verdict"] == "no_data"


def test_activity_ratio_and_flat_columns():
    assert ind.activity(TREND, 0.05) == pytest.approx(0.03 * 1800 / 365.25 / 0.05)
    assert np.isnan(ind.activity(TREND, 0.0)) and np.isnan(ind.activity(TREND, None))
    # A star TESS finds quiet: the floor stands in for its amplitude, so the ratio is a lower limit.
    floor = tess_activity.quiet_floor(12.0)
    assert floor < tess_activity.quiet_floor(14.0) < tess_activity.quiet_floor(15.5) == tess_activity.quiet_floor(np.nan)
    assert ind.activity(TREND, 0.002, floor) == pytest.approx(0.03 * 1800 / 365.25 / floor)
    assert ind.activity(TREND, 0.05, floor) == ind.activity(TREND, 0.05)
    rng = np.random.default_rng(11)
    t = np.arange(-100.0, 3700.0, 4.0)
    lc = epochs(rng, t, 1 + TREND["slope"] * (t - TREND["t_first"]) / 365.25)
    row = ind.columns(ind.same_years(lc, TREND, floors=FLOORS), ind.after(lc, TREND, floors=FLOORS))
    assert row["same_years"] == "seen" and row["skymapper_same"] == "seen" and row["after"] == "went_on"
    assert {"same_k", "same_k_err", "skymapper_k", "skymapper_k_err", "skymapper_bins", "after_ratio",
            "after_ratio_err", "after_slope", "skymapper_after"} <= set(row)
