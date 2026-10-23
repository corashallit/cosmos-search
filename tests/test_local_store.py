"""The local data store: HATS mirrors read first, the persistent cache root, the local solar-system ephemeris."""

import json

import numpy as np
import pandas as pd
import pytest


def test_data_dir_and_cache_root(tmp_path, monkeypatch):
    from cosmos_search.net import cache, local

    monkeypatch.delenv("COSMOS_CACHE_DIR", raising=False)
    monkeypatch.setenv("COSMOS_DATA_DIR", str(tmp_path))
    assert local.data_dir() == tmp_path and local.persistent()
    assert cache.default_root() == tmp_path / "cache" / "http"
    assert cache.default_max_gb() == local.PERSISTENT_CACHE_GB
    monkeypatch.setenv("COSMOS_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert cache.default_root() == tmp_path / "elsewhere"


def _fake_mirror(root, name, url, columns):
    d = root / "mirror" / "hats" / name
    d.mkdir(parents=True)
    (d / "mirror.json").write_text(json.dumps({"name": name, "url": url, "columns": list(columns)}))
    return d


def test_mirror_lookup_needs_every_column(tmp_path, monkeypatch):
    from cosmos_search.xmatch import mirror
    from cosmos_search.xmatch.sources import HatsSource
    from cosmos_search.xmatch.surveys import SURVEYS

    monkeypatch.setenv("COSMOS_DATA_DIR", str(tmp_path))
    mirror._manifest.cache_clear()
    s = SURVEYS["vsx"]
    d = _fake_mirror(tmp_path, "vsx", s.hats + "/", (*s.columns, "_healpix_29"))
    _fake_mirror(tmp_path, "2mass", SURVEYS["2mass"].hats, SURVEYS["2mass"].columns[:3])
    assert mirror.local(name="vsx", columns=s.columns) == str(d)
    assert mirror.resolve(s.hats, ["OID", "Name"]) == str(d)            # trailing slash does not matter
    assert mirror.resolve(s.hats, ["OID", "not_mirrored"]) == s.hats   # a missing column goes remote
    assert HatsSource().location(s) == str(d)
    assert HatsSource().location(SURVEYS["2mass"]) == SURVEYS["2mass"].hats  # incomplete mirror: remote
    assert HatsSource(urls={"vsx": "https://x"}).location(s) == "https://x"  # explicit urls win
    monkeypatch.setenv("COSMOS_MIRROR", "off")
    assert mirror.local(name="vsx", columns=s.columns) is None


def test_mirror_is_read_like_the_remote_catalogue(tmp_path, monkeypatch):
    """A HATS catalogue written to the mirror path answers cones exactly as an in-memory copy does."""
    lsdb = pytest.importorskip("lsdb")
    from cosmos_search.xmatch import mirror
    from cosmos_search.xmatch.sources import AutoSource, MemorySource
    from cosmos_search.xmatch.surveys import SURVEYS

    monkeypatch.setenv("COSMOS_DATA_DIR", str(tmp_path))
    mirror._manifest.cache_clear()
    s = SURVEYS["asassn_vars"]  # TAP-only upstream: served from the mirror when one exists
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({"RAJ2000": 10 + rng.uniform(-1, 1, n), "DEJ2000": -5 + rng.uniform(-1, 1, n),
                       "ID": np.arange(n), "ASASSN-V": [f"J{i}" for i in range(n)], "Type": "EA",
                       "Per": rng.uniform(1, 10, n), "Amp": 0.5, "Vmag": 12.0, "Prob": 0.9,
                       "GaiaDR3": np.arange(n) + 10**15, "TIC": np.arange(n)})
    d = tmp_path / "mirror" / "hats" / "asassn_vars"
    lsdb.from_dataframe(df, ra_column="RAJ2000", dec_column="DEJ2000", catalog_name="asassn_vars").write_catalog(d)
    (d / "mirror.json").write_text(json.dumps({"name": "asassn_vars", "url": None,
                                                "columns": [*df.columns, "_healpix_29"]}))
    got = AutoSource().fetch(s, 10.2, -5.1, 0.3).sort_values("ID").reset_index(drop=True)
    want = MemorySource(df).fetch(s, 10.2, -5.1, 0.3).sort_values("ID").reset_index(drop=True)
    assert len(got) == len(want) > 0
    assert (got["ID"].to_numpy() == want["ID"].to_numpy()).all()


def test_two_body_propagation_round_trip():
    from cosmos_search.vetting.sso_local import GM_SUN, propagate

    # Circular orbit at 1 au: a quarter period moves x onto y.
    v = np.sqrt(GM_SUN)
    period = 2 * np.pi / np.sqrt(GM_SUN)
    r0, v0 = np.array([[1.0, 0, 0]]), np.array([[0, v, 0]])
    p, vel = propagate(r0, v0, period / 4)
    assert np.allclose(p, [[0, 1, 0]], atol=1e-10) and np.allclose(vel, [[-v, 0, 0]], atol=1e-12)
    # Eccentric orbit: forward then back returns the start; energy is conserved.
    r0, v0 = np.array([[1.3, 0.2, 0.1]]), np.array([[0.002, 0.016, 0.001]])
    p, vel = propagate(r0, v0, 37.0)
    q, w = propagate(p, vel, -37.0)
    assert np.allclose(q, r0, atol=1e-11) and np.allclose(w, v0, atol=1e-13)
    energy = lambda r, u: 0.5 * np.sum(u**2) - GM_SUN / np.linalg.norm(r)
    assert np.isclose(energy(p, vel), energy(r0, v0), rtol=1e-10)


def _tiny_ephemeris(root, body_state, epochs, observer_xyz):
    """A one-body ephemeris on disk in sso_local's layout; the Sun sits at the barycentre."""
    root.mkdir(parents=True)
    np.save(root / "epochs.npy", np.asarray(epochs, float))
    for k in range(len(epochs)):
        np.save(root / f"states_{k:04d}.npy", np.asarray([body_state[k]], float))
    grid = np.arange(epochs[0] - 1, epochs[-1] + 1, 1 / 24)
    obs = {"jd_tdb": grid, "sun": np.zeros((len(grid), 3)), "earth": np.tile(observer_xyz, (len(grid), 1)),
           "gaia": np.tile(observer_xyz, (len(grid), 1)), "tess": np.full((len(grid), 3), np.nan)}
    np.savez(root / "observers.npz", **obs)
    pd.DataFrame({"number": pd.array([99999], dtype="Int64"), "name": ["Testia"], "H": [15.0], "G": [0.15],
                  "U": ["0"], "class": ["MB"]}).to_parquet(root / "bodies.parquet")
    (root / "manifest.json").write_text("{}")


def test_local_cone_finds_a_body_on_the_line_of_sight(tmp_path):
    from cosmos_search.vetting import sso_local

    # A body at rest 2 au out along +x is effectively fixed for the test (no velocity: a radial plunge would
    # move it; give it a tiny tangential speed instead and look where it is at the epoch).
    epochs = [2457000.5, 2457020.5]
    state = [[2.0, 0.0, 0.0, 0.0, 1e-6, 0.0]] * 2
    root = tmp_path / "eph"
    _tiny_ephemeris(root, state, epochs, np.array([0.0, 0.0, 0.0]))
    eph = sso_local.Ephemeris(root)
    jd_utc = 2457000.5 - 67.184 / 86400  # UTC so that TDB is the snapshot epoch
    hits = sso_local.cone(jd_utc, 0.0, 0.0, 30.0, observer="gaia", eph=eph)
    assert len(hits) == 1 and hits[0]["name"] == "Testia" and hits[0]["sep_arcsec"] < 1.0
    assert hits[0]["source"] == "local" and 15 < hits[0]["v_mag"] < 20
    assert sso_local.cone(jd_utc, 90.0, 0.0, 30.0, observer="gaia", eph=eph) == []
    with pytest.raises(LookupError):  # no TESS positions in this ephemeris
        sso_local.cone(jd_utc, 0.0, 0.0, 30.0, observer="tess", eph=eph)


def test_explicit_client_still_uses_the_remote_services(monkeypatch):
    from cosmos_search.vetting import sso, sso_local

    monkeypatch.delenv("COSMOS_SSO", raising=False)
    monkeypatch.setattr(sso_local, "available", lambda root=None: True)
    monkeypatch.setattr(sso_local, "cone", lambda *a, **k: [{"local": True}])
    monkeypatch.setattr(sso, "_nearby_remote", lambda *a, **k: [{"remote": True}])
    assert sso.nearby_with_source(2457000.5, 0, 0, 15, "gaia") == ([{"local": True}], "local")
    assert sso.nearby_with_source(2457000.5, 0, 0, 15, "gaia", client=object())[1] == "skybot"
    monkeypatch.setenv("COSMOS_SSO", "remote")
    assert sso.nearby_with_source(2457000.5, 0, 0, 15, "gaia")[1] == "skybot"


@pytest.mark.skipif("not __import__('cosmos_search.vetting.sso_local', fromlist=['x']).available()",
                    reason="needs the local solar-system ephemeris (cache-tools/build_sso.py)")
def test_local_ephemeris_puts_sirene_where_gaia_saw_it():
    """Gaia DR3's own detection of (1009) Sirene on 2015-02-20 (the SkyBoT test above): the local ephemeris,
    seen from Gaia, should put it within an arcsecond of Gaia's measured position."""
    from cosmos_search.vetting import sso_local

    ra, dec = 239.09585284122534, -21.426396751900427
    hits = sso_local.cone(2457074.3949829205, ra, dec, 15.0, observer="gaia")
    sirene = [h for h in hits if h["name"] == "Sirene"]
    assert sirene and sirene[0]["sep_arcsec"] < 1.0
