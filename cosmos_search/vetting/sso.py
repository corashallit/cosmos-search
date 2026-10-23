"""Solar-system bodies at dip epochs: a SkyBoT cone search per detected dip (FP class solar_system_occultation).

A known asteroid near the line of sight changes the measured flux in two ways. Inside the photometric window it
adds light, which biases the baseline or fills in a dip. In a background estimate (TESS annuli) it is
over-subtracted and makes a dip. An asteroid exactly in front of the star occults it for seconds, which shows
as a single-epoch dip in Gaia when a transit catches it. All three need the body within about the window or
aperture of the star at the dip epoch, so the test is a cone search at each dip's time.

The result is evidence for the packet and Stage B, not a veto: whether a body explains a dip depends on its
brightness against the depth, and on geometry that SkyBoT's basic output only roughly gives.

Observer. SkyBoT's cone search is geocentric whatever the location code (checked on Gaia DR3's own detection of
(1009) Sirene on 2015-02-20: SkyBoT put it 544 arcsec from where Gaia saw it, with location 258 and with 500).
That is parallax: Gaia is ~0.01 au from the Earth, a 2000 arcsec shift for a body at 1 au. Miriade's ephemcc does
take the spacecraft ("@gaia", "@tess") and puts Sirene within 0.1 arcsec of Gaia's position. So the search is a
geocentric cone widened by the largest parallax, then one Miriade call per dip for a reference body. Every body
near one line of sight shifts by the same sky-plane vector divided by its distance, so the reference fixes the
shift for all of them.

Local fallback. SkyBoT and Miriade go down for hours at a time, so where this machine has the local ephemeris
(sso_local.py, built by scripts/build_sso.py) it is used first: the same cone, computed from MPCORB orbits
integrated with ASSIST, seen from the spacecraft directly. It agrees with JPL Horizons to ~0.1 arcsec, but has no
comets and ages with its MPCORB file. COSMOS_SSO=remote forces SkyBoT and Miriade; COSMOS_SSO=local refuses them.
Each dip records which was used ("source").

Times: candidates carry BTJD (BJD_TDB - 2457000). SkyBoT wants the epoch at the observer, so the barycentric
light-travel time towards the source is removed (up to +-8.3 min; Gaia at L2 and TESS are within ~5 light s of
the geocentre for this purpose). Main-belt bodies move ~30-40 arcsec/h, so a few minutes of timing slack or
SkyBoT's ephemeris uncertainty (Err column) sit inside the search radii below.
"""

import os

import numpy as np

URL = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
MIRIADE = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"
OBSERVER = {"gaia": "@gaia", "tess": "@tess"}  # Miriade spacecraft observers
# Largest observer-geocentre distance (au): Gaia's L2 orbit, TESS's apogee. Bodies nearer than MIN_DIST_AU are
# missed when the parallax exceeds the widened cone (NEOs on close approach).
OBSERVER_DIST_AU = {"gaia": 0.0102, "tess": 0.0026}
MIN_DIST_AU = 0.8
# Gaia: the G window is 0.7 x 2.1 arcsec, BP/RP 3.5 x 2.1; the radius adds timing and ephemeris slack.
# TESS: 21 arcsec pixels; apertures plus background annuli reach ~6 px.
RADIUS_ARCSEC = {"gaia": 15.0, "tess": 150.0}
BTJD0 = 2457000.0
V_MINUS_G = 0.3  # solar-coloured bodies are ~0.2-0.4 mag brighter in Gaia G than in V
TESS_MINUS_V = -0.7  # and ~0.7 mag brighter in the TESS band


def observer_jd_utc(btjd, ra, dec):
    """JD (UTC) at the geocentre for barycentric BTJD times of light from (ra, dec) in degrees."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    t = Time(np.asarray(btjd, float) + BTJD0, format="jd", scale="tdb",
             location=EarthLocation.from_geocentric(0, 0, 0, unit=u.m))
    ltt = t.light_travel_time(SkyCoord(ra, dec, unit="deg"), kind="barycentric")
    return (t - ltt).utc.jd


def _deg(ra_hms, dec_dms):
    from astropy.coordinates import Angle

    return Angle(ra_hms, unit="hourangle").deg, Angle(dec_dms, unit="deg").deg


def parse(text):
    """SkyBoT text output (-mime=text, -output=basic) to a list of dicts."""
    rows, header = [], None
    for line in text.splitlines():
        if line.startswith("# Num"):
            header = [h.strip() for h in line[1:].split("|")]
        elif header and line.strip() and not line.startswith("#"):
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < len(header):
                continue
            r = dict(zip(header, cells))
            ra, dec = _deg(r["RA(h)"], r["DE(deg)"])
            rows.append({"name": r["Name"], "number": r["Num"], "class": r["Class"], "v_mag": float(r["Mv"]),
                         "ra": ra, "dec": dec, "dist_au": float(r["Dg(ua)"]),
                         "pos_err_arcsec": float(r["Err(arcsec)"]), "geo_sep_arcsec": float(r["d(arcsec)"]),
                         "rate_arcsec_h": float(np.hypot(float(r["dRA(arcsec/h)"]), float(r["dDEC(arcsec/h)"])))})
    return rows


def cone(jd_utc, ra, dec, radius_arcsec, client=None):
    """Known bodies within radius of (ra, dec) at jd_utc, geocentric (see the module notes)."""
    from cosmos_search.net import get_client

    client = client or get_client()
    resp = client.get(URL, params={"-ep": f"{jd_utc:.6f}", "-ra": f"{ra:.7f}", "-dec": f"{dec:.7f}",
                                   "-rd": f"{radius_arcsec / 3600:.6f}", "-mime": "text", "-output": "basic",
                                   "-loc": "500", "-from": "cosmos-search"}, cache=True)
    if resp.status_code == 204:
        return []
    resp.raise_for_status()
    return parse(resp.text)


def ephemeris(body, jd_utc, observer, client=None):
    """(ra, dec, distance in au) of a SkyBoT body seen from observer at jd_utc, from Miriade."""
    from cosmos_search.net import get_client

    client = client or get_client()
    name = f"a:{body['number']}" if body["number"] not in ("", "-") else f"a:{body['name']}"
    resp = client.get(MIRIADE, params={"-name": name, "-ep": f"{jd_utc:.6f}", "-observer": observer,
                                       "-mime": "text", "-tscale": "UTC", "-from": "cosmos-search"}, cache=True)
    resp.raise_for_status()
    line = next(x for x in resp.text.splitlines() if x.strip() and not x.startswith("#"))
    tok = line.split()
    ra, dec = _deg(" ".join(tok[1:4]), " ".join(tok[4:7]))
    return ra, dec, float(tok[7])


def _offset(ra, dec, ra0, dec0):
    """Tangent-plane offset (arcsec, east and north) of (ra, dec) from (ra0, dec0)."""
    d_ra = ((ra - ra0 + 180) % 360 - 180) * np.cos(np.radians(dec0))
    return np.array([d_ra, dec - dec0]) * 3600


def nearby(jd_utc, ra, dec, radius_arcsec, arm, client=None):
    """Bodies within radius of (ra, dec) as seen from the arm's spacecraft, each with its observer-centric sep."""
    return nearby_with_source(jd_utc, ra, dec, radius_arcsec, arm, client)[0]


def nearby_with_source(jd_utc, ra, dec, radius_arcsec, arm, client=None):
    """(bodies, "local" or "skybot"): the local ephemeris when it covers the epoch (see the module notes). Passing
    a client asks for the remote services through it, unless COSMOS_SSO=local."""
    from . import sso_local

    mode = os.environ.get("COSMOS_SSO", "auto").lower()
    if mode == "local" or (mode != "remote" and client is None and sso_local.available()):
        try:
            return sso_local.cone(jd_utc, ra, dec, radius_arcsec, observer=arm), "local"
        except LookupError:  # epoch or spacecraft outside the local ephemeris
            if mode == "local":
                raise
    return _nearby_remote(jd_utc, ra, dec, radius_arcsec, arm, client), "skybot"


def _nearby_remote(jd_utc, ra, dec, radius_arcsec, arm, client=None):
    margin = np.degrees(OBSERVER_DIST_AU[arm] / MIN_DIST_AU) * 3600
    bodies = cone(jd_utc, ra, dec, radius_arcsec + margin, client)
    if not bodies:
        return []
    # Parallax: observer-centric = geocentric + v / distance, the same v for every body on this line of sight.
    ref = min(bodies, key=lambda b: (b["number"] in ("", "-"), b["geo_sep_arcsec"]))
    obs = ephemeris(ref, jd_utc, OBSERVER[arm], client)
    geo = ephemeris(ref, jd_utc, "500", client)  # from Miriade too, so the two ephemerides' offsets cancel
    v = (_offset(obs[0], obs[1], ra, dec) - _offset(geo[0], geo[1], ra, dec)) * geo[2]
    out = []
    for b in bodies:
        rel = _offset(b["ra"], b["dec"], ra, dec) + v / b["dist_au"]
        b["sep_arcsec"] = float(np.hypot(*rel))
        if b["sep_arcsec"] <= radius_arcsec:
            out.append(b)
    return out


def flux_ratio(v_mag, star_mag, arm):
    """A body's flux as a fraction of the star's, in the arm's band."""
    body = v_mag - V_MINUS_G if arm == "gaia" else v_mag + TESS_MINUS_V
    return float(10 ** (-0.4 * (body - star_mag)))


def check(cand, dips=None, client=None, radius_arcsec=None):
    """Per detected dip: the known bodies near the source at its epoch, and whether one could matter.

    "sso_bright" when a body's flux is at least a tenth of the dip depth (enough to bias it, or, for TESS,
    to make it from a background annulus); "sso_faint" when bodies are present but too faint; "clear" when
    none is in the radius. Candidates without a position return [] (the packet anonymiser has removed it).
    """
    src = cand.source
    ra, dec = src.get("ra"), src.get("dec")
    if ra is None or dec is None:
        return []
    dips = cand.dips if dips is None else dips
    star = src.get("phot_g_mean_mag") if cand.arm == "gaia" else src.get("tmag", src.get("phot_g_mean_mag"))
    radius = radius_arcsec or RADIUS_ARCSEC[cand.arm]
    times = [d["t_min"] for d in dips]
    if not times:
        return []
    out = []
    for d, jd in zip(dips, np.atleast_1d(observer_jd_utc(times, ra, dec))):
        bodies, source = nearby_with_source(float(jd), ra, dec, radius, cand.arm, client)
        for b in bodies:
            b["flux_ratio"] = flux_ratio(b["v_mag"], star, cand.arm) if star is not None else None
        depth = abs(d.get("depth", 0.0)) or 1e-3
        bright = [b for b in bodies if b["flux_ratio"] is not None and b["flux_ratio"] >= 0.1 * depth]
        verdict = "sso_bright" if bright else "sso_faint" if bodies else "clear"
        out.append({"t_min": d["t_min"], "jd_utc": float(jd), "verdict": verdict, "bodies": bodies, "source": source})
    return out
