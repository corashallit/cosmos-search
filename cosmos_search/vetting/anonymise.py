"""Strip what would let a blind expert identify an object, keep what physics needs.

Stage A has no tools, so the only identification route left is memory: a model
that has read about KIC 8462852 may recognise it from a packet. We cannot
remove that route entirely (magnitude, colour, distance and a famous light curve
can be enough), so we coarsen what we can, scan for anything identifier-shaped,
and measure the rest with the recognition probe in every verdict
(docs/vetting.md §4).
"""

import math
import re

import numpy as np

# Source keys that identify an object or pin its position. Anything with "id" or "name" in it goes too.
DENY_KEYS = {"ra", "dec", "ra_deg", "dec_deg", "l", "b", "elon", "elat", "ecl_lon", "ecl_lat",
             "ra_error", "dec_error", "ref_epoch", "designation", "simbad", "vsx", "alias", "bibcode"}

LEAK_PATTERNS = {
    "catalogue_id": re.compile(r"\b(TIC|KIC|EPIC|HD|HIP|TYC|KOI|TOI|BD|CD|GSC|UCAC4?|HR|GJ|LHS|WD|PG)[ _-]?[+-]?\d", re.I),
    "gaia_id": re.compile(r"\bGaia\s*(E?DR\d)?\s*\d{6,}", re.I),
    "alert_name": re.compile(r"\b(Gaia|ZTF|ASASSN|ASAS-SN|ATLAS|AT|SN)[ -]?\d{2}[a-z]{2,4}\b", re.I),
    "jname": re.compile(r"\bJ\d{4,6}(\.\d+)?[+-]\d{4,6}"),
    # Digit runs inside a word (hex ids, hashes) are not identifiers; standalone runs of 7+ digits may be.
    "long_integer": re.compile(r"(?<![\w.])\d{7,}(?![\w.])"),
    "julian_date": re.compile(r"\b24[3-6]\d{4}(\.\d+)?\b"),
    "sexagesimal": re.compile(r"\b\d{1,2}[h:]\s?\d{2}[m:]\s?\d{2}(\.\d+)?s?\b"),
    "coordinate_label": re.compile(r"\b(ra|dec|r\.a\.|decl)\s*[=:]", re.I),
    "bibcode": re.compile(r"\b(19|20)\d{2}[A-Za-z&]{2,6}\.{0,4}\d"),
    "famous_name": re.compile(r"boyajian|tabby|przybylski|epsilon aur|eps aur|ee cep|8462852|vvv-wit", re.I),
}


def leak_scan(text):
    """Names of identifier patterns found in text; empty means clean."""
    return sorted(name for name, rx in LEAK_PATTERNS.items() if rx.search(text))


def sig(x, n=3):
    """Round to n significant figures; non-finite values become None."""
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return round(x, n - 1 - int(math.floor(math.log10(abs(x)))))


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


# ICRS -> Galactic rotation (Hipparcos convention).
_GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                 [+0.4941094279, -0.4448296300, +0.7469822445],
                 [-0.8676661490, -0.1980763734, +0.4559837762]])
_OBLIQUITY = math.radians(23.4392911)


def galactic_latitude(ra, dec):
    return math.degrees(math.asin(float(np.clip((_GAL @ _unit(ra, dec))[2], -1, 1))))


def ecliptic_latitude(ra, dec):
    x, y, z = _unit(ra, dec)
    return math.degrees(math.asin(float(np.clip(-math.sin(_OBLIQUITY) * y + math.cos(_OBLIQUITY) * z, -1, 1))))


def _bin(x, width):
    lo = math.floor(x / width) * width
    return f"{lo:g} to {lo + width:g}"


def sky_context(source):
    """Position reduced to what matters physically: |b| (extinction, crowding) and ecliptic latitude (scanning, SSOs)."""
    if "ra" not in source or "dec" not in source:
        return {}
    ra, dec = float(source["ra"]), float(source["dec"])
    return {
        "abs_galactic_latitude_deg": _bin(abs(galactic_latitude(ra, dec)), 5),
        "abs_ecliptic_latitude_deg": _bin(abs(ecliptic_latitude(ra, dec)), 10),
    }


def distance_context(source):
    plx, err = source.get("parallax"), source.get("parallax_error")
    if plx is None or err is None or not err > 0:
        return {}
    snr = float(plx) / float(err)
    if snr >= 5:
        return {"distance_pc": sig(1000.0 / float(plx), 2), "parallax_snr": sig(snr, 2)}
    return {"distance_pc": "poorly constrained", "parallax_snr": sig(snr, 2)}


def scrub_source(source):
    """Numeric source fields, rounded, minus identifiers and positions, plus coarse sky and distance context."""
    out = {}
    for k, v in source.items():
        kl = k.lower()
        if kl in DENY_KEYS or "id" in kl.split("_") or "name" in kl or kl.startswith(("parallax",)):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            continue
        out[k] = round(float(v), 1) if "mag" in kl else sig(v, 3)
    out.update(sky_context(source))
    out.update(distance_context(source))
    return out
