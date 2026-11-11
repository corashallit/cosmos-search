"""Upper limit on how common steady fades are among ordinary dwarfs (docs/arms/occurrence.md).

The one result the search gives with or without a find: of the Sun-like dwarfs searched, at most a fraction f is
fading steadily at a given rate. It needs three measured things and no reading of light curves:

- the stars searched: FGK dwarfs by parallax, absolute magnitude and colour (`fgk_dwarf`) with a measured trend;
- how many of them fade, and how many brighten, beyond a given trend significance (`cell_counts`);
- recall: steady fades injected into real light curves of the same stars and scored like any star
  (`cosmos_search.slow.recall`) give the share of faders at each rate that would pass (`efficiency`).

The limit is cut and count. A region is the stars brighter than some G with a trend beyond some significance
(`Config.g_edges` x `Config.z_cuts`). In a region n stars fade, and a fade at the rate in question would have put
`detectable` = sum over G ranges of (stars x recall) stars there. Every one of the n is counted as a possible
real fade: f < (Poisson upper limit on n) / detectable, which is 3.0 / (stars x recall) when no star is past the
cut. Stars that fade for other reasons (spots, eclipses, the instrument) make the limit weaker and never wrong, so
it needs no model of them.

Which region: for each rate, the one whose limit would be best if as many stars faded there as brighten
(`choose_region`). The fading counts play no part in the choice, so choosing takes nothing from the limit's
coverage, whatever the stars do (tests/test_slow_occurrence.py measures choice and limit together). Fast fades
end up in regions far out in significance, where few stars are; slow ones near the threshold among bright stars.

Beside the limit, for each rate, what subtracting the chance fades would give (`mirror_*` columns). If a star's
own variability and the instrument sent its trend either way equally often, the brightening count m would
estimate the chance fades among the n (statistics.md §2, mirror null): n Poisson with mean b + f x detectable, m
Poisson with mean b, and `single_limit` gives the limit on f. It is stronger for slow fades, and it is only as
good as that symmetry. On DR3 brightening stars outnumber fading ones 89 to 61 beyond 6 errors and 28 to 12
beyond 8, and fading ones lead 1,664 to 1,477 between 3 and 5 (docs/arms/occurrence.md), and in simulated counts
with chance fades at 0.6 of the chance brightenings this number fell below the true share in a fifth to all of
the samples (2026-11-10). So it is reported as what
symmetry would buy and is never the limit. With it:
- its region is chosen on fading + brightening together (`choose_mirror_region`), never on which way the stars
  went, so with no real fades and perfect symmetry the split of the chosen region is a fair coin;
- `fade_excess_p`: the chance of at least n fading stars out of n + m for that coin (exact binomial). A small
  value says more stars fade than brighten: real fades or a broken symmetry, and the n stars are the ones to read;
- the values chance splits of the same n + m would give (16%, median, 84%);
- `sky_dispersion`: whether the stars past the cut go the same way in the same patch of sky more often than
  independent stars would. The subtraction and the excess test take stars as independent.

Trend significance is each star's G trend over its error, measured against stars like it. On DR3 the drift
correction leaves +0.08% a year at G 12-13 (big-picture review, 2026-11-10), which would move every star there
towards brightening and hide fades. Two steps take that out. The search's own neighbour reference
(`summary.apply_reference`: the median trend of all stars near a star on the sky and in brightness) removes what
depends on brightness and place. What is left depends on colour: measured on DR3 on 2026-11-11, stars redder
than BP-RP 2 brighten by 0.06 to 0.22% a year against it and the FGK dwarfs at G 18-19 sit 0.29 errors low, and
an offset like that sends the region choice to where few stars brighten and many fade. So the searched dwarfs
are then centred on their own median at each brightness (`offset_hist`, `offsets`, `centre`), and injected
stars get the same two steps. A per-star table without a neighbour reference is measured against the running
median trend of the searched stars nearest in G (`brightness_reference`), which is both steps in one.

`reviewed_limit` is the limit after the list has been read: stars in a region that no check and no reviewer
could explain, over the stars a fade would have put there and the reviewers would have kept.
"""

import functools
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

LOG2 = float(np.log(2.0))
CONFIG_DIR = Path(__file__).resolve().parents[2] / "prereg" / "occurrence"


@dataclass(frozen=True)
class Config:
    """Everything that defines the limit. The frozen record for DR4 is prereg/occurrence/<name>.json."""

    name: str = "steady-fades-v1"
    # Fades injected: fraction of the star's flux at its first visit lost per year, the same in G, BP and RP.
    rates: tuple = (0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
    # Trend significance a region starts at (fading: -z, brightening: +z).
    z_cuts: tuple = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
    # G ranges, each with its own counts and recall; a region is every range up to one of the inner edges or the
    # last. Stars outside the first and last edge are not part of the limit.
    g_edges: tuple = (0.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0)
    cl: float = 0.95
    min_visits: int = 8
    # FGK dwarfs: parallax known to 1 / min_parallax_snr, colour range, and a band below the giants and above
    # the white dwarfs in absolute magnitude, M_G - ms_slope (BP-RP) in [ms_lo, ms_hi).
    min_parallax_snr: float = 5.0
    colour_lo: float = 0.5
    colour_hi: float = 1.8
    ms_slope: float = 3.2
    ms_lo: float = 0.5
    ms_hi: float = 5.5
    # Stars nearest in G whose median trend is each star's reference.
    reference_window: int = 401
    # HEALPix level of the sky pixels for `sky_dispersion` (6: 0.84 square degrees, about Gaia's field of view).
    sky_level: int = 6

    @property
    def n_zbins(self):
        return len(self.z_cuts)

    @property
    def n_strata(self):
        return len(self.g_edges) - 1

    def to_json(self):
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def load(cls, name_or_path="steady-fades-v1"):
        """A frozen config: a .json path, or a name in prereg/occurrence/."""
        p = Path(name_or_path)
        if p.suffix != ".json":
            p = CONFIG_DIR / f"{name_or_path}.json"
        d = json.loads(p.read_text())
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


# -- the stars ----------------------------------------------------------------------------------------------------

def absolute_magnitude(g, parallax_mas):
    """M_G from the parallax alone (no extinction); NaN where the parallax is not positive."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(g, float) + 5 * np.log10(np.asarray(parallax_mas, float)) - 10


def fgk_dwarf(aux, cfg=Config()):
    """FGK dwarfs among gaia_source rows (phot_g_mean_mag, bp_rp, parallax, parallax_error): the big-picture
    review's definition. Observed colour and no extinction, so in reddened fields it leans to nearby stars."""
    par, err = aux["parallax"].to_numpy(float), aux["parallax_error"].to_numpy(float)
    col = aux["bp_rp"].to_numpy(float)
    off = absolute_magnitude(aux["phot_g_mean_mag"], par) - cfg.ms_slope * col
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((par / err >= cfg.min_parallax_snr) & (col >= cfg.colour_lo) & (col < cfg.colour_hi)
                & (off >= cfg.ms_lo) & (off < cfg.ms_hi))


def sky_pixel(source_id, level):
    """HEALPix (nested) pixel at `level` from Gaia source_ids (level 12 is source_id // 2^35)."""
    return (np.asarray(source_id, dtype=np.int64) // 2 ** 35) // 4 ** (12 - level)


def stratum(g, cfg=Config()):
    """G range of each star (index into cfg.g_edges), -1 outside."""
    g = np.asarray(g, float)
    s = np.digitize(g, cfg.g_edges) - 1
    return np.where(np.isfinite(g) & (s >= 0) & (s < cfg.n_strata), s, -1)


def slope_error(table):
    """Trend error per star: `slope_err` when the table has it, else slope / slope_z."""
    if "slope_err" in table:
        return table["slope_err"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(table["slope"].to_numpy(float) / table["slope_z"].to_numpy(float))


def brightness_reference(g, slope, window=Config.reference_window):
    """Median trend of stars of similar brightness: the running median of `slope` over the `window` stars nearest
    in G (fewer at the bright and faint ends). Returns (G sorted, median) for `reference_at`.

    The median is taken over the searched stars themselves. A fader moves it by nothing measurable: one star in
    a window of 401 shifts a median by ~1/400 of the trend error. Faders making up several percent of the stars
    would move it, and each would then be found a little less often than injections into one star at a time
    measure; `plant` checks the limit with such shares put in."""
    g, slope = np.asarray(g, float), np.asarray(slope, float)
    ok = np.isfinite(g) & np.isfinite(slope)
    order = np.argsort(g[ok], kind="stable")
    gs, s = g[ok][order], pd.Series(slope[ok][order])
    if not len(s):
        return np.array([0.0]), np.array([0.0])
    w = int(min(window, len(s)))
    med = s.rolling(w, center=True, min_periods=max(1, w // 4)).median().to_numpy()
    return gs, med


def reference_at(g, ref):
    """Reference trend at each G (linear interpolation; the end values beyond the sample's range)."""
    return np.interp(np.asarray(g, float), ref[0], ref[1])


def trend_z(table, g, ref=None):
    """Trend significance against the reference: (slope - reference) / error. A table that already carries a
    neighbour reference (`slope_ref`) keeps its own `slope_z`."""
    if "slope_ref" in table:
        return table["slope_z"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (table["slope"].to_numpy(float) - reference_at(g, ref)) / slope_error(table)


# -- the searched stars' own zero -----------------------------------------------------------------------------

OFFSET_G = np.arange(9.0, 19.5 + 1e-9, 0.5)          # G ranges; brighter and fainter stars join the end ranges
OFFSET_EDGES = np.linspace(-0.01, 0.01, 4001)        # trend against the neighbour reference, per year
OFFSET_MIN_STARS = 400                               # a median from fewer stars is noisier than what it removes


def offset_range(g):
    """Index of each star's range of G for the offset (the pass's own magnitude, `g_mag`)."""
    g = np.nan_to_num(np.asarray(g, float), nan=OFFSET_G[-1])
    return np.clip(np.digitize(g, OFFSET_G) - 1, 0, len(OFFSET_G) - 2)


def offset_hist(table):
    """Counts of the searched stars' trends against their neighbour reference (slope - slope_ref), per G range.
    Histograms of batches add, so a whole-sky table is read one file at a time."""
    d = table["slope"].to_numpy(float) - np.nan_to_num(table["slope_ref"].to_numpy(float))
    ok = np.isfinite(d)
    h = np.zeros((len(OFFSET_G) - 1, len(OFFSET_EDGES) - 1), np.int64)
    k = np.clip(np.digitize(d[ok], OFFSET_EDGES) - 1, 0, len(OFFSET_EDGES) - 2)
    np.add.at(h, (offset_range(table["g_mag"].to_numpy(float)[ok]), k), 1)
    return h


def offsets(hist, min_stars=OFFSET_MIN_STARS):
    """Median of each G range's histogram (to within a bin, 0.0005% a year). A range with fewer than min_stars
    stars is pooled with its neighbours until it has them; 0 when the whole table has not."""
    out = np.zeros(len(hist))
    mid = (OFFSET_EDGES[:-1] + OFFSET_EDGES[1:]) / 2
    for i in range(len(hist)):
        for k in range(len(hist)):
            h = hist[max(0, i - k): i + k + 1].sum(axis=0)
            if h.sum() >= min_stars:
                c = np.cumsum(h)
                j = int(np.searchsorted(c, c[-1] / 2))
                before = c[j - 1] if j else 0
                out[i] = mid[j] + ((c[-1] / 2 - before) / h[j] - 0.5) * (OFFSET_EDGES[1] - OFFSET_EDGES[0])
                break
    return out


def centre(table, offset):
    """`table` (searched or injected stars measured against the neighbour reference) with the searched stars'
    own median taken out: slope_ref moved by the offset of each star's G range, slope_z recomputed."""
    out = table.copy()
    ref = np.nan_to_num(out["slope_ref"].to_numpy(float)) + np.asarray(offset)[offset_range(out["g_mag"])]
    ref_err = np.nan_to_num(out["slope_ref_err"].to_numpy(float)) if "slope_ref_err" in out else 0.0
    out["slope_ref"] = ref
    with np.errstate(invalid="ignore", divide="ignore"):
        out["slope_z"] = (out["slope"].to_numpy(float) - ref) / np.hypot(slope_error(out), ref_err)
    return out


# -- counts and recall --------------------------------------------------------------------------------------------

def zbin(stat, cfg=Config()):
    """Significance bin of each value (index into cfg.z_cuts; the last bin is open-ended), -1 below the first."""
    stat = np.asarray(stat, float)
    return np.where(np.isfinite(stat), np.digitize(stat, cfg.z_cuts) - 1, -1)


def cell_counts(strat, z, cfg=Config()):
    """Fading (n) and brightening (m) counts per (G range, significance bin), and the stars per G range."""
    strat, z = np.asarray(strat), np.asarray(z, float)
    ok = (strat >= 0) & np.isfinite(z)
    n = np.zeros((cfg.n_strata, cfg.n_zbins), int)
    m = np.zeros_like(n)
    for arr, b in ((n, zbin(-z, cfg)), (m, zbin(z, cfg))):
        k = ok & (b >= 0)
        np.add.at(arr, (strat[k], b[k]), 1)
    return n, m, np.bincount(strat[ok], minlength=cfg.n_strata)[: cfg.n_strata]


def efficiency(strat, z, cfg=Config()):
    """Per G range and significance bin: the share of injected fades landing in the fading bin; that share less
    the share landing in the same brightening bin (what a fade adds to fading minus brightening, for the
    subtraction); and the hosts per G range. One rate at a time: strat and z are the hosts' G ranges and the
    injected stars' significances; a host whose injected light curve could not be scored has z = NaN and counts
    as missed."""
    strat, z = np.asarray(strat), np.asarray(z, float)
    hosts = np.bincount(strat[strat >= 0], minlength=cfg.n_strata)[: cfg.n_strata]
    n, m, _ = cell_counts(strat, z, cfg)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(hosts[:, None] > 0, n / hosts[:, None], 0.0)
        net = np.where(hosts[:, None] > 0, (n - m) / hosts[:, None], 0.0)
    return share, net, hosts


def regions(cells):
    """Per-cell values summed over each region: out[k, j] covers G ranges 0..k and significance bins j and up."""
    return np.cumsum(np.cumsum(np.asarray(cells, float)[:, ::-1], axis=1)[:, ::-1], axis=0)


# -- the limit ----------------------------------------------------------------------------------------------------

def poisson_upper(n, cl=0.95):
    """Classical upper limit on a Poisson mean from a count n (3.0 for n = 0 at 95%)."""
    return 0.5 * stats.chi2.ppf(cl, 2 * (np.asarray(n, float) + 1))


EXACT_MAX = 1000   # n + m above which `single_limit` uses the normal form


@functools.lru_cache(maxsize=4096)
def single_limit(n, m, cl=0.95, grid=2001):
    """Upper limit on the number of real fades among n fading stars when m stars brighten.

    n ~ Poisson(b + s), m ~ Poisson(b). With the prior 1/b, b given m is Gamma(m), so the chance fades among the
    n are negative binomial with mean m (none when m = 0), and
        P(n | s, m) = sum_k NB(k; m, 1/2) x Poisson(n - k | s).
    The limit is the cl point of the posterior of s for a flat prior on s >= 0. Flat and Jeffreys priors on b put
    its mean at m + 1 and m + 1/2 and subtract that much more.

    Above EXACT_MAX stars the sum is replaced by its normal form, s ~ N(n - m, n + m) cut at zero, plus 1.6: the
    exact limit is 0.7-1.6 stars above the normal one from 100 stars up, so this errs high by under one star."""
    n, m = int(n), int(m)
    if n + m > EXACT_MAX:
        mu, sd = float(n - m), float(np.sqrt(n + m))
        return float(stats.truncnorm.ppf(cl, -mu / sd, np.inf, loc=mu, scale=sd)) + 1.6
    k = np.arange(n + 1)
    if m > 0:
        log_nb = special.gammaln(m + k) - special.gammaln(k + 1) - special.gammaln(m) - (m + k) * LOG2
    else:
        log_nb = np.where(k == 0, 0.0, -np.inf)
    s = np.linspace(0.0, n + 8 * np.sqrt(n + m + 4.0) + 16, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pois = np.where(n - k > 0, (n - k) * np.log(s[:, None]), 0.0) - s[:, None] - special.gammaln(n - k + 1)
    ll = special.logsumexp(log_nb + log_pois, axis=1)
    p = np.exp(ll - ll.max())
    cdf = np.r_[0.0, np.cumsum((p[1:] + p[:-1]) / 2)]
    return float(np.interp(cl, cdf / cdf[-1], s))


def fade_excess_p(n, m):
    """Chance of at least n fading stars among n + m when each is as likely to fade as to brighten. Exact for a
    region chosen on its total alone (`choose_mirror_region`)."""
    return float(stats.binom.sf(n - 1, n + m, 0.5)) if n + m > 0 else 1.0


def expected_limits(n, m, cl=0.95):
    """`single_limit` at the 16%, 50% and 84% points of a chance split of n + m."""
    t = int(n + m)
    return tuple(single_limit(q, t - q, cl) for q in stats.binom.ppf([0.16, 0.5, 0.84], t, 0.5).astype(int)) \
        if t > 0 else (single_limit(0, 0, cl),) * 3


def _best(expected_limit, counts, detectable):
    best, where = np.inf, None
    for k, j in np.ndindex(counts.shape):
        if detectable[k, j] > 0:
            e = expected_limit(int(counts[k, j])) / detectable[k, j]
            if e < best:
                best, where = e, (k, j)
    return where


def choose_region(brightening, detectable, cl=0.95):
    """The region (G-range index k, significance index j) for the limit: the one whose limit would be best if as
    many stars faded there as brighten, the Poisson limit on the brightening count over the stars a fade would
    put there. brightening and detectable per region (`regions`). The fading counts are not looked at, so the
    limit in the chosen region covers as it would in a region fixed in advance. None when no region has recall."""
    return _best(lambda m: float(poisson_upper(m, cl)), brightening, detectable)


def choose_mirror_region(total, detectable, cl=0.95):
    """The region for the subtracted limit: the best `single_limit` to be expected from the stars past its cut,
    whichever way they went (an even split of total = fading + brightening). Only the total enters, so under
    perfect symmetry and no real fades the split of the chosen region is still a fair coin."""
    def even(t):   # the normal form from 200 stars up: within a star of the exact value, and far quicker
        return single_limit(t // 2, t - t // 2, cl, grid=501) if t < 200 else \
            float(stats.truncnorm.ppf(cl, 0.0, np.inf, scale=np.sqrt(t))) + 1.6
    return _best(even, total, detectable)


def sky_dispersion(pix, fading, n_toys=2000, seed=0):
    """Whether the stars past a cut go the same way in the same patch of sky more often than independent stars
    would. pix: each star's sky pixel; fading: True for a fading star, False for a brightening one.

    Returns (ratio, p): the sum over pixels of (fading - expected)^2 over its value for independent stars with
    the sample's overall fading share q (expected = q x stars in the pixel, variance q (1 - q) x stars), and the
    chance of a ratio at least that large for independent stars. Near 1 for independent stars. Well above 1,
    chance fades come in groups (one scan miscalibrated over a field, say): the counts then scatter more than
    Poisson counts, and the subtracted limit and the excess test are both too confident. (nan, nan) with fewer than two
    pixels or when every star goes the same way."""
    pix, fading = np.asarray(pix), np.asarray(fading, bool)
    _, idx = np.unique(pix, return_inverse=True)
    t = np.bincount(idx)
    q = fading.mean() if len(fading) else np.nan
    if len(t) < 2 or not 0 < q < 1:
        return np.nan, np.nan
    var = q * (1 - q) * t.sum()
    ratio = float(np.sum((np.bincount(idx, fading) - q * t) ** 2) / var)
    rng = np.random.default_rng(seed)
    toys = np.sum((rng.binomial(t, q, size=(n_toys, len(t))) - q * t) ** 2, axis=1) / var
    return ratio, float((1 + np.sum(toys >= ratio)) / (1 + n_toys))


def reviewed_limit(unexplained, detectable, review_recall=1.0, cl=0.95):
    """Limit once the stars of a region have been read: `unexplained` of them survived every check, `detectable`
    is the number of searched stars a fade would have put there (stars x recall), and `review_recall` the share
    of real fades the review keeps (docs/vetting.md §14 measured 44/50)."""
    return float(poisson_upper(unexplained, cl) / (detectable * review_recall))


# -- putting it together ------------------------------------------------------------------------------------------

SIDE = (2.0, 3.0)                       # side band of the symmetry table, in errors
Z_HIST = np.linspace(-6.0, 6.0, 241)    # significance histogram behind the symmetry table's medians


@dataclass
class Tally:
    """Counts of searched stars, which add up over batches of stars (one file of a whole-sky pass at a time):
    fading and brightening per (G range, significance bin), stars per G range, the side band below the first
    cut, and a histogram of significance per G range after and before the reference."""

    fading: np.ndarray
    brightening: np.ndarray
    stars: np.ndarray
    side_fading: np.ndarray
    side_brightening: np.ndarray
    hist: np.ndarray
    hist_raw: np.ndarray

    def __add__(self, other):
        return Tally(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


def tally(strat, z, z_raw, cfg=Config()):
    """The `Tally` of one batch of stars: G range (`stratum`, -1 to leave a star out), significance against the
    reference, and significance before it."""
    strat, z, z_raw = np.asarray(strat), np.asarray(z, float), np.asarray(z_raw, float)
    n, m, stars = cell_counts(strat, z, cfg)
    ok = (strat >= 0) & np.isfinite(z)
    s, zz = strat[ok], z[ok]

    def per(mask):
        return np.bincount(s[mask], minlength=cfg.n_strata)[: cfg.n_strata]

    def hist(v):
        h = np.zeros((cfg.n_strata, len(Z_HIST) + 1), int)
        fin = np.isfinite(v)
        np.add.at(h, (s[fin], np.digitize(v[fin], Z_HIST)), 1)
        return h

    return Tally(n, m, stars, per((-zz >= SIDE[0]) & (-zz < SIDE[1])), per((zz >= SIDE[0]) & (zz < SIDE[1])),
                 hist(zz), hist(z_raw[ok]))


def _hist_median(h):
    """Median of the values behind one row of a `Tally` histogram, to within a bin (0.05)."""
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.nan
    i = int(np.searchsorted(c, c[-1] / 2))
    if i == 0 or i >= len(Z_HIST):
        return float(Z_HIST[0] if i == 0 else Z_HIST[-1])
    return float(Z_HIST[i - 1] + (c[-1] / 2 - c[i - 1]) / h[i] * (Z_HIST[i] - Z_HIST[i - 1]))


def symmetry(t, cfg=Config()):
    """Per G range: stars, median significance before and after the reference, and fading against brightening
    counts in a side band below the first cut and beyond it. The side band is not part of the limit; it shows
    whether chance fades and chance brightenings are equally common where they are plentiful."""
    return pd.DataFrame([{"g_lo": cfg.g_edges[s], "g_hi": cfg.g_edges[s + 1], "stars": int(t.stars[s]),
                          "median_z_raw": _hist_median(t.hist_raw[s]), "median_z": _hist_median(t.hist[s]),
                          "side_fading": int(t.side_fading[s]), "side_brightening": int(t.side_brightening[s]),
                          "fading": int(t.fading[s].sum()), "brightening": int(t.brightening[s].sum())}
                         for s in range(cfg.n_strata)])


def star_significances(stars, cfg=Config(), ref=None):
    """(z, z before the reference, G range) of searched stars. `ref`: a `brightness_reference`, for stars that
    carry no neighbour reference of their own; built from these stars when not given."""
    g = stars["phot_g_mean_mag"].to_numpy(float)
    if ref is None and "slope_ref" not in stars:
        ref = brightness_reference(g, stars["slope"].to_numpy(float), cfg.reference_window)
    z = trend_z(stars, g, ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_raw = stars["slope_z_raw"].to_numpy(float) if "slope_z_raw" in stars else \
            stars["slope"].to_numpy(float) / slope_error(stars)
    return z, z_raw, np.where(np.isfinite(z), stratum(g, cfg), -1)


def injection_significances(hosts, injections, cfg=Config(), ref=None):
    """(z, G range) of injected stars, each measured against its host's reference. hosts: the searched stars'
    rows (at least those of the hosts) with source_id and phot_g_mean_mag, and `slope_ref` (with `slope_ref_err`
    if there is one) when the stars carry a neighbour reference; otherwise `ref`, the `brightness_reference` the
    stars were measured against. Injections that were themselves scored against a neighbour reference (their
    own `slope_ref`) keep their `slope_z`. A host not among `hosts` gets G range -1 and is left out."""
    host = hosts.drop_duplicates("source_id").set_index("source_id").reindex(injections["source_id"].to_numpy())
    host_g = host["phot_g_mean_mag"].to_numpy(float)
    if "slope_ref" in host and "slope_ref" not in injections:
        ref_err = host["slope_ref_err"].to_numpy(float) if "slope_ref_err" in host else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            inj_z = (injections["slope"].to_numpy(float) - host["slope_ref"].to_numpy(float)) \
                / np.hypot(slope_error(injections), ref_err)
    else:
        inj_z = trend_z(injections, host_g, ref)
    return inj_z, stratum(host_g, cfg)


def significances(stars, injections, cfg=Config()):
    """`star_significances` of the stars and `injection_significances` of the injections against them."""
    g = stars["phot_g_mean_mag"].to_numpy(float)
    ref = None if "slope_ref" in stars else brightness_reference(g, stars["slope"].to_numpy(float),
                                                                 cfg.reference_window)
    return star_significances(stars, cfg, ref), injection_significances(stars, injections, cfg, ref)


def plant(stars, injections, rate, share, rng):
    """`stars` with a random `share` of them replaced by their own injected light curve's scores at `rate`: a
    sample in which that share of stars really fades. For checking on real stars that the limit stays above a
    share put in by hand and that the excess test sees it. Returns (stars, number planted)."""
    inj = injections[np.isclose(injections["rate"].to_numpy(float), rate)].drop_duplicates("source_id")
    inj = inj[inj["source_id"].isin(stars["source_id"])]
    picked = inj.iloc[rng.random(len(inj)) < share].set_index("source_id")
    out = stars.set_index("source_id")
    cols = [c for c in picked.columns if c in out.columns]
    out.loc[picked.index, cols] = picked[cols].astype(out[cols].dtypes.to_dict())
    if "slope_ref" in out and "slope_ref" not in picked and len(picked):
        # The stars carry a neighbour reference: the planted trends are measured against it like the rest.
        p = out.loc[picked.index]
        ref_err = p["slope_ref_err"].to_numpy(float) if "slope_ref_err" in p else 0.0
        err = slope_error(picked)
        out.loc[picked.index, "slope_z"] = (p["slope"].to_numpy(float) - p["slope_ref"].to_numpy(float)) \
            / np.hypot(err, ref_err)
        if "slope_z_raw" in out:
            out.loc[picked.index, "slope_z_raw"] = p["slope"].to_numpy(float) / err
    return out.reset_index(), len(picked)


def curve_from_tally(t, past, inj_z, inj_strat, inj_rate, cfg=Config(), span=np.nan, seed=0, rates=None):
    """The limit at each rate from counts: `t`, the `Tally` of the searched stars; `past`, the stars past the
    first cut either way (columns pix, z, strat), for `sky_dispersion`; the injected stars' significances, G
    ranges and rates; `span`, the length of the data in years. Returns (curve, cells) as `limit_curve`."""
    n_r, m_r = regions(t.fading).astype(int), regions(t.brightening).astype(int)
    stars_r = np.cumsum(t.stars)
    inj_rate = np.asarray(inj_rate, float)
    pix, z, strat = (past[c].to_numpy() for c in ("pix", "z", "strat"))

    curve, cells = [], []
    for rate in rates or cfg.rates:
        sel = np.isclose(inj_rate, rate)
        share, net, hosts = efficiency(inj_strat[sel], inj_z[sel], cfg)
        det, det_net = regions(t.stars[:, None] * share), regions(t.stars[:, None] * net)
        row = {"rate": rate, "total_fade": rate * span, "hosts": int(hosts.sum())}
        where = choose_region(m_r, det, cfg.cl)
        if where is None:
            curve.append({**row, "limit": np.nan})
            continue
        k, j = where
        nn, mm, d = int(n_r[k, j]), int(m_r[k, j]), float(det[k, j])
        lim = float(poisson_upper(nn, cfg.cl)) / d
        row.update(g_max=cfg.g_edges[k + 1], z_min=cfg.z_cuts[j], stars=int(stars_r[k]), fading=nn, brightening=mm,
                   recall=d / stars_r[k], detectable=d, limit=lim, one_in=1 / lim,
                   expected=float(poisson_upper(mm, cfg.cl)) / d)
        mirror = choose_mirror_region(n_r + m_r, det_net, cfg.cl)
        if mirror is not None:
            km, jm = mirror
            nn, mm, d = int(n_r[km, jm]), int(m_r[km, jm]), float(det_net[km, jm])
            lo, med, hi = (x / d for x in expected_limits(nn, mm, cfg.cl))
            sel_past = (strat <= km) & (np.abs(z) >= cfg.z_cuts[jm])
            disp, disp_p = sky_dispersion(pix[sel_past], z[sel_past] < 0, seed=seed)
            row.update(mirror_g_max=cfg.g_edges[km + 1], mirror_z_min=cfg.z_cuts[jm], mirror_fading=nn,
                       mirror_brightening=mm, mirror_detectable=d, mirror_limit=single_limit(nn, mm, cfg.cl) / d,
                       excess_p=fade_excess_p(nn, mm), mirror_expected_lo=lo, mirror_expected=med,
                       mirror_expected_hi=hi, sky_dispersion=disp, sky_dispersion_p=disp_p)
        curve.append(row)
        for kk, jj in np.ndindex(det.shape):
            cells.append({"rate": rate, "g_max": cfg.g_edges[kk + 1], "z_min": cfg.z_cuts[jj],
                          "stars": int(stars_r[kk]), "hosts": int(hosts[: kk + 1].sum()),
                          "fading": int(n_r[kk, jj]), "brightening": int(m_r[kk, jj]),
                          "recall": float(det[kk, jj] / stars_r[kk]) if stars_r[kk] else np.nan,
                          "recall_net": float(det_net[kk, jj] / stars_r[kk]) if stars_r[kk] else np.nan,
                          "chosen": (kk, jj) == (k, j), "chosen_mirror": (kk, jj) == mirror})
    return pd.DataFrame(curve), pd.DataFrame(cells)


def past_cut(stars, z, strat, cfg=Config()):
    """The stars past the first cut either way, with their significance, G range and sky pixel."""
    keep = (strat >= 0) & (np.abs(z) >= cfg.z_cuts[0])
    out = stars[keep]
    return out.assign(z=z[keep], strat=strat[keep], pix=sky_pixel(out["source_id"].to_numpy(), cfg.sky_level))


def limit_curve(stars, injections, cfg=Config(), seed=0, rates=None):
    """The limit at each injected rate (cfg.rates, or `rates`), for stars held in memory.

    stars: the searched stars (already the population: `fgk_dwarf`, enough visits), with source_id,
    phot_g_mean_mag, slope and slope_z (or slope_err); `slope_ref` if the trends are already measured against
    neighbours. injections: one row per host and rate with source_id, rate and the same trend columns for the
    injected light curve; hosts are looked up in `stars` for their G, and those not among `stars` are left out.

    Returns (curve, cells, sym). curve, one row per rate: the limit and its region (g_max, z_min, stars, fading,
    brightening, recall, detectable, limit, one_in, and `expected`, the limit had as many stars faded as
    brighten), then the same with chance fades subtracted (mirror_*, excess_p, sky_dispersion). cells: per rate
    and region the counts and recall. sym: the symmetry table."""
    (z, z_raw, strat), (inj_z, inj_strat) = significances(stars, injections, cfg)
    t = tally(strat, z, z_raw, cfg)
    span = float(np.nanmedian(stars["t_last"] - stars["t_first"])) / 365.25 if "t_last" in stars else np.nan
    curve, cells = curve_from_tally(t, past_cut(stars[["source_id"]], z, strat, cfg), inj_z, inj_strat,
                                    injections["rate"], cfg, span, seed, rates)
    return curve, cells, symmetry(t, cfg)
