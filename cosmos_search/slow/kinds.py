"""Kind of star, and the reading order of the fading list within each kind.

Ordered by how far a star is past its threshold (`search`'s strength), the DR3 trial's fading list starts with
known variables: 12,114 stars past five times a threshold on the whole sky, eclipsing binaries first, and in the
Andromeda field not one of 33,765 FGK dwarfs among them. An ordinary dwarf that fades by a few percent sits at
strength 1 to 3, far down such a list. So the list is read by kind of star instead:

- `star_kind` places each star by parallax, absolute magnitude and colour (no extinction correction): dwarfs of
  three colour ranges, giants and subgiants, white dwarfs, stars whose parallax is too uncertain to place, stars
  with no measurable motion (`search.motion`'s still: quasars, galaxies, stars in other galaxies), and stars
  with a BP/RP flux excess (`summary.flux_excess`: light in the BP and RP windows that the star's G flux does not
  account for, mostly a neighbour within about 2 arcsec) above EXCESS_MAX and above EXCESS_SIGMAS times the
  scatter of the flux ratio among stars of their magnitude and colour.
- `rank` measures each fade against the brightenings of the same kind. Every star has both a fading and a
  brightening strength, and for a star that is not changing either is as likely to be the larger, so the stars of a
  kind whose brightening wins estimate how many of its fades are chance or ordinary variability
  (`stats.mirror.stratified_knockoff_q`). `q_kind` is that estimated share among the kind's fades at least as
  strong as this one. On DR3 dwarfs brighten as often as they fade, so most have q_kind near 1 and the order
  within a kind falls back on brightness: a bright star with a good parallax is the cheapest to check.

Stars with a flux excess are a kind of their own because the comparison above fails for them, in three ways
measured on DR3's Andromeda field (2026-11-11; the kind holds 7.6% of the stars with G < 19):

- their G flux falls by 0.1 to 0.7% a year, BP's with it and RP's not, in stars whose G is otherwise steady: 320
  of this kind passed the fading trend threshold and 98 the brightening one, where stars within 0.006 of no
  excess split 713 to 753. The reference's flux-excess part (`summary.reference_table`) centres their median (to
  134 against 156) and leaves a longer fading side between 2 and 5 errors (1,392 against 941);
- a neighbour inside G's window at some scan directions and outside it at others gives jumps both ways: 5 to 9% of
  the stars with an excess above 0.1 are on a list at G 13 to 17, against 1% of other stars, and above twice a
  threshold they brighten more often than they fade (251 against 587);
- their G scatter is several times a single star's, and injected fades are found far less often: 26% of 1% a year
  fades against 94% for hosts without an excess.

Left among the dwarfs they took the first places of the dwarfs' reading order, and with them there FGK dwarfs
above twice a threshold brightened more often than they faded (131 against 202; without them 94 against 67). So
within this kind q_kind is not a share of chance fades. They stay on the lists and are read after every other kind
of moving star. A star with an excess below zero (BP and RP light missing for its G) brightens instead; the
reference centres those and they fill the lists no faster than other stars, so they keep their kind.

Reading order: kinds in READ_ORDER (ordinary dwarfs with good parallaxes first), stars with a variability class
after those without, then q_kind, then brightest first.
"""

import numpy as np
import pandas as pd

from cosmos_search.stats import mirror

PARALLAX_SNR = 5.0
# Dwarfs lie below this line in the colour-magnitude diagram, giants and subgiants above it.
LINE = (3.2, 0.5)            # M_G = 3.2 (BP-RP) + 0.5
SUBLUMINOUS = 5.0            # more than this many magnitudes below the line: white dwarfs, subdwarfs
FGK = (0.5, 1.8)             # BP-RP range of FGK dwarfs; bluer are "hot dwarf", redder "M dwarf"

EXCESS_MAX = 0.05            # a flux excess above this makes a star of the kind "flux excess", where it is also
EXCESS_SIGMAS = 2.0          # above this many times the flux ratio's scatter at the star's magnitude and colour

READ_ORDER = ("FGK dwarf", "M dwarf", "hot dwarf", "giant or subgiant", "white dwarf or subluminous",
              "parallax too uncertain", "no colour", "no astrometry", "flux excess", "no measurable motion")


def absolute_magnitude(g, parallax):
    """M_G from G and the parallax in mas, without extinction; NaN for parallax <= 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(g, float) + 5 * np.log10(np.asarray(parallax, float)) - 10


def star_kind(df, still=None, excess=None, scatter=None):
    """Kind of each star (a name in READ_ORDER) from parallax, parallax_error, bp_rp and G (phot_g_mean_mag, else
    g_mag). excess, scatter: each star's flux excess and the scatter of the flux ratio at its magnitude and
    colour (`summary.flux_excess`; NaN where it has none): stars above EXCESS_MAX and EXCESS_SIGMAS x scatter
    are "flux excess" whatever their parallax and colour. still: boolean array of stars with no measurable
    motion (`search.motion`), which take that kind whatever else they are."""
    g = df["phot_g_mean_mag"] if "phot_g_mean_mag" in df else df["g_mag"]
    plx, err = df["parallax"].to_numpy(float), df["parallax_error"].to_numpy(float)
    c = df["bp_rp"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        good = plx / err > PARALLAX_SNR
        below = absolute_magnitude(g, plx) - (LINE[0] * c + LINE[1])
    kind = np.full(len(df), "parallax too uncertain", dtype=object)
    kind[~np.isfinite(plx)] = "no astrometry"
    kind[good & ~np.isfinite(c)] = "no colour"
    kind[good & (below < 0)] = "giant or subgiant"
    dwarf = good & (below >= 0) & (below <= SUBLUMINOUS)
    kind[dwarf & (c <= FGK[0])] = "hot dwarf"
    kind[dwarf & (c > FGK[0]) & (c < FGK[1])] = "FGK dwarf"
    kind[dwarf & (c >= FGK[1])] = "M dwarf"
    kind[good & (below > SUBLUMINOUS)] = "white dwarf or subluminous"
    if excess is not None:
        floor = EXCESS_MAX if scatter is None else np.fmax(EXCESS_MAX, EXCESS_SIGMAS * np.asarray(scatter, float))
        with np.errstate(invalid="ignore"):
            kind[np.asarray(excess, float) > floor] = "flux excess"
    if still is not None:
        kind[np.asarray(still, bool)] = "no measurable motion"
    return pd.Series(kind, index=df.index, name="kind")


def rank(stars, kind="kind", fade="strength", rise="rise_strength", g="phot_g_mean_mag", classified=None):
    """Reading order within kind of star. stars: every listed star, fading or brightening, with its kind and both
    strengths. Returns a frame indexed like `stars`:

    - q_kind: the estimated share of chance and ordinary variability among the kind's fades at least as strong as
      this one (1 for stars whose brightening is the stronger);
    - n_fade_kind, n_rise_kind: stars of the kind whose fade (brightening) wins at this strength or above, the two
      counts behind q_kind;
    - read_order: 0 for the first star to read (READ_ORDER, unclassified before classified, q_kind, brightest).

    classified: boolean array, stars with a variability class (read after the others of their kind)."""
    f = np.nan_to_num(stars[fade].to_numpy(float), nan=-np.inf, neginf=-np.inf)
    r = np.nan_to_num(stars[rise].to_numpy(float), nan=-np.inf, neginf=-np.inf)
    k = stars[kind].to_numpy()
    q = mirror.stratified_knockoff_q(f, r, k)
    out = pd.DataFrame({"q_kind": q}, index=stars.index)
    w = mirror.competition(f, r)
    n_f, n_r = np.zeros(len(stars), np.int64), np.zeros(len(stars), np.int64)
    for name in pd.unique(k):
        m = k == name
        pos, neg = np.sort(w[m & (w > 0)]), np.sort(-w[m & (w < 0)])
        mine = np.abs(w[m])
        n_f[m] = len(pos) - np.searchsorted(pos, mine, "left")
        n_r[m] = len(neg) - np.searchsorted(neg, mine, "left")
    out["n_fade_kind"], out["n_rise_kind"] = n_f, n_r
    order = {name: i for i, name in enumerate(READ_ORDER)}
    key = pd.DataFrame({"kind": [order.get(x, len(order)) for x in k],
                        "classified": np.zeros(len(stars), bool) if classified is None else np.asarray(classified, bool),
                        "q": q, "g": stars[g].to_numpy(float) if g in stars else 0.0}, index=stars.index)
    pos = key.sort_values(["kind", "classified", "q", "g"], kind="stable").index
    out["read_order"] = pd.Series(np.arange(len(stars)), index=pos).reindex(stars.index).to_numpy()
    return out


def counts(stars, kind="kind", fade="strength", rise="rise_strength", steps=(1.0, 1.5, 2.0, 3.0, 5.0, 10.0)):
    """Per kind: stars fading and brightening above each multiple of the thresholds. Where the fading count stands
    clear of the brightening one is where a kind's list is worth reading star by star."""
    out = {}
    for name, d in stars.groupby(kind):
        f, r = d[fade].to_numpy(float), d[rise].to_numpy(float)
        out[name] = {str(x): {"fading": int(np.sum(f > x)), "rising": int(np.sum(r > x))} for x in steps}
    return out
