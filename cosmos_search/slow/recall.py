"""Recall for steady fades: inject them into real light curves and score the result like any star.

    out = injected_scores(lc, rates, scorer)   # one row per star of `lc` and rate, the scorer's columns plus `rate`

`lc` is a light-curve table (gaia.epoch.SCHEMA) holding whole stars, the hosts. Each host gets the fade in all
three bands (grey), keeps its source_id and its own noise, gaps and variability, and is scored by `scorer`, the
function the search itself runs on a light-curve table (today `search.score` with the rule's drift table). A
steady fade is deterministic, so a host gives one draw per rate; the randomness of the recall is in which stars
are hosts.

The fade: flux x (1 - rate x years since the star's first epoch), `inject.lightcurve.SecularTrend` with a
negative rate and the star's first epoch as reference. `rate` is the fraction of the star's flux at its first
visit lost per year, so a rate of 0.01 over DR4's 5.5 years is a star 5.5% fainter at the end. Errors scale with
the flux (as `inject.lightcurve.inject`).

Nothing here is blinded and nothing needs to be: no person or agent judges the injected stars. They pass through
the same arithmetic as the real ones, and only counts come out.
"""

import numpy as np
import pandas as pd
import pyarrow as pa

DAYS_PER_YEAR = 365.25


def fade_factor(t, t0, rate):
    """Flux factor of a steady fade of `rate` per year that starts from the star's level at t0 (days)."""
    return 1.0 - rate * (np.asarray(t, float) - np.asarray(t0, float)) / DAYS_PER_YEAR


def first_epoch(source_id, t):
    """Each row's star's first epoch (any band)."""
    return pd.Series(np.asarray(t, float)).groupby(np.asarray(source_id)).transform("min").to_numpy()


def inject_steady_fade(lc, rate):
    """The light-curve table with a grey steady fade of `rate` per year in every star, from its first epoch."""
    k = fade_factor(lc["t"], first_epoch(lc["source_id"], lc["t"]), rate)
    if np.any(k <= 0):
        raise ValueError(f"a fade of {rate} per year reaches zero flux inside the data")
    out = lc
    for name in ("flux", "err"):
        col = pa.array(np.asarray(lc[name], float) * k, lc.schema.field(name).type)
        out = out.set_column(out.schema.get_field_index(name), name, col)
    return out


def injected_scores(lc, rates, scorer, columns=None):
    """Scores of every star of `lc` with a steady fade injected, at each rate: the scorer's rows with a `rate`
    column. Every host has a row at every rate; one the scorer returned nothing for (too few good epochs once
    injected) has NaN scores and counts as missed. columns: the scorer's columns to keep (default all)."""
    hosts = pd.unique(np.asarray(lc["source_id"]))
    out = []
    for rate in rates:
        s = scorer(inject_steady_fade(lc, rate))
        if columns is not None and len(s):
            s = s[["source_id", *[c for c in columns if c in s and c != "source_id"]]]
        s = pd.DataFrame({"source_id": hosts}).merge(s, on="source_id", how="left") if len(s) else \
            pd.DataFrame({"source_id": hosts})
        out.append(s.assign(rate=float(rate)))
    return pd.concat(out, ignore_index=True)
