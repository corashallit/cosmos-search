"""Injected infrared excesses: known answers for the energy-budget test, put into real stars' photometry.

Each family is a recipe for (f, T, beta) components, drawn per host. The emission is added in flux to every band
(K, W1-W4) on top of the host's own photometry, using the host's predicted photosphere, so the real noise, upper
limits, flags and template errors all stay as they are. A band that was an upper limit becomes a detection when the
added flux would be measured at >= 3 sigma, with sigma = limit / 2 (the same reading of a WISE limit as
photometry.design uses); otherwise its limit grows to cover the added flux.

Natural analogues (what a dust dimmer looks like):
- yso_disk: a hot inner rim (900-1500 K) plus warm disk (150-400 K), each 2-20% of L* (class II young stars);
- rcb_shell: one shell at 600-1000 K carrying 10-60% of L* (R CrB stars);
- debris: one cold-to-warm belt at 50-300 K with 1e-5 to 1e-3 of L* (most are invisible to WISE, as in reality);
Structures:
- warm_structure: a blackbody at 200-400 K with 1e-3 to 0.5 of L* (waste heat at habitable-zone temperatures, the
  G-HAT case);
Nothing to see:
- cold: no excess at all, or a 20-50 K component that WISE cannot see; the "no waste heat in WISE" answer.

Dimmer pairs (`dimmer_plan`): a time-averaged dimming <delta> together with the IR that one physical arrangement
implies: shell (f = <delta>), thin (f = f_thin(T), the least absorbing orbit), or none (the occulter re-radiates
nothing WISE sees). The deficit test must never call the first two "missing waste heat" and should call the third
whenever the data can.
"""

import numpy as np
import pandas as pd

from .budget import BANDS, excess_per_f, f_thin

FAMILIES = {
    "yso_disk": [((900, 1500), (0.02, 0.2), 0.0), ((150, 400), (0.02, 0.2), 1.0)],
    "rcb_shell": [((600, 1000), (0.1, 0.6), 0.0)],
    "debris": [((50, 300), (1e-5, 1e-3), 1.0)],
    "warm_structure": [((200, 400), (1e-3, 0.5), 0.0)],
    "cold": [((20, 50), (1e-3, 0.3), 0.0)],
}
W_COLS = {"W1": "w1mpro", "W2": "w2mpro", "W3": "w3mpro", "W4": "w4mpro"}


def draw(family, n, rng):
    """List of components per host: [(T, f, beta), ...]."""
    out = []
    for _ in range(n):
        comps = []
        for (tlo, thi), (flo, fhi), beta in FAMILIES[family]:
            T = rng.uniform(tlo, thi)
            f = 10 ** rng.uniform(np.log10(flo), np.log10(fhi))
            comps.append((float(T), float(f), beta))
        out.append(comps)
    return out


def added_flux(design_rows, components):
    """(n, 5) flux to add per band (Vega-relative units), from the hosts' photosphere a0 * P_j."""
    teff = design_rows["teff"].to_numpy(float)
    phot = np.stack([design_rows["a0"].to_numpy(float) * design_rows[f"P_{b}"].to_numpy(float) for b in BANDS], -1)
    add = np.zeros_like(phot)
    for i, comps in enumerate(components):
        for T, f, beta in comps:
            if f > 0:
                add[i] += f * excess_per_f(teff[i:i + 1], [T], beta)[0, 0] * phot[i]
    return add


def apply(raw, design_rows, components):
    """Copy of canonical photometry `raw` with the components' emission added (see module docstring)."""
    raw = raw.reset_index(drop=True).copy()
    add = added_flux(design_rows.reset_index(drop=True), components)
    wq = raw["w_ph_qual"].astype(str).map(lambda s: list(s.ljust(4, "X")))
    for j, b in enumerate(BANDS):
        a = add[:, j]
        if b == "G":
            raw["phot_g_mean_mag"] = -2.5 * np.log10(10 ** (-0.4 * raw["phot_g_mean_mag"].to_numpy(float)) + a)
            continue
        if b == "K":
            m = raw["ks_m"].to_numpy(float)
            phi = 10 ** (-0.4 * m)
            new = phi + a
            raw["ks_msigcom"] = raw["ks_msigcom"].to_numpy(float) * phi / new
            raw["ks_m"] = -2.5 * np.log10(new)
            continue
        i = int(b[1]) - 1
        col, ecol = W_COLS[b], W_COLS[b] + "_error"
        m, e = raw[col].to_numpy(float), raw[ecol].to_numpy(float)
        q = np.array([s[i] for s in wq])
        phi = 10 ** (-0.4 * m)
        det = np.isin(q, list("ABC")) & np.isfinite(e)
        ul = q == "U"
        m_new, e_new, q_new = m.copy(), e.copy(), q.copy()
        # Detections keep their flux error (background-limited), so the magnitude error shrinks.
        nd = phi + a
        m_new[det] = -2.5 * np.log10(nd[det])
        e_new[det] = (e * phi / nd)[det]
        # Upper limits: noise sigma = limit / 2, measured flux ~ 0 before injection.
        sig = phi / 2
        snr = a / sig
        becomes = ul & (snr >= 3)
        m_new[becomes] = -2.5 * np.log10(a[becomes])
        e_new[becomes] = sig[becomes] / a[becomes] / (np.log(10) / 2.5)
        q_new[becomes] = np.where(snr[becomes] >= 10, "A", "B")
        stays = ul & ~becomes
        m_new[stays] = -2.5 * np.log10(np.maximum(phi, a + 2 * sig)[stays])
        raw[col], raw[ecol] = m_new, np.where(ul & ~becomes, np.nan, e_new)
        for k in range(len(wq)):
            wq.iloc[k][i] = q_new[k]
    raw["w_ph_qual"] = wq.map("".join)
    return raw


def dimmer_plan(design_rows, rng, n=None, delta_range=(1e-3, 0.3), t_range=(100.0, 1500.0)):
    """One dimmer per host: <delta> log-uniform in delta_range, an arrangement (shell, thin, none) and a re-emission
    temperature log-uniform in t_range. Returns a DataFrame (delta, arrangement, T, f) and the component lists."""
    n = n or len(design_rows)
    delta = 10 ** rng.uniform(*np.log10(delta_range), n)
    arr = rng.choice(["shell", "thin", "none"], n)
    T = 10 ** rng.uniform(*np.log10(t_range), n)
    teff = design_rows["teff"].to_numpy(float)[:n]
    f = np.where(arr == "shell", delta,
                 np.where(arr == "thin", np.array([f_thin(delta[i:i + 1], [T[i]], teff[i:i + 1])[0, 0]
                                                    for i in range(n)]), 0.0))
    plan = pd.DataFrame({"delta": delta, "arrangement": arr, "T": T, "f": f})
    comps = [[(float(T[i]), float(f[i]), 0.0)] if f[i] > 0 else [] for i in range(n)]
    return plan, comps
