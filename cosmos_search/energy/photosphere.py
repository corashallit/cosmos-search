"""The star's expected infrared colours, learnt from stars like it.

For each colour (K-W1, W1-W2, W1-W3, W3-W4) the prediction is the median colour of the K_NEIGHBOURS reference
stars nearest in (BP-RP, absolute G, and for K-W1 and W1-W2 apparent W1), and its error the neighbours' robust
scatter less their own measurement error. No spectral model enters, so giants, dwarfs and the saturation regime of
each band are handled by whatever the reference stars do there. Targets without a usable parallax get the same
from a model without absolute G, which mixes dwarfs and giants and is correspondingly wider.

Reference stars are clean by construction: both bands measured at SNR >= photometry.SNR_MEASURE, exactly the
condition under which a target's band counts as a measurement (so faint-end selection effects match), no
contamination flag, not extended, no other
Gaia star sharing the WISE or 2MASS source, not flagged variable by WISE, RUWE < 1.4. Each colour's photosphere has
to be detected whatever its excess, so the reference is cut on the bluer band's magnitude, never the redder one
(cutting on W3 or W4 detections would select stars with an excess):

- K-W1 and W1-W2 from the general sample with W1 fainter than W1_SAT (W1 saturates brighter than ~8.1). W1-W2
  drifts with W1 by ~0.01-0.02 mag over the range (measured on held-out stars), so apparent W1 is a feature;
- W1-W3 from the mid sample (G 9.5-12) with W1 in [W1_SAT, W13_W1_MAX], where W1 is unsaturated and a photosphere
  is >~ 20 sigma in W3. (The bright sample's W1-W3 drifts by 0.04 mag across W1 6-8 from saturation, and K-W3
  carries 0.06 A_V of reddening that the simultaneous W1-W3 does not.)
- W3-W4 from the bright sample with W3 in [W3_SAT, W34_W3_MAX], where W3 is unsaturated and a photosphere is
  >~ 15 sigma in W4.

A few per cent of reference stars have real excesses. The median and the MAD ignore them.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K_NEIGHBOURS = 64
W1_SAT = 8.2
W13_W1_MAX = 9.8
W3_SAT = 3.8
W34_W3_MAX = 5.5
SCALE = {"bp_rp": 0.05, "m_g": 0.3, "w1mpro": 1.0}
COLOURS = {
    # name: (bluer column, redder column, sample, features, bluer-band magnitude window)
    "K1": ("ks_m", "w1mpro", "general", ("bp_rp", "m_g", "w1mpro"), ("w1mpro", W1_SAT, 16.0)),
    "12": ("w1mpro", "w2mpro", "general", ("bp_rp", "m_g", "w1mpro"), ("w1mpro", W1_SAT, 16.0)),
    "13": ("w1mpro", "w3mpro", "mid", ("bp_rp", "m_g"), ("w1mpro", W1_SAT, W13_W1_MAX)),
    "34": ("w3mpro", "w4mpro", "bright", ("bp_rp", "m_g"), ("w3mpro", W3_SAT, W34_W3_MAX)),
}
ERR = {"ks_m": "ks_msigcom", "w1mpro": "w1mpro_error", "w2mpro": "w2mpro_error", "w3mpro": "w3mpro_error",
       "w4mpro": "w4mpro_error"}
W_INDEX = {"w1mpro": 0, "w2mpro": 1, "w3mpro": 2, "w4mpro": 3}
PLX_SNR_MIN = 5.0
MAX_ERR = 2.5 / np.log(10) / 5.0   # SNR >= 5, photometry.SNR_MEASURE
SIGMA_FLOOR = 0.01


def absolute_g(df):
    plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
    err = pd.to_numeric(df["parallax_error"], errors="coerce").to_numpy(float)
    g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mg = g + 5 * np.log10(plx) - 10
    return np.where((plx > 0) & (plx / err >= PLX_SNR_MIN), mg, np.nan)


def clean(df, cols):
    """Reference-quality rows for a colour made of `cols` (canonical names)."""
    ok = np.ones(len(df), bool)
    ok &= pd.to_numeric(df.get("ruwe", 1.0), errors="coerce").fillna(99).to_numpy() < 1.4
    wq = df["w_ph_qual"].astype(str)
    cc = df["w_cc_flags"].astype(str)
    var = df.get("w_var_flag", pd.Series("nnnn", index=df.index)).astype(str)
    ok &= pd.to_numeric(df["w_ext_flag"], errors="coerce").fillna(1).to_numpy() == 0
    ok &= pd.to_numeric(df.get("w_mates", 0), errors="coerce").fillna(1).to_numpy() == 0
    for c in cols:
        ok &= np.isfinite(pd.to_numeric(df[c], errors="coerce").to_numpy(float))
        e = pd.to_numeric(df[ERR[c]], errors="coerce").to_numpy(float)
        ok &= np.isfinite(e) & (e <= MAX_ERR)
        if c in W_INDEX:
            i = W_INDEX[c]
            ok &= wq.str[i].isin(list("ABC")).to_numpy() & (cc.str[i] == "0").to_numpy()
            ok &= var.str[i].isin(list("n012345")).to_numpy()
        else:
            ok &= df["t_ph_qual"].astype(str).str[2].isin(list("ABC")).to_numpy()
            ok &= pd.to_numeric(df.get("t_mates", 0), errors="coerce").fillna(1).to_numpy() == 0
    return ok


class Locus:
    def __init__(self, k=K_NEIGHBOURS):
        self.k = k
        self.models = {}

    def fit(self, samples):
        """samples: {"general", "mid", "bright": DataFrame} of canonical reference photometry."""
        for name, (blue, red, sample, feats, (wcol, lo, hi)) in COLOURS.items():
            df = samples[sample]
            mg = absolute_g(df)
            ok = clean(df, (blue, red))
            w = pd.to_numeric(df[wcol], errors="coerce").to_numpy(float)
            ok &= (w >= lo) & (w <= hi) & np.isfinite(df["bp_rp"].to_numpy(float))
            col = df[blue].to_numpy(float) - df[red].to_numpy(float)
            err = np.hypot(df[ERR[blue]].to_numpy(float), df[ERR[red]].to_numpy(float))
            sub = {}
            for kind, fs in (("full", feats), ("noplx", tuple(f for f in feats if f != "m_g"))):
                X = _features(df, fs, mg)
                good = ok & np.all(np.isfinite(X), axis=1)
                sub[kind] = (fs, cKDTree(X[good]), col[good], err[good])
            self.models[name] = sub
        return self

    def predict(self, df):
        """DataFrame with c_<colour>, s_<colour> (mag) and d_<colour> (median neighbour distance in feature
        units) for each row of df."""
        df = df.reset_index(drop=True)
        mg = absolute_g(df)
        out = {}
        for name, sub in self.models.items():
            c = np.full(len(df), np.nan)
            s = np.full(len(df), np.nan)
            dist = np.full(len(df), np.nan)
            for kind in ("full", "noplx"):
                fs, tree, col, err = sub[kind]
                X = _features(df, fs, mg)
                todo = np.all(np.isfinite(X), axis=1) & ~np.isfinite(c)
                if not todo.any():
                    continue
                dd, ii = tree.query(X[todo], k=min(self.k, len(col)))
                cc, ee = col[ii], err[ii]
                med = np.median(cc, axis=1)
                mad = 1.4826 * np.median(np.abs(cc - med[:, None]), axis=1)
                intr = np.sqrt(np.clip(mad ** 2 - np.median(ee, axis=1) ** 2, 0, None))
                d_med = np.median(dd, axis=1)
                # Far from every reference star: widen in proportion (a crude guard against extrapolation).
                widen = np.maximum(1.0, d_med / 2.0)
                c[todo] = med
                s[todo] = np.hypot(np.maximum(intr, SIGMA_FLOOR) * widen, mad / np.sqrt(len(cc[0])))
                dist[todo] = d_med
            out[f"c_{name}"], out[f"s_{name}"], out[f"d_{name}"] = c, s, dist
        return pd.DataFrame(out)


def _features(df, fs, mg):
    cols = []
    for f in fs:
        v = mg if f == "m_g" else pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        cols.append(v / SCALE[f])
    return np.column_stack(cols)
