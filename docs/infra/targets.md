# Target queries and freeze manifests (`cosmos_search.targets`)

Milestone I3, plus the DR4 adapters of G7 in `cosmos_search.gaia.dr4`.

DR4 epoch photometry is 7.2 TB and DataLink-only, so day one starts with a query that runs where the data
are: a prefilter on `vari_summary` over TAP, which picks the ~10⁶ sources whose light curves we then pull.
Everything here is written so the DR3 rehearsal and DR4 day one run the same code.

## The prefilter

For each source, from `vari_summary` columns that exist in both DR3 and the DR4 draft data model:

```
d_faint  = (max_mag_g_fov − median_mag_g_fov) / mad_mag_g_fov     deepest point, robust σ
d_bright = (median_mag_g_fov − min_mag_g_fov) / mad_mag_g_fov     its mirror
```

`mad_mag_g_fov` is already scaled by 1.4826 in both releases (DR4 draft data model eq. 54; for DR3 checked
against our own MAD on 5,345 sources, median ratio 1.000). On the same sources `d_faint` correlates at 0.988
with the full-light-curve `−min_z_rob`, and `num_selected_g_fov` equals our count of non-rejected G points
exactly.

**Rule v1** (`targets.prefilter.Rule`):

1. Histogram both tails per 0.5 mag bin of `median_mag_g_fov` (two aggregate queries).
2. In each bin, set the threshold T_b where the **bright** tail keeps a fraction α of the bin. Bins with
   fewer than 2,000 sources use the pooled threshold.
3. Select sources whose faint *or* bright d ≥ T_b, plus `MOD(source_id, 4093) = 0` controls (~0.024%).

Why this shape rather than a fixed nσ cut on skewness or depth:

- **Mirror null.** Instrumental outliers are roughly symmetric; occultation is one-sided. The bright tail is
  measured in the same magnitude bin, on the same day, with the same pipeline, so it calibrates the
  threshold without a noise model and without trusting the published errors.
- **Equal false-alarm rate across magnitude.** Gaia's noise tails change with magnitude (gating and
  saturation at G ≲ 13, background at G ≳ 19). A single threshold would spend most of the budget wherever
  tails happen to be heaviest.
- **Target–decoy.** The bright-side selections come along as decoys, as in proteomics database searches.
  Downstream stages see both, so their false-discovery rate is measured rather than assumed. (Astrophysical
  brightenings such as flares mean the bright side is a conservative null, not a perfect one.)
- **Controls independent of the prefilter**, so the prefilter's recall can be measured on DR4 itself by
  injecting into the controls' light curves.
- **Budget sets α.** About 2·α·N sources pass. The default α = 1.2 × 10⁻³ gives ~10⁶ on DR4's
  4.0 × 10⁸ `vari_summary` rows. That suits a bulk stream. DataLink now serves ≤ 50 sources per call at
  ~1.5 sources/s (docs/infra/gaia-epoch.md), so a day-one DataLink pull needs α ≈ 10⁻⁵ (~10⁴ sources).

The Gaia archive's ADQL rejects `CASE`, `LEAST`, `UNION` and expressions in `GROUP BY`, so bins come from
`FLOOR` inside a subquery and per-bin thresholds from an `OR` chain.

## DR3 rehearsal (2026-10-21)

The full run (two histograms plus the selection) took 28 s on all 11,754,237 DR3 `vari_summary` rows.
Thresholds came out at 13–37 robust σ, because the DR3 table is DPAC's variables only and eclipsing
binaries dominate the faint tail. It selected 132,604 sources: 116,214 over the faint threshold, 13,797 over
the bright one (≈ α N, as designed), and 2,865 controls. The faint/bright ratio of 8.4 is the astrophysical
excess, mostly known variable classes.

**Recall against full light curves.** `python -m cosmos_search.targets.validate` compares the selection with
features from the DR3 bulk pass. Preliminary numbers, from the first 400 of 3,386 bulk files (3.0 × 10⁶
sources, weighted toward the Galactic bulge, which the CDN lists first):

| Class, from full light curves | Sources | Prefilter recall | Recall at G < 13 |
|---|---|---|---|
| Deep dip (≤ −8σ) confirmed in the other FoV (partner ≤ −4σ) | 24,383 | 46% | 72% |
| Deep dip, partner transit disagrees (> −1σ) | 24,505 | 11% | 38% |
| Deep dip, no partner transit | 33,389 | 25% | 56% |
| Only DPAC-rejected faint points (≥ 3σ), otherwise quiet | 236,396 | 0.3% | 0.2% |
| Mirror: bright spike confirmed in the other FoV | 16,198 | 4% | 13% |
| Random controls | 30,045 | 1.1% | 3.2% |

Two things follow:

1. On DR3 the prefilter is far stricter than 8σ, because the per-bin thresholds are set by a variables-only
   population. On DR4, where most of the 4 × 10⁸ sources are not variable, the same α gives lower
   thresholds. The rule is frozen; the numbers it produces are not.
2. **The prefilter cannot see dips that DPAC's variability processing rejected.** `vari_summary` statistics
   use only non-rejected transits. Of the DR3 sources processed so far, 7.9% have at least one rejected point
   ≥ 3σ below their median while the rest of the light curve is quiet. Many will be genuine artefacts, but a
   rejection rule tuned on ordinary variables is exactly the kind of filter that would discard a rare,
   deep, one-off dip. DR4's `vari_epoch_flags` (DataLink-only) carries the same information. The defence is
   the random controls plus a budgeted random pull of full light curves, so that the rejected-dip channel
   is measured and searched directly rather than inherited from the prefilter.

Final numbers replace these when the full pass finishes.

## Freezing

`python -m cosmos_search.targets freeze --tag <tag> --out configs/targets/<tag>.json` records the rule, the
SHA-256 of every query template, and the git commit. `verify` re-hashes the current code and lists anything
that changed, so a pre-registered DR4 run can prove it ran the frozen queries. Each run writes its own
`manifest.json` with the rendered ADQL, its hash, the thresholds and row counts; the DR3 rehearsal manifest
is in `configs/targets/runs/`.

`python -m cosmos_search.targets check --release dr4` lists any required `vari_summary` column missing from
the archive; run it the moment DR4 is live. If the schema is not `gaiadr4`, change `targets/releases.py`.

## DR4 adapters (`cosmos_search.gaia.dr4`)

From the draft data model (2026-06-26):

- `epoch_photometry` is one row per FoV transit with scalar `g_flux`, `bp_flux`, … and flag words
  (`transit_acq_flags`, `transit_proc_flags`, `phot_ccd_proc_flags[12]`). `epoch_photometry_to_lc` maps it
  to the shared light-curve schema, so DR3 features run unchanged; the flag bits it adds (saturation,
  satellite event, cosmetic, multiple-source IPD, attitude) are listed in the module docstring.
- `epoch_photometry_ccd` has the nine AF fluxes per transit. `ccd_consistency` gives the reduced χ² of the
  nine CCDs about their mean for each transit: a dip lasting hours is common to all nine CCDs (~40 s
  apart), while a cosmic ray or CCD defect is not. This is the per-CCD invariance test that DR3 cannot do.
- `vari_summary` gains flux-space moments, per-CCD G statistics (`*_gccd`) and G/BP/RP correlations
  (`pearson_mag_g_bp`, …). The v1 prefilter uses none of them so that DR3 can rehearse it; a v2 that adds
  per-CCD agreement is a candidate once DR4 is out.
