# Vetting

Shared by the Gaia and TESS arms (milestones V1–V3). Code in `cosmos_search/vetting/`; decoys in
`cosmos_search/inject/classes/vetting_decoys.py`; expert agents in `.claude/agents/`.

## 1. The arithmetic that shapes everything

A vetter (deterministic test, LLM expert or person) is a classifier with a per-object rate α_c of calling an
explicable object of class c "unexplained". For a queue of N objects with class mix q, the shortlist carries about
N Σ_c q_c α_c false entries, and almost none of the queue is truly unexplained. With N = 10³ and a vetter as good as
α ≈ 2–3 %, that is ~25 false entries around at most one real one. Three consequences:

1. **No single stage can certify a shortlist.** To show a mix-weighted α ≤ 10⁻³ (one expected false entry per 10³)
   with no observed failures takes ~3,000 controls (`design.n_zero_failure(1e-3)`), three times the whole budget.
2. **Precision comes from stacking filters whose errors are independent**, and the strongest independence is a
   different instrument: a Gaia artefact does not recur in ZTF, ASAS-SN or TESS. Stage 0 tests, Stage A, Stage B and
   an independent-instrument check multiply only if their failures are uncorrelated, so we measure the joint failure
   rate on controls (`scoring.error_dependence`) instead of assuming the product.
3. **The number to report is the expected count of false entries on the shortlist, with an interval**, estimated
   from controls under label shift (§5). That figure stays honest even when the shortlist is mostly false, which it
   will be.

## 2. Pipeline

| Stage | Who | Sees | Output |
|---|---|---|---|
| 0. Automated FP tests | code | everything, unblinded | per-class p-values; hard vetoes only for exact nulls |
| A. Blind expert | no-tool LLM agent | one anonymised packet | class probabilities, held-out predictions, recognition probe |
| Commit | code | verdicts | SHA-256 and time recorded before any reveal |
| Reveal and score | code | batch key | decoys and controls scored, then dropped |
| B. Reconciler | LLM agent with tools | real ids, packet, Stage A verdict | literature, archival cross-checks, updated probabilities, verifiability |
| C. Independent instrument | code + Stage B | other surveys' photometry | dip present / absent / no coverage |

Input is one `Candidate` per object (`vetting/candidate.py`): light curves with per-epoch nuisance channels, dips,
detector scores, FP-test results and context. Both arms emit plain dicts in that shape.

## 3. False-positive classes

The class list (`vetting/taxonomy.py`) is what verdicts are distributions over. Each class needs a test, an owner
and a source of known-answer controls or decoys, or its α is unmeasured.

| Class | Signature | Deterministic test | Owner | Controls |
|---|---|---|---|---|
| blend_or_crowding | Flux tracks scan angle (Gaia) or pointing (TESS); BP/RP excess; centroid shifts | flux vs scan angle; `ipd_frac_multi_peak`, `phot_bp_rp_excess_factor`; TESS centroid z, aperture depth ratio | Gaia detectors (G2); TESS thread | close pairs from Gaia DR3 with known separations |
| scattered_light | Dip when background rises, near orbit ends | background z, straylight flags, segment-edge distance | TESS thread | TESS sectors with known scattered-light episodes |
| pointing_or_momentum | Brief excursions at dump times | `dump_dist` | TESS thread | stars at known dump epochs |
| detector_defect | Single epoch, one band or one CCD; G ≲ 13 gating | per-CCD agreement (DR4); magnitude stratification | Gaia detectors | random stars with flagged CCD transits |
| calibration_offset | Steps between FoVs, CCDs, sectors, calibration epochs | FoV pair test; step test at known calibration boundaries | Gaia detectors | step decoys |
| shared_systematic | Same epoch dims in unrelated stars | **cross-candidate coincidence** (`vetting/coincidence.py`, below); TESS common-mode per CCD | vetting; TESS thread | coincident-dip injections across hosts |
| processing | Detrending or background-model artefacts, cross-match errors | re-extraction with an independent pipeline (TGLC for TESS; DataLink re-pull for Gaia) | arm threads | — |
| solar_system_occultation | Single-epoch dip near the ecliptic; asteroid in TESS background annulus | **SkyBoT cone search at dip epochs** | vetting (pending the shared client) | known asteroid crossings |
| eclipsing_binary, planet_transit | Periodic, flat or V-shaped, grey | period search, odd/even, secondary | arm detectors | VSX/Gaia DR3 EBs |
| disk_occultation, yso_dipper, rcrb_dust, debris_transit | Long, reddened or irregular dips; IR excess | dip colour ratio R; energy budget vs WISE (G5) | Gaia detectors, cross-match thread | KIC 8462852, Gaia17bpp, EPIC 204278916, YSO dippers, R CrB stars, ZTF J0139+5245 |
| rotation_or_spots, pulsation_or_irregular, eruptive_or_be | Quasi-periodic or red-noise variability | periodograms, colour–magnitude slope | arm detectors | VSX/DR3 classified variables |
| extragalactic | Non-stellar SED, zero parallax and proper motion | astrometric and SED cuts | cross-match thread | QSOs in DR3 |

**Cross-candidate coincidence.** A real occulter belongs to one star; an instrument event touches every star
observed at that moment. For each dip with n dips of other candidates within ±W, k of them within ±τ, the null is
k ~ Binomial(n, τ/W): the local rate is taken as flat over W, which absorbs coverage gaps and seasonal changes in
how many stars are observed. Groups (camera/CCD, Gaia scan) restrict the comparison to dips that could share a
cause. This runs over the whole queue plus the random-control stream, which only the vetting stage sees at once.

Stage 0 vetoes outright only on exact nulls (per-CCD disagreement, FoV disagreement, coincidence with p below a
queue-wide Bonferroni threshold). Everything else goes into the packet as evidence, so Stage A's calibration
includes how it weighs the tests.

## 4. Stage A: blind expert

**Packets** (`vetting/packet.py`). Identifiers and positions removed; times re-zeroed to the first epoch;
magnitudes to 0.1 mag; distance to two significant figures; sky position reduced to |b| in 5° bins and ecliptic
latitude in 10° bins (extinction, crowding, scanning, solar-system bodies). Light curves up to 400 epochs per band
are shown in full with their nuisance channels; longer ones are binned, with zooms on each dip. Every packet is
scanned for identifier-shaped text (catalogue ids, alert names, J-names, JDs, sexagesimal coordinates, bibcodes,
famous names, long integers) and refused if anything matches.

**Agents.** Stage A runs with no tools and one packet inlined, so it cannot read files, run code or reach the
network. The preferred backend is the Messages API with no `tools` parameter (`python -m cosmos_search.vetting api`),
which is tool-free by construction and records token usage; it needs `ANTHROPIC_API_KEY` as an environment secret.
The fallback is the subagent `.claude/agents/expert-blind.md`, which deny-lists every built-in tool: Claude Code's
documentation does not define a zero-tool agent, so that list is a best effort to be checked once per session. Both
use the same text (`vetting/stage_a.md`; a test keeps them in sync).

**Why stateless agents change the decoy design.** For people, decoys must be mixed in at a hidden rate, because a
person who knows they are in a calibration session behaves differently. A Stage A agent sees one packet, keeps no
memory between packets and is never told which run it is in. Its behaviour on a packet depends only on that packet.
So what transfers calibration from controls to real candidates is **per-packet indistinguishability**, and the
mixing fraction does not matter. That lets most calibration happen in a separate, larger run (V3), with only a small
in-stream fraction on DR4 day to detect drift. The rule it puts on batch building (`vetting/batch.py`): every packet,
real or not, goes through the same detector, field selection and holdout. A decoy that lacks a score the real
packets carry is a labelled decoy.

**Indistinguishability audit.** Before a calibration run is trusted, fit a classifier on packet-level features
(summaries, field presence, cadence) to separate controls and decoys from real candidates, and test its AUC against
0.5 by permutation (`vetting/distinguish.py`). If it beats chance, fix the batch, not the statistics.

**Memorisation.** A model that has read about KIC 8462852 may recognise it from magnitude, colour, distance and a
famous light curve, and then score well by recall. Every verdict carries `recognised_as`; we report the recognition
rate on controls and score controls separately for recognised and unrecognised packets. Known-answer controls that
are recognised measure recall, not reasoning, and are excluded from the calibration fit.

## 5. Scoring and the shortlist false-positive figure

Verdicts are distributions over the leaf classes, summed into three groups (instrumental, astrophysical,
unexplained). Per batch, `vetting/report.py` gives:

| Quantity | From | Meaning |
|---|---|---|
| Brier and log score, reliability, ECE of P(unexplained) | controls + decoys | calibration |
| Confusion matrix over groups | controls + decoys | α_c for §1 |
| Confabulation rate: P(astrophysical) > 0.5 on incoherent decoys | decoys | inventing natural stories for signals no natural process makes |
| Probe sensitivity: "unexplained" wins on anomaly probes | decoys | would a real anomaly survive? |
| False "unexplained" on known-answer controls | controls | the shortlist's main contaminant |
| Recognition rate | controls | memorisation |
| Held-out prediction skill (§6) | all packets, including real ones | confabulation on real candidates |
| Format failures | all | reliability cost |

**Shortlist false entries** (`vetting/quantify.py`). Controls give C[c, d] = P(decision d | class c). If the vetter's
behaviour given the class is the same on controls and real candidates (the label-shift assumption, which the audit
in §4 tests), the queue's class mix q is the maximum-likelihood solution of its decision counts = Cᵀq (EM; Saerens
et al. 2002, Lipton et al. 2018). Each shortlist entry is then explicable with probability
f = Σ_{c≠u} q_c C[c,u] / Σ_c q_c C[c,u], and the false count is Binomial(k, f) for a shortlist of k. The reported
interval bootstraps controls within class and the queue, then draws that binomial, so it covers the realised count
rather than only its expectation.

Probabilities are recalibrated with a monotone (isotonic) map fitted on controls (`scoring.Isotonic`): a vetter that
says 0.9 is never mapped below one that says 0.8, the only structure a few hundred controls support.

## 6. Held-out prediction targets

Decoys measure confabulation on decoys. To measure it on real candidates, withhold part of each packet and ask the
expert to predict it from its own explanation, before giving class probabilities (`packet.HOLDOUTS`):

| Holdout | Target | Kind |
|---|---|---|
| `tail` | the last 20 % of the time span: does another ≥ 4σ dip occur; minimum flux | binary; 80 % interval |
| `colour` (Gaia) | RP/BP depth ratio at the deepest G epoch | 80 % interval |
| `ir` | mid-infrared excess > 3σ | binary |

Binary targets are scored by Brier skill against the batch base rate; intervals by coverage and the interval score
(Gneiting & Raftery 2007). The signature of confabulation is high stated confidence with no predictive skill, so
skill is reported separately for confident (max class probability ≥ 0.8) and unsure verdicts. The cost is a thinner
packet, so production runs two passes: A1 with the holdout, committed, then A2 with the full packet. The A1→A2
update is itself informative: a coherent explanation should not need to change much when the withheld data arrive.
Under a tail holdout, detector scores and tests computed on the full light curve are dropped from the packet, since
they would leak the tail.

## 7. Budget and design

Slots are objects that enter Stage A; the ceiling is ~10³, with ~10⁴ at the edge.

| Use | Objects | Notes |
|---|---|---|
| Calibration run (V3, on DR3) | ~400 | ~300 known-answer controls split by `design.neyman_allocation` (∝ q_c √(α_c(1−α_c)), floor 10 per class) plus ~100 decoys: 40 anomaly probes, 30 incoherent, 30 rare natural |
| DR4 production queue | ~10³ | includes ~5 % drift-check controls and decoys, and the 5–10 % random audit below threshold (statistics.md §5) |
| Replicates | — | a second independent Stage A pass on a random 20 % measures self-consistency; costs tokens, not slots |

A vetter configuration (model, prompt version, packet format) is accepted per control class by Bayesian sequential
stopping on its false-pass rate (`design.sequential_check`): accept when P(α < α_max) ≥ 0.95, reject when
P(α > α_max) ≥ 0.95, otherwise run another batch. Any change to the configuration is a new vetter and is calibrated
again; this is what the `dr3-frozen` and DR4 freeze tags pin.

**Simulation check** (`python -m cosmos_search.vetting.sim`): a vetter with α = 3 % (instrumental) and 2 %
(astrophysical), 70 % probe sensitivity, on a 10³ queue that is 60 % instrumental and 0.1 % truly unexplained. The
shortlist is ~27 entries, of which ~26 are false. 100 replications per row:

| Controls | Allocation (instr / astro / unexpl.) | Bias | RMSE | 90 % interval coverage | Mean width |
|---|---|---|---|---|---|
| 100 | 49 / 36 / 15 | −2.4 | 6.1 | 0.85 | 13 |
| 300 | 161 / 110 / 29 | −1.9 | 5.2 | 0.92 | 13 |
| 1000 | 551 / 370 / 79 | −2.1 | 4.8 | 0.92 | 10 |

The estimate is usable from ~300 controls, and more controls barely narrow it, because the binomial spread of a
26-entry shortlist dominates. Spending beyond ~300 controls buys less than lowering α through Stage 0, Stage B and
independent-instrument checks.

## 8. Blinding boundary

Unchanged from blinding.md: sealed truth lives only in the truth repo; Stage A has no tools; verdicts are committed
(`vetting/commit.py`: file, SHA-256, time, code commit) before `reveal()` will open a batch key, and it refuses if the
verdict file changed after its commit. In a sealed batch the key is age ciphertext for the reveal key; in a tuning
batch it is plaintext. Batches for the truth repo are built by the `vetting_batch` injector class, whose products
are the packets and whose reveal map is the scoring key, with each decoy's candidate id replaced by "decoy".

## Log

- 2026-10-21. Design written. Changes from statistics.md §6 as first drafted: calibration moved to a separate run
  (stateless agents), held-out prediction targets added as a confabulation measure on real candidates, anomaly
  probes added as their own decoy role (sensitivity, not only confabulation), and the shortlist figure defined as a
  label-shift estimate with a predictive interval. Decoys are redrawn until their sampled S/N ≥ 7, because a short
  dip injected into ~100 Gaia epochs usually lands between them.
