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
| detector_defect | Single epoch, one band or one CCD; G ≲ 13 gating | **G vs BP/RP band consistency and G error inflation** (`vetting/stage0.py`, §12); per-CCD agreement (DR4); magnitude stratification | vetting; Gaia detectors | G-only injections into DR3 light curves (`scripts/stage0_recovery.py`) |
| calibration_offset | Steps between FoVs, CCDs, sectors, calibration epochs | FoV pair test; step test at known calibration boundaries | Gaia detectors | step decoys |
| shared_systematic | Same epoch dims in unrelated stars | **cross-candidate coincidence** (`vetting/coincidence.py`, below); TESS common-mode per CCD | vetting; TESS thread | coincident-dip injections across hosts |
| processing | Detrending or background-model artefacts, cross-match errors | re-extraction with an independent pipeline (TGLC for TESS; DataLink re-pull for Gaia) | arm threads | — |
| solar_system_occultation | Single-epoch dip near the ecliptic; asteroid in TESS background annulus | **SkyBoT cone + Miriade parallax from the spacecraft** (`vetting/sso.py`, §13) | vetting | Gaia DR3 `sso_observation` |
| eclipsing_binary, planet_transit | Periodic, flat or V-shaped, grey | period search, odd/even, secondary | arm detectors | VSX/Gaia DR3 EBs |
| disk_occultation, yso_dipper, rcrb_dust, debris_transit | Long, reddened or irregular dips; IR excess | dip colour ratio R; energy budget vs WISE (G5) | Gaia detectors, cross-match thread | KIC 8462852, Gaia17bpp, EPIC 204278916, YSO dippers, R CrB stars, ZTF J0139+5245 |
| rotation_or_spots, pulsation_or_irregular, eruptive_or_be | Quasi-periodic or red-noise variability | periodograms, colour–magnitude slope | arm detectors | VSX/DR3 classified variables |
| extragalactic | Non-stellar SED, zero parallax and proper motion | astrometric and SED cuts | cross-match thread | QSOs in DR3 |

**Cross-candidate coincidence.** A real occulter belongs to one star; an instrument event touches every star
observed at that moment. For each dip with n dips of other candidates within ±W, k of them within ±τ, the null is
k ~ Binomial(n, τ/W): the local rate is taken as flat over W, which absorbs coverage gaps and seasonal changes in
how many stars are observed. Groups (camera/CCD, Gaia scan) restrict the comparison to dips that could share a
cause. This runs over the whole queue plus the random-control stream, which only the vetting stage sees at once.

Stage 0 vetoes outright only on exact nulls: per-CCD disagreement, FoV disagreement, G changes that the BP and RP
changes of the same transit rule out (§12), and coincidence with p below a queue-wide Bonferroni threshold.
Everything else goes into the packet as evidence, so Stage A's calibration includes how it weighs the tests.

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

## 9. Pilot 1 (Gaia DR3, 2026-10-21)

`scripts/vetting_pilot.py`. There were 24 packets with the tail held out. 12 were known-answer controls taken from
DPAC's classifier at score > 0.9 (two each of ECL, YSO, RCB, LPV, RS, AGN). 12 were decoys injected into six quiet
SOLAR_LIKE stars. The vetters were Stage A subagents, each told to make one Read of its packet and nothing else.
The transcripts confirm that every agent made exactly one Read plus its reply. All 24 verdicts parsed, and they were
committed before the key was opened.

| Measure | Result |
|---|---|
| False "unexplained" on controls | 0 / 12 (95% upper bound 0.19) |
| Group accuracy on controls | 11 / 12 (one AGN called a blend: no signal, `ipd_frac_multi_peak` = 1) |
| Recognition of controls | 0 / 12 named (none are famous) |
| Held-out tail minimum, 80% intervals | 79% coverage; Brier skill on the tail-dip question +0.52 for controls, ~0 for decoys |
| Highest P(unexplained) given to any packet | 0.15 |
| Planted-vs-real audit (§4) | AUC 0.96 (p = 0.002): decoy hosts were a different population from the controls |
| Cost | ~1.6k packet tokens; ~50k tokens and 25–120 s per vetter, including fixed overhead |

**The decoys mostly measured the injector, not the vetter.** Rechecked with the rules now in `visibility()`:

- 7 of 12 decoys were not visible in the epochs the packet showed. The signal was below the host's own variability,
  fell between epochs, or sat in the withheld tail. The S/N check had used the photometric errors over the full
  light curve, which is the wrong noise and the wrong epochs.
- All 4 "anomaly probes" (prime and geometric dip trains) were labelled "unexplained", but at DR3 cadence a train
  shows 1–3 events that a single period fits. Calling them eclipsing binaries was the right reading of the data,
  not a failure to see an anomaly.
- Only 3 decoys had a label the data supported. A one-band BP dip was called detector_defect (correct). A visible
  dust dip was called astrophysical (correct group, wrong leaf). A marginal (4.3σ) chromatic flip got a spot story,
  which is one confabulation out of two valid incoherent decoys.

So the pilot supports one claim: on 12 ordinary known variables, the vetter did not invent anomalies, its held-out
predictions were calibrated, and it ignored the many single-epoch BP/RP outliers (×20–60 flux spikes) in DR3 epoch
photometry. It says nothing yet about sensitivity or confabulation rates.

Changes made because of it:

1. Decoys are accepted only if the shown epochs carry them: at least two epochs ≥ 3σ and one ≥ 4σ, in units of
   the host's robust scatter, before the tail cut.
2. Anomaly probes must also be identifiable as anomalies from the sampled data: ≥ 3 distinct events, and no single
   period (brute-force grid) fits both the in-dip and out-of-dip epochs (`single_period_fits`). The general rule is
   that a decoy's label is what an ideal observer could conclude from the packet, not what the injector drew.
3. Decoys are spread over hosts by a shuffled round robin. A host that cannot show a decoy passes it to the next.
4. The tail-dip question is dropped when its 4σ threshold would be negative flux (R CrB, Miras).
5. Hosts must come from the same population as the packets they are compared with: the real queue in production,
   the controls' classes in a calibration run. This is what the audit failure says.
6. DR3 BP/RP epoch photometry has frequent single-epoch, single-band outliers. A Stage 0 flag for them is cheap and
   would save vetter attention.

## 10. Pilot 2 (Gaia DR3, 2026-10-21)

The setup was pilot 1's with its fixes and fresh sources (`--after 1000000000`). There were 26 packets. 14 were
controls, two each of ECL, YSO, RCB, LPV, RS, SOLAR_LIKE and AGN. 12 were decoys, two each of six specs, in 12
SOLAR_LIKE and RS hosts with G 12–17, so the hosts come from the controls' own classes. Every transcript shows one
Read and the reply. One launch had a mistyped file name; that agent read nothing, answered nothing, and was
relaunched. The audit (§4, signal-free fields only) gave AUC 0.69 (p = 0.10).

| Measure | Result |
|---|---|
| False "unexplained" on controls | 0 / 14 (95% upper bound 0.16) |
| Group / leaf accuracy on controls | 13 / 14 / 10 / 14 |
| P(unexplained): anomaly probes vs controls | probes 0.07, 0.20, 0.36, 0.40; every control ≤ 0.05; AUC 1.0 (p = 0.0005) |
| Probes where "unexplained" wins the group argmax | 0 / 4 (it was the top single class in 2) |
| Incoherent decoys given a natural story (rubric) | 3 / 4, but see below |
| Rare-natural decoys in an acceptable group | 3 / 4 (a dust dip on a host with `ipd_frac_multi_peak` = 1 was called a blend) |
| Held-out tail minimum, 80% intervals | 73% coverage (controls 64%, decoys 83%) |
| Cost | 40–62k tokens and 12–150 s per vetter |

**What it says about the vetter.** It ranks well. Every probe got a higher P(unexplained) than every control. The
two probes with 4–5 visible events got 0.36 and 0.40, with reasoning an astronomer would accept: identical grey
flat-bottomed dips, no period that fits the dip times and the out-of-dip coverage, and flat bottoms too long for
an orbital crossing of a dwarf at the allowed recurrence. Argmax over groups is the wrong summary for Stage A,
because "astrophysical" sums 10 leaves against 1. The report now has `unexplained_ranking`: the AUC of P(u) for
probes vs controls, and the share of probes above every control. The shortlist threshold is a cut on P(u) set from
these numbers, not an argmax.

**What it says about the injector, again.** The pilot exposed four problems in the decoys and the selection:

1. **Selection mismatch.** 10 of 12 decoys had "Detected dips" in the packet, against 2 of 14 controls. Real
   candidates are in the queue because the detector fired, so controls that never trip it calibrate a different
   population. Controls are now drawn through the same detector (`vetting_pilot.py fetch` keeps class members whose
   dips the detector finds before the tail cut). The TESS arm did the same (PR #19).
2. **Decoys never reached the tail.** Signals were drawn before the cut, so no decoy dipped in its withheld tail.
   The tail-dip question was then meaningless on decoys, and a repeating anomaly that stops at the cut is unreal.
   Signals are now drawn over the full span and judged on the shown epochs.
3. **Chromatic flips were not flips.** Both had RP still dimming slightly (by 2% and by 0.5–1%), so steep dust or
   spots were valid readings. With independent amplitudes, one half of a flip almost always swamps the other.
   The brightening now scales with the dip depth, and a flip decoy is accepted only if one transit shows BP ≥ 3σ
   down and RP ≥ 3σ up (`IDENTIFIABLE`).
4. **Which feature is being classified.** Both RP-only decoys were seen as RP artefacts in the explanation. One
   vetter still put the mass on the host's spot modulation, because the prompt never said what the classes are
   for. The prompt now says: classify the cause of the detected dips, and describe the star's other behaviour in
   the explanation. Single-band decoys now go into the detector's band (G), since a dip the detector cannot see
   never reaches vetting.

The TESS arm's pilot found two more leaks, both now fixed in `inject_candidate`. Decoys missed the alternate-aperture
fluxes in `meta`, and errors fell in proportion to flux inside the dip. Every flux channel is now injected, and
errors and in-dip scatter follow photon noise (√k).

**Identifiability at DR3 cadence is the real limit.** One draw from a dip-train family counts as identifiable
when it is visible, shows ≥ 4 events, and fits no periodic eclipsing system with a primary and a free secondary.
Pilot 2 vetters searched exactly that space. On the 12 pilot-2 hosts (23–126 epochs):

| Family | Draws | Visible | ≥ 3 events | ≥ 4 events | No eclipse solution |
|---|---|---|---|---|---|
| prime-spaced train | 2400 | 201 | 32 | 6 | 1 |
| geometric train | 2400 | 191 | 21 | 3 | 0 |

So 1 of 4,800 aperiodic trains drawn from these priors was recognisable as aperiodic in DR3 sampling. DR4 roughly
doubles the epochs and the baseline, which helps but does not change the picture. Timing alone will rarely
identify a Gaia anomaly. The evidence that sparse photometry can carry is per event: a dip whose depth, colour,
shape and duration are inconsistent with any crossing of this star, given its radius and density from the parallax
and colour, and with the recurrence the epochs allow. Pilot-2 vetters used exactly that argument. This points the
Gaia arm's anomaly score, and the next probe family, at per-event physics. Timing claims go to dense follow-up
(TESS, ZTF).

That family is `duration_tension` (`vetting/physics.py`). An occulter crossing a star of radius R* in time D moves
at v ≲ 2R*/D, so any orbit carrying it has P ≥ (π/4) G M* D³ / R*³. That is 0.9 d for a 106.5 min dip on the Sun
and 6.3 yr for a 1 d dip. Eccentricity relaxes this by up to ((1 − e)/(1 + e))^{3/2}, about 0.012 at e = 0.9.
The decoy is a set of identical grey, flat-bottomed dips 0.5–4 d long, recurring at less than a tenth of that
relaxed floor. It is accepted only when the shown epochs prove both halves. One event must show in-dip epochs
≥ 0.25 d apart. Events must be separated by an epoch back at baseline. The shortest spacing must sit a factor 3
inside the floor computed from the *observed* duration, which covers R* and M* estimated from G, BP − RP and
parallax. On the pilot-2 dwarf hosts about 1 draw in 250 qualifies, which is 20 times the timing probes. The
subgiant hosts give no tension, as the physics says they should.

## 11. Pilot 3 (Gaia DR3, 2026-10-21)

The fixes from §10 were in place, plus `duration_tension`, on fresh sources (`--after 1500000000`). There were 24
packets. 12 were controls drawn through the detector: it fired on 17/20 ECL and on 3/20 for each of YSO, LPV, RS,
SOLAR_LIKE and AGN. The one RCB past this point in `random_index` had no detected dip, so the pilot has none. 12
were decoys: 2 `duration_tension`, 1 prime train, 1 geometric train, 2 chromatic flips, 2 G-only dips, 2 dust dips
and 2 periodic grey trains. Every packet had detected dips, against 2/14 controls in pilot 2. Every transcript
shows one Read and the reply.

| Measure | Pilot 3 | Pilots 2 + 3 pooled |
|---|---|---|
| False "unexplained" on controls (argmax) | 0 / 12 | 0 / 26 (95% upper bound 0.11) |
| P(unexplained) AUC, probes vs controls | 0.95 (p = 0.006) | 0.97 (p = 0.0005), 8 probes vs 26 controls |
| Flagged at P(u) ≥ 0.25 | probes 2 / 4, controls 0 / 12 | probes 4 / 8, controls 0 / 26 (upper bound 0.09) |
| Incoherent decoys given a natural story | 0 / 4 (both flips seen as BP down with RP up) | |
| Rare-natural decoys in an acceptable group | 3 / 4 | 6 / 8 |
| Held-out tail minimum, 80% intervals | 75% (controls 67%) | controls 64–67% in both pilots: too narrow |

**The vetter.** The ranking result held up against harder, detector-selected controls. With these decoys,
confabulation fell from a rubric 3/4 to 0/4. The flip vetters argued it from first principles: "a real occultation
cannot brighten RP". One `duration_tension` probe drew the intended argument: "a 14.6 d dip on a dwarf would need an
occulter on a very wide orbit", with P(u) 0.28 as the top leaf. The other was read as an eclipse, with the long dip
taken for a partial one. The prime train drew the timing argument (P(u) 0.38). Control tail intervals were too
narrow in both pilots (64% and 67% against 80%). The fix is a split-conformal widening
(`scoring.conformal_widening`, reported per batch as `conformal_widening_80` on the controls). On the 26 pooled
controls the factor is 1.14, and leave-one-out coverage after widening is 81%. Most misses sit just outside the
interval (6 of 9 within 1.4 half-widths), so the vetter is slightly overconfident, not wrong about where the tail
goes. Decoy intervals covered 83% without widening.

**The labels, this time on the controls.** 4 of the 9 non-EB controls were called instrumental, and the vetters'
evidence was specific: a G-only dip with BP and RP normal at the same transit, a G error 30 times the median at the
dip epoch, a 63% G dip with BP and RP flat, and a RUWE-1.5 star whose G-only plateaus come and go between visits.
BP and RP together cover the G passband, so these are most likely real DR3 artefacts. DPAC's class labels the
*star*, but the detector fired on an *artefact*, so "rotation_or_spots" was the wrong answer key for those dips.
Two consequences follow:

1. **Control labels must describe the detected dip, not the star.** A control's class is its answer only when the
   class explains the dip: an EB dip at an eclipse phase of the DPAC period, or an LPV or AGN dip inside the smooth
   variation in all three bands. Detected dips that fail band consistency are not known answers, and are left out
   of calibration or labelled instrumental.
2. **Stage 0 should test band consistency.** A G deficit d_G with BP and RP flat at the same transit, each measured
   to better than d_G / 3, is instrumental. So is a dip epoch whose G error is far above the star's median. Both
   tests are cheap and need no model. Here they would have removed about half of the detector's non-EB triggers
   before any vetter saw them. G-only decoys then belong in the veto's own injection test, not in Stage A.

**Decoy hosts.** The audit gave AUC 0.81 (p = 0.008). The cause was cadence: decoys showed a median of 47 epochs
against the controls' 31. The host pool was better sampled (median 67 against 40), and acceptance favours
well-sampled hosts on top of that. The second part is legitimate, since real anomalies are also easier to see
there. The first part is a pool mismatch: `vetting_pilot.py fetch` now matches one host to each control in
log(epochs). Two dust dips were read as blends on hosts with `ipd_frac_multi_peak` = 1, in pilot 2 and again here.
Under §10's veto rule a blend-flagged host should not carry decoys.

## 12. Stage 0 band consistency (Gaia)

`vetting/stage0.py` tests each G epoch against the BP and RP epochs of the same transit (matched within 0.01 d).
G covers both passbands, so any change in the light from the source gives a G change near a flux-weighted mean of
the BP and RP changes, d_G ≈ w d_BP + (1 − w) d_RP, where w is the BP share of the G signal. Bounding w to
[0.1, 0.9] covers M dwarfs to A stars with no spectral model, and a chromatic flip (BP down, RP up) still passes.
An epoch whose G change lies outside that interval by more than 4σ is a measurement fault. Faults are excursions,
so the fault lies with whichever instrument moved more in units of its own scatter. `band_inconsistent` means G is
at fault, and that G epoch is dropped. `colour_fault` means BP or RP is at fault, and BP and RP are dropped at that
transit. Epochs where BP or RP is missing, or too noisy or too diluted to have shown the change, are `untested`.

**Dilution.** BP and RP are measured in a 3.5 × 2.1 arcsec window, G in a 0.7 × 2.1 arcsec one. A neighbour
inside the BP/RP window adds constant light there and scales the source's BP/RP changes by r ≤ 1. A real G dip
then lies outside the unblended interval by about 1/r. The first version of this test missed that, and it removed
real eclipses in a DR3 LMC chunk: 4659822847978414848 (P = 14.3 d) and 4659737665861241088 (P = 60.8 d), both in
`vari_eclipsing_binary`, with BP and RP blended in every transit. The test now allows d_G = mix / r for r in
[r_lo, 1], with BP/RP noise scaled by 1 / r_lo. `dilution_floor` measures r_lo per source from its G changes of
at least 10 G errors. For a source known to be unblended it takes 0.7 × the lower quintile of mix / d_G. For a
blended source it takes 0.7 × the minimum, because a neighbour enters the window only at some scan angles. A
quiet source gives no measurement: r_lo = 1 if it is unblended, else 0, which leaves it `untested`. "Unblended"
means a BP and RP blended-transit fraction below 5%, from `gaia_source` `phot_{bp,rp}_n_blended_transits` /
`n_obs`. Without those columns a quiet source is untested. On a blended source the test is therefore lenient by
design: a G-only change there cannot be told from a diluted real one. Leaving the epoch under test out of the
minimum would stop a single G fault from setting its own bound. It gained about 4 points of G-only recall, but
vetoed 2–8% of diluted injections, so the minimum keeps that epoch.

**Error inflation.** A G error more than 5 times the star's median is `inflated_error`, and that G epoch is
dropped, but only if no independent measurement corroborates the change. It is kept when BP or RP at that
transit, or G at the partner transit in the other field of view (within 0.1 d), changed the same way by more than
3 of its own errors. This rule has no physics behind it. A deep, brief change can inflate the error by itself:
DPAC's G error is the spread over the AF CCDs, so a change during the ~40 s crossing raises it. So the rule leans
towards keeping data.

The noise is the larger of the quoted errors and the star's own scatter in d_G − (d_BP + d_RP)/2. On the 50 DR3
stars from pilots 2 and 3, that scatter is about twice what the quoted errors predict (median ratio 2.1), so the
errors alone would over-flag. The DPAC reject flags were 0 on every epoch these stars kept.

**Injection test** (`scripts/stage0_recovery.py --sources`, the 68 controls and hosts of pilots 1–3, with their
blend counts, 20 single-transit injections each). "Diluted" is a grey dip that BP and RP see at k ~ U(0.1, 1) of
its depth, on the same star marked as blended in every transit.

| G deficit (σ_G) | G-only dips flagged: all (unblended stars) | grey, dust (β = 2), flip or diluted flagged |
|---|---|---|
| 3 | 0.20 (0.31) | 0 |
| 10 | 0.41 (0.59) | ≤ 0.001 |
| 20 | 0.45 (0.66) | ≤ 0.001 |

The first version flagged 71–77% of G-only dips at ≥ 10σ, but it would also have vetoed every diluted real change.
The fix gives up about half that recall to keep real dips on blended and on quiet unknown stars, and it vetoes
none of the 6,800 diluted injections. A veto that removes a real signal costs more than one that lets an artefact
through to Stage A. Recall on the 22 blended stars is about 0: untested, as designed. Known answers kept: the two
LMC eclipses above keep their deepest points (76% and 68%), and so does the third source the DR4 thread flagged
(4659940667527443712, 22%), whose deep G point is now corroborated by BP. On a 2,000-source DR3 chunk with blend
fractions, the archive thread found the same: all three keep their deep points, and cleaning now only moves
threshold-straddlers, 4 of 381 dips and 6 of 160 bright spikes at ≥ 8σ, each by a few tenths of a σ. The first
version had removed 29 of the 160 spikes and 6 of the dips. The test is symmetric in its residual and in the fault
attribution, so the mirror null stays valid.

**On the pilot-3 controls**, Stage 0 removes the dips behind two of the four instrumental readings: G errors 27 and
21 times the median, with BP and RP flat at that transit. The other two, a 5% G dip and a 4% G dip with BP and RP
flat, are `untested`. Those stars are unblended (fractions 0.015 and 0), but faint (G = 16.7), and the
band-residual scatter is about 3%, so the data do not show these dips to be faults. Their vetters'
"instrumental" is a plausible reading, but not a known answer. Across the 50 stars Stage 0 drops 1.6% of G epochs
for inflated errors, and BP/RP at 3.3% of transits for colour faults (first version). The colour faults are mostly
single BP or RP values tens of times the median.

Consequences for the pilots:

- `stage0.clean_long` runs the same test on `gaia.epoch`'s long light-curve table, matching bands by
  `transit_id`. It takes `blend` (source_id → `blend_fraction` of the `gaia_source` row) and returns per-source
  counts of each verdict as features. Dropped epochs are counted, never
  silently lost, because the fault rate per source and per sky region is an instrument diagnostic. Any statistic
  that reads the light curves should run after it, dim and bright sides alike so a mirrored null stays
  symmetric, and so should injections, so completeness is measured after cleaning. `epoch_verdicts` is the
  stable entry point for the DR4 stream's G-fault mask.
- `vetting_pilot.py fetch` cleans every control and host with `stage0.clean_long` before the detector runs, so a
  control's detected dip, and with it the label, is about the star. Control stars are no longer reused as hosts:
  pilots 2 and 3 each had two stars in both roles, because both lists come from the same `random_index` walk.
- `build` passes `stage0.veto` to `vetting_batch`, so every decoy must pass the queue's Stage 0.
- The `single_band_dip` decoy leaves the Gaia mix. Where BP and RP could show it, Stage 0 removes it. Where they
  cannot, the packet does not show it to be single-band, so its "instrumental" label would not be what an ideal
  observer can conclude. Its false-positive class is measured by the injection test above instead.

## 13. Solar-system bodies at dip epochs

`vetting/sso.py` looks for known asteroids near the source at each detected dip. A body inside the photometric
window adds light. One in a TESS background annulus is over-subtracted and makes a dip. One exactly in front of
the star occults it for seconds, which Gaia records as a single-epoch dip if a transit catches it. The result,
`sso_bright` (the body's flux is at least a tenth of the dip depth), `sso_faint` or `clear`, is evidence for the
packet and Stage B, not a veto.

SkyBoT's cone search is geocentric whatever observer code it is given. On Gaia DR3's own detection of (1009)
Sirene (`gaiadr3.sso_observation`, 2015-02-20), SkyBoT put the body 544 arcsec from where Gaia saw it, with
location codes 258, `@gaia` and 500 alike. That offset is parallax: Gaia is about 0.01 au from the Earth, so a body
at 3.6 au shifts by about 570 arcsec. Miriade's `ephemcc` does take spacecraft observers (`@gaia`, `@tess`). So
the check runs a geocentric SkyBoT cone widened by the largest parallax (Gaia at 0.0102 au, TESS at 0.0026 au,
bodies beyond 0.8 au), then makes two Miriade calls for one reference body. Every body near one line of sight
shifts by the same sky-plane vector divided by its distance, so that one body fixes the shift for all of them.
On 8 Gaia DR3 asteroid observations ((3200) Phaethon and (7000) Curie), the corrected search found the right body
0.4–1.6 arcsec from Gaia's position, against 500–830 arcsec geocentric. Search radii are 15 arcsec for Gaia (G
window 0.7 × 2.1, BP/RP 3.5 × 2.1, plus timing and ephemeris slack) and 150 arcsec for TESS. Dip times are
converted from BJD to observer UTC by removing the barycentric light time towards the source (up to ±8.3 min).
NEOs within 0.8 au can fall outside the widened cone.

## 14. Calibration run (V3) on slow-change candidates (written 2026-10-30, before the batch is drawn)

**Why on slow changes.** §7 set aside ~400 objects to calibrate the checks. The short-dip review has since had its
own 376-packet test (docs/dayone.md, "Confirmatory result"), and slow fades and long dimming episodes became the
main DR4 search (docs/arms/slow-changes.md). Nothing has yet measured how the per-candidate checks (S5) or the
blind review behave on that search's candidates. This run does, on DR3. Nothing below changes after the draw.

**What is measured.**

1. *The automated checks* (`slow/checks.py`): the colour of the change (BP depth over RP depth), NEOWISE W1/W2
   over the same dates, and ZTF from 2018 on. For each: how often it can be computed; whether its stated error
   holds, on fakes with a known answer; and how often natural slow changers of each class look grey or match each
   explanation's infrared prediction.
2. *The blind review*: one Claude subagent per object with the frozen Stage A text (`vetting/stage_a.md`,
   unchanged), told to make one Read of its prompt and one Write of its reply; the frozen shortlist rule (S =
   P(disk occultation + debris transit + other astrophysical + unexplained) ≥ 0.35, `dayone.vet.S_SHORTLIST`).
   How often it shortlists natural stars of each class, chance passes, and fakes of each kind.

**Stars.** Shard 4/10 of DR3's epoch photometry (`dr3_pass --shard 4/10`, every light curve kept: 350 bulk files),
which no slow-change calibration or test has used (shards 0-3 have). Detector, period search, shared-date test,
thresholds and drift table as frozen in `prereg/rules/slow-v3/`, applied unchanged; a star counts as flagged when
rule slow-v3 flags it (main list, not `early`). The period search costs ~40 ms a star, so it runs on every star of
a sampling frame of 120 of the 350 files (chosen with OS randomness before scoring, list kept with the batch) and on
the quiet stars of all 350. Natural stars are drawn from the frame. Stars in
`rehearsal-inputs/exclude/all_used_ids.txt` are left out.

*Amendment (2026-10-30 02:20 UTC, after scoring the shard and before any draw).* One bulk file of the shard,
`EpochPhotometry_041938-042766` (294,954 sources around M31, RA 11.8° Dec 39.8°), is DR3's Andromeda field, where
DPAC published light curves for every source whatever its variability. It holds 176,545 of the shard's 192,627
quiet stars, so drawn as written the fakes and chance passes would nearly all sit in one crowded patch of sky. It
is left out of both (it is not in the frame, so the natural stars are unaffected); its flagged share of quiet
stars (95 of 176,545) is reported for comparison. Outside it: 42 chance passes and 8,082 eligible hosts.

**Batch (about 410).**

| Group | Objects | Label for scoring | Measures |
|---|---|---|---|
| Natural stars flagged by slow-v3, by DPAC class | 280 | DPAC `best_class_name` (the star's own variability) | false shortlist per class; the checks on real slow changers |
| Quiet stars flagged by slow-v3 (all 350 files) | all of them (expected ~15-30) | no real change | false shortlist on chance passes |
| Fakes, grey (every band dims by the same fraction) | 25 | should be shortlisted | recall |
| Fakes, dust in front (fraction ∝ λ^−β, β 1-2.5; W1/W2 a few % of G) | 15 | should be shortlisted | recall |
| Fakes, dust that absorbs and re-emits (optical as dust; W1/W2 brighten at 300, 600 or 1000 K, `checks.reemission`) | 10 | should be shortlisted | recall |
| Fakes, change in the star's temperature (band ratios of a cooling blackbody at the host's Teff) | 25 | natural | reported only |
| Fakes, G alone (BP, RP, ZTF, W1, W2 unchanged) | 25 | instrumental | reported only (confabulation) |
| Fakes, grey, signal-to-noise 6-20 | 10 | — | reported only (recall near the threshold) |

- *Natural strata*: DPAC classes with at least 10 flagged stars in the frame get their own stratum, the rest are
  pooled as "other". The 280 are split by `design.neyman_allocation` with floor 10 (or all of a stratum, if fewer),
  mix = each stratum's share of flagged frame stars, and guessed false-pass rates 0.02 for LPV and ECL, 0.10 for RS,
  SOLAR_LIKE, YSO and "other", 0.05 for the rest. Drawn with OS randomness. Rates are reported per stratum, pooled,
  and weighted back to the frame's flagged mix (bootstrap within strata).
- *Fakes*: shapes drawn uniformly from long dip, late-starting fade, steady fade, staircase fade, accelerating fade
  and repeating long dip (`inject.lightcurve`; calibrated or spent shapes, none held out), made achromatic, then
  given the flavour's fraction of the G change in each band (G, BP, RP, ztf_g/r/i, W1, W2). Each goes into its own
  quiet star at G < 19 from the 230 files outside the frame (never also a natural star), injected into the raw
  epochs before normalisation and drift correction, and the same signal into that star's ZTF and NEOWISE exposures.
  A fake is kept only if slow-v3 flags it, with the host's real neighbours in the shared-date test, and if its
  signal-to-noise (S3: deepest G deficit at the visits × √(visits below half that depth) ÷ the host's visit noise)
  is at least 20 (6-20 for the last row). Hosts are visited in a shuffled order; a host whose draw fails passes to
  the next draw. The plan (hosts, signals, flavours) is sealed with age to Cora's key and the reveal key.

**Packets** (`vetting.packet.build`, no holdout). Drift-corrected G, BP and RP epochs; ZTF epochs (IRSA, catflags 0,
each ZTF object renormalised to its median after any injection); NEOWISE visit means as bands W1 and W2 (clean single
exposures from `neowiser_p1bs_psd`, `checks.neowise_clean`, averaged per ~6-month visit). The detected change goes in
as the packet's one detected dip (start, end, deepest visit, depth). Tests: the detector's scores, the colour of the
change, the NEOWISE change (or slope, for trends) beside what grey blocking, dust in front, and re-emission at 300 and
1000 K predict, and ZTF's level after 2020-01-20 against 2018-19; each test carries a description of what it
computes and what each explanation predicts, and none quotes a calibration figure. Context: AllWISE W1-W4 and 2MASS
Ks through `cosmos_search.xmatch` (limits renamed as in `dayone.vet.wise_limits`). The short-dip pipeline tests
(recurrence, single event) are not shown: they answer a different question. No variable-star catalogue type is shown,
because VSX carries Gaia DR3's own classes.

**Pass criteria** (one-sided 95% Clopper–Pearson, `vetting.power`):

| | Criterion | Passes a good rule | Passes a bad one |
|---|---|---|---|
| R1 | Grey, dust and re-emitting fakes at signal-to-noise ≥ 20 shortlisted: lower bound ≥ 50% (≥ 32 of 50) | 0.86 at recall 0.7 | 0.03 at 0.5 |
| R2 | Natural and chance-pass stars shortlisted, pooled: upper bound ≤ 10% (≤ 21 of 300; ≤ 23 of the 322 after the amendment) | 0.95 at 0.05 | 0.05 at 0.10 |
| C1 | Colour of the change on fakes: (BP/RP ratio − injected ratio) within 2 stated errors for ≥ 85% | | |
| C2 | NEOWISE change on fakes with a visit inside the change: (W1 change − injected) within 2 stated errors for ≥ 85% | | |

C1 and C2 are of the error bars: a Gaussian error gives 95%; errors that are half the true scatter give ~68%.

**Reported whatever the outcome**: per-stratum and frame-weighted shortlist rates; recall by flavour, shape and
signal-to-noise; the shortlist rate of temperature-change and G-only fakes; top classes; coverage of each check;
how many natural stars of each class look grey (|z_grey| < 2) or match the dust or grey infrared prediction within
2σ, which sets how far the checks alone could cut DR4's list; Brier score and recognition rate; transcript audit.

**What each outcome leads to.**

- R1 and R2 pass: the blind review with the frozen rule is the review step for DR4's slow-change candidates, and
  these rates are the ones quoted for it.
- R1 fails: the review misses anomalous slow changes. The misses are read after the reveal, and the choice (a
  slow-specific rule, which then needs its own fresh blind batch, or the rule as is with the lower recall stated)
  goes to Cora.
- R2 fails: likewise for false entries; DR4's review then has to come after automated cuts that keep the
  expected false entries within the ~10³ budget.
- C1 or C2 fails: the error of that check is fixed (for example from the star's own scatter), and the fix is
  checked on new fakes before DR4.

**Procedure.** `scripts/slow_cal.py`: `draw` (strata, natural ids, sealed fake plan), `build` (fetch NEOWISE,
ZTF and context; packets, prompts, sealed key), Stage A subagents (transcripts audited: a packet whose transcript is
not clean, or whose reply does not parse, is re-run once with a fresh agent; if it still fails it counts against the
rule: a fake not shortlisted, a natural star shortlisted), `vetting commit --archive prereg/verdicts/slow-cal-1`,
`git push`, then `score` (refuses unless the archive commit is on origin).

### Result (2026-10-30)

432 objects reviewed (280 natural, 42 chance passes, 110 fakes), one subagent each. Every reply parsed and every
transcript was one Read of its prompt and one Write of its reply (`audit.json`: 432 clean), so nothing was re-run.
The verdicts went to `prereg/verdicts/slow-cal-1` (commit 83b887f, pushed) before the key was opened.

| | Criterion | Result | |
|---|---|---|---|
| R1 | Grey, dust and re-emitting fakes at S/N ≥ 20 shortlisted, lower bound ≥ 50% | 44/50 (88%; one-sided bounds 78-95%) | pass |
| R2 | Natural and chance-pass stars shortlisted, upper bound ≤ 10% | 26/322 (8.1%; one-sided bounds 5.7-11.0%) | **fail** |
| C1 | Colour of the change within 2 stated errors of the injected ratio, ≥ 85% | 76/79 (96%) | pass |
| C2 | NEOWISE W1 change within 2 stated errors of the injected change, ≥ 85% | 72/78 (92%) | pass |

So, as written above: the review finds anomalous slow changes, but on DR4 it has to come after automated cuts that
keep its false entries within the ~10³ expert budget. The rates to quote for it are recall 88% (78-95%) at S/N ≥ 20
and 5/10 at S/N 6-20, and a false shortlist of 5.0% of flagged stars (2.8-7.5%, bootstrap) once the natural rates
are weighted back to the frame's mix of flagged stars; the pooled 8.1% is higher because the draw oversampled the
classes expected to be hard. At 5%, at most ~2×10⁴ flagged stars can reach the review.

*Natural stars.* None of the 128 AGN, ECL, ELL, LPV, RR and S stars was shortlisted. The 26 are CV 5/10, YSO
3/13, BE|GCAS|SDOR|WR 2/11, DSCT|GDOR|SXPHE 2/11, RS 5/41, SOLAR_LIKE 6/52, other 2/14 and chance passes 1/42.
Stars flagged by an episode are shortlisted more often (episode alone 7/81, with a trend 9/39, with a repeat 4/31)
than those flagged by a trend alone (4/109) or a repeat alone (2/60). 15 of the 26 sit at S 0.35-0.45.
Exploratory, chosen after the reveal and not adoptable without a fresh batch: at S ≥ 0.40 the same verdicts give
43/50 (lower bound 75%) and 16/322 (upper bound 7.4%).

*Detector artefact.* PR #78 found that the period search's straight-line detrend can be pulled by one visit far
above the rest, which makes spurious repeat flags with s_rep in the hundreds. Three natural stars here have that
signature (repeat only, s_rep 72-314, Abbé ≈ 1.1), all cataclysmic variables in outburst; one was shortlisted.
Without them R2 is 25/319 (upper bound 10.8%), still a fail.

*Fakes reported only.* Temperature changes were shortlisted 19/25, mostly as dust occultation (9) or unexplained
(5). At these depths (G 0.6-9%, median 3%) the infrared difference between dust in front and a cooling star is below NEOWISE's
errors (injected W1 changes ≤ 0.03, stated errors 0.002-0.05), so neither the checks nor the reviewer can tell
them apart. G-only changes were shortlisted 13/25, 12 of them as unexplained: the review takes a change seen in G
alone as unexplained rather than instrumental. The colour test gives the BP/RP ratio, which is undefined when
neither band moves, and no test says whether G's change agrees with BP's and RP's. Exploratory: G's depth lying
outside the range of the BP and RP depths by more than 3 combined errors marks 14 of the 25 G-only fakes (8 of the
13 shortlisted), none of the 85 other fakes and 3 of the 315 natural and chance stars with all three depths. That
would be a cheap automated cut before the review; it needs fresh fakes before it is used (done 2026-10-30:
docs/arms/slow-changes.md, "G against BP and RP").

*The checks on natural stars.* Coverage over the whole batch: colour 346/432, a NEOWISE visit inside the change
253/432, ZTF in at least one band 238/432. The checks alone would cut DR4's list very little: of the 253 natural and
chance stars with a measured colour, 235 are grey within 2σ and 207 dust-like, and only 13 are neither; of the 104
episodes with an infrared prediction, 6 match neither grey nor dust. C1's three misses are two temperature fakes
and a grey fake at S/N 18. Of C2's six, two grey fakes sit just outside (z 2.01) and four are hosts whose own W1
moved by 1.7-10% where almost nothing was injected (three temperature fakes, one G-only); one of those has a stated
W1 error of 0.002, too small for NEOWISE, so the error from visit scatter needs a look before DR4 (done
2026-10-30: docs/arms/slow-changes.md, "NEOWISE errors").

*Review quality.* Brier score of S against "should be shortlisted" over R1's fakes and the natural stars 0.064
(0.116 for always giving the base rate); AUC 0.977. One star (of 432) was named from its light curve and colours,
correctly (an eclipsing symbiotic star), and not shortlisted. Each review took 60-100 thousand tokens.

Data: `/mnt/project-files/slow-cal/` (sealed plan and key, review files, prompts, replies, transcripts, audit,
`verdicts_table.csv`, `report.json`).

## Log

- 2026-10-21. Design written. Changes from statistics.md §6 as first drafted: calibration moved to a separate run
  (stateless agents), held-out prediction targets added as a confabulation measure on real candidates, anomaly
  probes added as their own decoy role (sensitivity, not only confabulation), and the shortlist figure defined as a
  label-shift estimate with a predictive interval. Decoys are redrawn until their sampled S/N ≥ 7, because a short
  dip injected into ~100 Gaia epochs usually lands between them.
- 2026-10-21. Pilot 1 (§9): controls behaved, but most decoys were invisible or unidentifiable at DR3 cadence.
  Decoys now pass visibility and identifiability checks before they are accepted.
- 2026-10-21. Pilot 2 (§10): Stage A ranks probes above every control by P(unexplained), but the decoys and controls
  still differed in ways real candidates would not. Controls are now drawn through the detector. Decoys are drawn
  over the full span, injected into every flux channel with photon-noise errors, and must trigger the detector.
  Flips must show both halves, and anomaly probes must defeat eccentric-binary solutions with ≥ 4 events. The
  prompt names the detected dips as the thing to classify.
- 2026-10-21. Added the `duration_tension` anomaly-probe family: per-event physics that sparse cadence can show,
  where timing cannot.
- 2026-10-21. Pilot 3 (§11): with detector-selected controls and the fixed decoys, P(unexplained) still separates
  probes from controls (pooled AUC 0.97; at P(u) ≥ 0.25, 4/8 probes and 0/26 controls), and confabulation is 0/4.
  Half the detector's non-EB triggers were most likely DR3 artefacts (G-only dips), so control labels must be
  dip-level, and Stage 0 needs a band-consistency test. Decoy hosts are now matched to the controls' cadence.
- 2026-10-21. Stage 0 band consistency (§12): G epochs outside the BP–RP mixing interval, or with inflated
  errors, are dropped before detection. In an injection test, G-only dips were flagged 71–77% of the time at
  ≥ 10σ, and grey, dust and flip changes ≤ 0.3%. Pilot controls and hosts are cleaned first, control stars are
  no longer reused as hosts, and single-band decoys leave the Gaia mix.
- 2026-10-21. Held-out tail intervals are recalibrated by split-conformal widening fitted on controls: factor 1.14
  on pilots 2–3, which brings leave-one-out coverage from 65% to 81%.
- 2026-10-21. Stage 0 made dilution-aware (§12) after it removed real eclipses on BP/RP-blended LMC stars. The
  allowed G change scales with a per-source dilution bound, and quiet stars that may be blended are untested.
  Inflated-error epochs are kept when BP, RP or the partner transit corroborates them. G-only recall falls from
  71–77% to 41–45% at ≥ 10σ; none of 6,800 diluted injections is vetoed.
- 2026-10-21. Solar-system check at dip epochs (§13): SkyBoT is geocentric only, so positions are corrected to
  Gaia or TESS with Miriade. Gaia's own asteroid detections are recovered within 1.6 arcsec.
- 2026-10-21. Day-one dress rehearsal (docs/dayone.md): 44 packets from ranking v0 (31 real, 5 sealed injections
  the queue drew blind, 8 tier-3 EB controls), one blind subagent each, committed before the reveal. Controls 8/8
  astrophysical; injections 0/5 unexplained (max P(u) 0.20), so the pre-registered shortlist (P(u) ≥ 0.25) was
  empty. The agents placed injected grey dips in rare known classes (ε Aur-like disks), so the shortlist rule
  should count rare astrophysical classes as interesting. Packet tests computed from visible epochs only are now
  marked `visible_only` and survive the tail holdout; before, the holdout dropped every test.
- 2026-10-21. Shortlist rule v1 (S = P(disk) + P(debris) + P(other astrophysical) + P(u) ≥ 0.35), pre-registered
  before batch 2, failed on 47 fresh blind packets: 0/11 target decoys, 0/14 controls, AUC(S) 0.74 < AUC(P(u))
  0.85. Gaia's ~35 epochs show most injected dip trains as one or two dips, which an EB explains, so the agents
  called them EBs. The one real hit was a known RS CVn SB1. Packets need external light curves and known-variable
  context before the next rule (docs/dayone.md).
- 2026-10-21. Rule v2 (v1's rule on packets with ZTF and TESS light curves), pre-registered before batch 3, not
  adopted on 50 fresh blind packets: 4/16 target decoys (two of them secular fades), 0/14 controls, AUC(S) 0.69 ≥
  AUC(P(u)) 0.63. Only 3 of 14 dip decoys had any external epoch in dip, and the one with clear recurrence was
  called a periodic EB because an agent cannot run a period search. Next: a pipeline recurrence test in the
  packet, pre-registered before batch 4 (docs/dayone.md).
- 2026-10-21. Rule v3b (S ≥ 0.35 or the pipeline's `aperiodic` recurrence flag), pre-registered before batch 4,
  adopted for the D2 freeze on 48 fresh blind packets: 8/12 dip trains that leave ≥ 4 external dip events, 1/14
  controls (an irregular pulsator the flag cannot tell from an occulter), 0/4 single events. v3a (S alone) got
  7/12: the agents read the test and mostly follow it. Out of sample, 3 of 39 testable Gaia DR3 EBs are flagged
  (periods near 1 d, aliased by ZTF's cadence). The gain is conditional on external coverage (docs/dayone.md).
- 2026-10-21. Red-team review fixes (docs/dayone.md, "Review fixes and the confirmatory batch"). Control rates
  from batches 2–4 are in-sample (the same 14 controls and 16 real packets throughout, also used in calibration).
  Verdicts are now committed to `prereg/verdicts/` and must be on origin before a reveal; batches 1–4 were archived
  only afterwards. A reserved pool of 135 controls in 9 DPAC classes, including spotted, RS CVn, chemically
  peculiar stars and AGN, is sealed for one confirmatory batch, along with four held-out injection families and
  pass criteria with stated power (`vetting.power`). The vetter's text no longer quotes calibration figures.
- 2026-10-22. Rule v5 (S ≥ 0.35; no pipeline flag shortlists, review R16) adopted on the confirmatory batch, the
  first run under the full pre-registration procedure (verdicts on origin before the reveal). 283 fresh blind
  packets: held-out recall 22/48 (one-sided lower bound 33% vs the 30% criterion), reserved controls 3/135 (all
  RS or SOLAR_LIKE), queue sample 1/100. Rule v3b on the same packets: 28/48, 20/135, 8/100. Hiding the pipeline
  tests cut recall to 18/48. Most misses (15 of 26) and most of the queue (69 of 100) read as EBs (docs/dayone.md).
- 2026-10-30. Calibration run on slow-change candidates (§14), 432 objects from DR3 shard 4, verdicts pushed before
  the reveal. Recall of anomalous fakes 44/50 (pass), natural and chance-pass stars shortlisted 26/322 (upper bound
  11.0% against 10%: fail), colour and NEOWISE errors hold (96%, 92%). Weighted to the mix of flagged stars the
  false shortlist is 5.0% (2.8-7.5%). On DR4 the review comes after automated cuts. The review reads 12 of 25
  G-only changes as unexplained; no test compares G's change with BP's and RP's.
