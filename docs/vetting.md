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
transit. A G error more than 5 times the star's median is `inflated_error`, and that G epoch is dropped as well.
Epochs where BP or RP is missing, or too noisy to have shown the change, are `untested`.

The noise is the larger of the quoted errors and the star's own scatter in d_G − (d_BP + d_RP)/2. On the 50 DR3
stars from pilots 2 and 3, that scatter is about twice what the quoted errors predict (median ratio 2.1), so the
errors alone would over-flag. The DPAC reject flags were 0 on every epoch these stars kept.

**Injection test** (`scripts/stage0_recovery.py`, the 68 controls and hosts of pilots 1–3, 20 single-transit
injections each):

| G deficit (σ_G) | G-only dips flagged | grey, dust (β = 2) or flip flagged |
|---|---|---|
| 3 | 0.38 (G < 14: 0.68; G > 18: 0.01) | ≤ 0.003 |
| 10 | 0.71 (14–16: 0.89; G > 18: 0.41) | ≤ 0.001 |
| 20 | 0.77 (14–16: 0.91) | ≤ 0.003 |

Grey and flip injections were never flagged, and dust only at the 0.1–0.3% level. At 20σ the missed G-only dips
are 9% with no BP or RP epoch in the transit, 5% too noisy in BP/RP, and 10% on stars whose variability puts the
injected G change inside the interval. A veto that removes a real signal costs more than one that lets an artefact
through to Stage A, so the threshold stays at 4σ.

**On the pilot-3 controls**, Stage 0 removes the dips behind two of the four instrumental readings: G errors 27 and
21 times the median at the dip epoch. The other two, a 5% G dip and a 4% G dip with BP and RP flat, are `untested`.
The stars are faint (G = 16.7), and the band-residual scatter is about 3%, so the data do not show these dips to be
faults. Their vetters' "instrumental" is a plausible reading, but not a known answer. Across the 50 stars Stage 0
drops 1.6% of G epochs for inflated errors, and BP/RP at 3.3% of transits for colour faults. The colour faults are
mostly single BP or RP values tens of times the median.

Consequences for the pilots:

- `stage0.clean_long` runs the same test on `gaia.epoch`'s long light-curve table, matching bands by
  `transit_id`, and returns per-source counts of each verdict as features. Dropped epochs are counted, never
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
