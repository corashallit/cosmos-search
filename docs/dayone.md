# DR4 day-one dress rehearsal (DR3, 2026-10-21)

The whole day-one chain ran once on Gaia DR3 as if it were 2026-12-02, with 6,000 sealed injections in the
stream. Each stage used the code DR4 will use. This page records what broke, what each stage costs, how
many injections survived each stage, and what to change before the D2 freeze. Code: `cosmos_search/dayone/`,
`cosmos_search/inject/classes/gaia_stream.py`, `scripts/dr3_g3.py`.

## Chain as run

| Step | Command | Output |
|---|---|---|
| Seal injections | `python -m cosmos_search.inject.classes.gaia_stream plan --hosts … --blend … --n 6000` | `plan.json.age`, to `keys/cora.age.pub` plus a rehearsal reveal key |
| Bulk stream | `python -m cosmos_search.gaia.dr3_pass … --inject plan.json.age --keep-z 8` | features for every source, light curves kept at \|z\| ≥ 8 plus 1% controls, sketches |
| Prefilter | `python -m cosmos_search.targets run` (α = 10⁻⁴) | DataLink order |
| DataLink | `python -m cosmos_search.dayone.datalink --select … --inject plan.json.age` | the same features through the other route |
| G3 | `python scripts/dr3_g3.py <features> <out> <aux> [MIN_ABS_W]` | per-source q for `single`, `fov`, `single_raw` |
| Ranking v0 | `python -m cosmos_search.dayone.rank` | `ranked.parquet`, `queue.parquet` (30 ranked, 6 audit, 8 EB controls), `summary.json` with the urandom seed of the audit and control draw |
| Cross-match | `python -m cosmos_search.xmatch --context` | 11-survey context for the queue |
| Packets | `python -m cosmos_search.dayone.vet build [--no-sso]` | 44 packets with tail holdout, prompts |
| Stage A | one blind subagent per packet (Read the prompt, Write the reply) | 44 replies |
| Commit | `python -m cosmos_search.vetting commit` | `verdicts.jsonl` hash, before any reveal |
| Reveal and score | `python -m cosmos_search.dayone.vet score --plan plan.json.age` | verdict table, report, shortlist |

Ranking v0 was written down before G3 ran and was not changed after. Tiers: 3 = the deepest dip falls in the
core of a DPAC eclipsing-binary eclipse (within ±0.25 × duration of a primary or secondary, depth ≤ 1.5 ×
the model + 0.05 mag); 2 = a DPAC class that can dip; 1 = everything else. Flags: `fov` (dip confirmed in
both fields of view), `grey`, `tension` (dips spaced closer than a third of the shortest orbit the duration
allows), `bright`. The queue takes the top of tier 1, an audit draw with known inclusion probabilities, and
tier-3 EBs as known-answer controls.

## Bugs found and fixed

- **DataLink light curves carried no DPAC rejections.** The DataLink CSV writes flag arrays as
  `(true, false, …)`; the parser matched `True` exactly, so every flag read as False. Over half of 550
  sources compared had features that differed from the bulk route. After the fix, 450 sources pulled through
  DataLink match the bulk features exactly in all 62 numeric columns (`gaia/epoch.py`, test added).
- **A symmetric |z| cut breaks the `fov` mirror.** Deep EB dips keep more dim than bright rows below the cut,
  so `fov` gained 247 spurious selections at a cut of 12 (5,765 at 20). `dr3_g3.run(min_abs_w=…)` now masks
  each statistic at the cut; selections then equal the full-data ones (cut 8) or are a conservative subset.
- **Packets lost every test under the tail holdout,** and `build_batch` without a detect hook wiped them.
  Tests computed only from visible epochs are now marked `visible_only` and survive the holdout.
- **`in_eclipse` counted half of all phases as eclipse.** It now uses the eclipse core plus a depth check;
  the chance rate on non-dip epochs is ~19%.
- **Session setup did not run in project threads,** so `lsdb` and `pyrage` were missing. Run
  `CLAUDE_CODE_REMOTE=true bash scripts/session-setup.sh` by hand there.

## Throughput

| Stage | Measured | Note |
|---|---|---|
| Bulk stream | 3,386 files, 74 GB, 11,754,237 sources, 1.554 × 10⁹ rows in 63 min wall, 5.08 CPU-h on 4 cores (~20 MB/s) | DR4's 7.2 TB is ~100× DR3, so ~4.3 days in one container at this rate; split across sessions |
| Stream outputs | features 3.4 GB, sketches 190 MB, kept light curves 4.3 GB (1,077,979 sources, 9.2%) | the keep rule does not scale to DR4; see below |
| Prefilter | 17,125 sources at α = 10⁻⁴ in 25 s | |
| DataLink | 155–568 s per 50 sources while the stream ran, 150–190 s per 50 after (0.27 sources/s) | 5× slower than the 1.5 sources/s measured earlier the same day. 10⁴ sources ≈ 10 h at today's rate |
| G3 | 85 s for all of DR3 | |
| Ranking | 175 s (three TAP uploads for context, then light-curve physics) | |
| Cross-match | 1.1–1.6 s per target per survey (HATS partition reads) | 11 surveys for a 10³ queue ≈ 3 h |
| Stage A | 44 subagents, 25–90 s each, ~57k tokens each | transcripts audited: each made one Read of its prompt and one Write of its reply |

## Recall of the 6,000 injections by stage

Hosts: half uniform over the sample, half flat in G over 8–20.5. Families: grey dip 20%, dust dip 10%,
blue dip 5%, four dip-train families 35%, secular fade 7.5%, step 5%, chromatic flip 7.5%, G-only fault 10%.
Signals are drawn over DR3's whole span, so most dips fall between epochs, as real ones would.

| Stage | Injections | Note |
|---|---|---|
| In the stream | 6,000 | |
| Light curve kept (\|z\| ≥ 8 or control) | 603 (10%) | 9.2% of the background; most injections have no sampled dip, and of those sampled above 30σ the rule kept 97% |
| Prefilter (α = 10⁻⁴) | 18 | |
| G3 selected (q ≤ 0.1, `single` or `fov`) | 54 | ~10 expected by chance from host variability (background rate 0.17%) |
| of which a dip was sampled | 24 | the other 30 have no epoch inside the injected signal: selected for the host's own variability |
| selected because of the injection | 21 | checked by pulling all 54 hosts again through DataLink without the injection: the 30 with no sampled dip and 3 of the 24 pass G3 on their own |
| Ranked tier 3 (explained as an EB eclipse) | 27 of 54 | 22 of these have no injected dip above 3σ at any sampled epoch: real EB hosts, correctly explained |
| Queued for vetting | 5 | 4 with a sampled dip (depth 19–26%, SNR 220–720), 1 without |
| Vetted unexplained (P(u) ≥ 0.25) | 0 | |

What limits recall is mainly sampling, then the G3 threshold:

| Peak SNR of the injected dip at the sampled epochs (kept light curves) | n | G3 selected |
|---|---|---|
| ≤ 3 (no dip sampled) | 444 | 28 (host variability) |
| 3–10 | 25 | 2 |
| 10–30 | 27 | 1 |
| 30–100 | 51 | 2 |
| > 100 | 56 | 21 (38%) |

So on DR3's sparse sampling a dip has to land on an epoch, and once it does it needs to be roughly 100 times
the photometric error to beat the G3 threshold. The threshold is that high because the sample is DPAC's
11.75 M variable stars, whose robust scatter is set by their own variability. By G: 30/1,048 selected at
G ≤ 12, 18/714 at 12–14, 6/1,172 at 14–16, none fainter. Stage 0 does its job on the G-only fault family:
2 of 617 selected (chance expectation ~1), against 10 of 1,198 grey dips.

The `grey` flag is the sharpest discriminator in the chain. Of the 20 grey selections outside tier 3,
17 are injections. The three real ones: two are known EBs in VSX (ASAS J183135+1202.5, type ED,
P = 1.61 d; Gaia DR3 5591627238456331264, type E, P = 0.224 d). The third, TYC 2791-192-1
(Gaia DR3 381985255912902272, G = 11.4, RUWE 1.64, three dip episodes, ≥ 0.8 d long, 8% deep), has no VSX
entry and SIMBAD lists it only as a star. None of the three reached the queue. This is an exploratory
observation outside the pre-registered queue, so it is a lead for ranking v1, not a candidate.

## Vetting and the shortlist

44 packets: 31 real, 5 injected (the queue drew them blind), 8 EB controls. All verdicts were committed
(sha256 `cd5b4a61…`) before the plan or the key was opened.

- **Shortlist under the pre-registered rule (real packets with P(u) ≥ 0.25): empty.** The largest P(u) was
  0.20, for one real and one injected packet.
- **Controls:** 8 of 8 called astrophysical, top class EB; false-unexplained 0/8 (95% interval 0–0.26).
  With an empty shortlist, the label-shift FP estimate (`quantify.shortlist_false`) has nothing to estimate;
  its precision would rest on these 8 controls alone because V3's ~400-packet calibration run has not been
  done.
- **Injections:** 0 of 5 called unexplained. Mean P(u) 0.11 for injections against 0.07 for real packets
  and 0.06 for controls, so the ordering carries some signal while the absolute level sits far below the
  rule's threshold.
- **Held-out tail:** Brier skill on whether the hidden tail has a dip is −0.27 on real packets and +0.14 on
  controls; 80% intervals on the tail minimum covered 64% of cases.

The replies read well. For the injected 50-day, 19%-deep grey dip, the agent proposed an
ε Aurigae-like disk occultation, which is what an astronomer would say. The injection families sit inside the
space of known rare astrophysics, while the decision rule counts only P(unexplained). For a technosignature
search that is the wrong boundary: rare astrophysical classes (disk occultation, debris transit, YSO dipper)
are the interesting tail too. A post-hoc check, labelled as such: 1 − P(common classes: EB, rotation,
pulsation, instrumental) averages 0.40 for injections, 0.20 for real packets and 0.18 for controls, and the
two highest scores of the 44 are injections.

## Gaps

- G5 (energy budget against WISE) is not built; the packets carry WISE photometry without a test.
- V3's calibration run (~400 packets) has not happened, so the FP figure has no real precision yet.
- SkyBoT returned 502 and then tripped the client's breaker, so the packets went out without the
  solar-system test. It is a single external dependency on day one.
- 80% of tier 1 (1,747 of 2,190) has no DPAC classifier result, and VSX lists 9 of the 30 ranked sources as
  EA/ED/E types. DR4's classifications will lag its photometry, so day one needs its own EB filter.
- The coincidence test's p-value runs in the opposite sense to the other tests' description in the packet
  text ("p-value of the null 'this FP class does not explain the dip'").
- The commit record names the repository commit, but the working tree had uncommitted changes. The commit
  step should refuse a dirty tree or record its diff hash.
- Sealing inside an agent container is on trust: the same container holds the reveal key so the stream can
  decrypt the plan. Day one should have the plan applied by a process whose key the vetting agents never see.

## Changes to make before the D2 freeze

1. **Keep rule.** Replace "|z| ≥ 8 either side" (9% of sources, ~0.4 TB at DR4 scale) with the G3
   statistics computed in the stream and a per-G-bin keep threshold set from DR3, and store features for
   every source as now. Done after the rehearsal: see "Keep rule v2" below.
2. **Shortlist rule.** v1 (S = P(rare astrophysical) + P(u) ≥ 0.35) was pre-registered and failed on a fresh
   blind batch: 0/11 target decoys (below). The packets need external light curves before a rule can work.
3. **Grey and non-EB first.** Ranking v1 should sort on the `grey` flag within tier 1 and 2 and remove VSX/
   ASAS-SN eclipsing types before vetting.
4. **G3 power.** A per-source scale from a quiet reference (or a variability-class-specific null) would let
   dips well below 100σ through on quiet stars. DR4's denser sampling (~66 months of data) roughly doubles the
   epochs, which mainly helps the sampling term.
5. **DataLink capacity.** Re-measure on the day and run the pull in parallel sessions; at today's 0.27
   sources/s a 10⁴ pull takes most of a day.
6. **SSO fallback.** Cache a Miriade route or a local ephemeris for the dip epochs.
7. Run V3's calibration batch before December, so the shortlist FP figure has real precision.

## Shortlist rule v1 (pre-registered 2026-10-21, before batch 2 was built)

Written after the rehearsal batch and committed before the second batch's packets exist, so the rehearsal
numbers that suggested it do not count as its test.

- **Score.** S = P(disk_occultation) + P(debris_transit) + P(other_astrophysical) + P(unexplained), from each
  Stage A reply's leaf probabilities (`dayone.vet.interesting_score`). YSO dippers stay among the common
  classes, because they are frequent in star-forming regions and would flood the list.
- **Rule.** Shortlist = real packets with S ≥ 0.35 (`S_SHORTLIST`). On the rehearsal batch this would have
  taken 2 of 5 injections, 0 of 31 real packets and 0 of 8 controls; that is where 0.35 comes from.
- **Batch 2.** All injection-driven G3 selections not vetted yet (dip sampled, and not selected when pulled
  again without the injection); the 3 real grey selections outside tier 3; further tier-1 sources in ranking-v0
  order, excluding injected hosts, up to 16 real packets; 8 new tier-3 EB controls labelled at the dip; 6
  tier-2 controls with a non-EB DPAC class, labelled by star (an instrumental reading of their dip is also
  accepted). One blind subagent per packet, transcripts audited, verdicts committed before the reveal.
- **Measures** (`dayone.vet.score_v1`). Recall on target-family decoys (grey, blue, Poisson, prime and
  geometric dip trains, chromatic flip, secular fade); the flagged rate on controls, on the other decoys
  (periodic trains, dust dips, steps, G-only) and on real packets; AUC of S and of P(u) for target decoys
  against controls. All with 95% Wilson intervals.
- **Decision.** Adopt v1 for the D2 freeze if target recall ≥ 0.5, at most 1 of the 14 controls is flagged,
  and AUC(S) ≥ AUC(P(u)). Otherwise revise it and test again on another fresh batch. The batch is small, so
  the result decides which rule goes forward, while the shortlist FP figure still waits for V3's calibration run.

### Batch 2 result: v1 rejected

47 packets (11 target-family decoys, 6 other decoys, 16 real, 14 controls), one blind subagent each. Every
transcript shows exactly one Read of its prompt and one Write of its reply; verdicts committed at 830f1aa before
the reveal.

| measure | result | criterion |
|---|---|---|
| target-decoy recall | 0/11 (95% CI 0–0.26) | ≥ 0.5: fails |
| controls flagged | 0/14 (0–0.22) | ≤ 1: passes |
| AUC, target decoys vs controls | S 0.74, P(u) 0.85 | AUC(S) ≥ AUC(P(u)): fails |
| other decoys flagged | 1/6 (the step, S 0.80, read as a disk) | |
| real packets flagged | 1/16 | |

v1 does not go forward, and neither does v0 (P(u) ≥ 0.25 flagged nothing; the highest P(u) in the batch was 0.14).

**Why.** Eight of the 11 target decoys were called eclipsing binaries (the rest blends or a systematic), and on the evidence in the packet that is
the right call. In 7 of 11 the packet shows one dip event (one field-of-view pair, one or two epochs); the other 4
show a second dip, which an EB also explains. With ~35 epochs over ~760 d, an occulter that dips a few times
and an EB eclipse look the same, and a careful vetter prefers the common class. The agents' own
`discriminating_test` says what would separate them: denser photometry (ZTF, ATLAS, ASAS-SN, TESS) to look for
recurrence at a fixed period and a secondary eclipse. A threshold on Stage A's class probabilities cannot add
information the packet does not hold.

**The one real hit** is Gaia DR3 3047786003881272448 (G 12.7, b = +3.4°): a year-long, 10–16% dimming, deeper in
BP (17%) than RP (14%), read as an occulting disk (S 0.66). It is a known star: a Gaia DR3 SB1 (P = 39.8 d,
K₁ = 58 km/s, e = 0.25), an ATLAS LPV and an RS CVn star in VSX. A spot cycle on an active K-type binary is the
likelier cause, and a BP/RP depth ratio of 1.2 fits cool spots better than small-grain dust (≈ 1.7). The blind
packet carries no catalogue classes, so the agent could not know.

**Next (rule v2 needs more information in the packet, not a new threshold).**
1. Add an external light-curve block for each queued star: ZTF and ATLAS forced photometry, ASAS-SN, and a
   TESS cutout where the TESS arm has one, so "EB" becomes testable (recurrence, period, secondary).
2. A known-variable block (VSX type, Gaia NSS orbit, ATLAS class) is held back for now: 43 of the 47 batch-2
   stars match VSX, because VSX ingests every Gaia DR3 variable, so on DR3 the block would mostly repeat Gaia's own
   classification and would mark decoy hosts as known variables. It needs a design that works on DR4, whose
   epoch photometry covers non-variables too.
3. Pre-register rule v2 on the enlarged packet, then test it on a fresh batch built from injections into stars
   that have external coverage (below).

## Shortlist rule v2 and batch 3 (pre-registered 2026-10-21, before the decoy plan was drawn)

Batch 2 showed that the packet, not the rule, was the limit. v2 changes the packet and keeps the rule, so the
test isolates what the extra light curves do.

- **Packet.** Each Gaia packet also carries ZTF DR22 (ztf_g, ztf_r, ztf_i; catflags 0, normalised per field, CCD
  and filter) and TESS (TGLC aperture flux, 30 min bins, normalised per sector) epochs within 1.5" or from the
  star's own TGLC file, on the Gaia clock, with the TESS contamination ratio in the context
  (`dayone.external`). Long light curves are binned and also list their faintest epochs (≥ 4 robust σ below the
  median, at most 60), so short dips survive the binning. The tail holdout hides the window [t_cut, end of Gaia]
  in every band.
- **Rule.** Unchanged from v1: shortlist = real packets with S = P(disk_occultation) + P(debris_transit) +
  P(other_astrophysical) + P(unexplained) ≥ 0.35.
- **Decoys** (`dayone.decoys`). 20 hosts drawn from kept light curves that are not ranked, not injected in the
  rehearsal and not in a DPAC class that dips, matched in G to the real candidates, and with ≥ 100 epochs in at
  least one external band. Families fixed in advance: Poisson, prime and geometric dip trains 3 each, grey dips
  3, blue dip 1, chromatic flip 1, secular fade 2 (16 target), periodic trains 2, dust dip 1, step 1 (4 other).
  Signals are drawn over DR3's span and trains keep running under the same law to the end of the external data;
  TESS sees the change diluted by the contamination ratio. A decoy must move some visible G epoch by ≥ 5 robust
  σ. The plan is sealed (Cora's key and the rehearsal key) and injected at build time in memory.
- **Real packets and controls.** Batch 2's 16 real packets and 14 controls again, now with external data, so
  each gives a paired before/after verdict.
- **Measures.** As v1 (`score_v1`), plus the share of target decoys called eclipsing binaries, and the paired
  change in S on the real packets and controls.
- **Decision.** Adopt v2 for the D2 freeze if target recall ≥ 0.5, at most 1 of the 14 controls is flagged, and
  AUC(S) ≥ AUC(P(u)). If recall improves but stays below 0.5, the next step is a pipeline recurrence test on the
  external epochs (does the Gaia dip depth recur at a fixed period?), pre-registered before batch 4.

### Batch 3 result: v2 not adopted

50 packets (16 target decoys, 4 other decoys, 16 real, 14 controls), one blind subagent each. All 50 transcripts
show exactly one Read of the prompt and one Write of the reply; verdicts committed at 0a3882c before the reveal.
SkyBoT is unreachable from the cloud container, so the solar-system check did not run for this batch.

| measure | result | criterion |
|---|---|---|
| target-decoy recall | 4/16 (95% CI 0.10–0.49) | ≥ 0.5: fails |
| controls flagged | 0/14 (0–0.22) | ≤ 1: passes |
| AUC, target decoys vs controls | S 0.69, P(u) 0.63 | AUC(S) ≥ AUC(P(u)): passes |
| other decoys flagged | 2/4 (the step, S 0.50, unexplained; a periodic train, S 0.43, EB) | |
| real packets flagged | 1/16 (3047786003881272448 again, S 0.66 → 0.43) | |
| target decoys called EB | 8/16 (batch 2: 8/11) | |
| paired change in S, real | median −0.025 (4 up, 11 down) | |
| paired change in S, controls | median −0.025 (5 up, 9 down) | |

Recall rose from 0/11 to 4/16 but stays below 0.5, so by the pre-registered decision the next step is the pipeline
recurrence test. The rise is smaller than it looks: two of the four hits are secular fades (S 0.80, 0.36), a family
batch 2 did not have. On the families both batches share, it is 2/14 against 0/11, within noise.

**Why the external data barely helped.** For most decoys the external epochs never saw the signal. Of the 14
dip decoys (trains and single dips), only 3 have any external epoch moved by ≥ 5 robust σ: one Poisson train in
124 ZTF epochs, and two prime trains in 1 and 3. Single dips do not recur by design. Sparse trains with dips of
hours to days mostly fall between ZTF's nights and outside TESS's sectors (the TGLC route ends at BTJD 2824). In
the other 11 the external data said "no other dip", which the agents read either as a long-period EB that ZTF
missed or, in the two geometric trains near the threshold, as weak evidence for a one-off occulter.

The one decoy with clear external recurrence shows what is missing. The Poisson train (79 dips over the span, 124
ZTF epochs in dip) showed dozens of dips at a constant ~16% depth. The agent read that as a strict period near a
1-day alias and called an EB (S 0.06), while its own `discriminating_test` asked for a BLS search to recover the
period. An agent reading tables of epochs cannot run a period search; the pipeline can. That is the recurrence
test.

External data also made the agents slightly surer of mundane classes for real packets and controls alike. The
real hit fell because ZTF shows a ~39 d, 4.5% oscillation and slow 15–20% swings, which supports an evolved
pulsator (long secondary period or RVb) over an occulting disk; it is still a known SB1 (batch 2).

**Next (batch 4, pre-registered before its decoy plan is drawn).**

1. A recurrence test in the pipeline. For each packet with external epochs, mark in-dip epochs per band, then ask
   whether one strict period explains them: fold all epochs over a period grid and score how cleanly in-dip
   epochs separate from flat epochs at the same phase. An eclipse window holds only in-dip epochs; an aperiodic
   train's window fills with flat ones. The packet gets the best period, that purity, the number of distinct
   dips and their depth scatter. Calibrate on Gaia DR3 eclipsing binaries with ZTF data and on injected aperiodic
   trains before freezing a threshold.
2. Decoy dip trains must also be visible externally (some external epoch moved by ≥ 5σ), as they already must be
   in Gaia's visible epochs, so the batch measures what the test does when it has something to test. Single-dip
   families stay in the batch but are scored separately, since recurrence cannot help them.
3. Report how many of the real candidates have external epochs in dip, so the conditional recall can be scaled
   to DR4.

## Recurrence test and rule v3 (pre-registered 2026-10-21, before batch 4's decoy plan was drawn)

**The test** (`vetting.recurrence`, in every packet with two or more dip events as test `recurrence`). On the
visible epochs of every survey (Gaia G, ZTF, TESS; BP and RP share G's epochs and are left out), an epoch is in
dip when it is ≥ 4 robust σ low and deeper than the band's largest brightening (a mirror test: noise outliers
come in both signs, occultations in one). Dip epochs within 1 d form one event. A search over periods from 0.2 d
to half the span and window widths of 0.05–5 d finds the most events one strict period puts in up to two phase
windows (primary and secondary eclipse) whose epochs are in dip, counting against a window only epochs at
normal brightness (< 2 σ low) so that ingress and egress do not. It reports `n_events`, `n_aligned`, the period,
`aligned_fraction` = n_aligned / n_events, and `testable` (n_events ≥ 4). The pipeline flag is

    aperiodic = testable and n_events − n_aligned ≥ 2 and aligned_fraction ≤ 0.6

**Calibration** (`dayone.recurrence_cal`, ZTF and TESS epochs only), done before this rule was written:

| population | n | flagged aperiodic |
|---|---|---|
| Poisson, prime and geometric dip trains injected into 35 quiet decoy hosts, redrawn until ≥ 4 external dip events | 319 | 51% (Poisson 63%, prime 43%, geometric 39%) |
| strict-period dip trains, same hosts and condition | 122 | 0 |
| Gaia DR3 eclipsing binaries with ZTF light curves (`vari_eclipsing_binary`, G 11–15.5, random subset) | 63 (32 testable) | 1 (4 events) |
| batch 3's real candidates and controls with external data | 25 (11 testable) | 0 |

The Gaia EBs' recovered periods match DPAC's or a factor of 2 from it. Separation grows with the number of
events: with 15 or more, aperiodic trains align ≤ 0.5 of them and strict-period trains ≥ 0.94; with 4–5 events
the two overlap and the flag rarely fires. Two calibration fixes came from these runs: ZTF's heavy-tailed outliers
made one-epoch events in quiet hosts (fixed by the mirror test), and partly sampled eclipses failed purity
(fixed by not counting ingress epochs against it). 16 of the 51 hosts dip on their own (one is an unknown EB)
and were left out of the injection measure. Most trains under the injection priors leave fewer than 4 external
events: unconditioned, 0–10% are testable, which bounds what this test can do for a given survey overlap.
The IRSA light-curve service returned 504s through the run, so the EB sample stopped at 63 of 286; the rest are
reported out of sample when they arrive.

**Rule v3.** Two variants, scored on the same batch:

- v3a: unchanged S ≥ 0.35 on packets that now carry the recurrence test (does the agent use it?).
- v3b: S ≥ 0.35 or `aperiodic` (the pipeline flag can shortlist on its own).

**Batch 4.** 20 new decoys on batch 3's unused hosts (with external data), families fixed in advance: Poisson,
prime and geometric dip trains 4 each (12 trains, the primary measure), each redrawn until it leaves ≥ 4 dip
events in the external epochs as well as moving a visible G epoch by ≥ 5σ (`decoys plan --min-external-events
4`); grey dips 2, blue dip 1, chromatic flip 1 (4 single events, scored separately: recurrence cannot help them);
strict-period trains 2 (also ≥ 4 external events), dust dip 1, step 1 (4 other). Batch 3's 16 real packets and
14 controls again. Plan sealed as before.

**Measures and decision.** For v3a and v3b: recall on the 12 trains, controls flagged, other decoys flagged,
real packets flagged, AUC(S) and AUC(P(u)) for trains vs controls; single-event recall; how many real packets
are testable. Adopt v3b for the D2 freeze if train recall ≥ 0.5 and at most 1 of 14 controls is flagged.
Calibration puts the flag alone at about 0.5 on such trains, so the agent's S has to add some. If v3b fails on
recall alone, the next step is more overlap (ATLAS, ASAS-SN) rather than a lower threshold. v3a is reported to
learn whether the agents act on the test.

*Deviation, recorded before the build:* the plan drew 18 of 20 decoys. All 12 trains, the 4 single events and
the 2 strict-period trains are in; the dust dip and the step are not, because the 40 unused hosts ran out
(conditioned trains reject many hosts). Batch 4 has 48 packets.

### Batch 4 result: v3b adopted

48 packets (12 trains, 4 single events, 2 strict-period trains, 16 real, 14 controls), one blind subagent each.
All 48 transcripts show one Read of the prompt and one Write of the reply (plus the harness's hand-back); verdicts
committed at 2d15944 before the reveal. 43 packets carry the recurrence test; 35 are testable.

| measure | v3a (S ≥ 0.35) | v3b (S ≥ 0.35 or aperiodic) | criterion |
|---|---|---|---|
| train recall | 7/12 (95% CI 0.32–0.81) | 8/12 (0.39–0.86) | ≥ 0.5: passes |
| controls flagged | 0/14 (0–0.22) | 1/14 (0.01–0.31) | ≤ 1: passes |
| single-event recall | 0/4 | 0/4 | |
| other decoys flagged | 2/2 (both strict-period trains) | 2/2 | |
| real packets flagged | 1/16 (3047786003881272448) | 1/16 (the same) | |
| AUC, trains vs controls | S 0.79, P(u) 0.80 | | |

By the pre-registered decision, v3b goes into the D2 freeze. v3a passes too, because the agents use the test: the
pipeline flagged 7 of the 12 trains (calibration: about half), and the agents gave S ≥ 0.35 to 6 of those 7,
mostly as "unexplained". The seventh (a prime train the agent read as instrumental, S 0.23) is the one the flag
adds. Of the 5 trains the flag missed, the agents called 4 eclipsing binaries (S ≤ 0.08) and one a debris
transit (S 0.66). So recall here is mostly the flag's, and the agent adds about one train in twelve on top.

- **This is conditional recall.** Every train was drawn to leave ≥ 4 dip events in the external epochs. Under the
  injection priors, unconditioned, 0–10% of trains do (calibration above), so on DR4 the gain applies only to the
  candidates with enough ZTF or TESS coverage of their dips. More overlap (ATLAS, ASAS-SN, DR4's own longer
  baseline) widens it.
- **The flagged control** is a star the agent read as an irregular pulsator (S 0.12): irregular variability dips
  without a period, which the flag cannot tell from an occulter. This is the flag's expected false-positive class.
- **Real packets.** 13 of 16 are testable. Only 3047786003881272448 is flagged (aperiodic, S 0.43 → 0.51, disk
  occultation), the same star as batches 2 and 3. Paired S change from batch 3, median −0.01 for real and controls.
- **Gaia EBs out of sample.** 113 more of the EB sample arrived after the rule was written: 39 testable, 3 flagged
  aperiodic (DPAC periods 0.69, 1.17 and 1.30 d; ZTF's nightly cadence aliases near 1 d). With the 32 in sample,
  4 of 71 testable EBs (6%). Real dippers on DR4 will be mostly EBs, so expect about this share of testable EBs on
  the shortlist; it is a cost on the expert queue, not a reason to change the rule now.
- **Single events** (grey, blue, chromatic) got 0/4: by design recurrence cannot help them, and the agents
  called them EBs, spots or detector defects. They remain the gap the D2 rule does not close.

## Keep rule v2 (after the rehearsal)

The stream keeps a light curve when the source's G3 `single` or `fov` |W| reaches 0.6 × the |W| at which G3
selected on DR3, for its G bin and statistic (`features.keep_light_curve(table=…)`, table from
`python -m cosmos_search.dayone.keep table --q g3/q.parquet`, stream flag `--keep-table`). Bins where G3
selected nothing keep only the random controls, set to 0.05% for DR4 (~10⁶ sources, docs/storage.md).
Measured on all of DR3's features, with the injections in them:

| Rule | Sources kept | G3 selections kept | Injections sampled > 30σ kept | > 100σ kept |
|---|---|---|---|---|
| \|z\| ≥ 8 (rehearsal) | 8.25% | 100% | 97% | |
| \|W\| ≥ 16, any G | 2.9% | 98.9% | 51% | |
| v2, margin 0.6 | 0.37% (0.42% with controls) | 100% | 37% | 55% |

v2 keeps all of G3's selections at a twentieth of the old volume, ~20 GB of light curves at DR4 scale
against ~0.4 TB. The sampled injections it drops are ones G3 does not select either, so nothing downstream
loses them; a more sensitive detector would need its own keep table. A two-file stream run kept 39 of 10,140
sources (0.38%). DR4 thresholds will differ (about twice the epochs, other noise), so after G3 runs on DR4,
`python -m cosmos_search.dayone.keep missing` lists selections without a light curve for a DataLink pull.

## Single-event test and rule v4 (pre-registered 2026-10-21, before batch 5's decoy plan was drawn)

Batch 4 left single dips as the gap: 0 of 4 grey, blue and chromatic single events were shortlisted, because a
recurrence test needs something to recur. A single dip cannot be tested against an orbit's period, but it can be
tested against the claim that it is an eclipse. Kepler ties an eclipse's duration to its period, and the period
predicts repeats the other epochs should have caught.

**The test** (`vetting.single_event`, test `single_event` in every packet whose visible epochs in all surveys
show exactly one dip event). In-dip epochs are as in the recurrence test (≥ 4 robust σ low, deeper than the
band's second-largest brightening). A ZTF or TESS in-dip epoch with no other in-dip epoch of its band within 1 d
is ignored as a lone outlier. One event is split in two only by a normal epoch between in-dip epochs that would
have shown the dip at ≥ 5 σ, or by in-dip epochs from different surveys more than 1 d apart. The test reports:

- **The eclipse hypothesis.** A Monte Carlo of mundane eclipsing companions: stellar binaries (log-normal period,
  mean 10^5.03 d, σ 2.28 dex, 46%; mass ratio 0.1–1) and dark bodies (0.03–0.2 R☉, log-uniform period
  0.5–10⁴ d, 30%). Eccentricity is up to 0.7, ω is uniform, and b is uniform over [0, 1 + k]. The host's radius
  and mass come from `physics.stellar_params` with 0.15 dex scatter, or from a broad prior when the parallax is
  poor. The duration is drawn log-uniform between the event's bounds and fixes the period. Each draw is weighted
  by the population density, the transit probability and the catch probability T/P. A draw fits when a
  trapezoid eclipse matches every in-dip epoch to 25% + 3 σ and leaves the bracketing normal epochs below 5 σ.
  It is refuted when some other epoch in any survey would have shown a repeat at ≥ max(1.2 × threshold,
  threshold + 2 σ).
  - `p_eb` is the share of fitting weight that is not refuted.
  - `eb_rate` is the per-star rate of "this event and no repeat". With fewer than 20 fitting draws, an upper
    bound stands in for it, and `oneoff_vs_eb_is_lower_bound` says so.
- **The one-off hypothesis.** An occulter of unknown duration (log-uniform 0.01–1000 d), depth (log-uniform
  1e-3–1) and shape (trapezoid or Gaussian), scored with the same fit rules. `log10_oneoff_vs_eb` is the log
  ratio of the two rates. The one-off rate's normalisation is arbitrary, so only thresholds calibrated on data
  give the ratio a meaning.
- **Colour.** `colour_excludes_occultation` is true when, over the event, one of BP and RP dims and the other
  brightens, each at ≥ 3 σ. Removing light can only dim every band (an occulter's own glow is constant), so no
  eclipse, transit or dust occultation does this. A temperature change of the star can, and so can a colour
  error of the instrument.
- **Context for Stage A:** `quiet_ratio_g` (robust scatter of the other G epochs over their errors), the speed
  bound, the colour ratio, and the surviving periods as a follow-up ephemeris.

The pipeline flag is

    single_dip_flag = (log10_oneoff_vs_eb ≥ 2.5 or colour_excludes_occultation)
                      and n_in_dip ≥ 2, at least one in G
                      and beyond_one_transit (the in-dip epochs span > 0.08 d, or there are ≥ 3)
                      and bracketed (normal epochs on both sides) and quiet_ratio_g ≤ 3

The first condition is the evidence. The others make the event **testable**: they say whether the data can
carry that evidence at all.

**Calibration** (`dayone.single_cal`, packets' visible epochs under the tail holdout), done before this rule
was written. The table counts stars with exactly one dip event, those that are testable and quiet, and those
flagged.

| population | n | one event | testable, quiet | flagged |
|---|---|---|---|---|
| grey dips injected into 120 quiet hosts (3 rounds, ≥ 5 σ in some visible G epoch) | 120 | 96 | 31 | 30 |
| blue dips, same | 120 | 95 | 22 | 21 |
| chromatic flips, same | 120 | 66 | 14 | 12 |
| the same hosts without injection | 120 | 17 | 1 | 0 |
| DR3 sources with one low episode (min z ≤ −5) | 150 | 60 | 2 | 2 (both DPAC SOLAR_LIKE) |
| DPAC eclipsing binaries with P > 20 d | 120 | 15 | 2 | 0 |
| physical EBs injected into the quiet hosts (the EB-filter thread's injector, run from a scratch script; P log-uniform 10–3000 d) | 1,320 (11 rounds) | 279 | 15 | 0 (highest ratio 2.13, a dip of the host itself) |
| the EB-filter thread's joint Gaia + ZTF set (all DPAC classes) | 7,564 | 1,161 | 5 | 1 (SOLAR_LIKE, a 2% low state ≥ 52 d) |
| of which DPAC EBs | 1,352 | 102 | 2 | 0 |
| rehearsal tier-1 single-episode stars | 79 | 30 | 1 | 0 |
| batch 4's real candidates / controls | 16 / 14 | 1 / 4 | 0 / 0 | 0 / 0 |

- **Testable dips are recognised.** 63 of the 67 testable injected dips are flagged. The likelihood ratio does
  most of it. Colour exclusion adds one blue dip and one chromatic flip whose ratio is below 2.5.
- **Testability is the bottleneck.** Only 68 of 360 injected dips (19%) are testable at DR3 cadence. Most leave
  a single transit pair, or no normal epoch near enough on one side. DR4's doubled epochs and longer ZTF overlap
  raise this; the rule cannot.
- **Why the flag needs more than one transit.** The calibration runs first used a flag without
  `beyond_one_transit`. Events seen on one transit pair (two epochs 106.5 min apart) gave injected dips and
  eclipsing binaries overlapping ratios (injected median ~2.2; DPAC EBs at G 18–19.7, 2.9 and 3.3; an RS CVn,
  2.7). One look bounds the duration only at 106.5 min, which an eclipse of almost any period meets. A single
  bad transit can also fake a dip and its colour. On one transit, the ratio reflects the priors more than the
  data.
- **The false-positive class is stellar activity.** All three flagged real stars are DPAC SOLAR_LIKE (spotted,
  active) stars with 2–3% dips lasting a quarter of a day to weeks. The ratio says correctly that these are not
  eclipses. Ruling out a low state of an active star is Stage A's job, using the class and the colour.

**Rule v4.** Shortlist = v3b (S ≥ 0.35 or `aperiodic`) or `single_dip_flag`.

**Batch 5.**

- **Decoys.** 18 decoys on fresh hosts: quiet DR3 stars (robust scatter < 2× errors, no 3 σ epoch, not in a DPAC
  class that dips, not in the calibration), G-matched to batch 4's real and control stars and tier-1 singles,
  with ≥ 100 epochs in at least one external band.
  - The primary measure is 16 single events: grey dips 8, blue dips 4, chromatic flips 4. Each is redrawn until
    it moves a visible G epoch by ≥ 5 σ and the packet shows exactly one testable dip event
    (`single_cal plan5`).
  - 2 dust dips are other decoys, visibility only.
  - The plan is sealed to Cora's key and the reveal key, as before.
- **Controls.**
  - Batch 4's 14 controls.
  - Up to 8 single-eclipse EB controls (role `control_single_eb`, label eclipsing_binary). These are fresh DPAC
    EBs with P > 20 d whose packet shows exactly one dip event, falling inside DPAC's eclipse model, testable
    ones first.
- **Real packets.**
  - Batch 4's 16.
  - 8 rehearsal tier-1 stars whose packet shows one dip event, testable first, then at random.

**Measures and decision.** For v4 and v3b:

- recall on the 16 single-event decoys;
- the single-dip flag alone on the decoys;
- controls flagged, separately for the 14 old controls and the single-eclipse EB controls (with how many of
  those are testable);
- dust dips flagged;
- real packets flagged.

Adopt v4 for the DR4 shortlist if:

- single-event recall is ≥ 0.75 (calibration puts the flag alone at ~0.9 on testable dips);
- at most 1 of the 14 old controls is flagged (v3b flagged 1 in batch 4);
- at most 1 single-eclipse EB control is flagged.

If recall passes but an EB control is flagged, look at the case before changing anything, because the EB
controls are few. A real packet that v4 flags goes to the expert queue whatever the decision. The single-event
test is a packet test and a flag on top of v3b; it does not change the stream, the keep rule or the queue
inside the D2 freeze.

*Deviations, recorded before the build:*

- **Coincidence reference.** Batch 5 has no ranked list from a G3 run, so the coincidence test uses a stand-in
  (`single_cal.reference`): 6,000 random DR3 sources whose deepest G epoch is ≥ 5 robust σ low, plus the
  batch's real and control stars.
- **Host G distribution.** In DR3, epoch photometry exists only for variables, and quiet bright stars are rare
  in it (55 stars between G 10 and 13 out of 11.75 M have scatter < 2× their errors and no 3 σ epoch). The hosts therefore sit near G 13, about half a
  magnitude fainter than the real candidates.
- **Calibration overlap.** The tier-1 stars and the batch 4 controls were in the calibration, and none of them
  was flagged there.

### Step back: what limits single dips, and what to build for DR4 (written while batch 5 was being fetched)

Cora asked to re-check the assumptions one or two levels up. Only 19% of injected single dips were testable, so
the question is whether the limit sits in the rule or in the framing (one star at a time, argued not to be an EB).

**The limit is Gaia's visit structure, and DR4 does not change it.** On the real cadences of 300 calibration
stars:

- Half of consecutive G epochs are ≤ 0.18 d apart and a quarter are ≥ 29 d apart. Gaia sees a star in short
  visits (one transit pair, sometimes two or three scans 6 h apart) separated by weeks.
- A dip that lasts 0.5–10 d and is caught at all is caught on a single transit pair 71–76% of the time,
  whatever its length. Only dips longer than about a month reach multiple visits: 36% are testable at 30 d and
  84% at 100 d.
- DR4 doubles the baseline, not the visit structure. It will catch about twice as many dips and refute eclipse
  repeats over a longer span, but the testable share of caught dips stays near a quarter.

A one-pair dip bounds its duration only between 106.5 min and the neighbouring visits, so no per-star argument
from Gaia alone can separate it from an eclipse (calibration above). This is a property of the instrument, not of
the rule.

**Two levers do change it.**

1. **Concurrent external data, which DR3 lacked and DR4 has.**
   - **ZTF.** DR4 ends in January 2020, and 34% of its span overlaps ZTF (from March 2018). For calibration stars
     with ZTF in that window, a dip of 1, 5 or 10 d gets ≥ 2 ZTF epochs inside it 10%, 26% and 33% of the time.
   - **TESS.** It covers most stars for one or two 27-d sectors between July 2018 and January 2020 and resolves
     any dip longer than an hour when it overlaps. In the calibration fetch, though, TESS epochs inside that
     window are almost absent (about 1 per star), because the fetch keeps the first sectors TGLC returns.
   - **Before DR4:** the external fetch should take the TESS sectors and ZTF epochs that overlap Gaia's span
     first, and TGLC's coverage of sectors 1–20 needs checking. The per-star test already uses concurrent epochs
     as in-dip or refuting evidence.
2. **Population statistics for the one-look dips.** Most single dips will be one-look dips that no single star
   can argue. Their rate can still be tested.
   - **The EB expectation.** The single-event test already computes `eb_rate`, a per-star probability that a
     mundane companion produces such an event. Summed over stars in a bin (colour–magnitude, distance, sky
     position), it gives the expected count from eclipses.
   - **The artefact expectation.** The coincidence test (stars sharing a scan) and FoV/CCD splits give it.
   - **Activity.** The calibration's false-positive class, it can be modelled per DPAC class from DR4's own
     variable-star tables.
   - **The search.** An excess over the expectation that clusters in a way stellar physics does not predict
     would be a population signal. One example is a 3D scan statistic over position and distance, with
     permutations within CMD bins for the look-elsewhere effect.
   - **Verification.** DR4 offers a built-in check: find in one half of the epochs and replicate in the other.
   - **Follow-up.** Members of a significant cluster become TESS and ground follow-up targets, which makes the
     find individually verifiable afterwards.

**Batch 5 is still worth finishing, but it is confirmatory.** The flag's recall on testable dips is already
measured, so the batch mainly checks Stage A's behaviour and the controls. Its data are mostly fetched, and the
rule was pre-registered against it. The larger DR4 investments for single dips are the two levers above: a
concurrent-coverage fetch, and a one-look dip-rate census built on `eb_rate` (shared census tooling).
