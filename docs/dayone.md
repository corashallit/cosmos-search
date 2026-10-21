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
2. Add a known-variable block (VSX type, Gaia NSS orbit, ATLAS class) as context, kept separate from
   the identifiers so recognition stays measurable.
3. Pre-register rule v2 on the enlarged packet, then test it on a fresh batch built from injections into stars
   that have external coverage.

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
