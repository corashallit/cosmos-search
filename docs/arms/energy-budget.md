# Energy budget (G5): where did the blocked light go?

Status: **adopted as a ranking column and packet context (§7, 2026-10-22).** The commit that adds this file fixes the threshold rule, the half-B pass criteria and
the blind run on the rehearsal candidates (§5–§6) before any half-A, half-B or candidate result was looked at.
Results are in §7; changes after that commit are listed in §8 with their reason and date. How to run it on DR4 is
in §9.

Code: `cosmos_search/energy/` (`budget.py` physics and fit, `photosphere.py` expected colours, `photometry.py`
catalogue rules, `dimming.py` time-averaged dimming, `inject.py` known answers, `cal.py` calibration, `run.py`
per-star summary).

## 1. The question, from first principles

Light an occulter blocks is either absorbed or scattered. Absorbed light heats the occulter, which radiates it
away. That is true of dust, of rock and of anything built, unless it stores the energy (not for long), beams it
away from us, or radiates it far colder than WISE can see. So a dimming star has an infrared counterpart whose
size depends on two unknowns: **how the absorbers are spread around the star**, and **at what temperature they
re-radiate**.

Gaia sees only one sightline. A time-averaged dimming ⟨δ⟩ on that sightline gives:

| arrangement | fraction f of the star's light absorbed | when it applies |
|---|---|---|
| shell: absorbers spread over the whole sky as seen from the star | f = ⟨δ⟩ | swarms, envelopes, R CrB shells; the most absorbing case consistent with ⟨δ⟩ |
| thin: everything on one orbit through our sightline, no wider than the star | f = (π/4) ⟨δ⟩ R*/r | the least absorbing case |

Temperature fixes r for an absorber in equilibrium (flat plate facing the star: R*/r = (T/T*)²), so the thin
floor is f = (π/4)⟨δ⟩(T/T*)². Allowing small grains to run 2× hotter than a blackbody halves it:
**f_thin(T) = (π/8)⟨δ⟩(T/T*)²**. Both assume dark absorbers; an albedo ω scales them by (1 − ω).

Numbers that set expectations (Sun-like star, ⟨δ⟩ = 1%): the shell puts 1% of L* into the infrared, at 300 K
an excess of ~4 photospheres in W3 and ~20 in W4. The thin floor at 300 K is 4 × 10⁻⁶ of L*, a 0.2% W3 excess:
invisible. **So the energy budget can exclude envelopes and swarms; it cannot exclude a compact body on an
orbit.** That is the useful split: natural dust dimmers (young-star disks, R CrB shells, debris clouds) put dust
over a large solid angle and show it, while eclipsing companions, planets and compact structures need not.

The dip's duration constrains r only weakly with Gaia's sampling. A bound occulter crossing at least √δ R* within
a bracket D satisfies r ≤ 2GM*D²/(δR*²) (`budget.t_floor_kinematic`); for δ = 0.1 and D = 0.5 d around the Sun
that floor is 34 K, below anything WISE sees. The kinematic coupling is reported, not used.

## 2. Method

**Photometry** (`photometry.py`). 2MASS K and AllWISE W1–W4 per star, from the Gaia archive's best-neighbour
join (`reference.fetch_ids`) or from the xmatch context table (`photometry.from_context`). Rules: a WISE 'U' is
a 95% limit, read as 0 ± limit/2; a detection below SNR 5, a contamination flag, an extended source, a 2MASS or
WISE source shared with another Gaia star, or Gaia neighbours adding > 5% of the star's flux in the beam make the
band an **upper bound only** (a blend can fake an excess but cannot hide one, so the deficit test needs only upper
bounds). Saturated bands are dropped.

**Photosphere** (`photosphere.py`). The expected K−W1, W1−W2, W1−W3 and W3−W4 are the median of the 64 nearest
reference stars in (BP−RP, M_G, and apparent W1 for the first two), with their robust scatter as the error. Each
colour's reference set is cut on the bluer band only (W1−W3 from G 9.5–12 stars where W1 is unsaturated and the
photosphere is ≥ 20σ in W3; W3−W4 from G < 9.5 stars), so the reference is not selected on an excess. W1 is the
anchor because AllWISE measured W1–W4 at the same epochs; K, taken ten years earlier, carries the star's own
variability (from Gaia's epoch scatter) in its error. Held-out pull checks: K, W1, W2 within ±0.3σ of zero with
robust widths 0.2–1.3; W3/W4 detections of faint stars biased high, which is why those count as bounds.

**Fit** (`budget.fit`). For each T on a 50–1800 K grid and emissivity β ∈ {0, 1}: photosphere × scale plus an
emitter carrying f of L*, with trapezoid band responses and a blackbody star at T*(BP−RP). Outputs per star:

- `g5_z_excess`: √(χ²(photosphere) − min χ²(photosphere + emitter)), maximised over T and β;
- `g5_f_ul_<T>K`: largest f within Δχ² = 4 of the best fit at T (least constraining β);
- with ⟨δ⟩: `z_shell(T)`, `z_thin(T)`: how strongly the photometry rejects the f each arrangement needs (0 where
  it does not), the smaller over β; `g5_<geo>_excluded_above_K`, the lowest T from which every grid T up to 1500
  K is rejected at Z_DEFICIT; `g5_deficit_<geo>`: rejected over all of [150 K, 1500 K] and no excess.

`deficit_shell` reads: *whatever dims this star is not a cloud, envelope or swarm of absorbers at 150–1500 K
spread around it.* For a Sun-like star 150 K is ~3.5 au for a blackbody sphere.

**Time-averaged dimming** (`dimming.py`). Per visit group (epochs within 2 d), the mean deficit of epochs ≥ 3σ
below the median; ⟨δ⟩ is the mean over groups. Its lower bound keeps the mean depth of dimmed groups and replaces
the dimmed fraction k/N by its one-sided 95% Clopper–Pearson lower limit. The deficit test uses the lower bound.

## 3. Calibration design (`cal.py`)

Halves by `(source_id // 3) % 2`: A = 0, B = 1. The photosphere model scoring one half is fitted on the other
half's reference stars only. Hosts: 3,000 general-sample and 1,500 G 9.5–12 stars with G ≥ 10 per half (6,000 for
the dimmer runs), drawn with a seed from os.urandom.

- **null**: hosts unmodified. A few per cent of real stars have real excesses, so this bounds the false rate from
  above.
- **families** (`inject.FAMILIES`), added in flux to every band of the same hosts: yso_disk (900–1500 K plus
  150–400 K, each 2–20% of L*), rcb_shell (600–1000 K, 10–60%), debris (50–300 K, 10⁻⁵–10⁻³, β = 1),
  warm_structure (200–400 K, 10⁻³–0.5), cold (20–50 K, 10⁻³–0.3: the "invisible to WISE" answer).
- **dimmers** (`inject.dimmer_plan`): ⟨δ⟩ log-uniform 10⁻³–0.3, T log-uniform 100–1500 K, arrangement shell
  (f = ⟨δ⟩), thin (f = f_thin) or none (f = 0), equal shares; scored with the true ⟨δ⟩.
- **natural**: DPAC DR3 YSO-class stars (real dust) and ECL-class stars (dimmers without dust), unmodified.

A **false deficit** is a deficit claim for an injection whose emitter lies in the claimed range: deficit_shell on
a shell injection with T in [150, 1500] K, or deficit_thin on a shell or thin one in that range.

## 4. Threshold rule (applied to half A only)

`cal.choose_thresholds`, with the grid 2.0, 2.5, …, 10.0:

- **Z_EXCESS**: the smallest grid value with a null false-excess rate ≤ 1%;
- **Z_DEFICIT**: then the smallest grid value with a pooled (shell + thin) false-deficit rate ≤ 0.5%.

`run.Z_EXCESS` and `run.Z_DEFICIT` are set to these values in the commit that reports half A; nothing else is
tuned on A.

## 5. Half-B pass criteria (fixed now)

With the half-A thresholds, on half B (fresh seed):

| | criterion |
|---|---|
| B1 false excess, null | ≤ 2% (point), Wilson 95% upper bound ≤ 3% |
| B2 false deficit, shell + thin pooled | ≤ 1% (point) |
| B3 "excluded above T" cutting through an injected emitter (shell, T ≤ 1500 K) | ≤ 1% of shell injections |
| B4 excess recall, yso_disk and rcb_shell | ≥ 80% each |
| B5 cold family flagged as excess | ≤ B1's rate + 1 point (it is invisible to WISE by construction) |

G5 is adopted as a ranking column and packet context if B1–B5 all pass. If any fails, it is reported as failed
and the fix goes through a new pre-registration. Reported without a criterion: recall of debris and
warm_structure, deficit recall on "none" injections by ⟨δ⟩, excess rates of real DPAC YSO and ECL stars.

## 6. Blind run on real rehearsal candidates (fixed now)

The 4,002 tier-1/2 candidates of the DR3 dress rehearsal (`/mnt/project-files/dayone/ranked_tiers12_for_g5.parquet`,
written by the rehearsal thread without the reserved control pool or the confirmatory batch; a few are sealed
decoy hosts whose IR is their own). Photometry from `reference.fetch_ids` was downloaded before this commit and
not looked at. The rehearsal light curves are not at hand, so ⟨δ⟩ comes from the ranked table:

    N = max(matched_transits / 2, n_episodes)        (visit groups)
    ⟨δ⟩_lo = 0.5 |depth_frac| × CP_lower_95(n_episodes, N)

(half the deepest depth as the typical episode depth; the DR4 pipeline will use `dimming.mean_deficit` on the light
curves instead).

Scored with the photosphere model fitted on all reference stars and the half-A thresholds. Reported, all decided
now:

1. Excess rate among candidates against the null hosts reweighted to the candidates' G and BP−RP distribution.
   Expectation: higher, because dust dimmers (YSO, R CrB, disks) are among the explanations. One-sided
   two-proportion test, α = 0.05.
2. Excess rate by DPAC class (YSO, ECL, SOLAR_LIKE/RS, none, other).
3. Number of deficit_shell and deficit_thin flags. Expectation: deficit_thin ≈ 0 (§1).
4. The candidates with deficit_shell and no DPAC class, listed with their f_ul and ⟨δ⟩_lo as G5 leads. A lead is a
   star whose dimming cannot come from warm dust spread around it; it is not a claim of anything else.

## 7. Results

### Half A (thresholds)

Seed 11359754380648880456 (os.urandom), photosphere fitted on half B's reference stars. The rule of §4 gives
**Z_EXCESS = 5.0** and **Z_DEFICIT = 2.0** (the lowest grid value already met the 0.5% false-deficit target).

| measure | half A |
|---|---|
| false excess, null hosts | 36 / 4,500 = 0.8% (general 17/3,000, G 9.5–12 19/1,500) |
| excess recall: yso_disk, rcb_shell | 82%, 89% |
| excess recall: warm_structure, debris, cold | 54%, 3.0%, 2.7% |
| false deficit, shell / thin | 5 / 1,698, 0 / 3,425 |
| "excluded above T" cutting an injected emitter, shell / thin | 2 / 1,986, 0 / 4,005 |
| deficit_shell recall on "none", by ⟨δ⟩ 10⁻³–10⁻², 10⁻²–0.03, 0.03–0.1, 0.1–0.3 | 3%, 26%, 67%, 87% |
| deficit_thin recall on "none" | 0 / 1,995 (as §1 predicts) |
| real DPAC YSO / ECL stars with an excess | 10.1% / 1.3% |

Reading:

- The excess recall on injected disks is flat in f and in T: the ~11% misses at G < 16 are hosts whose WISE bands
  are all flagged as contaminated (diffraction spikes, persistence, halos) or blended, which by design count only as
  upper bounds. At G > 18 recall falls to ~40%, where W2 drops below SNR 5 and 2MASS K runs out.
- Only one DPAC "YSO" star in ten shows an excess, and raw WISE colours agree (W1−W2 > 0.3 in 11%). DPAC's DR3
  YSO class is selected from Gaia light curves alone and is mostly not disk-bearing, so it is not a clean positive
  control. Its excess rate is still 8× the EB rate.
- The deficit test does what §1 says: an envelope or swarm big enough to dim a star by ≳ 3% on average is
  excluded (or found) for most stars; an orbit-confined occulter never is.

### Half B (confirmation)

Seed 18161026498280999523, photosphere fitted on half A's reference stars, thresholds from half A.

| criterion | half B | |
|---|---|---|
| B1 false excess, null | 47 / 4,500 = 1.04% (Wilson 0.79–1.39%) | pass |
| B2 false deficit, pooled | 6 / 5,007 = 0.12% (shell 6/1,670, thin 0/3,337) | pass |
| B3 "excluded above T" cutting a shell emitter | 10 / 1,983 = 0.50% | pass |
| B4 excess recall yso_disk, rcb_shell | 82.8%, 89.5% | pass |
| B5 cold family flagged ≤ B1 + 1 point (2.04%) | 2.8% | **fail** |

Reported without criteria: warm_structure 54.9%, debris 3.8%; deficit_shell recall on "none" 3.8%, 31%, 67%, 93%
for ⟨δ⟩ 10⁻³–10⁻², 10⁻²–0.03, 0.03–0.1, 0.1–0.3; deficit_thin 1/2,064; DPAC YSO 9.6%, ECL 1.2% with an excess.

**Why B5 failed.** The criterion assumed the cold family (20–50 K, f up to 0.3) is invisible to WISE "by
construction". It is not at its top end: 0.3 of L* at 48 K is several photospheres in W4 around a G ≈ 11 star.
The 80 cold injections newly flagged (beyond the host's own null flag) have median T 47 K, f 0.15, an injected W4
excess of 4.8 photospheres, G 10.9. Binned by the injected W4 excess, the flag rate is 1.2% below 0.1 photospheres,
1.2% at 0.1–0.3, 1.1% at 0.3–1 (the null rate) and 8%, 17%, 25% above 1, 3 and 10. The test found real excesses;
the criterion was wrong about the physics. Under the rule of §5 this is still a failure, so the fix is
pre-registered in §8 and tested on fresh stars before G5 is adopted.

### Half C (B5', fresh stars)

Seed 7430908940562924249; 64,370 + 28,316 fresh reference stars (random_index slices disjoint from A and B), photosphere
fitted on all of A and B, half-A thresholds.

| | half C | |
|---|---|---|
| B5' injections below 0.3 photospheres in every band, flagged as excess | 47 / 6,547 = 0.72% (cold family alone 20 / 3,527 = 0.57%) | pass (null + 1 point = 1.78%) |
| B1 false excess, null | 35 / 4,500 = 0.78% | (as B) |
| B2 false deficit, pooled | 8 / 5,018 = 0.16% (shell 8/1,654, thin 0/3,364) | (as B) |
| B3 "excluded above T" cutting a shell emitter | 11 / 1,990 = 0.55% | (as B) |
| B4 yso_disk, rcb_shell | 82.3%, 88.8% | (as B) |

**G5 is adopted** (§5, §8) as a ranking column and packet context, with Z_EXCESS = 5.0 and Z_DEFICIT = 2.0. It
never shortlists on its own; a rule that uses it needs its own pre-registered blind batch.

### Blind run on the rehearsal candidates (§6)

Scored with the photosphere model fitted on all reference stars and the half-A thresholds. 3,256 of the 4,002 have
an AllWISE best neighbour in DPAC's table (the rest are scored NaN).

1. **Excess rate**: 130 / 3,078 candidates in G × BP−RP cells the null hosts cover (95% of the candidates' weight)
   have an excess, 4.2%, against 0.66% for the reweighted null hosts: one-sided binomial p = 7 × 10⁻⁶⁰. Dimmers carry
   infrared excesses six times as often as stars like them.
2. **By DPAC class**: LPV 66 / 211 (31%; dusty mass loss), the one RCB and the one symbiotic star, ECL 44 / 1,709 (2.6%),
   SOLAR_LIKE/RS 1 / 345, YSO 0 / 13, unclassified 24 / 1,604 (1.5%), remaining classes 2 / 117.
3. **Deficits**: deficit_shell 510, deficit_thin 0. deficit_shell falls mostly on eclipsing binaries (304 ECL,
   171 unclassified, 28 other, 4 YSO, 3 SOLAR_LIKE/RS).
4. **Leads (§6.4)**: the 171 unclassified deficit_shell stars are listed in
   `/mnt/project-files/energy/candidates_g5.parquet`. Read as §6 says: none of them is dimmed by warm dust spread around
   it. Their tier-1 light curves (4–12 episodes, depths 10–99%) look like eclipses, and an eclipsing companion is the
   textbook dimmer without waste heat, so they are not anomalies on G5's evidence.

What the blind run says about G5 itself, one step back: **the informative direction is the excess, not the
deficit.** A deficit_shell flag is what any compact occulter produces, and compact occulters (stars, mostly)
dominate dimmers, so it cannot promote anything on its own. An excess identifies the dust explanations: LPVs, R CrB,
symbiotics, and 24 unclassified dimmers and 44 eclipsers with warm (100–1000 K) excesses, which are the natural
candidates for the rare "disk_occultation" and "debris_transit" classes (ε Aur-like disks, collisional debris). The
useful pairing for technosignature ranking is deficit_shell with the EB filter and residual search saying "not an
eclipse": an aperiodic dimmer that is neither eclipse-like nor dust-like.

## 8. Changes after pre-registration

- 2026-10-22, during half A (before half B or the candidates were run): the first half-A pass (seed not kept)
  missed 20% of injected hot disks. Hot dust brightens K, W1 and W2 by similar factors, which the fit absorbed into
  the photosphere's scale because only 2MASS K, with the star's variability in its error, pinned that scale.
  Gaia G now anchors it too (G−W1 locus from the general sample, held-out scatter 0.065 mag, the same as K−W1;
  its error also carries the variability term). Half A was rerun with a fresh seed; the numbers above are that run.
  The change raised yso_disk recall only from 80% to 82%, because most remaining misses are contaminated photometry,
  found when checking the misses one by one.
- 2026-10-22, after half B (before any half-C star was scored): **B5 is replaced by B5'**, for the reason given
  in §7. B5': among family injections whose largest injected excess in any band is below 0.3 photospheres
  (`cal.INVISIBLE`; computed from the injected components and the host's photosphere), the excess flag rate is at
  most the null rate + 1 point, both measured on the same stars. It is tested on **half C**: fresh stars from
  disjoint random_index slices (`reference.SAMPLES` general_c, mid_c), scored with the photosphere fitted on all of
  A and B, a fresh os.urandom seed and the half-A thresholds. B1–B4 are re-reported on C for completeness but were
  already decided on B. G5 is adopted as a ranking column and packet context if B5' passes on C.

## 9. Running it

```
python -m cosmos_search.energy reference --out ref/            # reference samples (Gaia archive, ~10 min)
python -m cosmos_search.energy candidates --ref ref/ --ranked ranked.parquet --out g5.parquet [--plan decoys.json.age]
python -m cosmos_search.dayone.vet build ... --g5 g5.parquet   # packets carry the energy-budget context
```

`candidates` fetches K and W1–W4 through the Gaia archive's best-neighbour tables (`reference.fetch_ids`; DR4 will need
its own neighbour tables or `photometry.from_context` on the xmatch context table). With `--plan`, sealed decoys whose
injected story is dust get a dust excess (`inject.DECOY_IR`), so G5 cannot tell decoys from dust-like reals by their
infrared alone. ⟨δ⟩ comes from the ranked table's depth and episode count (§6); with light curves at hand,
`dimming.mean_deficit` is the better estimate.

DR4 to-dos: per-star ⟨δ⟩ from the DR4 light curves; NEOWISE epochs during a Gaia dip (2013–2024 overlaps DR4 fully)
would test grain size directly, since dust ≲ 1 μm dims W1 far less than G while bodies are grey from 0.6 to 4.6 μm;
unWISE/CatWISE W1–W2 go ~1 mag deeper than AllWISE for faint hosts, where recall drops (G > 18: ~40%).
