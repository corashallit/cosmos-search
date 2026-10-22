# Energy budget (G5): where did the blocked light go?

Status: **pre-registration.** The commit that adds this file fixes the threshold rule, the half-B pass criteria and
the blind run on the rehearsal candidates (§5–§6) before any half-A, half-B or candidate result was looked at.
Results go in §7; changes after this commit are listed in §8 with their reason and date.

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

(After the runs.)

## 8. Changes after pre-registration

(None yet.)
