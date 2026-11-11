# How common steady fades can be

Code: `cosmos_search/slow/occurrence.py` (the limit), `cosmos_search/slow/recall.py` (injected fades),
`scripts/slow_occurrence.py` (the steps). Frozen settings: `prereg/occurrence/steady-fades-v1.json`. Built
2026-11-10, item 4 of the big-picture review; joined to the bulk-file pass on 2026-11-11.

The slow-change search (docs/arms/slow-changes.md) may end without a find. One measured result comes out of it
either way: of the Sun-like dwarfs Gaia watched, at most 1 in N is fading steadily at a given rate. This page
covers how that number is computed, what was checked about it on DR3, and how to run it on DR4.

## What is reported

For each fade rate from 0.1% to 10% of the flux per year, an upper limit at 95% confidence on the share of FGK
dwarfs fading at that rate.

- **A steady fade**: the flux falls each year by a fixed fraction of its level at the first visit, by the same
  fraction in G, BP and RP, for the whole length of the data. At 1% a year a star ends DR4's 5.5 years 5.5%
  fainter.
- **FGK dwarfs**: parallax known to 20% or better, 0.5 ≤ BP−RP < 1.8, and 0.5 ≤ M_G − 3.2 (BP−RP) < 5.5, which
  is the band between the giants and the white dwarfs; G < 19 and at least 8 visits. Colours are as observed, with
  no correction for extinction, so in reddened fields the sample leans to nearby stars.

## How it is computed

1. **Trend of every star.** The search's own G trend and its error, from the per-star table the pass writes for
   every star (`slope`, `slope_err`; docs/arms/slow-changes.md, "Every star scored"), measured against stars
   like it in two steps. On DR3 the drift correction leaves +0.08% a year at G 12-13, which would push every star
   there towards brightening and hide fades. First the search's own reference: the median trend of all stars in
   the same sky pixel and 0.5 mag of G (`summary.apply_reference` with the flag step's `reference.parquet`).
   What that leaves depends on colour ("The leftover trend depends on colour" below), so the searched dwarfs'
   own median at each 0.5 mag of G is then taken out as well (`occurrence.centre`; `limit.json` lists it under
   `population_offset`). A per-star table without the search's reference is measured against the running median
   of the 401 searched stars nearest in G, which is both steps in one.
2. **Recall.** An FGK dwarf's own light curve is given the fade and scored again by the function the pass runs
   on every star (`summary.star_summary`, with the pass's bad-epoch test), at each rate
   (`recall.injected_scores`), and its trend goes through the same two steps as a real star's. The fade is
   `inject.lightcurve.SecularTrend` from the star's first epoch, in all three bands, with the errors scaled with
   the flux. The star keeps its noise, its gaps and its own variability, so recall is measured on the stars the
   limit is about. The hosts are every FGK dwarf of a sample, or a random share of the stars of a whole-sky run
   (the pass keeps their light curves). Nothing is judged by a reader, so nothing is blinded.
3. **Regions.** A region is the stars brighter than some G (13, 14, ..., 19) whose trend towards fading is at
   least z times its error (z = 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100): 112 regions. Below,
   "beyond 6 errors" means a trend of at least 6 times its error.
4. **The limit in a region.** n stars are in it. A fade at the rate in question would have put
   `detectable` = Σ over G ranges of (stars × recall) stars in it. Every one of the n is counted as a possible
   real fade:

   share fading < U(n) / detectable,

   with U the Poisson 95% upper limit (3.0 for n = 0, 4.7 for 1, 6.3 for 2). Stars that fade for other reasons
   (spots, eclipses, the instrument) make the limit weaker and cannot make it wrong.
5. **Which region.** For each rate, the region with the smallest U(m) / detectable, where m is the number of
   stars *brightening* past the same cut: the limit the region would give if as many stars faded there as
   brighten. The fading counts are not looked at when choosing, so choosing does not reduce the limit's
   coverage. Fast fades are measured far out in significance, where no star is; slow ones near the threshold
   among bright stars. Choosing on counts instead of on the true numbers of chance stars costs 4-21% in simulated
   counts.

## Why chance fades are not subtracted

The usual next step would be to subtract the stars that fade by chance, estimated from the stars that brighten
(the mirror null of docs/statistics.md §2), and so reach slower fades. The code computes that number
(`mirror_limit`, with its own region) and reports it beside the limit. It is not the limit, for two measured
reasons.

**On DR3, fading and brightening are not equally common by chance.** Among the 195,572 FGK dwarfs of the
Andromeda field (below), by trend significance after both steps of the reference:

| Trend beyond (errors) | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| Fading | 1,835 | 504 | 171 | 61 | 12 | 3 | 1 | 1 | 0 |
| Brightening | 1,662 | 452 | 185 | 89 | 28 | 14 | 6 | 2 | 0 |

The two sides differ, and in opposite directions at the two ends. Between 3 and 5 errors more stars fade, 1,664
to 1,477, most of the difference at G 15-17; a fair coin gives a split that uneven 0.05% of the time. Beyond 6
errors more stars brighten, 89 to 61, and beyond 8 errors 28 to 12 (1.4% and 0.8%). The 42 of the 150 stars
beyond 6 errors that DR3 classifies are all spotted, active stars (23 RS CVn type, 19 solar-like), and nothing
requires spots to appear and to fade at the same pace. With 28 brightening stars subtracted from 12 fading ones,
the conclusion would be that no star there can be a real fade, drawn from stars that are doing something else;
between 3 and 5 errors the same subtraction would count 187 stars as real fades. No cause was looked for.
Activity cycles are one candidate, since they are lopsided in time (the Sun's rise takes about 4 years and its
decline about 7).

The first run (2026-11-10, with DR3's rejected-epoch marks and a reference by brightness alone) gave the same
picture far out (85 to 55 beyond 6 errors, 30 to 10 beyond 8) and even counts near the threshold (1,701 to
1,637 beyond 3), so the excess of fading stars near the threshold moves with how the trends are measured and the
excess of brightening stars far out does not.

Directions used to come in groups on the sky. Before the reference, the median trend of the dwarfs in a
0.84 deg² sky pixel differs from pixel to pixel by 0.13 errors more than sampling allows; the search's
reference is made per sky pixel and leaves none of that. Among the 2,736 stars past 3 errors at G < 17 the
fading counts now scatter between pixels 1.11 times as much as independent stars would (chance 19%). Over all of
G < 19 the factor is 1.33 (chance 1.2%), so at the faint end some grouping is left. A subtraction that treats
stars as independent understates its own error wherever that factor is above 1.

**A subtracted limit is wrong when they are not.** Simulated counts in the 112 regions with a known share of
real fades, 1,500 samples for each row. The last four columns give how often each number fell below the true
share (a 95% limit may do so 5% of the time at most) and how far above the truth it sat in the median.

| Where the fade puts bright stars | Chance brightenings past 3 errors | Real fades | Chance fades per chance brightening | Limit below truth | Subtracted below truth | Limit / truth | Subtracted / truth |
|---|---|---|---|---|---|---|---|
| 60 errors | 0 | 20 | | 3.9% | 5.9% | 1.5 | 1.4 |
| 60 errors | 500 | 12 | 1 | 0.8% | 3.7% | 2.3 | 1.9 |
| 60 errors | 500 | 12 | 0.6 | 1.0% | 4.5% | 2.3 | 1.8 |
| 18 errors | 10,000 | 80 | 1 | 0.2% | 4.2% | 2.4 | 1.7 |
| 18 errors | 10,000 | 80 | 0.6 | 0.6% | 20.7% | 2.2 | 1.3 |
| 18 errors | 10,000 | 80 | 1.5 | 0.1% | 0.7% | 2.7 | 2.2 |
| 6 errors | 10,000 | 300 | 1 | 0.0% | 5.6% | 2.5 | 1.4 |
| 6 errors | 10,000 | 300 | 0.6 | 0.3% | 99.9% | 2.1 | 0.2 |
| 6 errors | 10,000 | 300 | 1.5 | 0.0% | 0.0% | 2.8 | 3.1 |
| 6 errors | 200,000 | 1,500 | 1 | 0.0% | 4.6% | 3.7 | 1.3 |
| 6 errors | 200,000 | 1,500 | 0.6 | 0.0% | 100% | 2.8 | 0.03 |
| 6 errors | 200,000 | 1,500 | 1.5 | 0.0% | 0.0% | 4.9 | 6.4 |

With symmetry the subtracted number is 1.2 to 2.8 times stronger, most where chance stars are plentiful. Without
it, with fewer chance fades than chance brightenings, it is below the truth in a fifth of the samples to all of
them, and more often the more stars there are, which is the direction DR4 goes. The limit holds in every row.
(These counts are simulated, so the table does not depend on how the real stars are scored.)

The subtracted number still has a use, as an alarm. With it comes the chance of seeing at least as many fading
stars among those past the cut if each were as likely to fade as to brighten (`excess_p`; its region is chosen
on fading and brightening stars together, so the test is exact under symmetry). A small value means more stars
fade than brighten at that rate. With symmetry imperfect, a small value is a reason to read those stars
(`past_cut.parquet` lists them) and falls short of a detection. `sky_dispersion` measures whether stars past
the cut go the same way in the same patch of sky more often than independent stars would; both the subtraction
and the test treat stars as independent.

## Checks

**Coverage in simulated counts.** The table above, and `tests/test_slow_occurrence.py`, which measures the
choice of region and the limit together on every run of the tests, with chance fades at 0.6, 1 and 1.5 times the
chance brightenings.

**Fades planted in real stars.** `slow_occurrence.py planted` gives a random share of the real stars their own
injected scores, so that share really fades, computes the limit, and repeats. On the 195,572 dwarfs of the DR3
field, 100 draws for each case, with shares between half and four times the limit the real stars give at that
rate:

| Fade rate (% a year) | Share planted | Limit below the planted share | Subtracted number below it |
|---|---|---|---|
| 0.1 | 4% | 0 of 100 | 0 |
| 0.1 | 16% | 0 | 100 |
| 0.2 | 0.5% | 0 | 0 |
| 0.2 | 2% | 0 | 0 |
| 0.5 | 0.03% | 0 | 8 |
| 0.5 | 0.12% | 4 | 23 |
| 2 | 0.003% | 0 | 1 |
| 2 | 0.012% | 0 | 1 |
| 10 | 0.001% | 0 | 0 |
| 10 | 0.004% | 0 | 1 |

The limit fell below the planted share in 4 of 1,000 draws, all in one case: 0.5% a year with 0.12% planted, 4
of 100, where a 95% limit may do so in up to 5. The region chosen at that rate holds no star before the
planting, so the limit there is the plain Poisson one and nothing ordinary adds to the count; in the other
cases the stars that fade for ordinary reasons keep the limit above the truth every time. The subtracted number
fell below the planted share in up to 100 of 100: at 0.5% a year, where its region holds more brightening than
fading stars (8 fading, 9 brightening), and at 0.1% a year with 16% planted, a share large enough to move the
dwarfs' own median. The check takes that median out again after the planting, as the limit step would on such a
sky, while recall still comes from fades injected into one star at a time. The first run's scoring gave the
same pattern (6 and 100 of 100 at 0.1% a year, 13 and 34 at 0.5%).

**The whole-sky route on the field.** The every-star run on DR4 takes the catalogue in sky slices and injects
only into the random share of stars whose light curves the pass keeps ("On DR4"). Run that way on the DR3 field,
with a random 5% of stars kept (9,650 hosts in place of 195,572): the limits are within 10% of the table below
at every rate, and with the same catalogue rows from slices as from one file the table is reproduced exactly.

## DR3: the Andromeda field

DR3 has light curves for stars not flagged variable in one place only, the 5.5° around M31 (Gaia's Andromeda
photometric survey). `slow_occurrence.py field` fetched every star there with G < 19 from the AIP mirror:
529,976 stars (153 level-6 sky pixels, four queries at once, 72 minutes). They were scored by the search's pass
as DR4 will arrive, with DR3's rejected-epoch marks ignored and our own bad-epoch test on (`slow_search.py pass
--no-marks --no-period`, then `flag`), with the thresholds and drift table made from this field's own quiet stars
(`/mnt/project-files/score-every-star/rules/rule_final`). 195,572 are FGK dwarfs by the definition above, with a
median of 20 visits over 2.74 years: half of DR4's span and about a fifth of its visits. The median trend error
is 0.07% a year at G 13-16, 0.10% at G 16-17, 0.15% at G 17-18 and 0.24% at G 18-19.

| Fade rate (% a year) | Fade over the 2.74 years | Region: G below, errors beyond | Stars | Recall | Fading there (brightening) | Fewer than 1 in |
|---|---|---|---|---|---|---|
| 0.1 | 0.27% | 16, 3 | 89,328 | 11.1% | 873 (820) | 11 |
| 0.2 | 0.55% | 15, 5 | 44,861 | 13.0% | 45 (57) | 101 |
| 0.3 | 0.82% | 15, 8 | 44,861 | 8.0% | 3 (4) | 463 |
| 0.5 | 1.4% | 15, 10 | 44,861 | 22.1% | 0 (0) | 3,314 |
| 1 | 2.7% | 16, 15 | 89,328 | 41.8% | 1 (0) | 7,879 |
| 2 | 5.5% | 19, 20 | 195,572 | 48.9% | 0 (0) | 31,938 |
| 3 | 8.2% | 19, 20 | 195,572 | 72.9% | 0 (0) | 47,565 |
| 5 | 13.7% | 19, 20 | 195,572 | 89.8% | 0 (0) | 58,623 |
| 10 | 27.4% | 19, 20 | 195,572 | 96.8% | 0 (0) | 63,215 |

Read as: of FGK dwarfs like these, fewer than 1 in 7,879 was fading at 1% a year between mid-2014 and
mid-2017. Recall is the share of the region's stars that would be past its cut with the fade. From 0.5% a year
up, except at 1%, no star is in the chosen region and the limit is the plain 3.0 / (stars × recall); at the slow
rates it is set by stars that fade for ordinary reasons.

The one star past the cut at 1% a year is Gaia DR3 388933104249558784 (G 15.3, BP−RP 1.56, 385 pc), fading at
6% a year, 18.5 errors; DR3 flags it variable. It was not looked at further here. The strongest trend in the
field is a brightening one (19.8 errors).

With chance fades subtracted the numbers would be 1 in 54, 234, 676, 4,052, 19,792, 32,960, 47,565, 58,622 and
63,214: 1.2 to 5 times stronger up to 1% a year and about the same from 2%. `excess_p` is 0.003 at 0.1-0.2% a year
(1,441 fading against 1,295 brightening at G < 17 beyond 3 errors) and 0.04 at 0.3% (422 against 373 beyond 4
errors); at every other rate it is above 0.6. That is the excess of fading stars near the threshold described
under "Why chance fades are not subtracted". At DR3's trend errors those stars fade by 0.2 to 0.5% a year; they
are listed in `past_cut.parquet` and were not read.

The first run of 2026-11-10 (the earlier scoring step, DR3's rejected-epoch marks, a reference by brightness
alone) gave 1 in 12, 110, 483, 1,627, 7,545, 15,941, 34,584, 51,304 and 61,511. The differences at 0.5% and 2% a
year are one or two stars in or out of the chosen region; recall at the fast rates is higher now (96.8% against
94.2% at 10% a year).

What DR3 cannot show: DR4 has twice the span and about five times the visits, so a given rate stands several
times further out in significance, and DR4 covers the whole sky.

Outputs: `/mnt/project-files/occurrence/dr3-m31/` (the limit's files and figure, the pass's per-star table and
reference, the injected scores, the stars' `gaia_source` rows, the planted-fade table, the trend by colour).

### The leftover trend depends on colour

Measured on 2026-11-11 on every star of the field with a measured trend. Median G trend after the drift
correction, in % of the flux per year, by magnitude and BP−RP:

| G | < 0.5 | 0.5-0.8 | 0.8-1.0 | 1.0-1.2 | 1.2-1.4 | 1.4-1.6 | 1.6-1.8 | 1.8-2.0 | 2.0-2.2 | 2.2-2.6 | 2.6-3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13-15 | +0.14 | +0.02 | +0.01 | +0.01 | +0.01 | 0.00 | +0.01 | +0.03 | +0.06 | +0.09 | +0.09 |
| 15-16 | +0.09 | +0.01 | +0.01 | 0.00 | 0.00 | 0.00 | +0.02 | +0.03 | +0.06 | +0.11 | +0.10 |
| 16-17 | +0.12 | +0.01 | +0.01 | +0.01 | 0.00 | −0.01 | 0.00 | +0.03 | +0.06 | +0.11 | +0.17 |
| 17-18 | +0.09 | +0.01 | +0.01 | +0.01 | 0.00 | −0.02 | −0.01 | +0.04 | +0.07 | +0.13 | +0.21 |
| 18-19 | 0.00 | +0.01 | +0.01 | +0.03 | +0.01 | −0.04 | −0.01 | +0.03 | +0.09 | +0.16 | +0.22 |

The standard errors are 0.001 to 0.02, and up to 0.07 in the two end columns (`colour_trend.csv` in the outputs
has them, with the star counts). Stars between BP−RP 0.5 and 1.8 sit within
0.04% a year of zero. Redder stars brighten by up to 0.2% a year and the bluest by about 0.1%. The drift
correction has three colour ranges (BP−RP below 0.8, 0.8 to 1.3, above 1.3), so inside the red one it takes out
the average and leaves the slope. The search's reference is by brightness and sky only, so where stars redder
than BP−RP 1.8 are many (16% of the stars at G 17-18, 30% at G 18-19) they pull its median up, and the FGK dwarfs
sit below it: by 0.10 errors at G 17-18 and 0.29 at G 18-19, with 1,713 against 1,116 and 300 against 100 stars between 2 and 3 errors either
way. The limit does not need the two sides to match. Its region is chosen where few stars brighten, though, and
an offset of that kind makes a region look empty on the brightening side while it is full on the fading side.
Taking the searched dwarfs' own median out per 0.5 mag of G (step 1) puts the medians at zero and the counts
between 2 and 3 errors at 1,418 against 1,318 and 185 against 190.

For the search's own lists this is a separate matter, raised with the bulk-file scoring work: if DR4's
calibration leaves the same, its trend errors, about 2.7 times smaller, put stars redder than BP−RP 2.2 near +1
error and the bluest stars higher still.

## After the list is read

The limit above uses no reading of light curves. Once the stars of a region have been through the checks and the
review (docs/arms/slow-changes.md S5, docs/vetting.md), the stars that something explains no longer count:

share fading < U(stars left unexplained) / (detectable × share of real fades the review keeps),

`occurrence.reviewed_limit`. The review's recall on anomalous fakes was 44 of 50 (docs/vetting.md §14). With
nothing left unexplained in a region, its limit improves by U(n) / 3.0, a large factor at the slowest rates. The
checks took 7.5 hours of archive calls for 10³ stars in the DR3 trial (docs/arms/slow-changes.md), so regions of
up to about that size can be read.

## What the limit does not say

- It is about fades that are steady for the whole of the data, linear in flux and the same in all three bands.
  A fade that starts part-way, speeds up, or reddens the star has a different recall, not measured here.
- A fade shared by every star of similar brightness is removed with the reference and cannot be seen. One
  shared by several percent of them moves the reference a little, so each is found slightly less often than
  injections into one star at a time measure; the planted check above includes shares of 4% and 16%.
- It is about stars with a parallax good to 20%. Distant and faint dwarfs without one are not in the sample.
- Unresolved binaries count as one star.
- The injected fade is added to the published fluxes. A real fade large enough to change how Gaia observes the
  star (a different window or gate, near G = 13 and G = 16) would not look exactly like that; at the rates
  here the change over the data is under a magnitude at 10% a year and a few hundredths at 1%.
- An injected star is counted in the G range of the catalogue magnitude of the star it was injected into. A real
  fader's would be fainter, by up to 0.15 mag at 10% a year over DR3's span, which moves a few stars across the
  edge of a G range. (Its reference and its magnitude and colour cell for the drift correction do follow its own
  faded light curve, as a real fader's would.)
- Recall is measured on all the stars searched, or on a sample of them. A sample must be a fair one: the same
  mix of brightness, number of visits and sky as the stars the counts come from. Recall is applied per G range.

## On DR4

Settings are frozen in `prereg/occurrence/steady-fades-v1.json` (rates, cuts, G ranges, the definition of an FGK
dwarf, confidence level); `test_frozen_config_is_the_code_default` keeps the code's defaults equal to it.

**Day one, on the threshold sample.** Step 2 of "Release day, end to end" (docs/arms/slow-changes.md) leaves the
light curves of a random 0.2% of the bulk files, whole patches of sky, in `S/lightcurves`. As soon as the rule
directory `RULE` exists:

    python scripts/slow_search.py pass --out S/run --lightcurves S/lightcurves --rule RULE --no-period --keep-lightcurves none
    python scripts/slow_search.py flag --run S/run --rule RULE --out S/lists
    python scripts/slow_occurrence.py aux --lightcurves S/lightcurves --schema gaiadr4 --out S/aux_plx.parquet
    python scripts/slow_occurrence.py inject --lightcurves S/lightcurves --aux S/aux_plx.parquet --rule RULE --out O
    python scripts/slow_occurrence.py limit --stars S/run/summary --reference S/lists/reference.parquet \
        --aux S/aux_plx.parquet --injections O/injections --out O/limit
    python scripts/slow_occurrence.py planted --stars S/run/summary --reference S/lists/reference.parquet \
        --aux S/aux_plx.parquet --injections O/injections --out O/planted

`aux` fetches parallaxes, which no light curve carries (uploaded id lists to ESA's archive, or `--mirror` by
source_id range). `limit` writes `limit.json` (settings, inputs, the table, the symmetry tables, the dwarfs' own
median per G range), `curve.csv`, `regions.parquet` (every region's counts and recall at every rate),
`past_cut.parquet` (every star past the first cut either way) and the figure. This limit is a first number and a
check that everything runs on DR4.

**Every star.** The number worth reporting comes from the whole-sky pass and its lists (steps 3 and 4 of
"Release day, end to end": `R` and `L`):

    python scripts/slow_occurrence.py aux --sky 4 --schema gaiadr4 --out A
    python scripts/slow_occurrence.py inject --lightcurves R/lightcurves --random-share 0.001 --aux A --rule RULE --out O
    python scripts/slow_occurrence.py limit --stars R/summary --reference L/reference.parquet --aux A \
        --injections O/injections --out O/limit

- `aux --sky 4` asks the archive for the FGK-dwarf candidates of each of the 3,072 level-4 HEALPix pixels, one
  query and one file each, with the magnitude, colour and parallax cuts in the query, so a few 10⁸ rows come
  back and not 2 × 10⁹. A second run resumes; `--pixels FIRST-LAST` splits the work. On DR3 a slice of 20,762
  rows took 5 s at ESA with the project's login and 15 to 65 s at the AIP mirror, with the same rows from both.
  `inject` and `limit` read only the slices a file of stars falls in.
- `inject --random-share 0.001` uses as hosts only the pass's random share of stars, whose light curves it keeps
  in `R/lightcurves` beside those of stars near a threshold (which are not a fair sample). That is of the order
  of 10⁵ FGK dwarfs over the sky; with a few thousand per G range recall is known to a few percent. Give
  `inject` the pass's own `--no-marks` and `--no-own-test` if the pass was run with them.
- `limit` reads the per-star table one file at a time, twice (once for the dwarfs' own median per G range), with
  the catalogue slices that file needs, so neither table has to fit in memory. Rows the pass marks as not a fair
  sample of their magnitude (`ref` false) are left out, and `occurrence.tally` gives counts that add up over
  files.

On the DR3 field the pass scored 22,000 stars a second on 4 cores without the period search, the 195,572 dwarfs
were injected at 9 rates in 0.8 minutes, and `limit` took 11 s; DR4's light curves are about five times longer.

Not tried at full size: the catalogue query for all 3,072 slices (14 were run), and a per-star table larger
than the field's. `planted` holds the searched stars in memory, so on DR4 it runs on the day-one sample or on a
part of `R/summary`.

The limit improves in proportion to the number of stars only while no star is past the cut of the chosen
region. With 10⁸ dwarfs some ordinary variable will be past any cut; from there the limit improves by moving to
stricter cuts, which is why they go up to 100 errors, or by reading the list.

To read on the day: the table; whether `excess_p` is small at any rate, and if so the stars behind it; the
symmetry tables and the dwarfs' own median per G range (`population_offset`: on DR3 it is under 0.01% a year
brighter than G 17 and −0.07% at G 18-18.5); `planted`.

## Log

- **2026-11-10.** Built, with the DR3 run above. Three designs were tried and measured before this one. A
  likelihood over all bins with a background per bin: its answer moved by a factor of 2-3 with the prior put on
  the backgrounds (13, 20 and 41 detected stars for flat, Jeffreys and 1/b priors on the same counts). One region
  with chance fades subtracted, chosen on the brightening counts: 8% of no-fade samples called an excess at the
  5% level. The same with the region chosen on fading and brightening together: sound under symmetry, and below
  the truth without it (tables above). The grid of cuts was made denser (12, 20, 30, 50, 80 added) after the
  first DR3 run and before freezing; the DR3 table did not change with it.
- **2026-11-11.** Joined to the bulk-file pass (PR #89), which replaced the scoring step this was built on.
  `limit` and `planted` read the pass's per-star table with the flag step's reference; injected stars are
  scored by the pass's own function; the catalogue for the whole sky comes in sky slices; hosts can be the
  pass's random share. The DR3 field was run again as DR4 will arrive (tables above; the first run's numbers are
  kept beside them). Two things changed with the measurement, neither in the frozen settings: the dwarfs sit
  below a reference made from all stars, because the leftover trend depends on colour, so their own median is
  taken out; and near the threshold more dwarfs now fade than brighten, where the first run had even counts.

