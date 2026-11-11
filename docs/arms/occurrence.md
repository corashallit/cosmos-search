# How common steady fades can be

Code: `cosmos_search/slow/occurrence.py` (the limit), `cosmos_search/slow/recall.py` (injected fades),
`scripts/slow_occurrence.py` (the steps). Frozen settings: `prereg/occurrence/steady-fades-v2.json`. Built
2026-11-10, item 4 of the big-picture review; joined to the bulk-file pass on 2026-11-11, and the same day
stars whose light is mixed with a neighbour's were taken out of the searched population (settings v2).

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
- **Not searched: stars whose light is mixed with a neighbour's.** A dwarf whose (BP + RP) / G flux ratio is
  more than 0.05 from the dwarfs' own median at its colour is left out: 3.6% of the dwarfs on DR3. Gaia's G
  photometry of such stars drifts ("Stars mixed with a neighbour" below). This came with settings v2.

## How it is computed

1. **Trend of every star.** The search's own G trend and its error, from the per-star table the pass writes for
   every star (`slope`, `slope_err`; docs/arms/slow-changes.md, "Every star scored"), measured against stars
   like it in two steps. On DR3 the drift correction leaves +0.08% a year at G 12-13, which would push every star
   there towards brightening and hide fades. First the search's own reference: the median trend of all stars in
   the same sky pixel and 0.5 mag of G, and of the run's stars of the same colour (`summary.apply_reference`
   with the flag step's `reference.parquet`; the colour term came with PR #93, after "The leftover trend depends
   on colour" below). Since PR #98 that reference also has a part by flux excess, made from stars of every
   kind; the limit leaves that part out, because the next step does the same among the dwarfs themselves.
   Then the searched dwarfs' own median is taken out as well (`occurrence.centre`;
   `limit.json` lists it under `population_offset`), because the dwarfs are one kind of star and the reference
   is made from all kinds. The median is taken among dwarfs of the same 0.5 mag of G and the same step of flux
   excess (11 steps inside the cut), since the G trend follows the excess all the way through zero. On DR3 that
   last step moves a trend by under 0.01% a year for a star brighter than G 17 whose flux ratio is within 0.006
   of the median, and by up to 0.1% a year at the two ends of the kept range. A per-star table without the
   search's reference is measured against the running median of the 401 searched stars nearest in G, which is
   the first two steps in one.
2. **Recall.** An FGK dwarf's own light curve is given the fade and scored again by the function the pass runs
   on every star (`summary.star_summary`, with the pass's bad-epoch test), at each rate
   (`recall.injected_scores`), and its trend goes through the same two steps as a real star's, with its
   host's flux excess: a fade shared by the three bands leaves the ratio of their mean fluxes as it was. The fade is
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

**On DR3, fading and brightening are not equally common by chance.** Among the 188,485 searched dwarfs of the
Andromeda field (below), by trend significance after both steps of the reference:

| Trend beyond (errors) | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| Fading | 1,524 | 432 | 158 | 55 | 12 | 3 | 1 | 1 | 0 |
| Brightening | 1,594 | 457 | 183 | 80 | 26 | 13 | 6 | 2 | 0 |

Near the threshold the two sides agree: between 3 and 5 errors 1,366 stars fade and 1,411 brighten, and a fair
coin gives a split at least that uneven 40% of the time. Far out they do not. Beyond 6 errors more stars
brighten, 80 to 55, beyond 8 errors 26 to 12 and beyond 10 errors 13 to 3 (3.9%, 3.4% and 2.1% of the time). With 26 brightening stars subtracted
from 12 fading ones, the conclusion would be that no star there can be a real fade, drawn from stars that are
doing something else. In the run before the colour term, the 42 of the 150 stars beyond 6 errors that DR3
classifies were all spotted, active stars (23 RS CVn type, 19 solar-like), and nothing requires spots to appear
and to fade at the same pace. No cause was looked for. Activity cycles are one candidate, since they are
lopsided in time (the Sun's rise takes about 4 years and its decline about 7).

The excess of brightening stars far out has been there however the trends were measured. The counts near the
threshold moved each time, and both moves that were traced had a cause in the instrument:

| Run | Fading / brightening between 3 and 5 errors | A fair coin is that uneven | Beyond 6 errors |
|---|---|---|---|
| First run, 2026-11-10 (earlier scoring step, DR3's rejected-epoch marks, reference by brightness alone) | 1,549 / 1,467 | 14% of the time | 55 / 85 |
| Bulk-file pass, reference by sky and brightness (PR #91) | 1,664 / 1,477 | 0.09% | 61 / 89 |
| With the colour term in the reference (PR #93) | 1,631 / 1,512 | 3.5% | 59 / 87 |
| With stars mixed with a neighbour left out and the rest measured by flux excess (settings v2) | 1,366 / 1,411 | 40% | 55 / 80 |

So the excess of fading stars near the threshold that the second run showed was the instrument twice over: a
trend that depends on colour, and a drift in the G photometry of stars with a neighbour. A subtraction would
have counted 187 of those stars as real fades.

Directions used to come in groups on the sky. Before the reference, the median trend of the dwarfs in a
0.84 deg² sky pixel differs from pixel to pixel by 0.13 errors more than sampling allows; the search's
reference is made per sky pixel and leaves none of that. Among the 2,510 stars past 3 errors at G < 17 the
fading counts now scatter between pixels 1.10 times as much as independent stars would (chance 23%). Over all of
G < 19 the factor is 1.41 (chance 0.2%), so at the faint end some grouping is left. A subtraction that treats
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
injected scores, so that share really fades, computes the limit, and repeats. On the 188,485 searched dwarfs of
the DR3 field, 100 draws for each case, with shares between one and four times the limit the real stars give at
that rate:

| Fade rate (% a year) | Share planted | Limit below the planted share | Subtracted number below it |
|---|---|---|---|
| 0.1 | 4% | 0 of 100 | 100 |
| 0.1 | 16% | 0 | 100 |
| 0.2 | 0.5% | 0 | 0 |
| 0.2 | 2% | 0 | 75 |
| 0.5 | 0.03% | 4 | 13 |
| 0.5 | 0.12% | 6 | 34 |
| 2 | 0.003% | 0 | 0 |
| 2 | 0.012% | 1 | 1 |
| 10 | 0.001% | 0 | 0 |
| 10 | 0.004% | 0 | 1 |

The limit fell below the planted share in 11 of 1,000 draws: 4 and 6 of 100 at 0.5% a year and 1 at 2%, where a
95% limit may do so in up to 5 (6 or more of 100 turn up by chance 38% of the time). The regions chosen at those
rates hold no star before the planting, so the limit there is the plain Poisson one and nothing ordinary adds
to the count; in the other cases the stars that fade for ordinary reasons keep the limit above the truth every
time. The subtracted number fell below the planted share in up to 100 of 100. At 0.5% a year its region holds
more brightening than fading stars (8 fading, 9 brightening). At 0.1% and 0.2% a year the planted shares are
large enough to move the dwarfs' own median, which the check takes out again after the planting, as the limit
step would on such a sky, while recall still comes from fades injected into one star at a time. With the two
sides now even near the threshold the subtracted number has no slack for that: it falls below a planted 4% at
0.1% a year in every draw, where the run before the colour term, with its excess of fading stars, never did
(that run: 0 and 100 of 100 at 0.1% a year, 8 and 23 at 0.5%; the limit 4 of 1,000).

**The whole-sky route on the field.** The every-star run on DR4 takes the catalogue in sky slices and injects
only into the random share of stars whose light curves the pass keeps ("On DR4"). Run that way on the DR3 field,
with a random 5% of stars kept (9,650 hosts in place of 195,572): the limits were within 10% of the full run's
at every rate, and with the same catalogue rows from slices as from one file its table was reproduced exactly
(tried on the run of PR #91; the table below takes its catalogue rows from slices).

## DR3: the Andromeda field

DR3 has light curves for stars not flagged variable in one place only, the 5.5° around M31 (Gaia's Andromeda
photometric survey). `slow_occurrence.py field` fetched every star there with G < 19 from the AIP mirror:
529,976 stars (153 level-6 sky pixels, four queries at once, 72 minutes). They were scored by the search's pass
as DR4 will arrive, with DR3's rejected-epoch marks ignored and our own bad-epoch test on (`slow_search.py pass
--no-marks --no-period`, then `flag`), with the thresholds and drift table made from this field's own quiet stars
(`/mnt/project-files/score-every-star/colour/rule`, the rule with the drift by colour of PR #93). The catalogue
rows came as on DR4, from ESA's archive in 14 level-4 sky slices (`aux --sky 4`: 475,224 rows in 0.3 minutes).
195,572 stars are FGK dwarfs by the definition above; 7,087 of them (3.6%) are mixed with a neighbour and
188,485 are searched, with a median of 20 visits over 2.74 years: half of DR4's span and about a fifth of its
visits. The median trend error is 0.07% a year at G 13-16, 0.10% at G 16-17, 0.15% at G 17-18 and 0.24% at
G 18-19.

| Fade rate (% a year) | Fade over the 2.74 years | Region: G below, errors beyond | Stars | Recall | Fading there (brightening) | Fewer than 1 in |
|---|---|---|---|---|---|---|
| 0.1 | 0.27% | 16, 3 | 86,846 | 11.1% | 779 (821) | 12 |
| 0.2 | 0.55% | 15, 5 | 43,601 | 13.0% | 41 (56) | 107 |
| 0.3 | 0.82% | 15, 8 | 43,601 | 8.0% | 3 (4) | 452 |
| 0.5 | 1.4% | 15, 10 | 43,601 | 22.8% | 0 (0) | 3,318 |
| 1 | 2.7% | 16, 15 | 86,846 | 43.1% | 1 (0) | 7,883 |
| 2 | 5.5% | 19, 20 | 188,485 | 50.7% | 0 (0) | 31,886 |
| 3 | 8.2% | 19, 20 | 188,485 | 75.1% | 0 (0) | 47,264 |
| 5 | 13.7% | 19, 20 | 188,485 | 91.9% | 0 (0) | 57,843 |
| 10 | 27.4% | 19, 20 | 188,485 | 98.2% | 0 (0) | 61,797 |

Read as: of FGK dwarfs like these, fewer than 1 in 7,883 was fading at 1% a year between mid-2014 and
mid-2017. Recall is the share of the region's stars that would be past its cut with the fade. From 0.5% a year
up, except at 1%, no star is in the chosen region and the limit is the plain 3.0 / (stars × recall); at the slow
rates it is set by stars that fade for ordinary reasons.

The one star past the cut at 1% a year is Gaia DR3 388933104249558784 (G 15.3, BP−RP 1.56, 385 pc), fading at
6.3% a year, 18.5 errors; DR3 flags it variable. Two more dwarfs brighter than G 16 fade beyond 10 errors
(368158828473101696 and 380438621009726976, at 3.9% and 4.7% a year). None was looked at further here. The
strongest trend in the field is a brightening one (19.8 errors).

With chance fades subtracted the numbers would be 1 in 137, 608, 1,532, 4,064, 19,790, 32,783, 47,264, 57,843
and 61,796: 1.2 to 12 times stronger up to 1% a year and the same from 2%. `excess_p` is 0.63 at 0.1-0.2% a year
(1,247 fading against 1,263 brightening at G < 17 beyond 3 errors) and above 0.6 at every other rate, so at no
rate do more stars fade than brighten by more than chance. The stars past the first cut are listed in
`past_cut.parquet` and were not read.

How the number has moved between runs of the same field, as 1 in N:

| Fade rate (% a year) | First run, 2026-11-10 | Bulk-file pass (PR #91) | Colour term in the reference (PR #93) | Mixed stars left out (v2, the table above) |
|---|---|---|---|---|
| 0.1 | 12 | 11 | 11 | 12 |
| 0.2 | 110 | 101 | 102 | 107 |
| 0.3 | 483 | 463 | 447 | 452 |
| 0.5 | 1,627 | 3,314 | 2,159 | 3,318 |
| 1 | 7,545 | 7,879 | 8,760 | 7,883 |
| 2 | 15,941 | 31,938 | 31,960 | 31,886 |
| 3 | 34,584 | 47,565 | 47,590 | 47,264 |
| 5 | 51,304 | 58,623 | 58,653 | 57,843 |
| 10 | 61,511 | 63,215 | 63,219 | 61,797 |

The moves at 0.5% to 2% a year are one or two stars in or out of the chosen region. Leaving out the mixed stars
changes the DR3 limit by a tenth at most at the other rates: it takes 3.6% of the stars, and the fakes injected
into the stars that stay are found a little more often (98.2% against 96.8% at 10% a year). What it changes is
the count of fading stars near the threshold, which matters on DR4 ("Stars mixed with a neighbour").

What DR3 cannot show: DR4 has twice the span and about five times the visits, so a given rate stands several
times further out in significance, and DR4 covers the whole sky.

Outputs: `/mnt/project-files/occurrence/dr3-m31/` (the limit's files and figure, the pass's per-star table and
reference, the injected scores, the catalogue slices, the planted-fade table, the trend by colour and by flux
excess; the runs before this one are kept in dated folders beside them).

### The leftover trend depends on colour

Measured on 2026-11-11 on every star of the field with a measured trend, on the run before PR #93 (drift
table with three colour ranges, reference by sky and brightness). Median G trend after the drift correction, in
% of the flux per year, by magnitude and BP−RP:

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

The bulk-file scoring work took this up the same day (PR #93, docs/arms/slow-changes.md "Drift by colour"):
narrow colour ranges in the drift table and a colour term in the reference. With both, every colour range of the
field sits within 0.013 errors of zero against the reference, and the limit runs on the new reference with no
change to its own code. For the dwarfs the offsets fell about threefold and did not go to zero at the faint
end (−0.02% a year at G 18-18.5, −0.09% at G 18.5-19). What was left there did not follow colour. It followed a
neighbour's light.

### Stars mixed with a neighbour

Measured 2026-11-11 on the field's 195,572 FGK dwarfs, with the colour term in the reference.

BP and RP fluxes are sums over a window 3.5 arcsec long. G comes from a fit to the star's image in a window of
0.7 to 1.1 arcsec. Light from a second source within about 2 arcsec therefore adds more to BP + RP than to G,
and the ratio (BP + RP) / G of the mean fluxes sits above that of single stars of the same colour (DR3 publishes
the ratio as `phot_bp_rp_excess_factor`). Here a star's **flux excess** is its ratio less the median ratio of
the field's dwarfs at its colour, which runs from 1.180 at BP−RP 0.5-0.6 to 1.296 at 1.7-1.8.

| Flux excess | Stars | Median G trend against the reference (% a year) | BP, RP trend against stars with no excess (% a year) | Fading / brightening between 2 and 5 errors |
|---|---|---|---|---|
| below −0.05 | 383 | +0.20 | +2.4, +2.5 | 3 / 61 |
| −0.05 to −0.02 | 4,090 | +0.07 | +0.8, +0.7 | 77 / 320 |
| −0.02 to −0.006 | 25,726 | +0.02 | +0.3, +0.3 | 737 / 1,214 |
| −0.006 to 0.006 | 134,370 | 0.00 | 0, 0 | 5,006 / 5,114 |
| 0.006 to 0.02 | 17,731 | −0.01 | 0.0, +0.1 | 872 / 485 |
| 0.02 to 0.05 | 6,568 | −0.04 | 0.0, +0.1 | 352 / 108 |
| 0.05 to 0.1 | 2,791 | −0.09 | −0.1, +0.1 | 253 / 30 |
| 0.1 to 0.2 | 1,657 | −0.20 | −0.1, 0.0 | 256 / 17 |
| 0.2 to 0.5 | 1,303 | −0.36 | −0.3, 0.0 | 289 / 7 |
| above 0.5 | 953 | −0.72 | −0.4, +0.2 | 190 / 9 |

(G trends are known to 0.001-0.07, BP and RP to 0.01-0.1, and to 0.4 in the first row. The counts are with
every dwarf measured against the median of the kept dwarfs of its G.)

- **With a neighbour's light, G fades.** The G trend falls steadily with the excess, by 1.7% a year per unit
  of excess near zero for stars brighter than G 17, and the fall is there at every magnitude. RP holds and BP
  falls less than half as fast. A star that was really fading would fade in all three, so this is the G
  measurement of a star with a neighbour, and the star is steady. Above an excess of 0.05, 988 dwarfs fade
  between 2 and 5 errors and 63 brighten. The cause was not established. A guess: the share of the neighbour's
  light that the G fit takes in changed as the sharpness of Gaia's images changed over the mission.
- **Below zero the stars are faint ones** (median G 17.2 to 17.8) whose BP and RP fluxes rise by 0.3 to 2.5%
  a year, which leaves their mean BP + RP low, while G rises a tenth as fast. That is the instrument too, since
  a real brightening would show the same in G. It was not traced further.
- **The excess says nothing about a real fade.** A fade shared by G, BP and RP leaves the ratio of their mean
  fluxes as it was, so choosing or correcting stars by it cannot hide the fades this limit is about. A fade
  that reddens the star moves the ratio a little: roughly a few thousandths for a fade of 5% (an estimate, not
  measured).
- **Other signs of a neighbour pick out the same stars less sharply.** Dwarfs with RUWE between 1.05 and 1.4
  (19% of them) split 1,640 fading to 1,249 brightening between 2 and 5 errors, and 1,267 to 1,219 once the
  stars with an excess beyond 0.05 are out. Those with double-peaked images in 5 to 20% of their transits
  (`ipd_frac_multi_peak`) split 188 to 27, and 22 to 10 after.

What the limit does with it (settings v2; `occurrence.flux_excess`, `unblended`, `offset_index`):

1. **Stars with an excess beyond ±0.05 are not searched**: 6,704 above and 383 below, 3.6% of the dwarfs
   (2.7 to 3.5% of those brighter than G 17, 5% at G 17-18, 12% at G 18-19).
2. **The rest are measured against dwarfs of their own excess**, in 11 steps, because the trend follows the
   excess inside the cut too. Between 2 and 5 errors that takes the stars with an excess of 0.006 to 0.02 from
   872 fading against 485 brightening to 702 against 583, those at 0.02 to 0.05 from 352 against 108 to 232
   against 162, and those below −0.006 to 934 against 979 and 154 against 169. Above zero more stars still
   fade than brighten: the drift differs from star to star, so taking out its median leaves its tail.
3. **The median ratio by colour comes from the run's own catalogue rows**, so DR4's passbands need no number
   from DR3. The catalogue step asks the archive for the ratio of the mean fluxes, which DR4's `gaia_source`
   has; DR4 does not publish `phot_bp_rp_excess_factor`.

On DR3 this changes the limit little (table above), because DR3's trend errors, 0.07 to 0.24% a year, are
larger than most of these drifts. DR4's are about 2.7 times smaller. A drift of 0.04% a year is then over one
error for a bright star, and without these two steps the mixed stars would fill the fading side near the
threshold, which is where the slow rates are measured. The cut stays at 0.05 on DR4 whatever its table shows; a
tighter one would be a new settings file, reported beside v2.

The search's own lists held the same stars. That was passed to the bulk-file scoring work on 2026-11-11 and
changed there in PR #98 (docs/arms/slow-changes.md, "Stars with a flux excess"): the pass keeps each star's
flux ratio from its own median fluxes, the search's reference has a part by flux excess, and stars with an
excess above 0.05 are read as a kind of their own, after the others. That work found the same trend in every
kind of star of the field (−0.11% a year at an excess of 0.05 to 0.1, −0.65% above 0.5).

The limit does not use the reference's part by flux excess (`flux_ratio=False` in the script's `_referenced`).
Its real stars would be placed there by the catalogue's ratio while its injected stars carry none, and the
dwarfs' own median by excess already takes the trend out. Run on a pass and a reference made with PR #98, the
DR3 limit, its counts and its offsets are the same to the last digit as in the tables above.

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
- It is about dwarfs whose light is not mixed with a neighbour's, 96.4% of them on DR3. A dwarf with a second
  source within about 2 arcsec, a companion or a chance alignment, is not searched.
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

Settings are frozen in `prereg/occurrence/steady-fades-v2.json` (rates, cuts, G ranges, the definition of an FGK
dwarf, the cut on the flux excess, confidence level); `test_frozen_config_is_the_code_default` keeps the code's
defaults equal to it. `steady-fades-v1.json`, frozen on 2026-11-10 without the cut on the flux excess, stays
beside it, and `--config steady-fades-v1` runs it.

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

`aux` fetches parallaxes and the flux ratio, which no light curve carries (uploaded id lists to ESA's archive,
or `--mirror` by source_id range). `limit` writes `limit.json` (settings, inputs, the table, the symmetry
tables, the dwarfs' median flux ratio by colour, their median trend by flux excess, and the median taken out
per G range and step of excess), `curve.csv`, `regions.parquet` (every region's counts and recall at every rate),
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
  back and not 2 × 10⁹. Each row carries the ratio of the mean fluxes, computed by the archive. A second run
  resumes; `--pixels FIRST-LAST` splits the work. On DR3 a slice of 20,762
  rows took 5 s at ESA with the project's login and 15 to 65 s at the AIP mirror, with the same rows from both.
  `inject` and `limit` read only the slices a file of stars falls in.
- `inject --random-share 0.001` uses as hosts only the pass's random share of stars, whose light curves it keeps
  in `R/lightcurves` beside those of stars near a threshold (which are not a fair sample). That is of the order
  of 10⁵ FGK dwarfs over the sky; with a few thousand per G range recall is known to a few percent. Give
  `inject` the pass's own `--no-marks` and `--no-own-test` if the pass was run with them.
- `limit` reads the per-star table one file at a time, twice (once for the dwarfs' own median per G range and
  step of flux excess), with the catalogue slices that file needs, so neither table has to fit in memory. Before
  that it reads every catalogue slice once for the dwarfs' median flux ratio by colour. It stops if the
  catalogue rows carry no flux ratio. Rows the pass marks as not a fair
  sample of their magnitude (`ref` false) are left out, and `occurrence.tally` gives counts that add up over
  files.

On the DR3 field the pass scored 22,000 stars a second on 4 cores without the period search, the 195,572 dwarfs
were injected at 9 rates in 0.8 minutes, and `limit` took 12 s; DR4's light curves are about five times longer.

Not tried at full size: the catalogue query for all 3,072 slices (14 were run), and a per-star table larger
than the field's. `planted` holds the searched stars in memory, so on DR4 it runs on the day-one sample or on a
part of `R/summary`.

The limit improves in proportion to the number of stars only while no star is past the cut of the chosen
region. With 10⁸ dwarfs some ordinary variable will be past any cut; from there the limit improves by moving to
stricter cuts, which is why they go up to 100 errors, or by reading the list.

To read on the day: the table; whether `excess_p` is small at any rate, and if so the stars behind it; the
symmetry tables; the median trend by flux excess (`trend_by_flux_excess`: on DR3 it runs from +0.07% a year at
an excess of −0.05 to −0.02, through zero, to −0.04% at 0.02 to 0.05 and −0.72% above 0.5), the share of dwarfs
left out as mixed (3.6% on DR3) and the medians taken out (`population_offset`); `planted`.

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
- **2026-11-11, later.** Run again on the pass with the colour term (PR #93): the limit needed no change, and
  the dwarfs' leftover median fell about threefold. What stayed at the faint end led to the stars whose light is
  mixed with a neighbour's: their G trend falls with the excess of (BP + RP) over G, they held the whole excess
  of fading stars near the threshold, and RP shows they are steady. Settings v2 leave out the 3.6% of dwarfs
  beyond an excess of ±0.05 and measure the rest against dwarfs of their own brightness and excess. The DR3
  limit moves by a tenth at most outside the rates where one star decides it; between 3 and 5 errors the counts
  are now 1,366 fading to 1,411 brightening. The cut alone was tried first: inside it the stars at an excess of
  0.02 to 0.05 still split 352 to 108, so the centring by excess was added. RUWE and the share of double-peaked
  images were tried as the indicator and are less sharp.
- **2026-11-11, PR #98.** The search's reference gained a part by flux excess and the pass's per-star table a
  flux ratio. The limit leaves the reference's new part out and keeps its own centring among the dwarfs. The
  field was run again through the new pass, reference and limit: every number of the DR3 tables is unchanged.
