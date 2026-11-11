# How common steady fades can be

Code: `cosmos_search/slow/occurrence.py` (the limit), `cosmos_search/slow/recall.py` (injected fades),
`scripts/slow_occurrence.py` (the steps). Frozen settings: `prereg/occurrence/steady-fades-v1.json`. Built
2026-11-10, item 4 of the big-picture review.

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

1. **Trend of every star.** The search's own G trend and its error (`slope`, `slope_z`; docs/arms/slow-changes.md
   S3), measured against stars of similar brightness: the median trend of the 401 stars nearest in G is
   subtracted. On DR3 the drift correction leaves +0.08% a year at G 12-13, which would push every star there
   towards brightening and hide fades. A per-star table that already carries a reference from similar stars
   (`slope_ref`, the bulk-file pass) is used as it is.
2. **Recall.** Every FGK dwarf's own light curve is given the fade and scored again, at each rate
   (`recall.injected_scores`). The fade is `inject.lightcurve.SecularTrend` from the star's first epoch, in all
   three bands, with the errors scaled with the flux. The star keeps its noise, its gaps and its own variability,
   so recall is measured on the stars the limit is about. Nothing is random and nothing is judged by a reader, so
   nothing is blinded.
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
Andromeda field (below), by trend significance after the reference:

| Trend beyond (errors) | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| Fading | 1,701 | 457 | 152 | 55 | 10 | 5 | 1 | 1 | 0 |
| Brightening | 1,637 | 427 | 170 | 85 | 30 | 13 | 6 | 2 | 1 |

Near the threshold the two agree. Beyond 6 errors brightening stars outnumber fading ones 85 to 55, and beyond
8 errors 30 to 10; a fair coin gives splits that uneven 1.4% and 0.2% of the time. The 50 of those 140 stars that
DR3 classifies are all spotted, active stars (31 RS CVn type, 19 solar-like), and nothing requires spots to
appear and to fade at the same pace. With 30 brightening stars subtracted from 10 fading ones, the conclusion
would be that no star there can be a real fade, drawn from stars that are doing something else.

Directions also come in groups on the sky. The median trend of the dwarfs in a 0.84 deg² sky pixel differs from
pixel to pixel by 0.10 errors more than sampling allows, and among the 2,700 stars past 3 errors at G < 17 the
fading counts scatter between pixels 1.6 times as much as independent stars would (chance 0.05%). A subtraction
that treats stars as independent then understates its own error.

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
field, 100 draws for each case, shares chosen at about half and twice the limit the real stars give at that
rate:

| Fade rate (% a year) | Share planted | Limit below the planted share | Subtracted number below it |
|---|---|---|---|
| 0.1 | 4% | 0 of 100 | 6 |
| 0.1 | 16% | 0 | 100 |
| 0.2 | 0.5% | 0 | 0 |
| 0.2 | 2% | 0 | 0 |
| 0.5 | 0.03% | 0 | 13 |
| 0.5 | 0.12% | 0 | 34 |
| 2 | 0.003% | 0 | 0 |
| 2 | 0.012% | 0 | 1 |
| 10 | 0.001% | 0 | 0 |
| 10 | 0.004% | 0 | 1 |

The limit stayed above the planted share in all 1,000 draws. A 95% limit may fall below the truth in up to 5%
of draws; here it never did, because the stars that fade for ordinary reasons add to every count. The subtracted
number fell below it in up to 100 of 100, at the rates where its region holds more brightening than fading
stars (0.5% a year: 7 fading, 9 brightening) or where the planted share is large enough to move the reference.

## DR3: the Andromeda field

DR3 has light curves for stars not flagged variable in one place only, the 5.5° around M31 (Gaia's Andromeda
photometric survey). `slow_occurrence.py field` fetched every star there with G < 19 from the AIP mirror.
529,976 stars came back (153 level-6 sky pixels, four queries at once, 72 minutes), 529,394 have enough epochs
to be scored and 195,572 are FGK dwarfs by the definition above. They have a median of 19 visits over 2.74
years: half of DR4's span and about a fifth of its visits. The median trend error is 0.07% a year at G 13-16,
0.10% at G 16-17, 0.16% at G 17-18 and 0.26% at G 18-19. Scores used the drift table of the DR3 trial
(`/mnt/project-files/slow-changes/trial-dr3/rule-dr3trial`), without the period search.

| Fade rate (% a year) | Fade over the 2.74 years | Region: G below, errors beyond | Stars | Recall | Fading there (brightening) | Fewer than 1 in |
|---|---|---|---|---|---|---|
| 0.1 | 0.27% | 15, 4 | 44,861 | 4.2% | 133 (131) | 12 |
| 0.2 | 0.55% | 15, 5 | 44,861 | 13.5% | 43 (54) | 110 |
| 0.3 | 0.82% | 15, 8 | 44,861 | 8.4% | 3 (5) | 483 |
| 0.5 | 1.4% | 15, 10 | 44,861 | 22.8% | 2 (1) | 1,627 |
| 1 | 2.7% | 16, 15 | 89,328 | 40.1% | 1 (0) | 7,545 |
| 2 | 5.5% | 16, 15 | 89,328 | 84.7% | 1 (0) | 15,941 |
| 3 | 8.2% | 19, 25 | 195,572 | 53.0% | 0 (0) | 34,584 |
| 5 | 13.7% | 19, 25 | 195,572 | 78.6% | 0 (0) | 51,304 |
| 10 | 27.4% | 19, 25 | 195,572 | 94.2% | 0 (0) | 61,511 |

Read as: of FGK dwarfs like these, fewer than 1 in 7,545 was fading at 1% a year between mid-2014 and
mid-2017. Recall is the share of the region's stars that would be past its cut with the fade. At the fast rates
the limit is the plain 3.0 / (stars × recall); at the slow ones it is set by stars that fade for ordinary
reasons.

The one star past the cut at 1-2% a year is Gaia DR3 388933104249558784 (G 15.3, BP−RP 1.56, 385 pc), fading at
6% a year, 17.6 errors; DR3 flags it variable. It was not looked at further here. The strongest trend in the
field is a brightening one (21.8 errors).

With chance fades subtracted the numbers would be 1 in 72, 298, 1,479, 4,357, 18,685, 31,685, 45,693, 57,898 and
63,173: 2 to 6 times stronger up to 2% a year and about the same from 3%. `excess_p` is 0.047 at 0.1-0.2% a
year (1,394 fading against 1,306 brightening at G < 17 beyond 3 errors), in the region where directions cluster
on the sky by the factor 1.6 above, which is enough to produce it; at every other rate it is above 0.5.

What DR3 cannot show: DR4 has twice the span and about five times the visits, so a given rate stands several
times further out in significance, and DR4 covers the whole sky.

Outputs: `/mnt/project-files/occurrence/dr3-m31/` (the limit's files and figure, the per-star scores, the
injected scores, the stars' `gaia_source` rows, the planted-fade table).

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
- An injected star keeps the catalogue magnitude of the star it was injected into. A real fader's would be
  fainter, by up to 0.15 mag at 10% a year over DR3's span, which moves a few stars across the edge of a G range.
- Recall is measured on all the stars searched, or on a sample of them. A sample must be a fair one: the same
  mix of brightness, number of visits and sky as the stars the counts come from. Recall is applied per G range.

## On DR4

Settings are frozen in `prereg/occurrence/steady-fades-v1.json` (rates, cuts, G ranges, the definition of an FGK
dwarf, confidence level); `test_frozen_config_is_the_code_default` keeps the code's defaults equal to it.

On the threshold sample of release day (`slow_search.py sample`, step 2 of "Release day, end to end"), with
`RULE` the release-day rule directory:

    python scripts/slow_occurrence.py aux --lightcurves S/lightcurves --schema gaiadr4 --out S/aux_plx.parquet
    python scripts/slow_search.py score --lightcurves S/lightcurves --aux S/aux.parquet --rule RULE --out S/run --no-period
    python scripts/slow_occurrence.py inject --lightcurves S/lightcurves --aux S/aux_plx.parquet --rule RULE --out O
    python scripts/slow_occurrence.py limit --stars S/run/stats --aux S/aux_plx.parquet --injections O/injections --out O/limit
    python scripts/slow_occurrence.py planted --stars S/run/stats --aux S/aux_plx.parquet --injections O/injections --out O/planted

`aux` adds parallaxes (the sample's own `aux.parquet` has none). `limit` writes `limit.json` (settings, inputs,
the table, the symmetry tables), `curve.csv`, `regions.parquet` (every region's counts and recall at every
rate), `past_cut.parquet` (every star past the first cut either way) and the figure. On the DR3 field, scoring
took 1.2 ms per star per core without the period search and each injected light curve 1.3 ms (195,572 stars at
9 rates: 37 core-minutes); DR4's light curves are about five times longer.

The sample is small (300 patches of 0.013 deg²: tens of thousands of dwarfs), so its limit is a first number
and a check that everything runs on DR4. The number worth reporting comes from every star, from the per-star
table of the bulk-file pass (R1 of docs/plan.md, in review as this is written). What is in place for it: `limit`
counts a per-star table one file at a time when the table carries its reference from similar stars
(`slope_ref`), so the scores of the whole sky never have to fit in memory; each injected star is measured
against the reference of the star it was injected into; rows marked as not a fair sample of their magnitude
(`ref` false) are left out; and `occurrence.tally` gives counts that add up over batches of stars. The stars the
fades are injected into must be a fair sample of the stars counted (above); with a few thousand of them per G
range, recall is known to a few percent.

Two things are still to be joined on once R1 is merged, and are not built: the pass stores trends without their
reference, so its reference table has to be applied to the stored rows first (`summary.apply_reference`); and
the pass has no parallaxes, so the FGK dwarfs of the whole sky need their `gaia_source` rows from a catalogue
query per sky slice, read one slice at a time (`aux` as it stands uploads lists of ids and holds its result in
memory, which suits a sample).

The limit improves in proportion to the number of stars only while no star is past the cut of the chosen
region. With 10⁸ dwarfs some ordinary variable will be past any cut; from there the limit improves by moving to
stricter cuts, which is why they go up to 100 errors, or by reading the list.

To read on the day: the table; whether `excess_p` is small at any rate, and if so the stars behind it; the
symmetry tables; `planted`.

## Log

- **2026-11-10.** Built, with the DR3 run above. Three designs were tried and measured before this one. A
  likelihood over all bins with a background per bin: its answer moved by a factor of 2-3 with the prior put on
  the backgrounds (13, 20 and 41 detected stars for flat, Jeffreys and 1/b priors on the same counts). One region
  with chance fades subtracted, chosen on the brightening counts: 8% of no-fade samples called an excess at the
  5% level. The same with the region chosen on fading and brightening together: sound under symmetry, and below
  the truth without it (tables above). The grid of cuts was made denser (12, 20, 30, 50, 80 added) after the
  first DR3 run and before freezing; the DR3 table did not change with it.
