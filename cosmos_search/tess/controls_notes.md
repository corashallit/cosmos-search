# TESS dimming control set: notes

Built 2026-10-21 from public archives and literature only. Re-run with `python3 scripts/build.py && python3 scripts/finalize.py`. HTTP responses are cached in `cache/`, and the downloaded paper sources and catalogues are in `../dl/`.

## Files
- `controls.csv` has 196 rows and 193 unique TICs. The columns start with the requested ones, `class,name,tic,gaia_dr3,ra,dec,tmag,event_sectors,qlp_sectors,bibcode,notes`. Six extra columns follow:
  - `all_sectors`: every sector TESScut reports as observed.
  - `qlp_event_sectors`: the event sectors that have a QLP file.
  - `event_basis`: how `event_sectors` was obtained.
  - `extra_bibcodes`
  - `id_resolution`: how the TIC and Gaia IDs were obtained.
  - `predicted_future_sectors`: sectors that tess-point predicts but TESScut does not have yet.
- `ra`, `dec` and `tmag` come from TIC v8.2, with positions at epoch J2000. `qlp_sectors` lists the observed sectors where a HEAD request to the QLP S3 file returned 200.
- `scripts/`:
  - `ads.py`: ADS helper. About 30 queries were used.
  - `svc.py`: clients for MAST TIC, TESScut, the Gaia archive, SIMBAD TAP and QLP HEAD requests.
  - `targets_manual.py`: the hand-curated literature targets.
  - `build.py`: builds the table.
  - `finalize.py`: sets SIMBAD display names and adds cross-notes.
  - `qlp_versions.py`: S3 listing of the QLP file version used in each sector.

## Counts
| class | n | with any QLP | rows with event sectors | event sectors with QLP |
|---|---|---|---|---|
| 1_yso_dipper | 39 | 38 | 39 | 38 |
| 2_aperiodic_deep_dimmer | 24 | 24 | 19 | 19 |
| 3_rcb | 40 | 39 | 20 (in decline during TESS) | 20 |
| 4_uxor_haebe | 48 | 46 | 30 | 30 |
| 5_long_or_disk_eclipse | 8 | 6 | 7 | 5 |
| 5b_detached_eb_reference | 30 | 30 | 30 (all sectors) | 30 |
| 6_false_alarm | 7 | 7 | 6 | 6 |
| **total** | **196** | **190** | 151 | **148** |

## Sources by class
1. **YSO dippers (39).**
   - 32 come from Capistrant+2022 (2022ApJS..263...14C; VizieR J/ApJS/263/14). Selection: VarF=D (asymmetry M>0.25), Tmag<13.5, all "previously known" dippers, then the highest-M new ones. `event_sectors` is the sector that Capistrant inspected. Other sectors are not verified.
   - 5 Lupus dippers from Bredall+2020 (2020MNRAS.496.3257B), seen in TESS S12.
   - AA Tau (2021AJ....161...61C), in its dim state since 2011, so all sectors count as dim.
   - 2M0632 = TIC 167303776 (2022MNRAS.514.1386G), with more than 300 dips over 25 sectors.
   - Stars that Cody+2025 lists as Herbig Ae/Be stars, and the debris-disk dippers, were moved to classes 4 and 2.
2. **Aperiodic deep dimmers (24).**
   - Boyajian's star: no dips >0.5% in S14–55 (2026RNAAS..10..115Y). This is a quiet control.
   - TIC 400799224 (S10).
   - HD 240779: dimming in S5, gone by about 2020 (2019MNRAS.488.4465G and 2022RNAAS...6...49G).
   - TYC 8830-410-1 (S1 and S28).
   - TIC 43488669, the runaway old dipper.
   - Long ASAS-SN dimming events:
     - ASASSN-21qj
     - ASASSN-24fw (S87, during the occultation)
     - ASASSN-21js
     - ASASSN-25bv (S89)
     - J202402+383938
     - the JoHantgen+2026 "dippers"
   - BD+05 4868 A, a disintegrating planet (S55 and S82).
   - Exocomets: β Pic (S5–6), RZ Psc (S17, S57, S84), and five candidates from Norazman+2025.
3. **RCB stars (40).** The input list is VSX type `RCB`, excluding the uncertain `RCB:` entries, with Tmag<13.5. Decline epochs come from the per-star decline onset and end tables of Crawford+2025 (2025MNRAS.537.2635C; Zenodo 15671659). A sector counts as "in decline" if its span overlaps any decline from onset to end. I kept the 20 stars that were in decline during at least one TESS sector, plus the brightest of the rest. `notes` lists the sectors that were at maximum light (not in decline). The Crawford light curves end around JD 2460570 (2024-09). Later sectors are marked as unknown in `event_basis`.
4. **UXors / Herbig Ae/Be stars (48).**
   - 30 rows come from Cody+2025 (2025ApJ...994..253C, tables parsed from arXiv:2510.08633). These are sectors classified as APD/QPD (confirmed, 13 stars) or APDC/QPDC (candidates), keeping candidates with amplitude ≥0.05 that are not blended.
   - 18 classic UXors come from VSX only: CQ Tau, RR Tau, T Ori, KK Oph, VV Ser and others. For these rows `event_sectors` is empty because their dimming during TESS has not been checked.
5. **Long or disk eclipses (8).**
   - TIC 114936199: a 12-day outer-orbit eclipse in S14 (2022ApJ...938..133P).
   - ZTF-CBO-1: egress in S60 and ingress in S73. It is Tmag 14.6, so no QLP.
   - KH 15D (Tmag 15.0): "all sectors" is inferred from the 48-day period, not verified.
   - EE Cep: TESS missed both the 2020 and the 2025 eclipse, so it is only an out-of-eclipse control.
   - 4 long-period eclipsing binaries from JoHantgen+2026 (2026OJAp....956224J). For three of them the paper names the TESS sector of the eclipse (S58; S38; S39 and S65). For J094848-545959 (S9–10), the overlap was computed from the paper's dip date.
   - 5b, detached EBs (30): from Prsa+2022 (J/ApJS/258/16), with morphology <0.4, P = 1–15 d, primary depth 0.05–0.7, and 6 random stars per 1-mag bin over Tmag 8–13 (seed 20261021).
6. **False alarms (7).**
   - HW Lup: scattered-Earthshine dips in S12 near BTJD ~1627.5 and ~1641.5 (Bredall+2020).
   - ASASSN-24fw S87: a real event, but the photometry is unreliable because of scattered light.
   - Norazman+2025 rejected exocomet events: two blends, a neighbour-star event, an SB1/trend case, and the S28 trend dip of TIC 280832588.

## How IDs were resolved (counts are rows)
- 113 rows: TIC quoted in the paper or catalogue, then checked to exist in TIC v8.2 through the MAST `Mast.Catalogs.Filtered.Tic` service. Where the paper also gave coordinates, the separation was checked.
- 8 rows: a Gaia DR3 ID from the paper was mapped to DR2 with `gaiadr3.dr2_neighbourhood`, the DR3 position was used for a TIC cone search, and the TIC `GAIA` (DR2) field was required to match.
- 68 rows: coordinates from the paper or VSX, then a TIC cone search of radius 10.8″, taking the nearest STAR that is not an ARTIFACT, SPLIT or DUPLICATE. All separations are 1.5″ or less. No TIC star within 1 mag of the target lies in that radius.
- 7 rows: SIMBAD name to coordinates to TIC cone. Every case agreed with SIMBAD's own TIC identifier.
- Gaia DR3 IDs came from the TIC DR2 ID through `dr2_neighbourhood`, taking the nearest match. 19 rows had several matches. TIC 400799224 has no Gaia ID in TIC (it is a 0.62″ binary), so its DR3 ID comes from Fores-Toribio+2025.
- Two Capistrant TICs (336119914, 11197417) have TIC disposition SPLIT. Their QLP files still exist.

## Sectors and QLP
- Observed sectors come from the TESScut sector API (radius 0). tess-point v0.9 gives the sector mid-times, and each sector span is approximated as the mid-time ±13.7 d. That approximation is only used for the "window" (ground-based event date) overlaps and for the RCB decline overlaps.
- **Surprise: S74–S79 QLP files are named `..._tess_v02_llc.fits`, not v01.** An S3 prefix listing showed only v02 in those sectors and only v01 everywhere else. The v01-only pattern from the task brief would wrongly report no QLP file for S74–79. `svc.qlp_url` now handles this.
- No QLP files exist yet for sectors after S98 (checked 2026-10-21). TESScut already has data up to S109.
- TESScut lists a "sector 1751". This is the special 3I/ATLAS pointing in January 2026. It has no QLP products.

## Caveats
- **Label quality is not uniform.** `event_basis` gives one of four bases:
  - "documented in reference": strong.
  - "persistent behaviour -> all observed sectors": medium. Dippers switch on and off, see HD 240779.
  - "computed: ... window": weak. Only the ground-based dates were overlapped with the sector spans, and nobody has checked that the TESS data actually show the event.
  - "class-only": no TESS label.

  Treat the window-based and persistent labels as soft labels.
- **Circularity.** Capistrant's dippers were found by eye in QLP, CDIPS, SPOC and PATHOS light curves, and many of the Cody and Norazman events come from the same kinds of pipelines. Recall measured on these stars will be optimistic, especially for events picked from QLP. The ASAS-SN-discovered events (classes 2 and 5) and the RCB declines are independent of TESS selection, so use them for an unbiased check.
- **Within-star controls are the most useful negatives.** The same TIC with and without the behaviour removes star-specific and position-specific systematics, which is better than comparing different stars:
  - HD 240779: dimming in S5, quiet in S32 and later.
  - RCBs: declines in some sectors, maximum light in others (listed in `notes`).
  - Boyajian's star: S14–55 quiet.
  - TIC 280832588: S1 real candidate, S28 trend.
  - ASASSN-24fw: S7 and S33 quiet, S87 event.

  I suggest scoring at the sector level, not the object level.
- **RCB caveats.**
  - The Crawford decline frames are built from ground-based data with gaps. Flag-1 measurements (gap longer than 1 yr) were kept, so their onset and end times can be off by a lot.
  - The "not in decline" sectors assume the star's ground-based coverage reached that date. A single global end date (about JD 2460570) was used.
  - TIC Tmag for an RCB can be the Gaia-epoch (2014–17) brightness, possibly taken during a decline. For example, R CrB has Tmag 8.4 while it is about 5.7 at maximum in V. QLP's magnitude cut may therefore be inconsistent for these stars.
- **Class overlaps and identities found.**
  - JoHantgen's J114712-621057 is ASASSN-21js, and J080327-261620 has the same TIC as ASASSN-24fa. Both were merged.
  - HD 290380 (TIC 4206238) is PDS 110, a periodic disc-eclipse system. It is kept in class 4 with a note.
  - TYC 8830-410-1 (a debris disk about 600 Myr old) and HD 240779 are also in Capistrant's "dipper" table. Capistrant's dipper list is therefore not purely YSOs.
  - Six Capistrant dippers are Cody HAeBe stars and were put in class 4.
  - ASASSN-24fw and TIC 280832588 appear in both class 2 and class 6, in different roles.
- ASASSN-21qj: the TESS sectors (S34–35 and S61–62) seem to bracket the main 2021–22 optical eclipse, so `event_sectors` is empty. Look at S61–62 by eye before using it either way.
- β Pic is Tmag 3.7, saturated, but QLP files exist. RZ Psc's exocomet depths of 1–20% come from eleanor, not QLP.
- TIC 229790952: Norazman+2025 lists it as a candidate in its final table, but a commented-out draft passage notes quasi-periodic repeats. It is low confidence.
- Some references are 2026 RNAAS notes or ATels, which are not refereed: Yang 2026 on Boyajian's star, Greenberg 2026 on TIC 43488669, and the ATels for ASASSN-24fa and ASASSN-25bv.
- Class 5 is small. Few long eclipses have published TESS coverage, and most of the JoHantgen single eclipses fall between sectors (I checked by computing the overlaps). The epsilon Aur, gamma Per and V773 Tau eclipses were not covered by TESS.
- Not done: no light curves were downloaded (HEAD checks only), and no blend check beyond 10.8″ was made for rows whose TIC came from a paper. Dippers in crowded fields (Orion, Lupus, ρ Oph) can be contaminated in a 21″-pixel aperture.
