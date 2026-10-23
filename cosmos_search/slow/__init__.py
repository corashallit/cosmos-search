"""Slow changes: stars that fade or brighten over years, and dimming episodes lasting weeks to years.

The main DR4 search since 2026-10-23 (docs/arms/slow-changes.md). Gaia's sampling catches nearly every change
longer than ~100 days, on several visits, and measures a star's long-term trend to a few hundredths of a percent
per year, so this is where its data say the most.

- `known`: known answers (Boyajian's star, fading Gaia Science Alerts, R CrB / UX Ori / DY Per stars), matched to
  Gaia DR3 source ids.
- `drift`: the calibration drift shared by stars of similar magnitude, colour and sky position, measured from
  their light curves and removed before any trend is fitted.
"""
