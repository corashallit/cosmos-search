"""Eclipsing-binary filter for the DR4 dimmers queue (docs/arms/eb-filter.md).

`search` asks whether one strict period describes which epochs are in dip, as a compression test with a
cadence-aware false-alarm bound; `decide` adds the physics (eclipse duration against the orbit the host allows) and
the condition that the selected dip itself is explained, and returns the filter's verdict.
"""
