"""Proper-motion-aware, calibrated cross-matching from Gaia to other surveys.

    from cosmos_search.xmatch import crossmatch
    res = crossmatch(gaia_table, "allwise")      # HATS or TAP chosen per survey
    res.matches, res.best, res.report           # pairs with P, best per target, false-match accounting

See docs/infra/xmatch.md for the method, the survey table and measured false-match rates.
"""

from .astrometry import Astrometry
from .match import Candidates, ErrorModel, gather, posteriors
from .run import Result, crossmatch
from .sources import AutoSource, HatsSource, MemorySource, TapSource
from .surveys import SURVEYS, Survey

__all__ = [
    "Astrometry", "AutoSource", "Candidates", "ErrorModel", "HatsSource", "MemorySource", "Result",
    "SURVEYS", "Survey", "TapSource", "crossmatch", "gather", "posteriors",
]
