"""Public compatibility exports for GNX orbit and SIS utilities.

The historical module surface was broad because this package re-exported
everything from `gnx_py.orbits.config`. The explicit `__all__` below preserves
that wildcard-import behavior for compatibility while documenting which names
are true SIS/orbits API and which are candidates for a future major cleanup.
"""

from . import config
from .config import *


# Active orbits/SIS public API.
_CORE_API = [
    "SISConfig",
    "SISController",
]

# Public constants used by tests, validation scripts and BDS status handling.
_CONSTANT_API = [
    "BDS_CLOCK_CONVENTION_NOTE",
    "BDS_LEGACY_BROADCAST_TGD_SIGNALS",
    "BDS_LEGACY_IF_PRECISE_CLOCK_MODE",
    "BDS_MODERN_BROADCAST_SIGNALS",
    "BDS_SISRE_ALPHA_BETA",
    "DEFAULT_SISRE_COEFFICIENTS",
]

# Compatibility exports retained because they were previously visible through
# `from gnx_py.orbits import *`. These are not recommended as new public API.
_COMPATIBILITY_EXPORTS = [
    "BrdcGenerator",
    "DDPreprocessing",
    "DefaultConfig",
    "GNSSDataProcessor2",
    "List",
    "Optional",
    "Path",
    "SP3InterpolatorOptimized",
    "Union",
    "arange_datetime",
    "bds_orbit_type",
    "config",
    "dataclass",
    "datetime",
    "guarded_session_run",
    "is_bds_geo",
    "mode_ionosphere_free_coefficients",
    "mode_signals",
    "np",
    "pd",
    "re",
    "read_sp3",
    "signal_spec",
    "warnings",
]

__all__ = _CORE_API + _CONSTANT_API + _COMPATIBILITY_EXPORTS
