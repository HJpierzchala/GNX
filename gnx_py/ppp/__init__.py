"""Public PPP package exports.

The package re-exports session routing, combined PPP filters, uncombined PPP
filters, PPP-AR helpers, and preprocessing utilities for compatibility with the
historical ``gnx_py.ppp`` public API. New code should prefer importing concrete
classes from their defining modules when the PPP variant matters.

Status:
    Public compatibility surface. The broad star exports intentionally include
    active and legacy classes; changing this file would be an API change and is
    outside the documentation/status cleanup stages.

Public API notes:
    The explicit ``__all__`` below is a compatibility snapshot of the current
    public surface, including historical accidental exports such as imported
    third-party modules and helper objects. Future cleanup may narrow wildcard
    exports, but that should happen only after a dedicated API migration.
"""

from .config import *
from .ppp_gnss import *
from .ppp_single import *
from .ppp_uduc import *
from .pppar import *

# Keep this compatibility import even though the module currently only
# documents the historical preprocessing surface.
from .preprocessing import *

# Compatibility snapshot for ``from gnx_py.ppp import *``. Keep this broad for
# now: it preserves the historical public surface while tests document which
# names are intended PPP API and which are candidates for future pruning.
__all__ = (
    "Any",
    "BroadcastInterp",
    "COMMON_AR_REASONS",
    "CSDetector",
    "CustomWrapper",
    "DDPreprocessing",
    "DUAL_FREQUENCY_MODES",
    "ExtendedKalmanFilter",
    "FREQ_DICT_ALL",
    "Final",
    "GIMReader",
    "GNSSDataProcessor2",
    "GNSS_MODE_SIGNALS",
    "Generic",
    "List",
    "Literal",
    "MODE_SIGNALS",
    "Optional",
    "PPPARDiagnostics",
    "PPPARSettings",
    "PPPConfig",
    "PPPDualFreqMultiGNSS",
    "PPPDualFreqSingleGNSS",
    "PPPFilterMultiGNSSIonConst",
    "PPPFilterMultiGNSSIonConstGEC",
    "PPPResult",
    "PPPSession",
    "PPPSingleFreqMultiGNSS",
    "PPPSingleFreqSingleGNSS",
    "PPPUdGenericMixedGNSS",
    "PPPUdMixedGNSS",
    "PPPUdMultiGNSS",
    "PPPUdSingleGNSS",
    "PPPUducSFMultiGNSS",
    "PPPUducSFSingleGNSS",
    "Path",
    "SIGNALS",
    "SIGNAL_COLUMNS",
    "SINGLE_FREQUENCY_MODES",
    "SP3InterpolatorOptimized",
    "STECMonitor",
    "Sequence",
    "Set",
    "TECConfig",
    "Union",
    "advance_arc_age",
    "annotations",
    "apply_conventional_pppar",
    "apply_indexed_uncombined_pppar",
    "apply_uncombined_pppar",
    "arange_datetime",
    "arc_age_array",
    "bias_column_m",
    "build_sd_matrix",
    "calculate_distance",
    "combined_if_pppar_enabled",
    "config",
    "dataclass",
    "decorrel",
    "ecef2geodetic",
    "ecef_to_enu",
    "emission_interp",
    "field",
    "frequency_hz",
    "guarded_session_run",
    "has_satellite_osb",
    "lagrange_emission_interp",
    "lagrange_reception_interp",
    "lambda_ils",
    "ldldecom",
    "make_ionofree",
    "mode_ionosphere_free_coefficients",
    "mode_layout",
    "mode_signals",
    "np",
    "osb_m",
    "parse_sinex",
    "pd",
    "phase_residuals_outliers",
    "ppp_code_screening",
    "ppp_gnss",
    "ppp_helpers",
    "ppp_single",
    "ppp_trace",
    "ppp_trace_epoch",
    "ppp_uduc",
    "pppar",
    "pppar_diagnostic_columns",
    "preprocessing",
    "read_sp3",
    "signal_spec",
    "split_dual_code_dsb_corrections_m",
    "ssearch",
    "warnings",
)
