"""Correction models of GNX-py: solid Earth and ocean tide loading, troposphere, phase wind-up and relativistic
corrections, satellite attitude, ANTEX antenna models, a priori positions and code and phase biases.

The modules were directly under ``gnx_py`` before v1.0.0; their old paths (``gnx_py.tides``, ...) still work.
The name ``gnx_py.models`` meant the module ``gnx_py.ionosphere.models`` before; its names
(``gnx_py.models.klobuchar``, ...) still resolve as attributes of this package.
"""

#: the modules of this package: never resolved by ``__getattr__``, so ``from gnx_py.models import attitude`` loads one
_MODULES = frozenset({"antex", "apriori", "attitude", "biases", "corrections", "ocean_loading", "tides", "troposphere"})


def __getattr__(name: str):
    # Before v1.0.0 the name gnx_py.models was the module gnx_py.ionosphere.models (bound by the star import of
    # gnx_py.ionosphere); its names still resolve as attributes of the package (gnx_py.models.klobuchar).
    if not name.startswith("__") and name not in _MODULES:
        from ..ionosphere import models as ionosphere_models

        try:
            return getattr(ionosphere_models, name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
