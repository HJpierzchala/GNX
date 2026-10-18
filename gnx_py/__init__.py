from ._version import __version__
from .core.conversion import *
from .core.coordinates import *
from .models.corrections import *
from .io import *
from .ionosphere import *
from .ionosphere import *
from .orbits import *
from .ppp import *
from .models.tides import *
from .core.tools import *
from .models.troposphere import *
from .core.utils import *
from .core.configuration import *
from .core.session_errors import *
from .spp import *
from .downloaders import *
from .viz import *
from .downloaders import _lazy_attribute as _downloaders_lazy_attribute


def _install_old_module_paths():
    # Package layout of v1.0.0: the modules that were directly under gnx_py moved to core, models, io and spp
    # (gnx_py.io and gnx_py.spp are the former io.py and spp.py) and download.py to downloaders. Every old path
    # gnx_py.<module> stays importable and is the same module object as the new path (a second sys.modules key),
    # so imports, private names and monkeypatching through an old path reach the module. The modules that
    # ``import gnx_py`` loads get their old path (and package attribute) here; the others, loaded on first use
    # as before, get it from the finder below when they are imported by their old path, or from __getattr__
    # once they are imported by the new one.
    import importlib
    import importlib.util
    import sys
    from importlib.machinery import ModuleSpec

    moved = {
        "_notes": "core._notes", "_proc": "core._proc", "configuration": "core.configuration",
        "conversion": "core.conversion", "coordinates": "core.coordinates", "cycle_slips": "core.cycle_slips",
        "gnss": "core.gnss", "session_errors": "core.session_errors", "session_guards": "core.session_guards",
        "time": "core.time", "tools": "core.tools", "utils": "core.utils", "validation": "core.validation",
        "antex": "models.antex", "apriori": "models.apriori", "attitude": "models.attitude",
        "biases": "models.biases", "corrections": "models.corrections", "ocean_loading": "models.ocean_loading",
        "tides": "models.tides", "troposphere": "models.troposphere",
        "compression": "io.compression", "erp": "io.erp", "orbex": "io.orbex", "rinex2_adapter": "io.rinex2_adapter",
        "rinex_clk": "io.rinex_clk", "rinex_nav": "io.rinex_nav", "rinex_obs": "io.rinex_obs",
        "sp3_reader": "io.sp3_reader", "download": "downloaders.download",
    }

    class OldPathLoader:
        """Returns the module of the new path; the import system then registers it under the old one."""

        def __init__(self, new_name):
            self.new_name = new_name

        def create_module(self, spec):
            self.module = importlib.import_module(self.new_name)
            self.module_spec = self.module.__spec__
            return self.module

        def exec_module(self, module):
            module.__spec__ = self.module_spec  # the import system set the old-path spec on the module

    class OldPathFinder:
        """sys.meta_path finder of the old module paths gnx_py.<module>."""

        gnx_py_old_module_paths = True

        def __init__(self, package):
            self.package = package

        def find_spec(self, fullname, path=None, target=None):
            package, _, old = fullname.rpartition(".")
            if package != self.package or old not in moved:
                return None
            # the origin is the file of the new path, as importlib.util.find_spec("gnx_py.<module>") gave before
            new_spec = importlib.util.find_spec(f"{package}.{moved[old]}")
            return ModuleSpec(fullname, OldPathLoader(f"{package}.{moved[old]}"),
                              origin=new_spec.origin if new_spec is not None else None)

        def loaded(self, old):
            """The module of the old path ``old`` if its new path is already imported, else None."""
            module = sys.modules.get(f"{self.package}.{moved[old]}") if old in moved else None
            if module is not None:
                sys.modules[f"{self.package}.{old}"] = module
            return module

    for old in moved:
        module = sys.modules.get(f"{__name__}.{moved[old]}")
        if module is not None:
            sys.modules[f"{__name__}.{old}"] = module
            # a star import above may already hold the name, as before (gnx_py.utils is gnx_py.ionosphere.utils)
            globals().setdefault(old, module)
    # ``from .ionosphere import *`` binds gnx_py.ionosphere.models to ``models``; the subpackage takes the name
    globals()["models"] = sys.modules[f"{__name__}.models"]
    sys.meta_path[:] = [f for f in sys.meta_path if not (getattr(f, "gnx_py_old_module_paths", False)
                                                          and getattr(f, "package", None) == __name__)]
    sys.meta_path.insert(0, OldPathFinder(__name__))


_install_old_module_paths()
del _install_old_module_paths


def __getattr__(name: str):
    import sys

    # The old path of a module that ``import gnx_py`` does not load, once that module is imported under its new
    # path (as an imported module was an attribute of the package before v1.0.0).
    for finder in sys.meta_path:
        if getattr(finder, "gnx_py_old_module_paths", False) and finder.package == __name__:
            module = finder.loaded(name)
            if module is not None:
                globals()[name] = module
                return module
    # The deprecated ProjectDownload* aliases and the lazily loaded
    # fetch_earthdata_token are kept out of gnx_py.downloaders.__all__, so the
    # star import above neither warns nor loads earthdata_auth; resolve them here.
    return _downloaders_lazy_attribute(__name__, name)
