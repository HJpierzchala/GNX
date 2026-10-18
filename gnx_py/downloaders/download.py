"""Compatibility import surface for GNSS archive downloads.

Re-exports the public :mod:`gnx_py.downloaders` API. The lazily loaded
``fetch_earthdata_token`` and the deprecated ``ProjectDownload*`` aliases are
not part of that package's ``__all__``, so they are forwarded on attribute
access instead of being copied by the wildcard import.
"""

from . import *
from . import _lazy_attribute


def __getattr__(name: str):
    # the module path of the deprecation and error messages is the documented one, gnx_py.download
    return _lazy_attribute(__name__.rsplit(".", 2)[0] + ".download", name)
