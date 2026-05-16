"""Compatibility placeholder for historical PPP preprocessing exports.

Status:
    Legacy compatibility module. The active PPP preprocessing pipeline is
    orchestrated from ``gnx_py.ppp.config.PPPSession`` through shared GNX tools,
    but ``gnx_py.ppp.__init__`` still imports this module to preserve the broad
    historical ``gnx_py.ppp`` public surface.

Notes:
    This module intentionally exposes no runtime helpers today. Do not delete
    it or remove its package import without auditing external imports such as
    ``gnx_py.ppp.preprocessing``.
"""

from __future__ import annotations
