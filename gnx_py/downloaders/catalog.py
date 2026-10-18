"""Product catalogue of the GNSS download engine.

What the archives publish, kept apart from URL generation so that the
configuration checks, the candidate generator and the product-set choice all
read one table.

Contract:

* :data:`NAV_PRODUCTS`: the broadcast-navigation vocabulary accepted in
  ``product_preferences["nav"]``, mapped to the RINEX version of the file;
* :data:`LINKED_TYPES`: the product types that must come from one analysis
  centre solution (``product_sets``);
* :func:`publishes`: whether a product id publishes a data type
  (``True``/``False``, or ``None`` when the catalogue does not know).

Behind :func:`publishes` sits :data:`PRODUCT_CATALOGUE`, one
:class:`ProductEntry` per known product id with:

* the data types the product publishes and the types it is known NOT to
  publish (everything else is unknown);
* the file-name forms per type (SP3 and CLK samplings, the ERP span; a span
  of ``07D`` means a weekly file stamped at the GPS-week start day, ``03L``
  a quarterly file stamped at the first day of the calendar quarter, see
  :func:`file_day`);
* ``verified``: where the entry comes from (an archive download or listing,
  repository data, documentation, or "assumed").

A product id that is not in the table gets the generic forms of
:data:`DEFAULT_SUFFIXES` (the names generated before v1.0.0, plus the 15M
SP3 sampling after the 05M one).

Product ids and archive file names are plain tokens (:func:`product_id`,
:func:`is_plain_file_name`): the candidate generator refuses anything that
could leave the archive directory or the project folder.

This module is a leaf: it imports nothing from :mod:`gnx_py.downloaders`, so
``core``, ``validation`` and ``product_sets`` may all import it at module
level.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "DEFAULT_SUFFIXES",
    "LINKED_TYPES",
    "NAV_PRODUCTS",
    "PRODUCT_CATALOGUE",
    "PRODUCT_TYPES",
    "ProductEntry",
    "analysis_centre",
    "file_day",
    "ionex_samplings",
    "is_mgex_campaign",
    "is_plain_file_name",
    "product_entry",
    "product_id",
    "product_suffixes",
    "publishes",
]

#: Broadcast-navigation products -> RINEX version of the daily file.
#: ``BRDC00IGS``: merged multi-GNSS RINEX 3; ``BRD400DLR``: RINEX 4 (carries
#: the BeiDou-3 BDGIM parameters); ``BRDM00DLR``: multi-GNSS RINEX 3 (DLR);
#: ``BRDC``: the legacy GPS-only RINEX 2 ``brdcDDD0.YYn`` file.
NAV_PRODUCTS: Mapping[str, str] = MappingProxyType({
    "BRDC00IGS": "3",
    "BRD400DLR": "4",
    "BRDM00DLR": "3",
    "BRDC": "2",
})

#: Product types tied to one analysis-centre solution (orbits, clocks, OSB,
#: ORBEX attitude), in lead order: the first requested one leads the choice.
LINKED_TYPES: tuple[str, ...] = ("sp3", "clk", "bia", "att")

#: Data types with long product names ``{ID}_{YYYYDDD}0000_{suffix}``.
PRODUCT_TYPES: tuple[str, ...] = ("sp3", "clk", "erp", "bia", "dcb", "att")

_GPS_EPOCH = date(1980, 1, 6)

# A product id or IONEX analysis-centre code: letters and digits only.
_PRODUCT_ID = re.compile(r"[A-Z0-9]+")
# An archive file name: no directory part, no leading dot, no "..".
_PLAIN_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _gz_then_plain(*forms: str) -> tuple[str, ...]:
    return tuple(name for form in forms for name in (form + ".gz", form))


#: File-name suffixes (after ``{ID}_{YYYYDDD}0000_``) for a product id that is
#: not in :data:`PRODUCT_CATALOGUE`, in trial order. SP3 tries the 05M
#: sampling, then 15M; ERP keeps the three names generated before v1.0.0;
#: ORBEX attitude is published compressed only.
DEFAULT_SUFFIXES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "sp3": _gz_then_plain("01D_05M_ORB.SP3", "01D_15M_ORB.SP3"),
    "clk": _gz_then_plain("01D_30S_CLK.CLK"),
    "erp": ("01D_12H_ERP.ERP.gz",) + _gz_then_plain("01D_01D_ERP.ERP"),
    "bia": _gz_then_plain("01D_01D_OSB.BIA"),
    "dcb": _gz_then_plain("01D_01D_DCB.BSX", "01D_01D_DCB.BIA"),
    "att": ("01D_30S_ATT.OBX.gz",),
})

#: GIM samplings tried for an analysis centre (3-letter code) in IONEX names.
_DEFAULT_IONEX_SAMPLINGS: tuple[str, ...] = ("01H", "02H")
_IONEX_SAMPLINGS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    # High-rate maps after the common samplings: UPC 15M (UQRG), CAS 30M
    # (verified: assumed; plan C3-7).
    "UPC": ("01H", "02H", "15M"),
    "CAS": ("01H", "02H", "30M"),
})


@dataclass(frozen=True)
class ProductEntry:
    """One product id of the catalogue.

    Attributes:
        product_id: 10-character IGS long-name id, e.g. ``"COD0MGXFIN"``.
        types: data types the product is known to publish.
        absent: data types it is known NOT to publish (no candidate is
            generated for them).
        names: per-type file-name suffix overrides (after
            ``{ID}_{YYYYDDD}0000_``), in trial order; types without an
            override use :data:`DEFAULT_SUFFIXES`, except SP3, whose
            override is the exact sampling of a listed product.
        verified: source of the entry.
    """

    product_id: str
    types: frozenset
    absent: frozenset = frozenset()
    names: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    verified: str = "assumed"

    def suffixes(self, data_type: str) -> tuple[str, ...]:
        if data_type in self.names:
            return tuple(self.names[data_type])
        return DEFAULT_SUFFIXES[data_type]


def _entry(product_id: str, types, absent=(), verified="assumed", **names) -> ProductEntry:
    return ProductEntry(
        product_id=product_id,
        types=frozenset(types),
        absent=frozenset(absent),
        names=MappingProxyType({key: tuple(value) for key, value in names.items()}),
        verified=verified,
    )


_SP3_05M = _gz_then_plain("01D_05M_ORB.SP3")

#: The known products. Unknown ids are generated with DEFAULT_SUFFIXES.
PRODUCT_CATALOGUE: Mapping[str, ProductEntry] = MappingProxyType({
    entry.product_id: entry
    for entry in (
        _entry(
            "IGS0OPSFIN",
            types=("sp3", "clk", "erp", "ionex"),
            absent=("bia", "dcb", "att"),
            sp3=_gz_then_plain("01D_15M_ORB.SP3"),
            clk=_gz_then_plain("01D_30S_CLK.CLK"),
            erp=_gz_then_plain("07D_01D_ERP.ERP"),
            verified=(
                "BKG listing IGS/products/2300 and CDDIS listing products/2300 (checked 2026-10): SP3 15M, "
                "CLK 30S (also 05M), weekly ERP 07D_01D stamped at the week start; SP3 and ERP downloaded; "
                "GIM 02H in CDDIS products/ionex/2024/035 and sample_data"
            ),
        ),
        _entry(
            "COD0MGXFIN",
            types=("sp3", "clk", "erp", "bia", "att"),
            absent=("ionex",),
            sp3=_SP3_05M,
            verified=(
                "CDDIS listing products/2300 (checked 2026-10-14): SP3 05M, CLK 30S, ERP 01D_12H, OSB 01D, "
                "ATT 30S; the SP3 of 2024-034..036 downloaded; sample_data SP3 05M"
            ),
        ),
        _entry(
            "GRG0MGXFIN",
            types=("sp3", "clk", "erp", "bia", "att"),
            absent=("ionex",),
            sp3=_SP3_05M,
            # Weekly ERP only (07D_01D, stamped at the GPS-week start), like IGS0OPSFIN (DL-038).
            erp=_gz_then_plain("07D_01D_ERP.ERP"),
            verified=(
                "CDDIS listing products/2300 (checked 2026-10-14): SP3 05M, CLK 30S, OSB 01D, ATT 30S, weekly "
                "ERP 07D_01D (no daily ERP); the OSB downloaded (2024-035 and 2026-213)"
            ),
        ),
        _entry(
            "GFZ0MGXRAP",
            types=("sp3", "clk", "erp", "bia", "att"),
            absent=("ionex",),
            sp3=_SP3_05M,
            verified=(
                "CDDIS listing products/2300 (checked 2026-10-14): SP3 05M, CLK 30S, ERP 01D_01D, OSB 01D, "
                "ATT 30S; DCB.BSX 01D in CDDIS products/bias/2024 for days 021-048 only (GFZ0OPSRAP "
                "DCB.BIA covers the rest of 2024); sample_data OSB 01D"
            ),
        ),
        _entry(
            "COD0OPSFIN",
            types=("sp3", "clk", "erp", "bia", "att", "ionex"),
            sp3=_SP3_05M,
            # Weekly ERP only (07D_01D, stamped at the GPS-week start), like IGS0OPSFIN (DL-036).
            erp=_gz_then_plain("07D_01D_ERP.ERP"),
            verified=(
                "BKG listing IGSac/products/2300 and CDDIS listing products/2300 (checked 2026-10): SP3 05M, "
                "CLK 30S (also 05S), ATT 30S, weekly ERP 07D_01D only; OSB 01D at CDDIS, not in BKG IGSac in "
                "2024 (only the rapid COD0OPSRAP OSB.BIA there); GIM 01H in IGSac and CDDIS "
                "products/ionex/2024/035; BKG IGSac/products/2429 (2026-10-12): OSB 01D"
            ),
        ),
        _entry(
            "CAS0MGXRAP",
            types=("bia", "dcb"),
            absent=("sp3", "clk", "erp", "att"),
            verified=(
                "CDDIS listing products/bias/2024 (checked 2026-10-14): DCB.BSX 01D for days 021-048 only "
                "(CAS0OPSRAP DCB.BIA covers the rest of 2024), downloaded; "
                "no CAS0MGXRAP OSB in CDDIS products/bias/2024 or products/2300 (OSB assumed)"
            ),
        ),
        # The rapid DCB successors (author decision D9, bug-fix stage 13d): one
        # daily Bias-SINEX DCB file each, named ``..._01D_01D_DCB.BIA`` only.
        _entry(
            "CAS0OPSRAP",
            types=("dcb", "ionex"),
            absent=("sp3", "clk", "erp", "att"),
            dcb=_gz_then_plain("01D_01D_DCB.BIA"),
            verified=(
                "CDDIS listing products/bias/2024 (checked 2026-10-14): DCB.BIA 01D on 346 days of 2024 (not "
                "doy 085, 181-182, 196-199, 210, 259, 267, 274, 302, 308, 315, 318, 362-366); "
                "BKG listing IGS/products/2300: DCB.BIA of 2024-035..041; GIM 30M in CDDIS "
                "products/ionex/2024/035; DCB.BIA of 2024-035/131/132 read (sample_data: 2024-035); OSB unknown"
            ),
        ),
        _entry(
            "GFZ0OPSRAP",
            types=("sp3", "clk", "erp", "dcb"),
            sp3=_gz_then_plain("01D_05M_ORB.SP3"),
            clk=_gz_then_plain("01D_05M_CLK.CLK"),
            erp=_gz_then_plain("01D_01D_ERP.ERP"),
            dcb=_gz_then_plain("01D_01D_DCB.BIA"),
            verified=(
                "CDDIS listing products/bias/2024 (checked 2026-10-14): DCB.BIA 01D on 364 days of 2024 (not "
                "doy 021-022); BKG listing IGSac/products/2300 and CDDIS listing products/2300: SP3 05M, "
                "CLK 05M, ERP 01D_01D; no DCB at BKG in 2024; OSB, ATT and GIM unknown. "
                "DCB content (2024-085 read from CDDIS on 2026-10-16; CDDIS is the only archive found to "
                "serve it): Bias-SINEX DSB, one satellite "
                "pair per system (GPS C1W-C2W, Galileo C1C-C5Q, BeiDou C2I-C6I, QZSS C1C-C2C, GLONASS "
                "C1P-C2P) plus receiver DSBs; no B1C/B2a/B2b, E5b/E6, L5 or X pairs. "
                "Only the names seen at BKG and CDDIS are generated: the CLK 30S of the GFZ data centre "
                "(ISDC rapid/w2300, not a GNX source) is not tried; before v1.0.0 the generic names "
                "(SP3 05M then 15M, CLK 30S, ERP 12H then 01D) were tried and the CLK name matched no "
                "BKG or CDDIS listing"
            ),
        ),
        _entry(
            "CAS1OPSRAP",
            types=("dcb",),
            absent=("sp3", "clk", "erp", "att"),
            dcb=_gz_then_plain("01D_01D_DCB.BIA"),
            verified=(
                "CDDIS listing products/bias/2024 (checked 2026-10-14): DCB.BIA 01D on 332 days of 2024 (not doy "
                "001-014, 033, 037, 074, 105, 112, 129, 190, 196-199, 213, 302, 315, 329, 362-366); BKG listing "
                "IGS/products/2300: DCB.BIA of 2024-035..041; DCB.BIA of 2024-035/131/132 read (sample_data: "
                "2024-035)"
            ),
        ),
        _entry(
            "CAS1MGXRAP",
            types=("bia", "dcb"),
            absent=("sp3", "clk", "erp", "att"),
            verified=(
                "assumed (bias/DCB provider, default dcb list of version 0.1.0); not in CDDIS "
                "products/bias/2024 (checked 2026-10-14), which lists CAS1OPSRAP DCB.BIA instead"
            ),
        ),
        _entry(
            "DLR0MGXFIN",
            types=("bia", "dcb"),
            absent=("sp3", "clk", "erp", "att"),
            # Quarterly (03L) files stamped at the quarter start first, then the
            # daily names generated before v1.0.0.
            bia=("03L_01D_OSB.BIA.gz",) + DEFAULT_SUFFIXES["bia"],
            dcb=("03L_01D_DCB.BSX.gz",) + DEFAULT_SUFFIXES["dcb"],
            verified=(
                "assumed: quarterly 03L_01D files stamped at the first day of the quarter, then "
                "the daily names; not in CDDIS products/bias/2024 (checked 2026-10-14), where DLR "
                "publishes DLR0OPSFIN instead"
            ),
        ),
        _entry(
            "DLR0OPSFIN",
            types=("dcb",),
            absent=("sp3", "clk", "erp", "att"),
            # Quarterly DCB in the Bias-SINEX .BIA format, stamped at the quarter
            # start, then the daily names generated for unknown ids (DL-041).
            dcb=("03L_01D_DCB.BIA.gz",) + DEFAULT_SUFFIXES["dcb"],
            verified=(
                "CDDIS listing products/bias/2024 (checked 2026-10-14): DLR0OPSFIN_20240010000_03L_01D_DCB.BIA.gz "
                "(and a 03L_07D variant, not generated); OSB unknown"
            ),
        ),
    )
})


def product_entry(product_id: str) -> "ProductEntry | None":
    """The catalogue entry of ``product_id`` (case-insensitive), or ``None``."""
    return PRODUCT_CATALOGUE.get(str(product_id).strip().upper())


def publishes(product_id: str, data_type: str) -> "bool | None":
    """Whether ``product_id`` (e.g. ``"COD0MGXFIN"``) publishes ``data_type``.

    Returns ``True`` when the catalogue lists the type for that product,
    ``False`` when it lists the product as not publishing that type (no
    candidate is generated and the product-set choice treats the item as
    unavailable), and ``None`` when the product or the type is not in the
    catalogue (candidates are generated with the generic names).
    """
    entry = product_entry(product_id)
    if entry is None:
        return None
    if data_type in entry.types:
        return True
    if data_type in entry.absent:
        return False
    return None


def product_suffixes(product_id: str, data_type: str) -> tuple[str, ...]:
    """File-name suffixes of ``product_id`` for ``data_type``, in trial order.

    The full name is ``f"{product_id}_{YYYYDDD}0000_{suffix}"`` where the
    stamp day comes from :func:`file_day`. Raises ``ValueError`` for a type
    without long product names.
    """
    if data_type not in DEFAULT_SUFFIXES:
        raise ValueError(f"Unsupported product data type {data_type!r}.")
    entry = product_entry(product_id)
    if entry is None:
        return DEFAULT_SUFFIXES[data_type]
    return entry.suffixes(data_type)


def file_day(day: date, suffix: str) -> date:
    """The day a product file covering ``day`` is stamped with.

    The span is the first three characters of ``suffix`` (``01D`` daily,
    ``07D`` weekly, ``03L`` quarterly). A weekly file is stamped with the
    start day (Sunday) of the GPS week holding ``day``, which may lie in the
    previous year; a file of ``NNL`` months (``NN`` dividing 12) with the
    first day of the ``NN``-month period of the calendar year holding
    ``day`` (``03L``: 1 January, 1 April, 1 July or 1 October); every other
    file with ``day`` itself.
    """
    span = suffix[:3]
    if span == "07D":
        delta = (day - _GPS_EPOCH).days
        if delta < 0:
            raise ValueError("GPS week is undefined before 1980-01-06.")
        return day - timedelta(days=delta % 7)
    if span[2:] == "L" and span[:2].isdigit() and int(span[:2]) and 12 % int(span[:2]) == 0:
        months = int(span[:2])
        return date(day.year, (day.month - 1) // months * months + 1, 1)
    return day


def analysis_centre(product_id: str) -> str:
    """The 3-letter analysis-centre code of a product id or IONEX AC code."""
    return str(product_id).strip().upper()[:3]


def is_mgex_campaign(product_id: str) -> bool:
    """Whether ``product_id`` is an MGEX campaign product (``AAA?MGX???``)."""
    product_id = str(product_id).strip()
    return len(product_id) == 10 and product_id[4:7].upper() == "MGX"


def product_id(value, label: str = "product id") -> str:
    """``value`` as an upper-case product id or IONEX analysis-centre code.

    Raises ``ValueError`` (naming ``label``) unless the stripped value holds
    letters and digits only, so that a product id can never add a directory,
    a host or a path step to a candidate URL or local path.
    """
    text = str(value).strip().upper()
    if not _PRODUCT_ID.fullmatch(text):
        raise ValueError(
            f"{label}: {value!r} is not a product id (letters and digits only, e.g. 'COD0MGXFIN')"
        )
    return text


def is_plain_file_name(name: str) -> bool:
    """Whether ``name`` is a plain archive file name (no directory, host or ``..`` part)."""
    return bool(_PLAIN_FILE_NAME.fullmatch(name)) and ".." not in name


def ionex_samplings(analysis_center: str) -> tuple[str, ...]:
    """GIM samplings tried for an analysis centre, in trial order."""
    return _IONEX_SAMPLINGS.get(analysis_centre(analysis_center), _DEFAULT_IONEX_SAMPLINGS)
