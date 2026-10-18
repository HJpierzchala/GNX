"""Compact Markdown previews of GNX result frames.

GNX result frames are wide and long: printing one in a notebook or a manual
buries the point. These helpers show a trimmed corner of a frame as a Markdown
table, with the float precision and string lengths capped.

The rendering is split from the formatting. By default (``show=True``) both
functions render the table and return None, so a call on the last line of a
notebook cell shows the table once: rendered as Markdown where the front-end
supports it (Jupyter), shown as Markdown source otherwise (terminal IPython,
plain Python). With ``show=False`` nothing is rendered
and the Markdown source is returned, which is what scripts and tests use.
IPython is not a GNX runtime dependency.
"""
from __future__ import annotations

import pandas as pd

__all__ = [
    "df_head",
    "df_tail",
]


def _float_digits(floatfmt: str) -> int | None:
    """Extract the digit count from a ``.3f``-style format, or None."""
    if floatfmt.startswith(".") and floatfmt[1:].rstrip("f").isdigit():
        return int(floatfmt[1:].rstrip("f"))
    return None


def _render_markdown(md: str) -> None:
    """Render Markdown under an active IPython shell, else print the source."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ModuleNotFoundError:
        print(md)
        return
    # IPython can be importable in a plain interpreter; only a live shell can
    # display output. Terminal IPython and jupyter console render only
    # text/plain, so the Markdown source doubles as the plain-text form.
    if get_ipython() is None:
        print(md)
        return
    display({"text/markdown": md, "text/plain": md}, raw=True)


def _to_markdown_corner(
    df: pd.DataFrame | pd.Series,
    *,
    head: bool,
    nrows: int | None,
    ncols: int | None,
    index: bool,
    floatfmt: str,
    truncate_str: int | None,
    reset_index: bool,
) -> str:
    """Format a corner of ``df`` as a Markdown table. See :func:`df_head`."""
    from ..ionosphere.utils import _check_number  # lazy: no import cycle with ionosphere

    # Arguments first: a negative count would otherwise slice from the other
    # end of the frame and a negative truncate_str cut strings from the end.
    # None keeps every row or column.
    nrows = _check_number("nrows", nrows, ge=0, integer=True, allow_none=True)
    ncols = _check_number("ncols", ncols, ge=0, integer=True, allow_none=True)
    truncate_str = _check_number("truncate_str", truncate_str, ge=0, integer=True, allow_none=True)

    if isinstance(df, pd.Series):
        df = df.to_frame() if df.name is not None else df.to_frame(name="value")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame or Series, got {type(df)!r}.")

    sub = df.copy()
    # An unnamed RangeIndex is only row numbers; resetting it would spend a
    # column slot on an "index" column that carries no information.
    positional = isinstance(sub.index, pd.RangeIndex) and sub.index.name is None
    if reset_index and not positional:
        sub = sub.reset_index()

    # The tail starts at a non-negative position: ``iloc[-0:]`` would be the
    # whole frame, not zero rows.
    if nrows is not None:
        start = 0 if head else max(len(sub) - nrows, 0)
        sub = sub.iloc[start : start + nrows]
    sub = sub.iloc[:, :ncols].copy()

    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = ["_".join(map(str, level)).strip() for level in sub.columns]

    # Strings are cut before integers become object below, so the integer
    # columns are not turned into strings here.
    if truncate_str is not None:
        for col in sub.select_dtypes(include=["object", "string"]).columns:
            sub[col] = sub[col].astype(str).map(
                lambda s: (s[:truncate_str] + "…") if len(s) > truncate_str else s
            )

    # Only floating columns follow floatfmt. Integer and boolean columns go
    # to object so tabulate prints them as-is (12, not 12.000); missing
    # values of nullable integers become None, which tabulate leaves blank.
    # Columns are addressed by position so duplicate names are handled.
    ndigits = _float_digits(floatfmt)
    for pos in range(sub.shape[1]):
        values = sub.iloc[:, pos]
        if pd.api.types.is_float_dtype(values.dtype):
            # Nullable Float64/Float32 go to float64 (pd.NA -> nan): tabulate
            # does not apply floatfmt to extension values and cannot compare
            # pd.NA. This holds for any floatfmt, not only ".Nf".
            values = values.astype("float64")
            if ndigits is not None:
                values = values.round(ndigits)
            sub.isetitem(pos, values)
        elif pd.api.types.is_integer_dtype(values.dtype) or pd.api.types.is_bool_dtype(
            values.dtype
        ):
            values = values.astype(object)
            sub.isetitem(pos, values.where(values.notna(), None))

    return sub.to_markdown(index=index, tablefmt="github", floatfmt=floatfmt)


def df_head(
    df: pd.DataFrame | pd.Series,
    *,
    nrows: int | None = 10,
    ncols: int | None = 6,
    index: bool = False,
    floatfmt: str = ".3f",
    truncate_str: int | None = 30,
    reset_index: bool = True,
    show: bool = True,
) -> str | None:
    """Show the leading rows of a frame as a Markdown table.

    Parameters
    ----------
    df:
        Frame to preview. A Series is shown as a one-column frame, named
        ``"value"`` when the Series has no name.
    nrows:
        Number of leading rows to show, an integer >= 0 or None. Zero shows
        only the header; None or a value above the frame length shows every
        row.
    ncols:
        Number of leading columns to show, an integer >= 0 or None. None
        shows every column.
    index:
        Include the frame index as a table column.
    floatfmt:
        Float format, such as ``".3f"``. Floating columns are rounded to
        match; integer and boolean columns are shown without decimals.
    truncate_str:
        Maximum length of string values, an integer >= 0; longer values
        keep their first ``truncate_str`` characters and are suffixed with
        an ellipsis. None disables truncation.
    reset_index:
        Move the index (including a MultiIndex) into columns first, which is
        usually more readable for GNX ``(sv, time)`` frames. An unnamed
        RangeIndex holds only row numbers and is never moved.
    show:
        Render the table instead of returning it: as Markdown where the
        front-end supports it (Jupyter), as Markdown source otherwise.

    Returns
    -------
    str or None
        None when ``show`` is True, else the Markdown table source.

    Raises
    ------
    ValueError
        When ``nrows``, ``ncols`` or ``truncate_str`` is neither None nor an
        integer >= 0.
    TypeError
        When ``df`` is neither a DataFrame nor a Series.
    """
    md = _to_markdown_corner(
        df,
        head=True,
        nrows=nrows,
        ncols=ncols,
        index=index,
        floatfmt=floatfmt,
        truncate_str=truncate_str,
        reset_index=reset_index,
    )
    if not show:
        return md
    _render_markdown(md)
    return None


def df_tail(
    df: pd.DataFrame | pd.Series,
    *,
    nrows: int | None = 10,
    ncols: int | None = 6,
    index: bool = False,
    floatfmt: str = ".3f",
    truncate_str: int | None = 30,
    reset_index: bool = True,
    show: bool = True,
) -> str | None:
    """Show the trailing rows of a frame as a Markdown table.

    Parameters
    ----------
    df:
        Frame to preview. A Series is shown as a one-column frame, named
        ``"value"`` when the Series has no name.
    nrows:
        Number of trailing rows to show, an integer >= 0 or None. Zero shows
        only the header; None or a value above the frame length shows every
        row.
    ncols:
        Number of leading columns to show, an integer >= 0 or None. None
        shows every column.
    index:
        Include the frame index as a table column.
    floatfmt:
        Float format, such as ``".3f"``. Floating columns are rounded to
        match; integer and boolean columns are shown without decimals.
    truncate_str:
        Maximum length of string values, an integer >= 0; longer values
        keep their first ``truncate_str`` characters and are suffixed with
        an ellipsis. None disables truncation.
    reset_index:
        Move the index (including a MultiIndex) into columns first, which is
        usually more readable for GNX ``(sv, time)`` frames. An unnamed
        RangeIndex holds only row numbers and is never moved.
    show:
        Render the table instead of returning it: as Markdown where the
        front-end supports it (Jupyter), as Markdown source otherwise.

    Returns
    -------
    str or None
        None when ``show`` is True, else the Markdown table source.

    Raises
    ------
    ValueError
        When ``nrows``, ``ncols`` or ``truncate_str`` is neither None nor an
        integer >= 0.
    TypeError
        When ``df`` is neither a DataFrame nor a Series.
    """
    md = _to_markdown_corner(
        df,
        head=False,
        nrows=nrows,
        ncols=ncols,
        index=index,
        floatfmt=floatfmt,
        truncate_str=truncate_str,
        reset_index=reset_index,
    )
    if not show:
        return md
    _render_markdown(md)
    return None
