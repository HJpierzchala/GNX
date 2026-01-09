import pandas as pd
from IPython.display import Markdown, display

def df_head(
    df: pd.DataFrame,
    nrows: int = 10,
    ncols: int = 6,
    index: bool = False,
    floatfmt: str = ".3f",
    truncate_str: int | None = 30,
    reset_index: bool = True,
):
    """
    Wyświetl DataFrame jako czytelną tabelę Markdown (do PDF),
    z ograniczeniem liczby wierszy/kolumn, cyfr i długości stringów.

    Parametry:
    - nrows: ile pierwszych wierszy pokazać
    - ncols: ile pierwszych kolumn pokazać
    - index: czy pokazywać index w tabeli
    - floatfmt: format liczb zmiennoprzecinkowych (np. '.2f', '.3f')
    - truncate_str: maksymalna długość stringów (None = bez ucinania)
    - reset_index: czy wrzucić MultiIndex do kolumn (True = zwykle czytelniej)
    """
    sub = df.copy()

    if reset_index:
        sub = sub.reset_index()

    sub = sub.iloc[:nrows, :ncols].copy()

    # Spłaszczenie MultiIndex kolumn, jeśli jest
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = [
            "_".join(map(str, lvl)).strip()
            for lvl in sub.columns
        ]

    # 1) Zaokrąglanie kolumn numerycznych na podstawie floatfmt
    ndigits = None
    # oczekujemy formatu typu '.3f' albo '.3'
    if floatfmt.startswith(".") and floatfmt[1:].rstrip("f").isdigit():
        ndigits = int(floatfmt[1:].rstrip("f"))

    if ndigits is not None:
        num_cols = sub.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            sub[num_cols] = sub[num_cols].astype(float).round(ndigits)

    # 2) Ucinanie długich stringów TYLKO w kolumnach tekstowych
    if truncate_str is not None:
        obj_cols = sub.select_dtypes(include=["object", "string"]).columns

        def _truncate(s: str) -> str:
            return (s[:truncate_str] + "…") if len(s) > truncate_str else s

        for col in obj_cols:
            sub[col] = sub[col].astype(str).map(_truncate)

    # 3) Generacja Markdown – floatfmt nadal działa na kolumnach numerycznych
    md = sub.to_markdown(
        index=index,
        tablefmt="github",
        floatfmt=floatfmt,
    )

    display(Markdown(md))
