import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib as mpl
import seaborn as sns
# Styl publikacyjny
sns.set_context("paper", font_scale=1.3)
sns.set_style("ticks")
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
import pandas as pd
from IPython.display import Markdown, display

def df_tail(
    df: pd.DataFrame,
    nrows: int = 10,
    ncols: int = 6,
    index: bool = False,
    floatfmt: str = ".3f",
    truncate_str: int | None = 30,
    reset_index: bool = True,
):
    """
    Wyświetl końcowe wiersze DataFrame jako czytelną tabelę Markdown (do PDF),
    z ograniczeniem liczby wierszy/kolumn, cyfr i długości stringów.

    Parametry:
    - nrows: ile ostatnich wierszy pokazać
    - ncols: ile pierwszych kolumn pokazać
    - index: czy pokazywać index w tabeli
    - floatfmt: format liczb zmiennoprzecinkowych
    - truncate_str: maksymalna długość stringów (None = bez ucinania)
    - reset_index: czy wrzucić MultiIndex do kolumn
    """
    sub = df.copy()

    if reset_index:
        sub = sub.reset_index()

    sub = sub.iloc[-nrows:, :ncols].copy()

    # Spłaszczenie MultiIndex kolumn, jeśli jest
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = [
            "_".join(map(str, lvl)).strip()
            for lvl in sub.columns
        ]

    # 1) Zaokrąglanie kolumn numerycznych wg floatfmt
    ndigits = None
    if floatfmt.startswith(".") and floatfmt[1:].rstrip("f").isdigit():
        ndigits = int(floatfmt[1:].rstrip("f"))

    if ndigits is not None:
        num_cols = sub.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            sub[num_cols] = sub[num_cols].astype(float).round(ndigits)

    # 2) Ucinanie długich stringów w kolumnach tekstowych
    if truncate_str is not None:
        obj_cols = sub.select_dtypes(include=["object", "string"]).columns

        def _truncate(s: str) -> str:
            return (s[:truncate_str] + "…") if len(s) > truncate_str else s

        for col in obj_cols:
            sub[col] = sub[col].astype(str).map(_truncate)

    # 3) Generacja Markdown
    md = sub.to_markdown(
        index=index,
        tablefmt="github",
        floatfmt=floatfmt,
    )

    display(Markdown(md))

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

@dataclass
class STECModelComparator:
    """
    Kompaktowy porównywacz modeli STEC:
    - liczy statystyki: Bias, RMS, STD, MAE, Corr, R2, slope, intercept
    - rysuje: hexbin (model vs pomiar) + linia y=x i linia regresji
    - rysuje: histogram błędów (model - pomiar)

    Parametry
    ---------
    df : pd.DataFrame
        Dane z kolumną pomiarową oraz kolumnami modeli.
    x_col : str
        Nazwa kolumny z pomierzonym STEC (np. 'leveled_tec').
    model_cols : List[str]
        Nazwy kolumn modeli (np. ['ion', 'klobuchar', 'ntcm']).
    model_names : Optional[List[str]]
        Etykiety do wykresów (np. ['GIM', 'Klobuchar', 'NTCM G']).
        Jeśli None, używa wartości z model_cols.
    scale : float
        Skala nakładana na wartości modeli, np. K=1/0.16.
    gridsize : int
        Rozdzielczość hexbina.
    mincnt : int
        Minimalna liczba punktów w heksie, by był rysowany.
    """
    df: pd.DataFrame
    x_col: str = 'leveled_tec'
    model_cols: List[str] = field(default_factory=lambda: ['ion', 'klobuchar', 'ntcm'])
    model_names: Optional[List[str]] = None
    scale: float = 1.0
    gridsize: int = 80
    mincnt: int = 1

    def _get_xy(self, model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.df[self.x_col].to_numpy()
        y = (self.df[model] * self.scale).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask], mask

    @staticmethod
    def _metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # e = model - observation
        e = y - x
        n = e.size
        if n < 2:
            return {k: np.nan for k in
                    ["Bias", "RMS", "STD", "MAE", "Corr", "R2", "Slope", "Intercept", "N"]}
        bias = np.mean(e)
        rms = np.sqrt(np.mean(e**2))
        std = np.std(e, ddof=0)
        mae = np.mean(np.abs(e))
        # korelacja i R^2
        if np.std(x) == 0 or np.std(y) == 0:
            corr = np.nan
            r2 = np.nan
            slope = np.nan
            intercept = np.nan
        else:
            slope, intercept, r, p, stderr = stats.linregress(x, y)
            corr = r
            r2 = r**2
        return {
            "Bias": bias,
            "RMS": rms,
            "STD": std,
            "MAE": mae,
            "Corr": corr,
            "R2": r2,
            "Slope": slope,
            "Intercept": intercept,
            "N": int(n),
        }

    def compute_stats(self) -> pd.DataFrame:
        rows = []
        labels = self.model_names or self.model_cols
        for col, name in zip(self.model_cols, labels):
            x, y, _ = self._get_xy(col)
            rows.append({"Model": name, **self._metrics(x, y)})
        stats_df = pd.DataFrame(rows)
        # kolejność i zaokrąglenia przyjazne na publikację
        order = ["Model", "N", "Bias", "STD", "RMS", "MAE", "Corr", "R2", "Slope", "Intercept"]
        for c in ["Bias", "STD", "RMS", "MAE", "Corr", "R2", "Slope", "Intercept"]:
            if c in stats_df:
                stats_df[c] = stats_df[c].astype(float).round(3)
        return stats_df[order]

    def plot_hexbin(self, suptitle: Optional[str] = None,
                    share_limits: bool = True,
                    figsize: Tuple[int, int] = (18, 6),
                    show: bool = True) -> Tuple[plt.Figure, List[plt.Axes]]:
        labels = self.model_names or self.model_cols
        n = len(self.model_cols)
        fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True, sharey=True)
        if n == 1:
            axes = [axes]

        # wspólne limity osi (opcjonalnie)
        x_all = self.df[self.x_col].to_numpy()
        finite_x = np.isfinite(x_all)
        xmin = np.nanmin(x_all[finite_x]) if finite_x.any() else 0.0
        xmax = np.nanmax(x_all[finite_x]) if finite_x.any() else 1.0

        hbs = []
        for ax, col, name in zip(axes, self.model_cols, labels):
            x, y, _ = self._get_xy(col)
            if x.size == 0:
                ax.set_title(f"{name} (brak danych)")
                continue
            hb = ax.hexbin(x, y, gridsize=self.gridsize, cmap='plasma', mincnt=self.mincnt)
            hbs.append(hb)
            # y=x
            xx = np.array([xmin, xmax])
            ax.plot(xx, xx, ls='--', color='red', lw=1)

            # regresja
            if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
                slope, intercept, r, p, stderr = stats.linregress(x, y)
                ax.plot(xx, slope*xx + intercept, color='orange',
                        label=f"Fit: y={slope:.2f}x+{intercept:.2f}\nR²={r**2:.2f}")
                ax.legend(fontsize=9)

            ax.set_title(name)
            ax.grid(True, ls=':', alpha=0.6)
            ax.set_xlabel("Measured STEC [TECU]")
        axes[0].set_ylabel("Modelled STEC [TECU]")

        # colorbar wspólny
        if hbs:
            cbar = fig.colorbar(hbs[0], ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
            cbar.set_label("Point Count")

        if suptitle:
            fig.suptitle(suptitle, y=1.02)

        # fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_error_hist(self, bins: int = 60,
                        suptitle: Optional[str] = "Error histogram",
                        figsize: Tuple[int, int] = (18, 5),
                        density: bool = False,
                        show: bool = True) -> Tuple[plt.Figure, List[plt.Axes]]:
        labels = self.model_names or self.model_cols
        n = len(self.model_cols)
        fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True, sharey=True)
        if n == 1:
            axes = [axes]

        for ax, col, name in zip(axes, self.model_cols, labels):
            x, y, _ = self._get_xy(col)
            e = y - x
            if e.size == 0:
                ax.set_title(f"{name} (lack of data)")
                continue
            ax.hist(e, bins=bins, alpha=0.85, edgecolor='k', density=density)
            mu = np.mean(e)
            sd = np.std(e)
            ax.axvline(0, color='red', ls='--', lw=1)
            ax.axvline(mu, color='black', ls='-', lw=1)
            ax.set_title(f"{name}\nBias={mu:.2f}, STD={sd:.2f}")
            ax.set_xlabel("Error [TECU]")
            ax.grid(True, ls=':', alpha=0.6)
        axes[0].set_ylabel("Count" if not density else "Density")

        if suptitle:
            fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def report(self,
               title: str = "Modelled vs Measured STEC",
               show_hex: bool = True,
               show_hist: bool = True,
               return_stats: bool = True):
        """
        Szybki raport: liczy statystyki i rysuje wykresy.
        """
        stats_df = self.compute_stats()
        if show_hex:
            self.plot_hexbin(suptitle=title, show=True)
        if show_hist:
            self.plot_error_hist(show=True)
        if return_stats:
            return stats_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VTECZenithEstimator:
    """
    Prosta estymacja VTEC nad stacją (zenit) z obserwacji STEC.

    Wymagane kolumny w df:
      - time : datetime64[ns]
      - ev   : elewacja satelity [deg]
      - leveled_tec : STEC [TECU] (po własnych korektach/DСB/levelling)

    Opcjonalne:
      - sv : identyfikator satelity (pomocne do diagnostyki)
      - az : azymut [deg] (tu nieużywany)

    Kroki:
      1) VTEC = STEC / M(e), M(e) – obliquity factor dla h_ion_m.
      2) W każdej epoce wybór satelity o max(ev) >= min_elev_deg.
      3) Złożenie serii czasowej VTEC_zenith i interpolacja liniowa w czasie.
    """
    df: pd.DataFrame
    time_col: str = "time"
    elev_col: str = "ev"
    stec_col: str = "leveled_tec"
    sv_col: Optional[str] = "sv"
    h_ion_m: float = 450e3
    R_earth_m: float = 6371e3
    min_elev_deg: float = 30.0  # ignorujemy bardzo płaskie ścieżki
    resample_freq: Optional[str] = None  # np. '30S' – jeśli None, interpolujemy po nieregularnych czasach
    rolling_median_win: Optional[int] = None  # np. 3 – delikatne wygładzenie po interpolacji
    # smooth_method: 'median' | 'ema' | 'savgol' | 'whittaker'

    smooth_method: str = "whittaker"
    smooth_params: dict = field(default_factory=dict)

    # ----------------------------- core geometry -----------------------------
    def _obliquity_factor(self, elev_deg: np.ndarray) -> np.ndarray:
        """
        M(e) = 1 / cos(z'), cos(z') = sqrt(1 - (Re/(Re+h))^2 * cos^2(e))
        """
        e_rad = np.deg2rad(elev_deg)
        rho = self.R_earth_m / (self.R_earth_m + self.h_ion_m)
        cosz_prime_sq = 1.0 - (rho ** 2) * (np.cos(e_rad) ** 2)
        # zabezpieczenie numeryczne
        cosz_prime_sq = np.clip(cosz_prime_sq, 1e-8, None)
        M = 1.0 / np.sqrt(cosz_prime_sq)
        return M

    # ----------------------------- pipeline ---------------------------------
    def compute_vtec_per_obs(self) -> pd.DataFrame:
        """Dodaje kolumnę 'M' i 'vtec' do kopii ramki wejściowej."""
        df = self.df.copy()
        # maska na poprawne wartości
        m = np.isfinite(df[self.stec_col]) & np.isfinite(df[self.elev_col])
        df.loc[~m, [self.stec_col, self.elev_col]] = np.nan

        M = self._obliquity_factor(df[self.elev_col].to_numpy())
        df["M"] = M
        df["vtec"] = df[self.stec_col] / M  # [TECU]
        return df

    def pick_nearest_zenith(self, df_v: pd.DataFrame) -> pd.DataFrame:
        """
        Wybiera w każdej epoce rekord z najwyższą elewacją (>= min_elev_deg).
        Działa dla:
          - MultiIndex z poziomem 'time' (np. index = ['sv','time'])
          - zwykłego indeksu + kolumny 'time'
        """
        elev = self.elev_col
        tcol = self.time_col

        # --- MultiIndex z poziomem 'time' ---
        if isinstance(df_v.index, pd.MultiIndex) and (tcol in df_v.index.names):
            # indeksy (etykiety) wierszy o maks. elewacji w każdej epoce
            idx = (
                df_v[df_v[elev].notna()]
                .groupby(level=tcol)[elev]
                .idxmax()
                .dropna()
            )
            # klucz: używamy listy krotek jako etykiet do .loc
            chosen = df_v.loc[list(idx)].copy().reset_index()  # 'sv' i 'time' wracają jako kolumny

            # liczba satelitów >= próg na epokę (MultiIndex: grupujemy po level=time)
            cnt = (
                df_v.loc[df_v[elev] >= self.min_elev_deg]
                .groupby(level=tcol)[elev]
                .size()
                .rename("n_sat_ge_thr").astype(int)
            )

            chosen = chosen.merge(cnt, left_on=tcol, right_index=True, how="left")

        # --- Zwykły DataFrame (time jako kolumna) ---
        else:
            # upewnij się, że 'time' jest kolumną
            if tcol not in df_v.columns:
                if df_v.index.name == tcol:
                    df_v = df_v.reset_index()
                else:
                    raise ValueError(f"Nie widzę kolumny ani poziomu indeksu '{tcol}' w df_v.")
            tmp = (
                df_v[df_v[elev].notna()]
                .sort_values([tcol, elev], ascending=[True, False])
                .groupby(tcol, as_index=False)
                .head(1)
                .copy()
                .reset_index(drop=True)
            )
            cnt = (
                df_v.loc[df_v[elev] >= self.min_elev_deg]
                .groupby(tcol)[elev]
                .size()
                .rename("nsat").astype(int)
            )
            chosen = tmp.merge(cnt, left_on=tcol, right_index=True, how="left")

        # próg elewacji → jeśli < min_elev_deg, to nie ufamy tej próbce
        chosen.loc[chosen[elev] < self.min_elev_deg, "vtec"] = np.nan

        # kluczowe kolumny
        cols = [self.time_col, "vtec", self.elev_col, "M", "nsat"]
        if self.sv_col and self.sv_col in chosen.columns:
            cols.insert(1, self.sv_col)

        # posortuj po czasie
        chosen[self.time_col] = pd.to_datetime(chosen[self.time_col], errors="coerce")
        return chosen[cols].sort_values(self.time_col)


    # --- DODAJ DO __init__ klasy pola konfiguracyjne (np. w dataclass) ---
    # smooth_method: 'median' | 'ema' | 'savgol' | 'whittaker'
    # smooth_params: dict z parametrami metody
    # przykład w konstruktorze:
    # smooth_method: str = "savgol"
    # smooth_params: dict = field(default_factory=dict)

    def interpolate_timeseries(self, ts: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolacja po czasie + wygładzanie wg self.smooth_method.
        """
        out = ts.copy()

        # upewnij się, że 'time' jest kolumną
        if self.time_col not in out.columns:
            if isinstance(out.index, pd.MultiIndex) and self.time_col in out.index.names:
                out = out.reset_index(self.time_col)
            elif out.index.name == self.time_col:
                out = out.reset_index()
            else:
                raise ValueError(f"Brak kolumny '{self.time_col}' w ts.")

        out[self.time_col] = pd.to_datetime(out[self.time_col], errors="coerce")
        out = out.set_index(self.time_col).sort_index()

        # stała siatka czasu (opcjonalnie)
        if self.resample_freq:
            num_cols = out.select_dtypes(include="number").columns
            out = out[num_cols].resample(self.resample_freq).mean()

        # liniowe domknięcie luk (czasowe)
        if "vtec" in out:
            out["vtec"] = out["vtec"].interpolate(method="time", limit_direction="both")

        # --- AKTYWNE WYGŁADZANIE ---
        y = out["vtec"].to_numpy()
        y_s = self._apply_smoothing(out.index, y, method=getattr(self, "smooth_method", "savgol"),
                                    params=getattr(self, "smooth_params", {}))
        out["vtec_smooth"] = y_s

        return out.reset_index()

    def _apply_smoothing(self, t_index: pd.Index, y: np.ndarray,
                         method: str = "savgol", params: dict = None) -> np.ndarray:
        """
        Wspólna bramka do wygładzania. Obsługuje NaN-y (wstępnie je uzupełnia).
        Metody:
          - 'median'    : okno N (param 'window')
          - 'ema'       : dwustronna EWM (param 'alpha' lub 'span' lub 'halflife')
          - 'savgol'    : Savitzky–Golay (param 'window', 'polyorder')
          - 'whittaker' : wygładzanie Eilersa (param 'lam', 'order' [1 lub 2])
        """
        params = params or {}
        y = y.astype(float)
        # uzupełnij ewentualne NaN-y prostą interpolacją
        if np.isnan(y).any():
            s = pd.Series(y, index=t_index).interpolate(method="time", limit_direction="both")
            y = s.to_numpy()

        n = len(y)
        if n < 5:
            return y  # za mało danych, nic nie rób

        # pomoc: oszacuj krok czasowy (sekundy) – przyda się do doboru okna
        try:
            t_sec = pd.to_datetime(t_index).view("int64") / 1e9
            dt = np.median(np.diff(t_sec))
            if not np.isfinite(dt) or dt <= 0:
                dt = None
        except Exception:
            dt = None

        method = (method or "savgol").lower()

        if method == "median":
            win = int(params.get("window", 7))
            win = max(1, win)
            s = pd.Series(y).rolling(win, center=True, min_periods=1).median().to_numpy()
            return s

        elif method == "ema":
            # dwustronna EMA (forward + backward) zmniejsza przesunięcie fazowe
            if "alpha" in params:
                alpha = float(params["alpha"])
            elif "span" in params:
                alpha = 2.0 / (float(params["span"]) + 1.0)
            elif "halflife" in params:
                hl = float(params["halflife"])
                alpha = 1 - np.exp(-np.log(2) / hl)
            else:
                # domyślnie: ok. ~10 min stałej czasowej jeśli znamy dt
                if dt is not None:
                    tau = params.get("tau_seconds", 600.0)
                    alpha = 1 - np.exp(-dt / float(tau))
                else:
                    alpha = 0.2  # bez wiedzy o dt
            # forward
            yf = np.copy(y)
            for i in range(1, n):
                yf[i] = alpha * y[i] + (1 - alpha) * yf[i - 1]
            # backward
            yb = np.copy(y)
            for i in range(n - 2, -1, -1):
                yb[i] = alpha * yb[i] + (1 - alpha) * yb[i + 1]
            return 0.5 * (yf + yb)

        elif method == "savgol":
            # okno: jeśli nie podano, wybierz ~10 min / dt (lub 11 jako bezpieczna wartość)
            poly = int(params.get("polyorder", 3))
            if "window" in params:
                win = int(params["window"])
            else:
                if dt is not None:
                    target_sec = float(params.get("window_seconds", 600.0))  # 10 min
                    win = max(5, int(round(target_sec / dt)))
                else:
                    win = 11
            # okno musi być nieparzyste i <= n
            if win % 2 == 0:
                win += 1
            if win > n:
                win = n - (1 - n % 2)  # największe nieparzyste <= n
            win = max(win, poly + 2 | 1)  # zapewnij > poly i nieparzyste
            return savgol_filter(y, window_length=win, polyorder=poly, mode="interp")


        elif method == "whittaker":

            order = int(params.get("order", 2))

            order = max(1, min(order, len(y) - 1))

            lam = float(params.get("lam", 2000.0))

            return self._whittaker(y, lam=lam, order=order)


        else:
            # nieznana metoda -> brak zmian
            return y

    def _whittaker(self, y: np.ndarray, lam: float = 1000.0, order: int = 2) -> np.ndarray:
        """
        Whittaker–Eilers smoother: min ||y - z||^2 + lam * ||D^d z||^2
        Budujemy D (d-tej różnicy) z malejącymi wymiarami, by uniknąć konfliktów.
        """
        n = len(y)
        if n < 3:
            return y

        # Bezpieczny zakres rzędu
        order = max(1, min(int(order), n - 1))

        E = sparse.eye(n, format="csc")

        # Pierwsza różnica: (n-1) x n
        D = sparse.diags([1, -1], [0, 1], shape=(n - 1, n), format="csc")

        # Wyższe rzędy: każda kolejna różnica działa na wynik poprzedniej
        for k in range(1, order):
            rows, cols = D.shape  # po k-tej iteracji D ma rozmiar (n-k, n)
            Di = sparse.diags([1, -1], [0, 1], shape=(rows - 1, rows), format="csc")
            D = Di @ D  # finalnie D ma rozmiar (n - order, n)

        A = E + lam * (D.T @ D)
        z = spsolve(A, y)
        return np.asarray(z, dtype=float)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Zwraca:
          df_v : ramka z VTEC dla wszystkich obserwacji (kolumny: ...,'M','vtec')
          zen  : seria czasowa VTEC nad stacją (po wyborze sat. najbliżej zenitu i interpolacji)
        """
        df_v = self.compute_vtec_per_obs()
        zen_raw = self.pick_nearest_zenith(df_v)
        zen = self.interpolate_timeseries(zen_raw)
        return df_v, zen

    # ----------------------------- quick plot --------------------------------
    def plot(self, zen: pd.DataFrame, use_smoothed: bool = True, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Szybki wykres VTEC nad stacją w czasie.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        y = "vtec_smooth" if (use_smoothed and "vtec_smooth" in zen.columns) else "vtec"
        ax.plot(zen[self.time_col], zen[y], lw=1.5)
        ax.set_title("VTEC nad stacją (zenit)")
        ax.set_xlabel("Czas")
        ax.set_ylabel("VTEC [TECU]")
        ax.grid(True, ls=":", alpha=0.6)
        return ax


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import List, Dict, Optional, Tuple


def plot_stec(
    obs_tec: pd.DataFrame,
    prn: str,
    time_col: str = "LT",
    figsize: Tuple[float, float] = (8, 6),
    left_series: Optional[List[Dict]] = None,
    right_series: Optional[List[Dict]] = None,
    title: Optional[str] = None,
    grid: bool = True,
    tight_layout: bool = True,
):
    """
    Uniwersalny wykres STEC z możliwością pełnej konfiguracji serii danych.

    Parameters
    ----------
    obs_tec : pd.DataFrame
        DataFrame z obserwacjami. Musi mieć indeks zawierający PRN (np. MultiIndex)
        tak, aby obs_tec.loc[prn, ...] wybierało dane jednej satelity.
    prn : str
        Klucz/PRN używany w obs_tec.loc[prn, :].
    time_col : str, default "LT"
        Nazwa kolumny z czasem (datetime).
    figsize : tuple, default (8, 6)
        Rozmiar figury w calach.
    left_series : list of dict, optional
        Konfiguracja serii rysowanych na lewej osi Y.
        Każdy słownik może mieć klucze:
            - "col" (str, wymagane): nazwa kolumny w obs_tec,
            - "label" (str, opcjonalne),
            - "color" (str, opcjonalne),
            - "marker" (str, default "o"),
            - "s" (float, default 10): rozmiar markerów,
            - "alpha" (float, default 0.7),
            - "edgecolor" (str, default "k"),
            - "y_label" (str, opcjonalne): opis osi Y (tylko pierwsza seria się liczy),
            - "y_color" (str, opcjonalne): kolor etykiet i osi Y (domyślnie color pierwszej serii).
    right_series : list of dict, optional
        Analogicznie jak left_series, ale dla prawej osi Y (ax2).
        Jeśli None lub pusta lista – oś prawa nie jest tworzona.
    title : str, optional
        Tytuł wykresu. Jeśli None, używane jest "Slant TEC  PRN: {prn[:3]}".
    grid : bool, default True
        Czy rysować grid na głównej osi (lewej).
    tight_layout : bool, default True
        Czy wywołać plt.tight_layout() na końcu.

    Przykład użycia
    ---------------
    # Domyślna konfiguracja (jak Twoja stara funkcja):
    plot_stec(obs_tec, "G01")

    # Własna konfiguracja:
    left = [
        {"col": "code_tec", "label": "CODE STEC", "color": "tab:blue",
         "y_label": "CODE STEC [TECU]", "y_color": "tab:blue"},
        {"col": "leveled_tec", "label": "LEVELED", "color": "tab:green"},
    ]
    right = [
        {"col": "phase_tec", "label": "PHASE STEC", "color": "tab:red",
         "y_label": "PHASE STEC [TECU]", "y_color": "tab:red"}
    ]
    plot_stec(obs_tec, "G01", left_series=left, right_series=right)
    """

    # --- Domyślna konfiguracja odpowiadająca Twojej starej funkcji ---
    if left_series is None:
        left_series = [
            {
                "col": "code_tec",
                "label": "Code STEC",
                "color": "tab:blue",
                "y_label": "CODE STEC [TECU]",
                "y_color": "tab:blue",
            },
            {
                "col": "leveled_tec",
                "label": "Phase levelled STEC",
                "color": "tab:green",
            },

            {
                "col": "leveled_tec_sg",
                "label": "S-G filter smoothed STEC",
                "color": "orange",
            },
        ]

    if right_series is None:
        right_series = [
            {
                "col": "phase_tec",
                "label": "Phase STEC",
                "color": "tab:red",
                "y_label": "PHASE STEC [TECU]",
                "y_color": "tab:red",
            }
        ]

    # --- Wycięcie danych dla danego PRN ---
    data = obs_tec.loc[prn]
    t = data[time_col]

    fig, ax = plt.subplots(figsize=figsize)

    # --- LEWA OŚ Y ---
    lns_left = []
    labs_left = []

    y_label_left = None
    y_color_left = None

    for i, series in enumerate(left_series):
        col = series["col"]
        label = series.get("label", col)
        color = series.get("color", None)
        marker = series.get("marker", "o")
        s = series.get("s", 10)
        alpha = series.get("alpha", 0.7)
        edgecolor = series.get("edgecolor", "k")

        sc = ax.scatter(
            t,
            data[col],
            label=label,
            color=color,
            marker=marker,
            alpha=alpha,
            edgecolors=edgecolor,
            s=s,
        )
        lns_left.append(sc)
        labs_left.append(label)

        if i == 0:
            # jeśli podano, bierzemy z serii; w przeciwnym wypadku domyślnie
            y_label_left = series.get("y_label", f"{label} [TECU]")
            y_color_left = series.get("y_color", color)

    if y_label_left is not None:
        ax.set_ylabel(y_label_left, color=y_color_left, fontsize=8)
        ax.tick_params(axis="y", labelcolor=y_color_left)
        ax.spines["left"].set_color(y_color_left)

    # --- PRAWA OŚ Y (opcjonalnie) ---
    lns_right = []
    labs_right = []

    if right_series:
        ax2 = ax.twinx()

        y_label_right = None
        y_color_right = None

        for i, series in enumerate(right_series):
            col = series["col"]
            label = series.get("label", col)
            color = series.get("color", None)
            marker = series.get("marker", "o")
            s = series.get("s", 10)
            alpha = series.get("alpha", 0.7)
            edgecolor = series.get("edgecolor", "k")

            sc = ax2.scatter(
                t,
                data[col],
                label=label,
                color=color,
                marker=marker,
                alpha=alpha,
                edgecolors=edgecolor,
                s=s,
            )
            lns_right.append(sc)
            labs_right.append(label)

            if i == 0:
                y_label_right = series.get("y_label", f"{label} [TECU]")
                y_color_right = series.get("y_color", color)

        if y_label_right is not None:
            ax2.set_ylabel(y_label_right, color=y_color_right, fontsize=8)
            ax2.tick_params(axis="y", labelcolor=y_color_right)
            ax2.spines["right"].set_color(y_color_right)
    else:
        ax2 = None

    # --- Oś X, format czasu ---
    ax.set_xlabel("Local Time [LT]", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate()

    # --- Grid ---
    if grid:
        ax.grid(True, which="major", linestyle="--", alpha=0.6)

    # --- Tytuł ---
    if title is None:
        title = f"Slant TEC  PRN: {prn[:3]}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    # --- Legenda łączona z obu osi ---
    lns = lns_left + lns_right
    labs = labs_left + labs_right
    ax.legend(lns, labs, loc="best", fontsize=8, frameon=True)

    if tight_layout:
        plt.tight_layout()

    plt.show()



