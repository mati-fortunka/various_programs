import os
import re
import glob
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ==============================================================================
# KONFIGURACJA CZCIONEK I STYLU (SPÓJNA Z FIGURAMI 5.1-5.4)
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"  # POGRUBIONA KURSYWA (\mathsf)
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

FS_LABEL = 14.5      # etykiety osi głównych
FS_RES_LABEL = 12.0  # etykiety osi residuals
FS_TICK = 13         # podziałki osi głównych
FS_RES_TICK = 11     # podziałki osi residuals
FS_LEGEND = 11.5     # legendy
FS_TITLE = 13        # tytuły paneli, bold
FS_PANEL = 14.5      # litery paneli A-D, bold

FIG55_RC = {
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
}

# Szerokość = 12.0 in jak w pozostałych figurach rozdziału 5; ściśnięta wysokość
FIG55_SIZE = (12.0, 9.8)

# Parametry rysowania punktów, linii dopasowania i wygładzania rezyduów
POINT_SIZE = 6
POINT_ALPHA = 0.55
FIT_LW = 1.65

RES_POINT_SIZE = 4.5
RES_POINT_ALPHA = 0.35
RES_SMOOTH_WINDOW_S = 75.0   # szerokość okna Savitzky-Golay [s] (spójna dla CD i FL)
RES_SMOOTH_POLYORDER = 2
RES_SMOOTH_LW = 1.75
RES_SMOOTH_ALPHA = 0.95

# Normalizacja krzywych fluorescencji do zakresu [0, 1] dla czytelnego porównania
# krzywizny i rezyduów trzech białek o różnej intensywności bazowej
NORMALIZE_FL_SIGNALS = True

# Wspólne okno czasowe analizy i wykresu [s] (wszystkie krzywe mają dead_time = 30 s)
DEAD_TIME_S = 30.0
CD_FIT_END_S = 2000.0
FL_FIT_END_S = 2000.0


def darker(color, factor=0.70):
    """Ciemniejszy odcień koloru dla linii dopasowania/trendu na tle punktów."""
    r, g, b = mpl.colors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def smooth_residuals_sg(t, residuals, window_s=RES_SMOOTH_WINDOW_S, polyorder=RES_SMOOTH_POLYORDER):
    """Wygładza rezydua filtrem Savitzky-Golay z oknem dobranym w sekundach."""
    t = np.asarray(t, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n = len(t)
    if n < polyorder + 3:
        return residuals
    dt = float(np.median(np.diff(t)))
    win = int(round(window_s / dt)) if dt > 0 else 15
    if win % 2 == 0:
        win += 1
    min_win = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
    max_win = n if n % 2 == 1 else n - 1
    win = max(min_win, min(win, max_win))
    return savgol_filter(residuals, window_length=win, polyorder=polyorder)


def style_legend(ax, handles=None, labels=None, loc="upper right", markerscale=2.6, ncol=1):
    """Legenda w stylu Fig. 5.1-5.4: biała ramka bez krawędzi, uchwyt po prawej."""
    anchor = (1, 0.03) if "lower" in loc else (1, 0.97)
    kw = dict(
        loc=loc, bbox_to_anchor=anchor, fontsize=FS_LEGEND,
        frameon=True, facecolor="white", framealpha=0.92, edgecolor="none",
        markerfirst=False, handlelength=2.4, handletextpad=0.7,
        scatterpoints=1, markerscale=markerscale, ncol=ncol,
    )
    leg = ax.legend(handles, labels, **kw) if handles is not None else ax.legend(**kw)
    for text_obj in leg.get_texts():
        text_obj.set_ha("right")
    return leg


# ==============================================================================
# KONFIGURACJA ŚCIEŻEK I WYBRANYCH TRAJEKTORII (STĘŻENIA CD POBIERANE STĄD)
# ==============================================================================
PATHS = {
    "fig_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch05",
    "tables_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/tables",
}

# Rząd 1 (Panele A i B): Far-UV CD — wartość "conc_uM" brana bezpośrednio stąd
CD_TRACES = [
    {
        "key": "FsTrmD-Tm1570",
        "label": r"$\mathit{Fs}$TrmD–Tm1570",
        "color": "#1F78B4",
        "n_res": 447,
        "pathlength_mm": 1.0,
        "conc_uM": 3,
        "search_dirs": ["/home/matifortunka/Downloads/PhD/rozdział_V/F8/CD"],
        "file": "/home/matifortunka/Downloads/PhD/rozdział_V/F8/CD/F8_2000s_probka200001.csv",
    },
    {
        "key": "CnTm1570",
        "label": r"$\mathit{Cn}$Tm1570",
        "color": "#1B9E77",
        "n_res": 224,
        "pathlength_mm": 1.0,
        "conc_uM": 6.0,
        "search_dirs": ["/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/CD/5µM"],
        "file": "/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/CD/5µM/kinetyka_2000s_00000.csv",
    },
    {
        "key": "CnTrmD-Tm1570",
        "label": r"$\mathit{Cn}$TrmD–Tm1570",
        "color": "#7570B3",
        "n_res": 447,
        "pathlength_mm": 1.0,
        "conc_uM": 3.7,
        "search_dirs": ["/home/matifortunka/Downloads/PhD/rozdział_V/Cn_fusion/cd/5µM"],
        "file": "/home/matifortunka/Downloads/PhD/rozdział_V/Cn_fusion/cd/5µM/fuzja_kin00002.csv",
    },
]

# Rząd 2 (Panele C i D): Intrinsic Fluorescence (CnTrmD, CnTm1570, CnTrmD-Tm1570)
FL_TRACES = [
    {
        "key": "CnTrmD",
        "label": r"$\mathit{Cn}$TrmD",
        "color": "#1F78B4",
        "search_dirs": ["/home/matifortunka/Downloads/PhD/rozdział_V/TrmD/2µM"],
        "file": "/home/matifortunka/Downloads/PhD/rozdział_V/TrmD/2µM/Kinetic Sample1 Scan2_TrmD.csv",
    },
    {
        "key": "CnTm1570",
        "label": r"$\mathit{Cn}$Tm1570",
        "color": "#1B9E77",
        "search_dirs": ["/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/fluo/waw"],
        "file": "/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/fluo/waw/Kinetic Sample1 Scan8_Tm.csv",
    },
    {
        "key": "CnTrmD-Tm1570",
        "label": r"$\mathit{Cn}$TrmD–Tm1570",
        "color": "#7570B3",
        "search_dirs": ["/home/matifortunka/Downloads/PhD/rozdział_V/Cn_fusion/fluo/waw"],
        "file": "/home/matifortunka/Downloads/PhD/rozdział_V/Cn_fusion/fluo/waw/Kinetic Sample1 Scan3_fuzja.csv",
    },
]


# ==============================================================================
# MODELE KINETYCZNE, DOPASOWANIE I METRYKI JAKOŚCI (R2, RMSE, AIC, BIC)
# ==============================================================================
def double_exp(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def mono_exp_drift(t, A, k, c, m):
    return A * np.exp(-k * t) + c + m * t


def calc_fit_metrics(y_true, y_pred, n_params):
    n = len(y_true)
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(residuals)))
    max_abs_res = float(np.max(np.abs(residuals)))
    rmse = float(np.sqrt(ss_res / max(n, 1)))
    if ss_res > 0 and n > n_params:
        aic = n * np.log(ss_res / n) + 2 * n_params
        bic = n * np.log(ss_res / n) + n_params * np.log(n)
    else:
        aic, bic = np.nan, np.nan
    return {
        "residuals": residuals,
        "R2": r2,
        "RMSE": rmse,
        "AIC": aic,
        "BIC": bic,
        "MAE": mae,
        "Max_Abs_Res": max_abs_res
    }


def grid_init_double_exp(t, y):
    """Wyznacza optymalny punkt startowy (A1, k1, A2, k2, c) metodą przeszukiwania
    siatki (k1, k2) z liniowym dopasowaniem amplitud (lstsq)."""
    k1_grid = np.geomspace(0.006, 0.25, 18)
    k2_grid = np.geomspace(0.00015, 0.008, 18)
    best_sse = np.inf
    best_p0 = [0.5 * (y[0] - y[-1]), 0.03, 0.5 * (y[0] - y[-1]), 0.001, y[-1]]
    ones = np.ones_like(t)

    for k1 in k1_grid:
        e1 = np.exp(-k1 * t)
        for k2 in k2_grid:
            if k1 <= 1.8 * k2:
                continue
            e2 = np.exp(-k2 * t)
            X = np.column_stack([e1, e2, ones])
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if not np.all(np.isfinite(coeffs)):
                continue
            sse = float(np.sum((y - X @ coeffs) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_p0 = [float(coeffs[0]), float(k1), float(coeffs[1]), float(k2), float(coeffs[2])]
    return best_p0


def grid_init_mono_drift(t, y):
    """Wyznacza optymalny punkt startowy (A, k, c, m) metodą przeszukiwania siatki k."""
    k_grid = np.geomspace(0.0005, 0.25, 35)
    best_sse = np.inf
    best_p0 = [y[0] - y[-1], 0.01, y[-1], 0.0]
    ones = np.ones_like(t)

    for k in k_grid:
        ek = np.exp(-k * t)
        X = np.column_stack([ek, ones, t])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        if not np.all(np.isfinite(coeffs)):
            continue
        sse = float(np.sum((y - X @ coeffs) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_p0 = [float(coeffs[0]), float(k), float(coeffs[1]), float(coeffs[2])]
    return best_p0


def fit_both_models(t, y):
    """Dopasowuje oba konkurencyjne modele do tego samego przebiegu (t, y)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Double exponential: A1 * exp(-k1 * t) + A2 * exp(-k2 * t) + c
    p0_de = grid_init_double_exp(t, y)
    bounds_de = ([-np.inf, 1e-4, -np.inf, 1e-5, -np.inf], [np.inf, 1.0, np.inf, 0.2, np.inf])
    popt_de, _ = curve_fit(double_exp, t, y, p0=p0_de, bounds=bounds_de, maxfev=80000)
    if popt_de[1] < popt_de[3]:
        popt_de = np.array([popt_de[2], popt_de[3], popt_de[0], popt_de[1], popt_de[4]])
    y_fit_de = double_exp(t, *popt_de)
    metrics_de = calc_fit_metrics(y, y_fit_de, n_params=5)
    res_de = {
        "y_fit": y_fit_de,
        "popt": popt_de,
        "t_half1": np.log(2) / popt_de[1],
        "t_half2": np.log(2) / popt_de[3],
        **metrics_de,
    }

    # 2. Mono exponential + linear drift: A * exp(-k * t) + c + m * t
    p0_md = grid_init_mono_drift(t, y)
    bounds_md = ([-np.inf, 1e-4, -np.inf, -np.inf], [np.inf, 1.0, np.inf, np.inf])
    popt_md, _ = curve_fit(mono_exp_drift, t, y, p0=p0_md, bounds=bounds_md, maxfev=60000)
    y_fit_md = mono_exp_drift(t, *popt_md)
    metrics_md = calc_fit_metrics(y, y_fit_md, n_params=4)
    res_md = {
        "y_fit": y_fit_md,
        "popt": popt_md,
        "t_half1": np.log(2) / popt_md[1],
        "drift_m": popt_md[3],
        **metrics_md,
    }

    return res_de, res_md


# ==============================================================================
# WCZYTYWANIE PLIKÓW CD I FL (Z OBSŁUGĄ WSZYSTKICH FORMATÓW)
# ==============================================================================
def read_cd_kinetics(filepath):
    """Wczytuje pojedynczą trajektorię kinetyczną CD (ProDataCSV lub CSV)."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    if lines and lines[0].strip() == "ProDataCSV":
        data_block_start = -1
        df_full = None
        for i, line in enumerate(lines):
            if line.strip() == "CircularDichroism":
                data_block_start = i
            if data_block_start != -1 and line.strip().startswith("Time") and "Wavelength" in line:
                df_full = pd.read_csv(StringIO("".join(lines[i + 1:])), sep=",", header=None)
                break
        if df_full is None:
            raise ValueError(f"Brak sekcji 'CircularDichroism' w pliku {filepath}")
        df_full = df_full.iloc[:, :2].copy()
        df_full.columns = ["Time", "CircularDichroism"]
    else:
        first_line = lines[0].strip()
        second_line = lines[1].strip() if len(lines) > 1 else ""
        is_min = ("Time (min)" in first_line) or ("Time (min)" in second_line)
        skiprows = 0 if re.match(r"^-?\d+(\.\d*)?(?:[eE][+\-]?\d+)?", first_line.split(",")[0]) else 1
        df_full = pd.read_csv(StringIO("".join(lines)), skiprows=skiprows, sep=",").dropna(how="all", axis=1)
        df_full = df_full.iloc[:, :2].copy()
        df_full.columns = ["Time", "CircularDichroism"]
        if is_min:
            df_full["Time"] = pd.to_numeric(df_full["Time"], errors="coerce") * 60.0

    df_full["Time"] = pd.to_numeric(df_full["Time"], errors="coerce")
    df_full["CircularDichroism"] = pd.to_numeric(df_full["CircularDichroism"], errors="coerce")
    df_full = df_full.dropna()

    t_vals = df_full["Time"].values
    wrap_idx = np.where(np.diff(t_vals) <= 0)[0]
    if wrap_idx.size > 0:
        df_full = df_full.iloc[:wrap_idx[0] + 1].copy()

    return df_full.drop_duplicates(subset=["Time"], keep="first").reset_index(drop=True)


def read_fl_kinetics(filepath):
    """Wczytuje pojedynczą trajektorię kinetyczną fluorescencji (Horiba Labels/ns lub CSV)."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_lines = lines[:15]
    is_new_format = any(
        line.startswith(("Labels,", "Type,", "XAxis,", "YAxis,", "Start,"))
        for line in header_lines
    )

    if is_new_format:
        start_idx = None
        for i, line in enumerate(lines[:25]):
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    start_idx = i
                    break
                except ValueError:
                    continue

        if start_idx is None:
            return pd.DataFrame(columns=["Time", "Intensity"])

        times, intensities = [], []
        prev_t = -np.inf
        for line in lines[start_idx:]:
            s = line.strip()
            if not s:
                if times:
                    break
                continue
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            if len(parts) < 2:
                if times:
                    break
                continue
            try:
                t_val = float(parts[0])
                y_val = float(parts[1])
            except ValueError:
                break

            if not (np.isfinite(t_val) and np.isfinite(y_val)):
                continue
            if t_val <= prev_t and len(times) > 1:
                break
            if y_val <= 0.0 or y_val > 1e10:
                continue

            times.append(t_val)
            intensities.append(y_val)
            prev_t = t_val

        df = pd.DataFrame({"Time": times, "Intensity": intensities})
        if not df.empty and df["Time"].max() > 1e5:
            df["Time"] = df["Time"] / 1e9  # ns -> s
        return df.reset_index(drop=True)
    else:
        first_line = header_lines[0].strip() if len(header_lines) > 0 else ""
        second_line = header_lines[1].strip() if len(header_lines) > 1 else ""
        is_min = ("Time (min)" in first_line) or ("Time (min)" in second_line)

        df_full = pd.read_csv(filepath, skiprows=2, header=None, sep=",").dropna(how="all", axis=1)
        df_full = df_full.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()

        time_col = df_full.iloc[:, 0].copy()
        if is_min:
            time_col *= 60.0

        if not time_col.empty:
            initial_time = time_col.iloc[0]
            wrap_index_series = time_col[1:].sub(initial_time).abs().lt(1e-6)
            if not wrap_index_series.empty and wrap_index_series.any():
                wrap_index = wrap_index_series.idxmax()
                if wrap_index > 1:
                    df_full = df_full.iloc[:wrap_index].copy()
                    time_col = time_col.iloc[:wrap_index]

        df_full.iloc[:, 0] = time_col
        df_full.columns = ["Time", "Intensity"]
        return df_full.reset_index(drop=True)


def find_representative_file(search_dirs, explicit_filename, reader_fn, min_duration_s=1200.0):
    """Znajduje wskazany plik lub automatycznie wybiera reprezentatywny plik CSV."""
    if explicit_filename and os.path.isfile(explicit_filename):
        return explicit_filename

    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        if explicit_filename:
            direct = os.path.join(d, explicit_filename)
            if os.path.isfile(direct):
                return direct
        for fp in sorted(glob.glob(os.path.join(d, "*.csv"))):
            fname = os.path.basename(fp)
            if fname.lower().startswith(("fitting", "fitted", "multiple", "spectra")):
                continue
            candidates.append(fp)

    best_fp = None
    best_score = (-1, -1.0)
    for fp in candidates:
        try:
            df = reader_fn(fp)
            if len(df) < 30:
                continue
            dur = float(df["Time"].iloc[-1] - df["Time"].iloc[0])
            meets_min = 1 if dur >= min_duration_s else 0
            score = (meets_min, min(dur, 4000.0))
            if score > best_score:
                best_score = score
                best_fp = fp
        except Exception:
            continue

    return best_fp


def plot_residuals_with_sg(ax_res, t, residuals, color):
    """Rysuje półprzezroczyste punkty surowych rezyduów oraz gładką linię Savitzky-Golay."""
    ax_res.scatter(
        t, residuals, s=RES_POINT_SIZE, color=color,
        alpha=RES_POINT_ALPHA, linewidths=0, rasterized=True, zorder=2
    )
    res_smooth = smooth_residuals_sg(t, residuals)
    ax_res.plot(
        t, res_smooth, color=darker(color, factor=0.82),
        lw=RES_SMOOTH_LW, alpha=RES_SMOOTH_ALPHA, zorder=4
    )


# ==============================================================================
# RYSOWANIE PANELI Z PODWYKRESAMI REZYDUÓW (2x2 -> 4 PANELE A-D)
# ==============================================================================
def make_figure_model_selection():
    os.makedirs(PATHS["fig_out_dir"], exist_ok=True)
    os.makedirs(PATHS["tables_out_dir"], exist_ok=True)

    plt.rcParams.update(FIG55_RC)

    fig = plt.figure(figsize=FIG55_SIZE, dpi=300)
    outer_gs = fig.add_gridspec(2, 2, wspace=0.14, hspace=0.21)

    def build_panel_pair(grid_cell):
        inner_gs = grid_cell.subgridspec(2, 1, height_ratios=[3.3, 1.0], hspace=0.05)
        ax_main = fig.add_subplot(inner_gs[0])
        ax_res = fig.add_subplot(inner_gs[1], sharex=ax_main)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.tick_params(axis="x", length=0)
        return ax_main, ax_res

    axA_main, axA_res = build_panel_pair(outer_gs[0, 0])  # CD: Double-exponential
    axB_main, axB_res = build_panel_pair(outer_gs[0, 1])  # CD: Mono-exponential + drift
    axC_main, axC_res = build_panel_pair(outer_gs[1, 0])  # FL: Double-exponential
    axD_main, axD_res = build_panel_pair(outer_gs[1, 1])  # FL: Mono-exponential + drift

    comparison_rows = []

    # --------------------------------------------------------------------------
    # RZĄD 1: FAR-UV CD (222 nm) — Panel A (Double-Exp) vs Panel B (Mono + Drift)
    # --------------------------------------------------------------------------
    cd_res_abs_max = 0.0
    cd_max_t = 0.0

    for cfg in CD_TRACES:
        fp = find_representative_file(cfg["search_dirs"], cfg["file"], read_cd_kinetics, min_duration_s=1200.0)
        if fp is None:
            print(f"⚠️ Nie znaleziono pliku CD dla {cfg['key']}: {cfg['file']}")
            continue

        # Stężenie pobierane bezpośrednio ze słownika w skrypcie
        conc_uM = float(cfg["conc_uM"])
        df = read_cd_kinetics(fp)
        t = df["Time"].values + DEAD_TIME_S
        y_mrw = (df["CircularDichroism"].values * 1e6) / (conc_uM * cfg["pathlength_mm"] * cfg["n_res"])
        y_scaled = y_mrw / 1e3  # [theta]_MRW * 10^-3

        t_end = CD_FIT_END_S if CD_FIT_END_S is not None else float(np.max(t))
        mask = (t >= DEAD_TIME_S) & (t <= t_end)
        t_m, y_m = t[mask], y_scaled[mask]
        if len(t_m) < 15:
            continue

        cd_max_t = max(cd_max_t, float(np.max(t_m)))
        res_de, res_md = fit_both_models(t_m, y_m)

        # Panel A: Double-exponential
        axA_main.scatter(
            t_m, y_m, s=POINT_SIZE, color=cfg["color"], alpha=POINT_ALPHA,
            linewidths=0, rasterized=True, label=cfg["label"]
        )
        axA_main.plot(t_m, res_de["y_fit"], color=darker(cfg["color"]), ls="--", lw=FIT_LW, zorder=3)
        plot_residuals_with_sg(axA_res, t_m, res_de["residuals"], cfg["color"])

        # Panel B: Mono-exponential + linear drift
        axB_main.scatter(
            t_m, y_m, s=POINT_SIZE, color=cfg["color"], alpha=POINT_ALPHA,
            linewidths=0, rasterized=True, label=cfg["label"]
        )
        axB_main.plot(t_m, res_md["y_fit"], color=darker(cfg["color"]), ls="--", lw=FIT_LW, zorder=3)
        plot_residuals_with_sg(axB_res, t_m, res_md["residuals"], cfg["color"])

        cd_res_abs_max = max(
            cd_res_abs_max,
            float(np.nanpercentile(np.abs(res_de["residuals"]), 99.5)),
            float(np.nanpercentile(np.abs(res_md["residuals"]), 99.5)),
        )

        comparison_rows.append({
            "Construct": cfg["key"], "Modality": "MM_CD", "File": os.path.basename(fp),
            "Conc_uM": conc_uM,
            "R2_double_exp": res_de["R2"], "RMSE_double_exp": res_de["RMSE"],
            "AIC_double_exp": res_de["AIC"], "BIC_double_exp": res_de["BIC"],
            "R2_mono_drift": res_md["R2"], "RMSE_mono_drift": res_md["RMSE"],
            "AIC_mono_drift": res_md["AIC"], "BIC_mono_drift": res_md["BIC"],
            "Delta_AIC_mono_minus_double": res_md["AIC"] - res_de["AIC"],
            "Delta_BIC_mono_minus_double": res_md["BIC"] - res_de["BIC"],
            "t_half1_DE": res_de["t_half1"],
            "t_half2_DE": res_de["t_half2"],
            "MAE_double_exp": res_de["MAE"],
            "MaxRes_double_exp": res_de["Max_Abs_Res"],
            "t_half_MD": res_md["t_half1"],
            "MAE_mono_drift": res_md["MAE"],
            "MaxRes_mono_drift": res_md["Max_Abs_Res"],
        })
        print(
            f"[CD | {cfg['key']}] {os.path.basename(fp)} (c={conc_uM:g} uM): "
            f"R2_DE={res_de['R2']:.4f} vs R2_MD={res_md['R2']:.4f} | "
            f"ΔBIC={res_md['BIC'] - res_de['BIC']:+.1f}"
        )

    for ax_m, ax_r in [(axA_main, axA_res), (axB_main, axB_res)]:
        if cd_max_t > 0:
            ax_m.set_xlim(0.0, cd_max_t)
        ax_m.grid(True, linestyle=":", alpha=0.5)
        ax_r.axhline(0.0, color="black", ls="--", lw=0.9, alpha=0.75)
        ax_r.grid(True, linestyle=":", alpha=0.5)
        ax_r.set_xlabel("Time (s)")
        ax_r.tick_params(axis="both", labelsize=FS_RES_TICK)
        if cd_res_abs_max > 0:
            lim_r = 1.25 * cd_res_abs_max
            ax_r.set_ylim(-lim_r, lim_r)

    axA_main.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ (\mathrm{deg\cdot cm^2\cdot dmol^{-1}})$", fontsize=FS_LABEL)
    axA_res.set_ylabel("Residuals", fontsize=FS_RES_LABEL)
    style_legend(axA_main, loc="lower right", markerscale=2.8)
    style_legend(axB_main, loc="lower right", markerscale=2.8)

    axA_main.set_title(r"Far-UV CD ($\mathbf{222\ nm}$) – Double-Exponential Fit", fontweight="bold", fontsize=FS_TITLE, pad=8)
    axB_main.set_title(r"Far-UV CD ($\mathbf{222\ nm}$) – Single-Exponential + Drift", fontweight="bold", fontsize=FS_TITLE, pad=8)

    # --------------------------------------------------------------------------
    # RZĄD 2: FLUORESCENCE — Panel C (Double-Exp) vs Panel D (Mono + Drift)
    # --------------------------------------------------------------------------
    fl_res_abs_max = 0.0
    fl_max_t = 0.0

    for cfg in FL_TRACES:
        fp = find_representative_file(cfg["search_dirs"], cfg["file"], read_fl_kinetics, min_duration_s=1200.0)
        if fp is None:
            print(f"⚠️ Nie znaleziono pliku FL dla {cfg['key']}: {cfg['file']}")
            continue

        df = read_fl_kinetics(fp)
        t = df["Time"].values + DEAD_TIME_S
        y_raw = df["Intensity"].values

        t_end = FL_FIT_END_S if FL_FIT_END_S is not None else float(np.max(t))
        mask = (t >= DEAD_TIME_S) & (t <= t_end)
        t_m, y_m = t[mask], y_raw[mask].copy()
        if len(t_m) < 15:
            continue

        if NORMALIZE_FL_SIGNALS:
            y_min, y_max = float(np.nanmin(y_m)), float(np.nanmax(y_m))
            if y_max > y_min:
                y_m = (y_m - y_min) / (y_max - y_min)

        fl_max_t = max(fl_max_t, float(np.max(t_m)))
        res_de, res_md = fit_both_models(t_m, y_m)

        # Panel C: Double-exponential
        axC_main.scatter(
            t_m, y_m, s=POINT_SIZE, color=cfg["color"], alpha=POINT_ALPHA,
            linewidths=0, rasterized=True, label=cfg["label"]
        )
        axC_main.plot(t_m, res_de["y_fit"], color=darker(cfg["color"]), ls="--", lw=FIT_LW, zorder=3)
        plot_residuals_with_sg(axC_res, t_m, res_de["residuals"], cfg["color"])

        # Panel D: Mono-exponential + linear drift
        axD_main.scatter(
            t_m, y_m, s=POINT_SIZE, color=cfg["color"], alpha=POINT_ALPHA,
            linewidths=0, rasterized=True, label=cfg["label"]
        )
        axD_main.plot(t_m, res_md["y_fit"], color=darker(cfg["color"]), ls="--", lw=FIT_LW, zorder=3)
        plot_residuals_with_sg(axD_res, t_m, res_md["residuals"], cfg["color"])

        fl_res_abs_max = max(
            fl_res_abs_max,
            float(np.nanpercentile(np.abs(res_de["residuals"]), 99.5)),
            float(np.nanpercentile(np.abs(res_md["residuals"]), 99.5)),
        )

        comparison_rows.append({
            "Construct": cfg["key"], "Modality": "MM_FL", "File": os.path.basename(fp),
            "Conc_uM": np.nan,
            "R2_double_exp": res_de["R2"], "RMSE_double_exp": res_de["RMSE"],
            "AIC_double_exp": res_de["AIC"], "BIC_double_exp": res_de["BIC"],
            "R2_mono_drift": res_md["R2"], "RMSE_mono_drift": res_md["RMSE"],
            "AIC_mono_drift": res_md["AIC"], "BIC_mono_drift": res_md["BIC"],
            "Delta_AIC_mono_minus_double": res_md["AIC"] - res_de["AIC"],
            "Delta_BIC_mono_minus_double": res_md["BIC"] - res_de["BIC"],
            "t_half1_DE": res_de["t_half1"],
            "t_half2_DE": res_de["t_half2"],
            "MAE_double_exp": res_de["MAE"],
            "MaxRes_double_exp": res_de["Max_Abs_Res"],
            "t_half_MD": res_md["t_half1"],
            "MAE_mono_drift": res_md["MAE"],
            "MaxRes_mono_drift": res_md["Max_Abs_Res"],
        })
        print(
            f"[FL | {cfg['key']}] {os.path.basename(fp)}: "
            f"R2_DE={res_de['R2']:.4f} vs R2_MD={res_md['R2']:.4f} | "
            f"ΔBIC={res_md['BIC'] - res_de['BIC']:+.1f}"
        )

    for ax_m, ax_r in [(axC_main, axC_res), (axD_main, axD_res)]:
        if fl_max_t > 0:
            ax_m.set_xlim(0.0, fl_max_t)
        if NORMALIZE_FL_SIGNALS:
            ax_m.set_ylim(-0.05, 1.08)
        ax_m.grid(True, linestyle=":", alpha=0.5)
        ax_r.axhline(0.0, color="black", ls="--", lw=0.9, alpha=0.75)
        ax_r.grid(True, linestyle=":", alpha=0.5)
        ax_r.set_xlabel("Time (s)")
        ax_r.tick_params(axis="both", labelsize=FS_RES_TICK)
        if fl_res_abs_max > 0:
            lim_r = 1.25 * fl_res_abs_max
            ax_r.set_ylim(-lim_r, lim_r)

    fl_ylabel = "Normalized Fluorescence (a.u.)" if NORMALIZE_FL_SIGNALS else "Fluorescence Intensity (a.u.)"
    axC_main.set_ylabel(fl_ylabel, fontsize=FS_LABEL)
    axC_res.set_ylabel("Residuals", fontsize=FS_RES_LABEL)
    style_legend(axC_main, loc="upper right", markerscale=2.8)
    style_legend(axD_main, loc="upper right", markerscale=2.8)

    axC_main.set_title("Intrinsic Fluorescence – Double-Exponential Fit", fontweight="bold", fontsize=FS_TITLE, pad=8)
    axD_main.set_title("Intrinsic Fluorescence – Single-Exponential + Drift", fontweight="bold", fontsize=FS_TITLE, pad=8)

    # Wyrównanie wertykalnego środka liter wszystkich opisów osi Y w lewej kolumnie
    LABEL_X = -0.145
    for ax, y_pos in [(axA_main, 0.56), (axA_res, 0.50), (axC_main, 0.54), (axC_res, 0.50)]:
        ax.yaxis.label.set_va("center")
        ax.yaxis.set_label_coords(LABEL_X, y_pos)

    # Litery paneli A, B, C, D podniesione nad górną krawędź wykresów
    for ax_m, letter in zip([axA_main, axB_main, axC_main, axD_main], ["A", "B", "C", "D"]):
        ax_m.text(-0.11, 1.135, letter, transform=ax_m.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.065)

    out_pdf = os.path.join(PATHS["fig_out_dir"], "Figura_5_5_Model_Selection.pdf")
    out_png = os.path.join(PATHS["fig_out_dir"], "Figura_5_5_Model_Selection.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    # plt.show()

    if comparison_rows:
        df_comp = pd.DataFrame(comparison_rows)
        comp_csv = os.path.join(PATHS["tables_out_dir"], "model_selection_comparison.csv")
        df_comp.to_csv(comp_csv, index=False)
        print("\n--- PORÓWNANIE MODELI (DOUBLE-EXP vs MONO-EXP + DRIFT) ---")
        cols_show = [
            "Construct", "Modality", "File", "Conc_uM",
            "R2_double_exp", "R2_mono_drift",
            "RMSE_double_exp", "RMSE_mono_drift",
            "Delta_BIC_mono_minus_double",
        ]
        print(df_comp[cols_show].to_string(index=False))
        print(f"\n✅ Zapisano tabelę porównawczą:\n  {comp_csv}")

    print(f"\n✅ Zapisano figurę:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    make_figure_model_selection()