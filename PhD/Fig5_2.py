import os
import re
import glob
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

# ==============================================================================
# KONFIGURACJA CZCIONEK I PALETY BARW (ZGODNA Z ROZDZIAŁEM 4 ORAZ FIG. 5.1 I 5.3)
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"  # POGRUBIONA KURSYWA (\mathsf)
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

# Rozmiary czcionek identyczne jak w Figurach 4.3, 5.1 oraz 5.3
FS_LABEL = 15        # etykiety osi
FS_TICK = 13         # podziałki osi
FS_LEGEND = 11.5     # legendy
FS_TITLE = 13        # tytuły paneli, bold
FS_PANEL = 14.5      # litery paneli A-F, bold
FS_CBAR_TICK = 10.5  # podziałki colorbara
FS_CBAR_LABEL = 11   # opis colorbara

FIG52_RC = {
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
}

# Szerokość = 12.0 in jak w Fig. 4.3, 5.1 i 5.3 (identyczna skala czcionek w LaTeX)
FIG52_SIZE = (12.0, 14.2)

# True -> tytuły paneli zaczynają się od "CnTm1570 – ..."
SHOW_CONSTRUCT_IN_TITLES = True

plasma_trimmed = LinearSegmentedColormap.from_list(
    "plasma_trimmed", plt.cm.plasma(np.linspace(0.0, 0.88, 256))
)


def darker(color, factor=0.7):
    """Ciemniejszy odcień koloru (linia dopasowania widoczna na tle punktów)."""
    r, g, b = mpl.colors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def style_legend(ax, handles=None, labels=None, loc="upper right", markerscale=1.0, ncol=1):
    """Legenda w stylu Fig. 4.3 / 5.1 / 5.3: biała ramka bez krawędzi, uchwyt po prawej."""
    anchor = (1, 0.02) if "lower" in loc else (1, 0.98)
    kw = dict(
        loc=loc, bbox_to_anchor=anchor, fontsize=FS_LEGEND,
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="none",
        markerfirst=False, handlelength=2.6, handletextpad=0.8,
        scatterpoints=1, markerscale=markerscale, ncol=ncol,
    )
    leg = ax.legend(handles, labels, **kw) if handles is not None else ax.legend(**kw)
    for text_obj in leg.get_texts():
        text_obj.set_ha("right")
    return leg


# ==============================================================================
# ŚCIEŻKI I PARAMETRY KONSTRUKTU Tm1570 (CnTm1570)
# ==============================================================================
PATHS = {
    "mm_cd_base_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/CD",
    "cd_sk_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/CD/SK",
    "mm_fl_cam_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/fluo/cam",
    "mm_fl_waw_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/fluo/waw",
    "fl_sk_file": "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/fluo/Tm1570/spectra_kinetics/MultipleEmScan1.csv",
    "fig_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch05",
    "tables_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/tables",
}

PROTEIN_PARAMS = {
    "construct": "Tm1570",
    "mw": 25573.18,
    "n_res": 224,                      # MRW = 25573.18 / 224 = 114.165982143
    "pathlength_mm": 1.0,
    "condition": "6.0 M GdmCl, 298 K",
}

# ==============================================================================
# KONFIGURACJA POSZCZEGÓLNYCH TECHNIK
# ==============================================================================

# 1. Simple Kinetics Fluorescence (cam: indywidualne zakresy; waw: do 2000 s)
MM_FL_CAM_MAX_TIMES = {
    "Tm_12h1": 3000.0,
    "Tm_12h2": 5000.0,
    "Tm_12h4": 1700.0,
    "Tm_12h6": 1250.0,
    "Tm_12h7": 600.0,
    "Tm_12h8": 1100.0,
    "Tm_12h9": 6600.0,
    "Tm_12h12": 4400.0,
}

MM_FL_SETTINGS = {
    "cam_default_dead_time": 30.0,     # Seria 1 (cam): domyślny czas martwy 30 s
    "waw_default_dead_time": 45.0,     # Seria 2 (waw): domyślny czas martwy 45 s
    "cam_boost_threshold": 12.0,       # Seria 1: jeśli y[0] < 12, pomnóż przez 1.5
    "cam_boost_factor": 1.5,
    "waw_boost_threshold": 2.0e5,      # Seria 2: jeśli y[0] < 2e5, pomnóż przez 2.0
    "waw_boost_factor": 2.0,
    "fit_start": 0.0,
    "waw_fit_end": 2000.0,
    "cam_default_fit_end": 4000.0,
    "plot_xlim": 2000.0,               # Zakres osi X na wykresie w Panelu A [s]
    "point_size": 5,                   # Rozmiar punktów surowych danych w Panelu A
    "point_alpha": 0.55,               # Przezroczystość punktów w Panelu A
    "fit_lw": 1.6,                     # Grubość linii dopasowania w Panelu A
    "color_cam": "#1F78B4",
    "color_waw": "#D95F02",
}

# 2. Spectra-Kinetics Fluorescence (340 nm, model double_exp_drift, zakres 0-14500 s)
FL_SK_CONFIG = {
    "label": "Scan #1",
    "time_interval": 32.0,
    "dead_time": 30.0,
    "fit_range": (0.0, 14500.0),
    "fit_model": "double_exp_drift",
    "target_wl": 340.0,
    "baseline_wl": 400.0,
    "smoothing": True,
    "window_size": 25,
    "poly_order": 3,
    "color": "#1F78B4",
}

# 3. Simple Kinetics Far-UV CD (wszystkie foldery xµM, model double_exponential)
MM_CD_SETTINGS = {
    "default_dead_time": 30.0,         # Czas martwy 30 s dla wszystkich pomiarów w Panelu D
    "use_dead_times_file": False,      # Wymusza 30 s dla wszystkich plików w Panelu D
    "exclude_plot_concs": [1.0],       # Stężenia pomijane na wykresie D (zostają w fitach i tabelach)
    "fit_start": 0.0,
    "fit_end": None,                   # Pełny zakres pliku
    "point_size": 6,                   # Rozmiar punktów surowych danych w Panelu D
    "point_alpha": 0.55,               # Przezroczystość punktów w Panelu D
    "fit_lw": 1.6,                     # Grubość linii dopasowania w Panelu D
    "conc_colors": {
        1.0: "#1B9E77",
        2.0: "#66A61E",
        3.0: "#7570B3",
        5.0: "#E7298A",
    },
    "fallback_palette": ["#1B9E77", "#7570B3", "#E7298A", "#D95F02", "#0721A6", "#E6AB02"],
}

# 4. Spectra-Kinetics Far-UV CD (folder CD/SK)
CD_SK_MAIN_SPECTRA_STEM = "Tm_1200003_clean"   # Plik pokazywany jako widma w Panelu E
CD_SK_DEFAULT_CONC_UM = 5.0                    # Domyślne stężenie [uM] dla wszystkich plików SK CD

CD_SK_FILE_OVERRIDES = {
    "Tm_30sp_12h00000":            {"max_time_s": 4500.0, "conc_uM": 5.0},
    "Tm1570_2h_spectra_2min00001": {"max_time_s": 3000.0, "conc_uM": 5.0},
    "Tm15_1200002_clean":          {"max_time_s": 4000.0, "conc_uM": 5.0},
    "Tm_1200003_clean":            {"max_time_s": 2000.0, "conc_uM": 8.0},
    "Tm15_12h0000000009":          {"max_time_s": None,   "conc_uM": 5.0},
    "Tm15_12h00000":               {"max_time_s": None,   "conc_uM": 5.0},
    "spectra_2h00000":             {"max_time_s": None,   "conc_uM": 4.0},
}

CD_SK_SETTINGS = {
    "dead_time": 30.0,
    "nm_per_sec": 0.4,
    "sec_per_spectrum": 109.65,
    "time_mode": "Repeats",
    "fit_model": "double_exponential",
    "hv_threshold": 950.0,
    "smoothing_window": 15,
    "smoothing_polyorder": 3,
    "target_wavelength": 222.0,
    "integration_range": (217.0, 250.0),
    "integration_sign": "negative",
    "baseline_correction": True,
    "baseline_wavelength": 250.0,
    "palette": ["#1B9E77", "#0721A6", "#75053B", "#D95F02", "#7570B3", "#E6AB02"],
}


# ==============================================================================
# MODELE MATEMATYCZNE I DOPASOWANIE
# ==============================================================================
def mono_exp(t, A, k, c):
    return A * np.exp(-k * t) + c


def mono_exp_drift(t, A, k, c, m):
    return A * np.exp(-k * t) + c + m * t


def double_exp(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def double_exp_drift(t, A1, k1, A2, k2, c, m):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c + m * t


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


def estimate_initial_k(t, y):
    A0 = y[0] - y[-1]
    C0 = y[-1]
    half_val = C0 + 0.5 * A0
    idx = int(np.abs(y - half_val).argmin())
    t_half = t[idx] - t[0]
    return np.log(2) / t_half if t_half > 0 else 0.002


def clip_p0_to_bounds(p0, bounds):
    """Utrzymuje parametry startowe ściśle wewnątrz przedziału bounds."""
    lb, ub = bounds
    p0_clipped = []
    for val, lo, hi in zip(p0, lb, ub):
        if np.isfinite(lo) and val <= lo:
            val = lo + 1e-6 * max(abs(lo), 1.0)
        if np.isfinite(hi) and val >= hi:
            val = hi - 1e-6 * max(abs(hi), 1.0)
        p0_clipped.append(val)
    return p0_clipped


def fit_trace(t, y, model_name):
    """Ujednolicone dopasowanie modeli kinetycznych z kontrolą granic i kolejności faz."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    A0 = y[0] - y[-1]
    C0 = y[-1]
    k0 = estimate_initial_k(t, y)

    if model_name == "mono_exp":
        bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, 0.5, np.inf])
        p0 = clip_p0_to_bounds([A0, max(k0, 1e-4), C0], bounds)
        popt, pcov = curve_fit(mono_exp, t, y, p0=p0, bounds=bounds, maxfev=30000)
        perr = np.sqrt(np.diag(pcov))
        y_fit = mono_exp(t, *popt)
        A1, k1, c = popt
        eA1, ek1, ec = perr
        return {
            "y_fit": y_fit, "R2": calc_r2(y, y_fit),
            "A1": A1, "A1_err": eA1,
            "k1": k1, "k1_err": ek1,
            "t_half1": np.log(2) / k1, "t_half1_err": (np.log(2) / k1 ** 2) * ek1,
            "A2": np.nan, "A2_err": np.nan,
            "k2": np.nan, "k2_err": np.nan,
            "t_half2": np.nan, "t_half2_err": np.nan,
            "c": c, "c_err": ec,
            "drift_m": np.nan, "drift_m_err": np.nan,
        }

    elif model_name == "mono_exp_drift":
        bounds = ([-np.inf, 1e-5, -np.inf, -np.inf], [np.inf, 0.5, np.inf, np.inf])
        p0 = clip_p0_to_bounds([A0, max(k0, 5e-4), C0, 0.0], bounds)
        popt, pcov = curve_fit(mono_exp_drift, t, y, p0=p0, bounds=bounds, maxfev=40000)
        perr = np.sqrt(np.diag(pcov))
        y_fit = mono_exp_drift(t, *popt)
        A1, k1, c, m = popt
        eA1, ek1, ec, em = perr
        return {
            "y_fit": y_fit, "R2": calc_r2(y, y_fit),
            "A1": A1, "A1_err": eA1,
            "k1": k1, "k1_err": ek1,
            "t_half1": np.log(2) / k1, "t_half1_err": (np.log(2) / k1 ** 2) * ek1,
            "A2": np.nan, "A2_err": np.nan,
            "k2": np.nan, "k2_err": np.nan,
            "t_half2": np.nan, "t_half2_err": np.nan,
            "c": c, "c_err": ec,
            "drift_m": m, "drift_m_err": em,
        }

    elif model_name == "double_exponential":
        bounds = ([-np.inf, 1e-4, -np.inf, 1e-5, -np.inf], [np.inf, 0.5, np.inf, 0.1, np.inf])
        p0 = clip_p0_to_bounds(
            [0.5 * A0, np.clip(k0 * 2.0, 0.005, 0.3), 0.5 * A0, np.clip(k0 * 0.2, 0.0002, 0.02), C0],
            bounds
        )
        popt, pcov = curve_fit(double_exp, t, y, p0=p0, bounds=bounds, maxfev=80000)
        perr = np.sqrt(np.diag(pcov))
        if popt[1] < popt[3]:
            popt = np.array([popt[2], popt[3], popt[0], popt[1], popt[4]])
            perr = np.array([perr[2], perr[3], perr[0], perr[1], perr[4]])
        y_fit = double_exp(t, *popt)
        A1, k1, A2, k2, c = popt
        eA1, ek1, eA2, ek2, ec = perr
        return {
            "y_fit": y_fit, "R2": calc_r2(y, y_fit),
            "A1": A1, "A1_err": eA1,
            "k1": k1, "k1_err": ek1,
            "t_half1": np.log(2) / k1, "t_half1_err": (np.log(2) / k1 ** 2) * ek1,
            "A2": A2, "A2_err": eA2,
            "k2": k2, "k2_err": ek2,
            "t_half2": np.log(2) / k2, "t_half2_err": (np.log(2) / k2 ** 2) * ek2,
            "c": c, "c_err": ec,
            "drift_m": np.nan, "drift_m_err": np.nan,
        }

    elif model_name in ("double_exp_drift", "double_exponential_with_drift"):
        bounds = ([-np.inf, 1e-4, -np.inf, 1e-5, -np.inf, -np.inf], [np.inf, 0.5, np.inf, 0.1, np.inf, np.inf])
        p0 = clip_p0_to_bounds(
            [0.5 * A0, np.clip(k0 * 2.0, 0.005, 0.3), 0.5 * A0, np.clip(k0 * 0.2, 0.0002, 0.02), C0, 0.0],
            bounds
        )
        popt, pcov = curve_fit(double_exp_drift, t, y, p0=p0, bounds=bounds, maxfev=90000)
        perr = np.sqrt(np.diag(pcov))
        if popt[1] < popt[3]:
            popt = np.array([popt[2], popt[3], popt[0], popt[1], popt[4], popt[5]])
            perr = np.array([perr[2], perr[3], perr[0], perr[1], perr[4], perr[5]])
        y_fit = double_exp_drift(t, *popt)
        A1, k1, A2, k2, c, m = popt
        eA1, ek1, eA2, ek2, ec, em = perr
        return {
            "y_fit": y_fit, "R2": calc_r2(y, y_fit),
            "A1": A1, "A1_err": eA1,
            "k1": k1, "k1_err": ek1,
            "t_half1": np.log(2) / k1, "t_half1_err": (np.log(2) / k1 ** 2) * ek1,
            "A2": A2, "A2_err": eA2,
            "k2": k2, "k2_err": ek2,
            "t_half2": np.log(2) / k2, "t_half2_err": (np.log(2) / k2 ** 2) * ek2,
            "c": c, "c_err": ec,
            "drift_m": m, "drift_m_err": em,
        }

    else:
        raise ValueError(f"Nieznany model dopasowania: {model_name}")


def mdeg_to_mrw(cd_mdeg, conc_uM):
    """Przelicza eliptyczność [mdeg] na [theta]_MRW [deg * cm^2 * dmol^-1]."""
    return (np.asarray(cd_mdeg, dtype=float) * 1e6) / (
        conc_uM * PROTEIN_PARAMS["pathlength_mm"] * PROTEIN_PARAMS["n_res"]
    )


def get_display_scale(max_val):
    """Dobiera potęgę 10^k i zwraca (dzielnik, kompletną etykietę jednostki w nawiasie)."""
    if max_val <= 0 or not np.isfinite(max_val):
        return 1.0, "(a.u.)"
    exp10 = int(np.floor(np.log10(max_val)))
    if exp10 >= 3:
        return 10.0 ** exp10, rf"($\times 10^{{{exp10}}}$ a.u.)"
    return 1.0, "(a.u.)"


# ==============================================================================
# FUNKCJE WCZYTUJĄCE DANE
# ==============================================================================
def read_dead_times(folder_path, default_dead_time):
    dead_times = {}
    dead_file = os.path.join(folder_path, "dead_times.txt")
    if not os.path.exists(dead_file):
        return dead_times

    with open(dead_file, "r") as f:
        for line in f:
            parts = [p for p in re.split(r"[\s,;]+", line.strip()) if p]
            if len(parts) >= 2:
                key = parts[0]
                match = re.match(r"^(\d+(?:\.\d+)?)[sS]?$", parts[1])
                if match:
                    val = float(match.group(1))
                    base = os.path.splitext(os.path.basename(key))[0]
                    dead_times[key] = val
                    dead_times[base] = val
    return dead_times


def discover_mm_cd_dirs(base_dir):
    """Wyszukuje wszystkie podfoldery postaci xµM (np. 1µM, 3µM, 5µM) i sortuje je wg stężenia."""
    if not os.path.exists(base_dir):
        print(f"⚠️ Katalog bazowy MM CD nie istnieje: {base_dir}")
        return []

    discovered = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*[uµμ]M", entry, flags=re.IGNORECASE)
        if match:
            conc_uM = float(match.group(1).replace(",", "."))
            discovered.append((conc_uM, full_path))

    discovered.sort(key=lambda item: item[0])
    return discovered


def resolve_cam_max_time(fname):
    """Dopasowuje maksymalny czas dopasowania dla pliku z folderu cam."""
    fbase = os.path.splitext(fname)[0]
    if fbase in MM_FL_CAM_MAX_TIMES:
        return MM_FL_CAM_MAX_TIMES[fbase]
    for key, val in MM_FL_CAM_MAX_TIMES.items():
        if re.match(rf"^{re.escape(key)}(?:_|$)", fbase):
            return val
    return MM_FL_SETTINGS["cam_default_fit_end"]


def read_fl_mm_data(filename):
    """Wczytuje przebiegi kinetyczne FL z folderów 'cam' (od wiersza 2) i 'waw' (Horiba CSV od wiersza 10).
    Zatrzymuje odczyt na końcu pierwszego skanu (gdy czas się zeruje lub zaczyna kolejny nagłówek)."""
    with open(filename, "r") as f:
        lines = f.readlines()

    header_lines = lines[:15]
    is_waw_format = any(
        line.startswith(("Labels,", "Type,", "XAxis,", "YAxis,", "Start,"))
        for line in header_lines
    )

    if is_waw_format:
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

        times = []
        intensities = []
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
        time_unit = "min" if ("Time (min)" in first_line or "Time (min)" in second_line) else "s"

        df_full = pd.read_csv(filename, skiprows=2, header=None, sep=",").dropna(how="all", axis=1)
        df_full = df_full.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()

        time_col = df_full.iloc[:, 0].copy()
        if time_unit == "min":
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


def read_cd_prodata_single(filepath):
    """Wczytuje pojedynczą trajektorię kinetyczną CD z pliku ProDataCSV."""
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
            raise ValueError(f"Nie znaleziono sekcji 'CircularDichroism' w pliku: {filepath}")
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


def extract_prodata_section(lines, section_name):
    start = None
    for i, line in enumerate(lines):
        if line.strip() == section_name:
            start = i + 2
            break
    if start is None:
        raise ValueError(f"Sekcja '{section_name}' nie została znaleziona.")
    section_lines = []
    for line in lines[start:]:
        if not line.strip():
            break
        section_lines.append(line)
    return section_lines


def load_cd_sk_dataset(filepath, sec_per_spec=None):
    """Wczytuje macierz widm czasowych CD oraz HV z pliku ProDataCSV
    z automatyczną detekcją transpozycji (osi długości fali 170-320 nm)."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    cd_lines = extract_prodata_section(lines, "CircularDichroism")
    hv_lines = extract_prodata_section(lines, "HV")

    cd_df = pd.read_csv(StringIO("".join(cd_lines)), skipinitialspace=True)
    hv_df = pd.read_csv(StringIO("".join(hv_lines)), skipinitialspace=True)

    cd_df.rename(columns={cd_df.columns[0]: "Wavelength"}, inplace=True)
    cd_df.dropna(axis=1, how="all", inplace=True)
    hv_df.rename(columns={hv_df.columns[0]: "Wavelength"}, inplace=True)
    hv_df.dropna(axis=1, how="all", inplace=True)

    row_axis = pd.to_numeric(cd_df.iloc[:, 0], errors="coerce").dropna()
    col_axis = pd.to_numeric(pd.Index(cd_df.columns[1:]), errors="coerce").dropna()
    rows_are_wl = (len(row_axis) > 5) and (row_axis.min() >= 170.0) and (row_axis.max() <= 320.0)
    cols_are_wl = (len(col_axis) > 5) and (col_axis.min() >= 170.0) and (col_axis.max() <= 320.0)

    need_transpose = cols_are_wl and not rows_are_wl

    if need_transpose:
        cd_times_raw = pd.to_numeric(cd_df.iloc[:, 0], errors="coerce")
        cd_wavelengths = pd.to_numeric(cd_df.columns[1:], errors="coerce")
        cd_values = cd_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
        cd_df = pd.DataFrame(cd_values.T, index=cd_wavelengths, columns=cd_times_raw).reset_index()
        cd_df.rename(columns={"index": "Wavelength"}, inplace=True)

        hv_times_raw = pd.to_numeric(hv_df.iloc[:, 0], errors="coerce")
        hv_wavelengths = pd.to_numeric(hv_df.columns[1:], errors="coerce")
        hv_values = hv_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
        hv_df = pd.DataFrame(hv_values.T, index=hv_wavelengths, columns=hv_times_raw).reset_index()
        hv_df.rename(columns={"index": "Wavelength"}, inplace=True)

    cd_df["Wavelength"] = pd.to_numeric(cd_df["Wavelength"], errors="coerce")
    hv_df["Wavelength"] = pd.to_numeric(hv_df["Wavelength"], errors="coerce")
    cd_df = cd_df.dropna(subset=["Wavelength"]).sort_values("Wavelength").reset_index(drop=True)
    hv_df = hv_df.dropna(subset=["Wavelength"]).sort_values("Wavelength").reset_index(drop=True)

    common_wl = np.intersect1d(cd_df["Wavelength"].values, hv_df["Wavelength"].values)
    cd_df = cd_df[cd_df["Wavelength"].isin(common_wl)].reset_index(drop=True)
    hv_df = hv_df[hv_df["Wavelength"].isin(common_wl)].reset_index(drop=True)
    wavelengths = cd_df["Wavelength"].values

    dt_spec = sec_per_spec if sec_per_spec is not None else CD_SK_SETTINGS["sec_per_spectrum"]
    if CD_SK_SETTINGS["time_mode"] == "Repeats":
        old_cd_cols = [c for c in cd_df.columns if c != "Wavelength"]
        old_hv_cols = [c for c in hv_df.columns if c != "Wavelength"]
        cd_col_map = {i * dt_spec: col for i, col in enumerate(old_cd_cols)}
        hv_col_map = {i * dt_spec: col for i, col in enumerate(old_hv_cols)}
    else:
        cd_col_map = {float(col): col for col in cd_df.columns if col != "Wavelength"}
        hv_col_map = {float(col): col for col in hv_df.columns if col != "Wavelength"}

    return wavelengths, cd_df, hv_df, cd_col_map, hv_col_map, need_transpose


def load_fl_sk_dataset(filename):
    """Wczytuje widma czasowe fluorescencji (MultipleEmScan1.csv)."""
    data_start_row = -1
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            if len(parts) > 1 and re.match(r"^-?\d+(?:\.\d+)?$", parts[0].strip()):
                data_start_row = i
                break
    if data_start_row == -1:
        raise ValueError(f"Nie znaleziono bloku danych w pliku: {filename}")

    df = pd.read_csv(filename, skiprows=data_start_row, header=None)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(axis=0, how="all", inplace=True)
    df.rename(columns={df.columns[0]: "Wavelength"}, inplace=True)
    df.dropna(subset=["Wavelength"], inplace=True)
    return df.reset_index(drop=True)


# ==============================================================================
# ZAPIS WYNIKÓW DO CSV (IDEMPOTENTNY)
# ==============================================================================
def record_fit_row(records_list, modality, file_label, conc_uM, model_name, range_str, res):
    records_list.append({
        "Construct": PROTEIN_PARAMS["construct"],
        "Modality": modality,
        "Condition": PROTEIN_PARAMS["condition"],
        "Concentration_uM": conc_uM,
        "File": file_label,
        "Model": model_name,
        "Fit_Range_s": range_str,
        "A1": res["A1"], "A1_err": res["A1_err"],
        "k1_s_inv": res["k1"], "k1_err": res["k1_err"],
        "t_half1_s": res["t_half1"], "t_half1_err": res["t_half1_err"],
        "A2": res["A2"], "A2_err": res["A2_err"],
        "k2_s_inv": res["k2"], "k2_err": res["k2_err"],
        "t_half2_s": res["t_half2"], "t_half2_err": res["t_half2_err"],
        "c": res["c"], "c_err": res["c_err"],
        "drift_m": res["drift_m"], "drift_m_err": res["drift_m_err"],
        "R2": res["R2"],
    })


def append_mean_row(mean_list, modality, model_name, range_str, fit_results):
    if not fit_results:
        return
    k1_arr = np.array([r["k1"] for r in fit_results], dtype=float)
    th1_arr = np.array([r["t_half1"] for r in fit_results], dtype=float)
    k2_arr = np.array([r["k2"] for r in fit_results], dtype=float)
    th2_arr = np.array([r["t_half2"] for r in fit_results], dtype=float)
    m_arr = np.array([r["drift_m"] for r in fit_results], dtype=float)

    ddof_val = 1 if len(fit_results) > 1 else 0
    mean_list.append({
        "Construct": PROTEIN_PARAMS["construct"],
        "Modality": modality,
        "Condition": PROTEIN_PARAMS["condition"],
        "Model": model_name,
        "Fit_Range_s": range_str,
        "N_runs": len(fit_results),
        "k1_mean_s_inv": np.nanmean(k1_arr),
        "k1_sd_s_inv": np.nanstd(k1_arr, ddof=ddof_val),
        "t_half1_mean_s": np.nanmean(th1_arr),
        "t_half1_sd_s": np.nanstd(th1_arr, ddof=ddof_val),
        "k2_mean_s_inv": np.nanmean(k2_arr) if not np.all(np.isnan(k2_arr)) else np.nan,
        "k2_sd_s_inv": np.nanstd(k2_arr, ddof=ddof_val) if not np.all(np.isnan(k2_arr)) else np.nan,
        "t_half2_mean_s": np.nanmean(th2_arr) if not np.all(np.isnan(th2_arr)) else np.nan,
        "t_half2_sd_s": np.nanstd(th2_arr, ddof=ddof_val) if not np.all(np.isnan(th2_arr)) else np.nan,
        "drift_m_mean": np.nanmean(m_arr) if not np.all(np.isnan(m_arr)) else np.nan,
        "drift_m_sd": np.nanstd(m_arr, ddof=ddof_val) if not np.all(np.isnan(m_arr)) else np.nan,
    })


def update_fits_tables(individual_rows, mean_rows, tables_dir):
    os.makedirs(tables_dir, exist_ok=True)
    fits_path = os.path.join(tables_dir, "kinetic_fits.csv")
    mean_path = os.path.join(tables_dir, "kinetic_fits_mean.csv")

    df_new = pd.DataFrame(individual_rows)
    if os.path.exists(fits_path):
        df_old = pd.read_csv(fits_path)
        keys = set(zip(df_new["Construct"], df_new["Modality"]))
        mask_keep = [(c, m) not in keys for c, m in zip(df_old["Construct"], df_old["Modality"])]
        df_combined = pd.concat([df_old[mask_keep], df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(fits_path, index=False)

    df_mean_new = pd.DataFrame(mean_rows)
    if os.path.exists(mean_path):
        df_mean_old = pd.read_csv(mean_path)
        keys_m = set(zip(df_mean_new["Construct"], df_mean_new["Modality"]))
        mask_keep_m = [(c, m) not in keys_m for c, m in zip(df_mean_old["Construct"], df_mean_old["Modality"])]
        df_mean_combined = pd.concat([df_mean_old[mask_keep_m], df_mean_new], ignore_index=True)
    else:
        df_mean_combined = df_mean_new
    df_mean_combined.to_csv(mean_path, index=False)

    print(f"\n✅ Zaktualizowano tabele:\n  {fits_path}\n  {mean_path}")


# ==============================================================================
# PANEL A: Simple Kinetics FL (surowe punkty + linie dopasowania, dwie osie Y)
# ==============================================================================
def plot_panel_a_mm_fl(ax_left, individual_records, mean_records):
    cam_files = sorted(glob.glob(os.path.join(PATHS["mm_fl_cam_dir"], "*.csv")))
    waw_files = sorted(glob.glob(os.path.join(PATHS["mm_fl_waw_dir"], "*.csv")))

    cam_dt_map = read_dead_times(PATHS["mm_fl_cam_dir"], MM_FL_SETTINGS["cam_default_dead_time"])
    waw_dt_map = read_dead_times(PATHS["mm_fl_waw_dir"], MM_FL_SETTINGS["waw_default_dead_time"])

    def prepare_cam_series(files, dt_map, default_dt):
        prepared = []
        for fp in files:
            fname = os.path.basename(fp)
            fbase = os.path.splitext(fname)[0]
            dt = dt_map.get(fname, dt_map.get(fbase, default_dt))
            max_t_file = resolve_cam_max_time(fname)

            df = read_fl_mm_data(fp)
            if df.empty:
                continue
            t = df["Time"].values + dt
            y = df["Intensity"].values
            t0 = MM_FL_SETTINGS["fit_start"] + dt
            mask = (t >= t0) & (t <= max_t_file)
            t_m, y_m = t[mask], y[mask].copy()
            if len(t_m) < 10:
                continue

            scale_factor = 1.0
            if y_m[0] < MM_FL_SETTINGS["cam_boost_threshold"]:
                scale_factor = MM_FL_SETTINGS["cam_boost_factor"]
                y_m = y_m * scale_factor

            prepared.append((f"cam/{fname}", dt, max_t_file, t_m, y_m, scale_factor))
        return prepared

    def prepare_waw_series(files, dt_map, default_dt):
        prepared = []
        for fp in files:
            fname = os.path.basename(fp)
            fbase = os.path.splitext(fname)[0]
            dt = dt_map.get(fname, dt_map.get(fbase, default_dt))
            max_t_file = MM_FL_SETTINGS["waw_fit_end"]

            df = read_fl_mm_data(fp)
            if df.empty:
                continue
            t = df["Time"].values + dt
            y = df["Intensity"].values
            t0 = MM_FL_SETTINGS["fit_start"] + dt
            mask = (t >= t0) & (t <= max_t_file)
            t_m, y_m = t[mask], y[mask].copy()
            if len(t_m) < 10:
                continue

            scale_factor = 1.0
            if y_m[0] < MM_FL_SETTINGS["waw_boost_threshold"]:
                scale_factor = MM_FL_SETTINGS["waw_boost_factor"]
                y_m = y_m * scale_factor

            prepared.append((f"waw/{fname}", dt, max_t_file, t_m, y_m, scale_factor))
        return prepared

    cam_series = prepare_cam_series(cam_files, cam_dt_map, MM_FL_SETTINGS["cam_default_dead_time"])
    waw_series = prepare_waw_series(waw_files, waw_dt_map, MM_FL_SETTINGS["waw_default_dead_time"])

    cam_max = max((float(np.nanmax(s[4])) for s in cam_series), default=1.0)
    waw_max = max((float(np.nanmax(s[4])) for s in waw_series), default=1.0)
    cam_div, cam_unit_label = get_display_scale(cam_max)
    waw_div, waw_unit_label = get_display_scale(waw_max)

    ax_right = ax_left.twinx()
    all_fit_results = []

    for idx, (label_file, dt, max_t_file, t_m, y_m, scale_factor) in enumerate(cam_series):
        lbl = rf"Series 1 ($n={len(cam_series)}$)" if idx == 0 else None
        ax_left.scatter(
            t_m, y_m / cam_div,
            s=MM_FL_SETTINGS["point_size"], color=MM_FL_SETTINGS["color_cam"],
            alpha=MM_FL_SETTINGS["point_alpha"], linewidths=0, rasterized=True, label=lbl
        )

        try:
            res = fit_trace(t_m, y_m, "double_exponential")
            ax_left.plot(
                t_m, res["y_fit"] / cam_div,
                color=darker(MM_FL_SETTINGS["color_cam"]), ls="--", lw=MM_FL_SETTINGS["fit_lw"], zorder=3
            )
            all_fit_results.append(res)
            record_fit_row(
                individual_records, "MM_FL", label_file, np.nan,
                "double_exponential", f"{int(round(dt))}-{int(round(max_t_file))}", res
            )
            scale_info = f", scale={scale_factor:g}x" if scale_factor != 1.0 else ""
            print(
                f"[MM FL cam] {label_file} ({int(round(dt))}-{int(round(max_t_file))}s, dt={dt}s, n={len(t_m)}{scale_info}): "
                f"t1/2_1 = {res['t_half1']:.1f} s, t1/2_2 = {res['t_half2']:.1f} s"
            )
        except Exception as e:
            print(f"❌ Dopasowanie MM FL nie powiodło się dla {label_file}: {e}")

    for idx, (label_file, dt, max_t_file, t_m, y_m, scale_factor) in enumerate(waw_series):
        lbl = rf"Series 2 ($n={len(waw_series)}$)" if idx == 0 else None
        ax_right.scatter(
            t_m, y_m / waw_div,
            s=MM_FL_SETTINGS["point_size"], color=MM_FL_SETTINGS["color_waw"],
            alpha=MM_FL_SETTINGS["point_alpha"], linewidths=0, rasterized=True, label=lbl
        )

        try:
            res = fit_trace(t_m, y_m, "double_exponential")
            ax_right.plot(
                t_m, res["y_fit"] / waw_div,
                color=darker(MM_FL_SETTINGS["color_waw"]), ls="--", lw=MM_FL_SETTINGS["fit_lw"], zorder=3
            )
            all_fit_results.append(res)
            record_fit_row(
                individual_records, "MM_FL", label_file, np.nan,
                "double_exponential", f"{int(round(dt))}-{int(round(max_t_file))}", res
            )
            scale_info = f", scale={scale_factor:g}x" if scale_factor != 1.0 else ""
            print(
                f"[MM FL waw] {label_file} ({int(round(dt))}-{int(round(max_t_file))}s, dt={dt}s, n={len(t_m)}, "
                f"y0={y_m[0]:.2e}{scale_info}): t1/2_1 = {res['t_half1']:.1f} s, t1/2_2 = {res['t_half2']:.1f} s"
            )
        except Exception as e:
            print(f"❌ Dopasowanie MM FL nie powiodło się dla {label_file}: {e}")

    append_mean_row(mean_records, "MM_FL", "double_exponential", "dt-2000 / custom", all_fit_results)

    ax_left.set_xlim(0.0, MM_FL_SETTINGS["plot_xlim"])
    ax_left.set_xlabel("Time (s)")
    ax_left.set_ylabel(f"Fluorescence — Series 1 {cam_unit_label}", color=MM_FL_SETTINGS["color_cam"])
    ax_right.set_ylabel(f"Fluorescence — Series 2 {waw_unit_label}", color=MM_FL_SETTINGS["color_waw"])
    ax_left.tick_params(axis="y", colors=MM_FL_SETTINGS["color_cam"])
    ax_right.tick_params(axis="y", colors=MM_FL_SETTINGS["color_waw"])
    ax_left.grid(True, linestyle=":", alpha=0.5)

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    if h1 or h2:
        style_legend(ax_left, h1 + h2, l1 + l2, loc="upper right", markerscale=3.0)


# ==============================================================================
# PANELE B I C: SK FL (Widma 0-14500 s oraz Spectra-Kinetics przy 340 nm)
# ==============================================================================
def plot_panels_bc_fl_sk(fig, ax_spectra, ax_kinetics, individual_records, mean_records):
    fp = PATHS["fl_sk_file"]
    fname = os.path.basename(fp)
    df = load_fl_sk_dataset(fp)
    wavelengths = df["Wavelength"].values
    scan_cols = df.columns[1:]

    t_start, t_end = FL_SK_CONFIG["fit_range"]
    target_idx = int(np.abs(wavelengths - FL_SK_CONFIG["target_wl"]).argmin())
    bl_idx = int(np.abs(wavelengths - FL_SK_CONFIG["baseline_wl"]).argmin())

    times_s = []
    int_340 = []
    spectra_to_plot = []

    for i, col in enumerate(scan_cols):
        t_s = FL_SK_CONFIG["dead_time"] + i * FL_SK_CONFIG["time_interval"]
        if t_s < t_start or t_s > t_end:
            continue

        y_raw = df[col].values.astype(float)
        y_corr = y_raw - y_raw[bl_idx] if FL_SK_CONFIG["baseline_wl"] else y_raw.copy()

        if FL_SK_CONFIG["smoothing"] and len(y_corr) >= FL_SK_CONFIG["window_size"]:
            w = FL_SK_CONFIG["window_size"]
            if w % 2 == 0:
                w += 1
            y_corr = savgol_filter(y_corr, w, FL_SK_CONFIG["poly_order"])

        times_s.append(t_s)
        int_340.append(y_corr[target_idx])
        spectra_to_plot.append((t_s, y_corr))

    times_s = np.array(times_s, dtype=float)
    int_340 = np.array(int_340, dtype=float)

    # Panel B: Widma emisji w zakresie 300-400 nm (skala * 10^4 a.u.)
    max_t_b = float(t_end)
    norm_b = Normalize(vmin=0.0, vmax=max_t_b)
    step = max(1, len(spectra_to_plot) // 50)
    max_spec_val = 0.0

    for idx_s, (t_s, spec_y) in enumerate(spectra_to_plot):
        wl_mask = (wavelengths >= 300.0) & (wavelengths <= 400.0)
        if np.any(wl_mask):
            max_spec_val = max(max_spec_val, float(np.nanmax(spec_y[wl_mask])))
        if idx_s % step == 0:
            ax_spectra.plot(
                wavelengths, spec_y / 1e4,
                color=plasma_trimmed(norm_b(t_s)), lw=1.25, alpha=0.88
            )

    upper_y_b = max(1.0, float(np.ceil((max_spec_val / 1e4) * 1.08)))
    ax_spectra.set_xlim(300.0, 400.0)
    ax_spectra.set_ylim(0.0, upper_y_b)
    ax_spectra.set_xlabel("Wavelength (nm)")
    ax_spectra.set_ylabel(r"Fluorescence Intensity ($\times 10^4$ a.u.)")
    ax_spectra.grid(True, linestyle=":", alpha=0.5)

    cax_b = ax_spectra.inset_axes([0.74, 0.50, 0.045, 0.42])
    sm_b = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_b)
    sm_b.set_array([])
    cbar_b = fig.colorbar(sm_b, cax=cax_b, orientation="vertical")
    cbar_b.solids.set_rasterized(True)
    cbar_b.set_ticks([0, 3500, 7000, 10500, 14500])
    cbar_b.set_ticklabels(["0", "3500", "7000", "10500", "14500"])
    cbar_b.ax.tick_params(labelsize=FS_CBAR_TICK)
    cbar_b.set_label("Time (s)", fontsize=FS_CBAR_LABEL, labelpad=5)

    # Panel C: Kinetyka przy 340 nm (0-14500 s, double_exp_drift)
    sk_fl_fits = []
    try:
        res = fit_trace(times_s, int_340, FL_SK_CONFIG["fit_model"])
        sk_fl_fits.append(res)
        range_str = f"{int(round(times_s[0]))}-{int(t_end)}"
        record_fit_row(
            individual_records, "SK_FL", fname, np.nan,
            FL_SK_CONFIG["fit_model"], range_str, res
        )
        print(
            f"[SK FL 340 nm - {FL_SK_CONFIG['fit_model']}] {fname} ({range_str}s): "
            f"t1/2_1 = {res['t_half1']:.1f} s, t1/2_2 = {res['t_half2']:.1f} s, m = {res['drift_m']:.4e}"
        )

        ax_kinetics.scatter(
            times_s, int_340 / 1e4, s=12, alpha=0.65,
            color=FL_SK_CONFIG["color"], label=FL_SK_CONFIG["label"]
        )
        ax_kinetics.plot(
            times_s, res["y_fit"] / 1e4,
            color=darker(FL_SK_CONFIG["color"]), ls="--", lw=1.6
        )
    except Exception as e:
        print(f"❌ Dopasowanie SK FL nie powiodło się dla {fname}: {e}")

    append_mean_row(mean_records, "SK_FL", FL_SK_CONFIG["fit_model"], f"{int(t_start)}-{int(t_end)}", sk_fl_fits)

    ax_kinetics.set_xlim(-50, float(t_end))
    ax_kinetics.set_xlabel("Time (s)")
    ax_kinetics.set_ylabel(r"Intensity at 340 nm ($\times 10^4$ a.u.)")
    ax_kinetics.grid(True, linestyle=":", alpha=0.5)


# ==============================================================================
# PANEL D: Simple Kinetics CD (surowe punkty + linie dopasowania, dt = 30 s)
# ==============================================================================
def plot_panel_d_mm_cd(ax, individual_records, mean_records):
    conc_dirs = discover_mm_cd_dirs(PATHS["mm_cd_base_dir"])
    if not conc_dirs:
        print(f"⚠️ Nie znaleziono podfolderów xµM w katalogu: {PATHS['mm_cd_base_dir']}")
        return

    found_summary = ", ".join(f"{c:g} uM ({os.path.basename(p)})" for c, p in conc_dirs)
    print(f"[MM CD] Znalezione foldery stężeń: {found_summary}")

    use_dt_file = MM_CD_SETTINGS.get("use_dead_times_file", False)
    exclude_plot_concs = MM_CD_SETTINGS.get("exclude_plot_concs", [])
    parent_dt_map = (
        read_dead_times(PATHS["mm_cd_base_dir"], MM_CD_SETTINGS["default_dead_time"])
        if use_dt_file else {}
    )
    fallback_palette = MM_CD_SETTINGS["fallback_palette"]

    all_fit_results = []
    max_t_plotted = 0.0
    min_y_plotted = np.inf
    max_y_plotted = -np.inf
    plotted_conc_count = 0

    for conc_idx, (conc_uM, folder_path) in enumerate(conc_dirs):
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        if not csv_files:
            print(f"⚠️ Brak plików CSV w folderze: {folder_path}")
            continue

        dt_map = dict(parent_dt_map)
        if use_dt_file:
            dt_map.update(read_dead_times(folder_path, MM_CD_SETTINGS["default_dead_time"]))

        color = MM_CD_SETTINGS["conc_colors"].get(
            conc_uM, fallback_palette[conc_idx % len(fallback_palette)]
        )
        show_on_plot = not any(np.isclose(conc_uM, ex) for ex in exclude_plot_concs)

        valid_traces = []
        for fp in csv_files:
            fname = os.path.basename(fp)
            fbase = os.path.splitext(fname)[0]
            dt = dt_map.get(fname, dt_map.get(fbase, MM_CD_SETTINGS["default_dead_time"]))

            df = read_cd_prodata_single(fp)
            t = df["Time"].values + dt
            y_mrw = mdeg_to_mrw(df["CircularDichroism"].values, conc_uM)

            t0 = MM_CD_SETTINGS["fit_start"] + dt
            t_end = MM_CD_SETTINGS["fit_end"] if MM_CD_SETTINGS["fit_end"] is not None else float(np.max(t))
            mask = (t >= t0) & (t <= t_end)
            t_m, y_m = t[mask], y_mrw[mask]
            if len(t_m) < 10:
                continue
            valid_traces.append((fname, dt, t_m, y_m, t_end))

        n_conc = len(valid_traces)
        if show_on_plot and n_conc > 0:
            plotted_conc_count += 1

        for idx, (fname, dt, t_m, y_m, t_end) in enumerate(valid_traces):
            y_plot = y_m / 1e3
            if show_on_plot:
                max_t_plotted = max(max_t_plotted, float(np.max(t_m)))
                min_y_plotted = min(min_y_plotted, float(np.nanmin(y_plot)))
                max_y_plotted = max(max_y_plotted, float(np.nanmax(y_plot)))

                lbl = rf"${conc_uM:g}\ \mu\mathrm{{M}}$ ($n={n_conc}$)" if idx == 0 else None
                ax.scatter(
                    t_m, y_plot,
                    s=MM_CD_SETTINGS["point_size"], color=color,
                    alpha=MM_CD_SETTINGS["point_alpha"], linewidths=0, rasterized=True, label=lbl
                )

            try:
                res = fit_trace(t_m, y_m, "double_exponential")
                if show_on_plot:
                    ax.plot(
                        t_m, res["y_fit"] / 1e3,
                        color=darker(color), ls="--", lw=MM_CD_SETTINGS["fit_lw"], zorder=3
                    )
                all_fit_results.append(res)
                range_str = f"{int(round(t_m[0]))}-{int(round(t_end))}"
                record_fit_row(
                    individual_records, "MM_CD", f"{conc_uM:g}uM/{fname}", conc_uM,
                    "double_exponential", range_str, res
                )
                hidden_note = " [ukryto na wykresie]" if not show_on_plot else ""
                print(
                    f"[MM CD {conc_uM:g} uM] {fname} (dt={dt}s){hidden_note}: "
                    f"t1/2_1 = {res['t_half1']:.1f} s, t1/2_2 = {res['t_half2']:.1f} s"
                )
            except Exception as e:
                print(f"❌ Dopasowanie MM CD nie powiodło się dla {fname}: {e}")

    append_mean_row(mean_records, "MM_CD", "double_exponential", "30s-full", all_fit_results)

    if max_t_plotted > 0:
        ax.set_xlim(0.0, max_t_plotted)
    if np.isfinite(min_y_plotted) and np.isfinite(max_y_plotted):
        span = max(max_y_plotted - min_y_plotted, 0.5)
        ax.set_ylim(min_y_plotted - 0.08 * span, max_y_plotted + 0.12 * span)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$")
    style_legend(ax, loc="lower right", markerscale=2.8, ncol=2 if plotted_conc_count > 3 else 1)
    ax.grid(True, linestyle=":", alpha=0.5)


# ==============================================================================
# PANELE E I F (+ FIGURA DO ANEKSU): SK CD
# ==============================================================================
def evaluate_fit_on_grid(res, t_grid):
    """Oblicza wartość dopasowanej krzywej na gęstej siatce czasu t_grid."""
    t_grid = np.asarray(t_grid, dtype=float)
    if np.isfinite(res.get("k2", np.nan)) and np.isfinite(res.get("A2", np.nan)):
        if np.isfinite(res.get("drift_m", np.nan)):
            return double_exp_drift(t_grid, res["A1"], res["k1"], res["A2"], res["k2"], res["c"], res["drift_m"])
        return double_exp(t_grid, res["A1"], res["k1"], res["A2"], res["k2"], res["c"])
    if np.isfinite(res.get("drift_m", np.nan)):
        return mono_exp_drift(t_grid, res["A1"], res["k1"], res["c"], res["drift_m"])
    return mono_exp(t_grid, res["A1"], res["k1"], res["c"])


def get_cd_sk_file_settings(fname):
    """Pobiera ustawienia (max_time_s, conc_uM, sec_per_spectrum) dla pliku w CD/SK."""
    fbase = os.path.splitext(fname)[0]
    override = CD_SK_FILE_OVERRIDES.get(fbase, {})
    is_main = (fbase == CD_SK_MAIN_SPECTRA_STEM) or fbase.startswith(CD_SK_MAIN_SPECTRA_STEM)
    return {
        "stem": fbase,
        "max_time_s": override.get("max_time_s", None),
        "conc_uM": override.get("conc_uM", CD_SK_DEFAULT_CONC_UM),
        "sec_per_spectrum": override.get("sec_per_spectrum", CD_SK_SETTINGS["sec_per_spectrum"]),
        "is_main_spectra": is_main,
    }


def process_all_cd_sk(fig_main, ax_e_spectra, ax_f_222, individual_records, mean_records):
    target_wl = CD_SK_SETTINGS["target_wavelength"]
    bl_wl = CD_SK_SETTINGS["baseline_wavelength"]
    nm_per_sec = CD_SK_SETTINGS["nm_per_sec"]
    base_dt = CD_SK_SETTINGS["dead_time"]
    hv_thresh = CD_SK_SETTINGS["hv_threshold"]
    w_smooth = CD_SK_SETTINGS["smoothing_window"]
    p_smooth = CD_SK_SETTINGS["smoothing_polyorder"]
    int_min, int_max = CD_SK_SETTINGS["integration_range"]
    int_sign = CD_SK_SETTINGS["integration_sign"].lower()
    palette = CD_SK_SETTINGS["palette"]

    sk_files = sorted(glob.glob(os.path.join(PATHS["cd_sk_dir"], "*.csv")))
    if not sk_files:
        print(f"⚠️ Nie znaleziono plików CSV w folderze: {PATHS['cd_sk_dir']}")
        return

    fits_222_list = []
    fits_int_list = []
    appendix_spectra_datasets = []
    appendix_int_datasets = []
    max_t_222 = 0.0
    min_y_222 = np.inf
    max_y_222 = -np.inf
    main_spectra_plotted = False

    for idx_file, fp in enumerate(sk_files):
        fname = os.path.basename(fp)
        fcfg = get_cd_sk_file_settings(fname)
        scan_label = f"Scan #{idx_file + 1}"
        color = palette[idx_file % len(palette)]
        conc_uM = float(fcfg["conc_uM"])
        max_time_s = fcfg["max_time_s"]

        wavelengths, cd_df, hv_df, cd_col_map, hv_col_map, was_transposed = load_cd_sk_dataset(
            fp, sec_per_spec=fcfg["sec_per_spectrum"]
        )
        cd_times = sorted(cd_col_map.keys())
        hv_times = sorted(hv_col_map.keys())

        print(
            f"[SK CD load] {fname} ({scan_label}): transposed={was_transposed}, "
            f"conc={conc_uM:g} uM, max_time={max_time_s if max_time_s else 'full'}, "
            f"wl={wavelengths.min():.1f}-{wavelengths.max():.1f} nm, scans={len(cd_times)}"
        )

        time_shift_spec = base_dt + (bl_wl - target_wl) / nm_per_sec
        spectra_curves = []
        for cd_t in cd_times:
            t_eff_spec = cd_t + time_shift_spec
            if max_time_s is not None and t_eff_spec > max_time_s:
                continue

            cd_col = cd_col_map[cd_t]
            hv_t = min(hv_times, key=lambda t: abs(t - cd_t))
            hv_col = hv_col_map[hv_t]

            cd_vals = cd_df[cd_col].values.astype(float)
            hv_vals = hv_df[hv_col].values.astype(float)
            mask = hv_vals <= hv_thresh
            x_wl = wavelengths[mask]
            y_mdeg = cd_vals[mask]

            if len(y_mdeg) >= w_smooth:
                y_mdeg = savgol_filter(y_mdeg, w_smooth, p_smooth)
            if CD_SK_SETTINGS["baseline_correction"] and len(x_wl) > 0:
                b_idx = int(np.argmin(np.abs(x_wl - bl_wl)))
                y_mdeg = y_mdeg - y_mdeg[b_idx]

            y_mrw_scaled = mdeg_to_mrw(y_mdeg, conc_uM) / 1e3
            spectra_curves.append((t_eff_spec, x_wl, y_mrw_scaled))

        if fcfg["is_main_spectra"] and spectra_curves:
            main_spectra_plotted = True
            t_secs_e = [s[0] for s in spectra_curves]
            max_t_e = float(max(t_secs_e))
            norm_e = Normalize(vmin=0.0, vmax=max_t_e)
            step_e = max(1, len(spectra_curves) // 55)
            for idx_s, (t_sec, x_wl, y_mrw_s) in enumerate(spectra_curves):
                if idx_s % step_e == 0 or idx_s == len(spectra_curves) - 1:
                    ax_e_spectra.plot(
                        x_wl, y_mrw_s,
                        color=plasma_trimmed(norm_e(t_sec)), lw=1.25, alpha=0.88
                    )

            ax_e_spectra.set_xlim(right=250.0)
            ax_e_spectra.set_xlabel("Wavelength (nm)")
            ax_e_spectra.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$")
            ax_e_spectra.grid(True, linestyle=":", alpha=0.5)

            cax_e = ax_e_spectra.inset_axes([0.75, 0.10, 0.045, 0.42])
            sm_e = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_e)
            sm_e.set_array([])
            cbar_e = fig_main.colorbar(sm_e, cax=cax_e, orientation="vertical")
            cbar_e.solids.set_rasterized(True)
            cbar_e.locator = MaxNLocator(nbins=5, integer=True)
            cbar_e.update_ticks()
            cbar_e.ax.tick_params(labelsize=FS_CBAR_TICK)
            cbar_e.set_label("Time (s)", fontsize=FS_CBAR_LABEL, labelpad=5)
        else:
            appendix_spectra_datasets.append((scan_label, fname, spectra_curves))

        closest_idx = int(np.argmin(np.abs(wavelengths - target_wl)))
        actual_wl = float(wavelengths[closest_idx])
        eff_dt_222 = base_dt + (bl_wl - actual_wl) / nm_per_sec

        t_222, y_222_mrw = [], []
        for cd_t in cd_times:
            t_eff = cd_t + eff_dt_222
            if max_time_s is not None and t_eff > max_time_s:
                continue

            cd_col = cd_col_map[cd_t]
            hv_t = min(hv_times, key=lambda t: abs(t - cd_t))
            hv_col = hv_col_map[hv_t]

            hv_val = float(hv_df[hv_col].values[closest_idx])
            if hv_val <= hv_thresh:
                curve = cd_df[cd_col].values.astype(float)
                if len(curve) >= w_smooth:
                    curve = savgol_filter(curve, w_smooth, p_smooth)
                if CD_SK_SETTINGS["baseline_correction"]:
                    b_idx = int(np.argmin(np.abs(wavelengths - bl_wl)))
                    val_mdeg = curve[closest_idx] - curve[b_idx]
                else:
                    val_mdeg = curve[closest_idx]

                t_222.append(t_eff)
                y_222_mrw.append(float(mdeg_to_mrw(val_mdeg, conc_uM)))

        sk_cd_model = CD_SK_SETTINGS["fit_model"]

        t_222 = np.array(t_222, dtype=float)
        y_222_mrw = np.array(y_222_mrw, dtype=float)
        if len(t_222) > 0:
            max_t_222 = max(max_t_222, float(np.max(t_222)))
            y_222_scaled = y_222_mrw / 1e3
            min_y_222 = min(min_y_222, float(np.nanmin(y_222_scaled)))
            max_y_222 = max(max_y_222, float(np.nanmax(y_222_scaled)))

            ax_f_222.scatter(t_222, y_222_scaled, s=16, color=color, alpha=0.8, label=scan_label)
            if len(t_222) >= 6:
                try:
                    res_222 = fit_trace(t_222, y_222_mrw, sk_cd_model)
                    t_222_dense = np.linspace(t_222[0], t_222[-1], 1000)
                    y_222_dense = evaluate_fit_on_grid(res_222, t_222_dense)
                    ax_f_222.plot(t_222_dense, y_222_dense / 1e3, color=darker(color), ls="--", lw=1.6)
                    fits_222_list.append(res_222)
                    range_str = f"{int(round(t_222[0]))}-{int(round(t_222[-1]))}"
                    record_fit_row(
                        individual_records, "SK_CD_222nm", fname, conc_uM,
                        sk_cd_model, range_str, res_222
                    )
                    print(
                        f"[SK CD 222 nm] {fname} ({scan_label}): "
                        f"t1/2_1 = {res_222['t_half1']:.1f} ± {res_222['t_half1_err']:.1f} s, "
                        f"t1/2_2 = {res_222['t_half2']:.1f} ± {res_222['t_half2_err']:.1f} s "
                        f"(R2 = {res_222['R2']:.3f})"
                    )
                except Exception as e:
                    print(f"❌ Dopasowanie SK CD (222 nm) nie powiodło się dla {fname}: {e}")

        mid_wl = 0.5 * (int_min + int_max)
        eff_dt_int = base_dt + (bl_wl - mid_wl) / nm_per_sec

        t_int, y_int_mrw = [], []
        for cd_t in cd_times:
            t_eff_i = cd_t + eff_dt_int
            if max_time_s is not None and t_eff_i > max_time_s:
                continue

            cd_col = cd_col_map[cd_t]
            hv_t = min(hv_times, key=lambda t: abs(t - cd_t))
            hv_col = hv_col_map[hv_t]

            cd_vals = cd_df[cd_col].values.astype(float)
            hv_vals = hv_df[hv_col].values.astype(float)
            mask = hv_vals <= hv_thresh
            x_sub = wavelengths[mask]
            in_rng = (x_sub >= int_min) & (x_sub <= int_max)
            x_sub = x_sub[in_rng]
            y_sub = cd_vals[mask][in_rng]

            if len(y_sub) >= w_smooth:
                y_sub = savgol_filter(y_sub, w_smooth, p_smooth)
            if CD_SK_SETTINGS["baseline_correction"] and len(x_sub) > 0:
                b_idx = int(np.argmin(np.abs(x_sub - bl_wl)))
                y_sub = y_sub - y_sub[b_idx]

            y_sub = np.where((y_sub > 0) if int_sign == "positive" else (y_sub < 0), y_sub, 0.0)
            if len(x_sub) > 1:
                area_mdeg_nm = trapezoid(y_sub, x_sub)
                area_mrw_nm = float(mdeg_to_mrw(area_mdeg_nm, conc_uM))
                t_int.append(t_eff_i)
                y_int_mrw.append(area_mrw_nm)

        t_int = np.array(t_int, dtype=float)
        y_int_mrw = np.array(y_int_mrw, dtype=float)
        res_int = None
        if len(t_int) >= 6:
            try:
                res_int = fit_trace(t_int, y_int_mrw, sk_cd_model)
                fits_int_list.append(res_int)
                range_str = f"{int(round(t_int[0]))}-{int(round(t_int[-1]))}"
                record_fit_row(
                    individual_records, "SK_CD_integrated", fname, conc_uM,
                    sk_cd_model, range_str, res_int
                )
            except Exception as e:
                print(f"❌ Dopasowanie SK CD (całkowanie) nie powiodło się dla {fname}: {e}")

        appendix_int_datasets.append((scan_label, fname, conc_uM, color, t_int, y_int_mrw, res_int))

    append_mean_row(mean_records, "SK_CD_222nm", sk_cd_model, "custom / full", fits_222_list)
    append_mean_row(mean_records, "SK_CD_integrated", sk_cd_model, "custom / full", fits_int_list)

    ax_f_222.set_xlim(0.0, max_t_222 * 1.02 if max_t_222 > 0 else 12000.0)
    if np.isfinite(min_y_222) and np.isfinite(max_y_222):
        span_f = max(max_y_222 - min_y_222, 0.5)
        ax_f_222.set_ylim(min_y_222 - 0.10 * span_f, max_y_222 + 0.12 * span_f)

    ax_f_222.set_xlabel("Time (s)")
    ax_f_222.set_ylabel(r"$[\theta]_{\mathrm{MRW},222} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$")
    style_legend(ax_f_222, loc="lower right", markerscale=1.8, ncol=2 if len(sk_files) > 4 else 1)
    ax_f_222.grid(True, linestyle=":", alpha=0.5)

    make_appendix_cd_sk_figure(appendix_spectra_datasets, appendix_int_datasets)


def make_appendix_cd_sk_figure(appendix_spectra_datasets, appendix_int_datasets):
    n_spec_panels = len(appendix_spectra_datasets)
    total_panels = n_spec_panels + 1
    ncols = min(3, total_panels)
    nrows = int(np.ceil(total_panels / ncols))

    TP = r"$\mathsf{Cn}$Tm1570 – " if SHOW_CONSTRUCT_IN_TITLES else ""

    fig_app, axes_grid = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), dpi=300)
    axes_flat = np.atleast_1d(axes_grid).ravel()

    for idx_ax, (scan_label, fname, spectra_curves) in enumerate(appendix_spectra_datasets):
        ax = axes_flat[idx_ax]
        y_vis_min, y_vis_max = np.inf, -np.inf
        if spectra_curves:
            t_secs = [s[0] for s in spectra_curves]
            max_ts = max(t_secs)
            norm = Normalize(vmin=0.0, vmax=max_ts)
            for t_sec, x_wl, y_mrw_s in spectra_curves:
                ax.plot(x_wl, y_mrw_s, color=plasma_trimmed(norm(t_sec)), lw=1.2, alpha=0.88)
                vis_mask = (x_wl >= 210.0) & (x_wl <= 250.0)
                if np.any(vis_mask):
                    y_vis_min = min(y_vis_min, float(np.nanmin(y_mrw_s[vis_mask])))
                    y_vis_max = max(y_vis_max, float(np.nanmax(y_mrw_s[vis_mask])))
            cax = ax.inset_axes([0.75, 0.10, 0.045, 0.42])
            sm = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm)
            sm.set_array([])
            cbar = fig_app.colorbar(sm, cax=cax, orientation="vertical")
            cbar.solids.set_rasterized(True)
            cbar.locator = MaxNLocator(nbins=5, integer=True)
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=FS_CBAR_TICK)
            cbar.set_label("Time (s)", fontsize=FS_CBAR_LABEL, labelpad=5)
        ax.set_xlim(210.0, 250.0)
        if np.isfinite(y_vis_min) and np.isfinite(y_vis_max):
            span_y = max(y_vis_max - y_vis_min, 0.5)
            ax.set_ylim(y_vis_min - 0.06 * span_y, y_vis_max + 0.06 * span_y)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$")
        ax.set_title(f"{TP}Far-UV CD Spectra ({scan_label})", fontweight="bold", fontsize=FS_TITLE, pad=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    ax_int = axes_flat[n_spec_panels]
    max_t = 0.0
    for scan_label, fname, conc_uM, color, t_int, y_int_mrw, res_int in appendix_int_datasets:
        if len(t_int) == 0:
            continue
        max_t = max(max_t, float(np.max(t_int)))
        ax_int.scatter(t_int, y_int_mrw / 1e4, s=16, color=color, alpha=0.8, label=scan_label)
        if res_int is not None:
            t_dense = np.linspace(t_int[0], t_int[-1], 1000)
            y_dense = evaluate_fit_on_grid(res_int, t_dense)
            ax_int.plot(t_dense, y_dense / 1e4, color=darker(color), ls="--", lw=1.6)

    ax_int.set_xlim(0.0, max_t * 1.02 if max_t > 0 else 12000.0)
    ax_int.set_xlabel("Time (s)")
    ax_int.set_ylabel(r"Integrated $[\theta]_{\mathrm{MRW}}\ (217\text{--}250\ \mathrm{nm}) \times 10^{-4}$")
    ax_int.set_title(f"{TP}Integrated Spectra-Kinetics CD", fontweight="bold", fontsize=FS_TITLE, pad=8)
    style_legend(ax_int, loc="lower right", markerscale=1.8, ncol=2 if len(appendix_int_datasets) > 4 else 1)
    ax_int.grid(True, linestyle=":", alpha=0.5)

    panel_letters = [chr(ord("A") + i) for i in range(total_panels)]
    for ax, lbl in zip(axes_flat[:total_panels], panel_letters):
        ax.text(-0.13, 1.06, lbl, transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top")

    for unused_ax in axes_flat[total_panels:]:
        unused_ax.set_visible(False)

    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    out_app_pdf = os.path.join(PATHS["fig_out_dir"], "Figura_A_SK_CD_Tm1570.pdf")
    out_app_png = os.path.join(PATHS["fig_out_dir"], "Figura_A_SK_CD_Tm1570.png")
    plt.savefig(out_app_pdf)
    plt.savefig(out_app_png, dpi=300)
    plt.close(fig_app)
    print(f"✅ Zapisano figurę do aneksu:\n  {out_app_pdf}\n  {out_app_png}")


# ==============================================================================
# GŁÓWNA FUNKCJA BUDUJĄCA FIGURĘ (6 PANELI A-F)
# ==============================================================================
def make_figure_tm1570():
    os.makedirs(PATHS["fig_out_dir"], exist_ok=True)
    os.makedirs(PATHS["tables_out_dir"], exist_ok=True)

    plt.rcParams.update(FIG52_RC)
    TP = r"$\mathsf{Cn}$Tm1570 – " if SHOW_CONSTRUCT_IN_TITLES else ""

    fig, axes = plt.subplots(3, 2, figsize=FIG52_SIZE, dpi=300)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]
    axE, axF = axes[2, 0], axes[2, 1]

    individual_records = []
    mean_records = []

    # Rząd 1: Simple Kinetics FL (A) oraz Fluorescence Spectra (B)
    plot_panel_a_mm_fl(axA, individual_records, mean_records)
    axA.set_title(f"{TP}Fluorescence Kinetics", fontsize=FS_TITLE, fontweight="bold", pad=8)

    # Rząd 1/2: Fluorescence Spectra (B) oraz Spectra-Kinetics Fluorescence przy 340 nm (C)
    plot_panels_bc_fl_sk(fig, axB, axC, individual_records, mean_records)
    axB.set_title(f"{TP}Fluorescence Spectra", fontsize=FS_TITLE, fontweight="bold", pad=8)
    axC.set_title(rf"{TP}Fluorescence Spectra-Kinetics ($\mathbf{{340\ nm}}$)", fontsize=FS_TITLE, fontweight="bold", pad=8)

    # Rząd 2 (prawy): Simple Kinetics Far-UV CD (D)
    plot_panel_d_mm_cd(axD, individual_records, mean_records)
    axD.set_title(rf"{TP}Far-UV CD Kinetics ($\mathbf{{222\ nm}}$)", fontsize=FS_TITLE, fontweight="bold", pad=8)

    # Rząd 3: Far-UV CD Spectra (E) oraz Spectra-Kinetics CD przy 222 nm (F)
    process_all_cd_sk(fig, axE, axF, individual_records, mean_records)
    axE.set_title(f"{TP}Far-UV CD Spectra", fontsize=FS_TITLE, fontweight="bold", pad=8)
    axF.set_title(rf"{TP}Far-UV CD Spectra-Kinetics ($\mathbf{{222\ nm}}$)", fontsize=FS_TITLE, fontweight="bold", pad=8)

    for ax, label in zip([axA, axB, axC, axD, axE, axF], ["A", "B", "C", "D", "E", "F"]):
        ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top")

    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    out_pdf = os.path.join(PATHS["fig_out_dir"], "Figura_5_2_Kinetyka_Tm1570.pdf")
    out_png = os.path.join(PATHS["fig_out_dir"], "Figura_5_2_Kinetyka_Tm1570.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)

    update_fits_tables(individual_records, mean_records, PATHS["tables_out_dir"])

    print("\n--- PODSUMOWANIE UŚREDNIONYCH DOPASOWAŃ (Tm1570) ---")
    df_summary = pd.DataFrame(mean_records)
    if not df_summary.empty:
        cols_show = [
            "Construct", "Modality", "Model", "N_runs",
            "t_half1_mean_s", "t_half1_sd_s",
            "t_half2_mean_s", "t_half2_sd_s"
        ]
        print(df_summary[cols_show].to_string(index=False))

    print(f"\n✅ Zapisano figurę główną:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    make_figure_tm1570()