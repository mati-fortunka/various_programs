import os
import re
import glob
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================================================================
# KONFIGURACJA CZCIONEK (ZGODNA Z ROZDZIAŁEM 4 ORAZ FIG. 5.1-5.3)
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"  # POGRUBIONA KURSYWA (\mathsf)
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"             # Wycisza ostrzeżenie o 'cursive'
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

# Rozmiary czcionek identyczne jak w Figurach 4.3, 5.1, 5.2 oraz 5.3
FS_LABEL = 15        # etykiety osi
FS_TICK = 13         # podziałki osi
FS_LEGEND = 11.5     # legendy
FS_TITLE = 13        # tytuły paneli, bold
FS_PANEL = 14.5      # litery paneli A-D, bold

FIG54_RC = {
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
}

# Szerokość = 12.0 in jak w Fig. 4.3, 5.1, 5.2 i 5.3 (identyczna skala czcionek w LaTeX)
FIG54_SIZE = (12.0, 9.6)

# Parametry wizualizacji punktów surowych danych i linii dopasowania
POINT_SIZE = 6
POINT_ALPHA = 0.55
FIT_LW = 1.6


def darker(color, factor=0.7):
    """Ciemniejszy odcień koloru (linia dopasowania widoczna na tle punktów)."""
    r, g, b = mpl.colors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def style_legend(ax, handles=None, labels=None, loc="upper right", markerscale=1.0, ncol=1):
    """Legenda w stylu Fig. 4.3 / 5.1 / 5.2 / 5.3: biała ramka bez krawędzi, uchwyt po prawej."""
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
# KONFIGURACJA ŚCIEŻEK I PARAMETRÓW
# ==============================================================================
PATHS = {
    "fs_cd_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/F8/CD",
    "fs_fl_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/F8/FL",
    "cd_cd_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/F6/CD",
    "cd_fl_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/F6/FL",
    "fig_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch05",
    "tables_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/tables",
}

# Wybór metody wyznaczania parametrów startowych (p0) dla curve_fit:
#   "heuristic" -> dotychczasowe szacowanie z kształtu pojedynczej krzywej (estimate_initial_k)
#   "table"     -> uśrednione stałe k1, k2 z kinetic_fits_mean.csv (lub TABLE_INITIAL_GUESSES)
INIT_GUESS_SOURCE = "table"

# Wymusza stały czas martwy (30.0 s) dla wszystkich pomiarów w panelach A, B, C i D
USE_DEAD_TIMES_FILE = False

# Kolory dla dwóch stężeń w Panelu D (identyczne jak w Panelu A Figury 5.1)
COLOR_5UM = "#1F78B4"  # Niebieski dla 5.0 uM (lewa oś Y)
COLOR_2UM = "#D95F02"  # Pomarańczowy dla 2.0 uM (prawa oś Y)

# Wbudowane wartości referencyjne z tabeli (fallback, gdyby nie było jeszcze wpisu w CSV)
TABLE_INITIAL_GUESSES = {
    ("FsTrmD-Tm1570", "MM_CD"): {"k1": 0.025036, "k2": 0.000865},
    ("FsTrmD-Tm1570", "MM_FL"): {"k1": 0.039076, "k2": 0.001208},
    ("CdTrmD-Tm1570", "MM_CD"): {"k1": 0.051904, "k2": 0.000873},
    ("CdTrmD-Tm1570", "MM_FL"): {"k1": 0.024466, "k2": 0.000978},
}

CONFIG = {
    "FsTrmD-Tm1570": {
        "n_res": 447,                  # Mw = 50189.16, MRW = 112.28
        "pathlength_mm": 1.0,
        "default_conc_uM": 2.0,
        "cd_default_dead_time": 30.0,
        "fl_default_dead_time": 30.0,
        "cd_fit_model": "double_exponential",
        "cd_fit_start": 0.0,
        "cd_fit_end": None,
        "cd_ylim": (-4.5, -2.0),
        "fl_fit_model": "double_exponential",
        "fl_fit_start": 0.0,
        "fl_fit_end": 1300.0,
        "color_raw": "#17BECF",        # Spójny z Rozdziałem 4 (FsTrmD-Tm1570)
        "color_fit": "#0E717B",
    },
    "CdTrmD-Tm1570": {
        "n_res": 447,                  # Mw = 49761.67, MRW = 111.32
        "pathlength_mm": 1.0,
        "default_conc_uM": 2.0,
        "cd_default_dead_time": 30.0,
        "fl_default_dead_time": 30.0,
        "cd_fit_model": "double_exponential",
        "cd_fit_start": 0.0,
        "cd_fit_end": None,
        "cd_ylim": (-4.5, -2.0),
        "fl_fit_model": "double_exponential",
        "fl_fit_start": 0.0,
        "fl_fit_end": 1500.0,
        "color_raw": "#7570B3",        # Kontrastowy fiolet dla CdTrmD-Tm1570 (Panel C)
        "color_fit": "#3F3B73",
    },
}


# ==============================================================================
# MODELE MATEMATYCZNE
# ==============================================================================
def exponential(t, A, k, c):
    return A * np.exp(-k * t) + c


def single_exponential_with_drift(t, A, k, c, m):
    return A * np.exp(-k * t) + c + m * t


def double_exponential(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


def estimate_initial_k(time, signal):
    """Heurystyczne szacowanie stałej szybkości k z czasu połowicznego zaniku."""
    A0 = signal[0] - signal[-1]
    C = signal[-1]
    half_val = C + A0 / 2.0
    idx = np.abs(signal - half_val).argmin()
    t_half = time[idx] - time[0]
    return np.log(2) / t_half if t_half > 0 else 0.01


def load_table_initial_guesses(tables_dir):
    """Wczytuje uśrednione parametry z kinetic_fits_mean.csv i uzupełnia je
    wbudowanym słownikiem TABLE_INITIAL_GUESSES."""
    guesses = dict(TABLE_INITIAL_GUESSES)
    mean_path = os.path.join(tables_dir, "kinetic_fits_mean.csv")
    if os.path.exists(mean_path):
        try:
            df_mean = pd.read_csv(mean_path)
            for _, row in df_mean.iterrows():
                key = (str(row["Construct"]), str(row["Modality"]))
                entry = {}
                if "k1_mean_s_inv" in row and pd.notna(row["k1_mean_s_inv"]):
                    entry["k1"] = float(row["k1_mean_s_inv"])
                if "k2_mean_s_inv" in row and pd.notna(row["k2_mean_s_inv"]):
                    entry["k2"] = float(row["k2_mean_s_inv"])
                if "drift_m_mean" in row and pd.notna(row["drift_m_mean"]):
                    entry["drift_m"] = float(row["drift_m_mean"])
                if entry:
                    guesses[key] = entry
        except Exception as e:
            print(f"⚠️ Nie udało się odczytać '{mean_path}': {e}. Używam wbudowanych wartości.")
    return guesses


def estimate_linear_amplitudes_double_exp(t, y, k1, k2, A0_fallback, C0_fallback):
    """Dla zadanych k1, k2 z tabeli wyznacza liniową metodą najmniejszych kwadratów
    startowe amplitudy A1, A2 oraz przesunięcie c dopasowane do skali danej krzywej."""
    try:
        X = np.column_stack([np.exp(-k1 * t), np.exp(-k2 * t), np.ones_like(t)])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        A1_est, A2_est, c_est = coeffs
        if np.all(np.isfinite(coeffs)):
            return float(A1_est), float(A2_est), float(c_est)
    except Exception:
        pass
    return 0.5 * A0_fallback, 0.5 * A0_fallback, C0_fallback


def fit_kinetic_trace(time, signal, model_name, table_guess=None):
    """Dopasowuje przebieg kinetyczny i zwraca ustandaryzowany słownik parametrów."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)

    A0 = y[0] - y[-1]
    C0 = y[-1]
    k0 = estimate_initial_k(t, y)

    use_table = (INIT_GUESS_SOURCE == "table") and (table_guess is not None)

    if model_name == "exponential":
        if use_table and "k1" in table_guess:
            p0 = [A0, float(table_guess["k1"]), C0]
        else:
            p0 = [A0, k0, C0]

        bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, 2.0, np.inf])
        popt, pcov = curve_fit(exponential, t, y, p0=p0, bounds=bounds, maxfev=30000)
        perr = np.sqrt(np.diag(pcov))
        y_fit = exponential(t, *popt)
        r2 = calc_r2(y, y_fit)
        A1, k1, c = popt
        eA1, ek1, ec = perr
        return {
            "y_fit": y_fit, "R2": r2,
            "A1": A1, "A1_err": eA1,
            "k1": k1, "k1_err": ek1,
            "t_half1": np.log(2) / k1, "t_half1_err": (np.log(2) / k1**2) * ek1,
            "A2": np.nan, "A2_err": np.nan,
            "k2": np.nan, "k2_err": np.nan,
            "t_half2": np.nan, "t_half2_err": np.nan,
            "c": c, "c_err": ec,
            "drift_m": np.nan, "drift_m_err": np.nan,
        }

    elif model_name == "single_exponential_with_drift":
        m0 = (y[-1] - y[0]) / max(t[-1] - t[0], 1.0)
        if use_table and "k1" in table_guess:
            p0 = [A0, float(table_guess["k1"]), C0, float(table_guess.get("drift_m", m0))]
        else:
            p0 = [A0, k0, C0, m0]

        bounds = ([-np.inf, 1e-6, -np.inf, -np.inf], [np.inf, 2.0, np.inf, np.inf])
        popt, pcov = curve_fit(single_exponential_with_drift, t, y, p0=p0, bounds=bounds, maxfev=30000)
        perr = np.sqrt(np.diag(pcov))
        y_fit = single_exponential_with_drift(t, *popt)
        r2 = calc_r2(y, y_fit)
        A1, k1, c, m = popt
        eA1, ek1, ec, em = perr
        return {
            "y_fit": y_fit, "R2": r2,
            "A1": A1, "A1_err": eA1,
            "k1": k1, "k1_err": ek1,
            "t_half1": np.log(2) / k1, "t_half1_err": (np.log(2) / k1**2) * ek1,
            "A2": np.nan, "A2_err": np.nan,
            "k2": np.nan, "k2_err": np.nan,
            "t_half2": np.nan, "t_half2_err": np.nan,
            "c": c, "c_err": ec,
            "drift_m": m, "drift_m_err": em,
        }

    elif model_name == "double_exponential":
        if use_table and ("k1" in table_guess) and ("k2" in table_guess):
            k1_init = float(table_guess["k1"])
            k2_init = float(table_guess["k2"])
            A1_init, A2_init, c_init = estimate_linear_amplitudes_double_exp(
                t, y, k1_init, k2_init, A0, C0
            )
            p0 = [A1_init, k1_init, A2_init, k2_init, c_init]
        else:
            p0 = [0.5 * A0, max(k0 * 2.0, 0.01), 0.5 * A0, max(k0 * 0.2, 0.0005), C0]

        bounds = ([-np.inf, 1e-5, -np.inf, 1e-6, -np.inf], [np.inf, 2.0, np.inf, 0.5, np.inf])
        popt, pcov = curve_fit(double_exponential, t, y, p0=p0, bounds=bounds, maxfev=40000)
        perr = np.sqrt(np.diag(pcov))
        # Uporządkowanie: k1 = faza szybsza, k2 = faza wolniejsza
        if popt[1] < popt[3]:
            popt = np.array([popt[2], popt[3], popt[0], popt[1], popt[4]])
            perr = np.array([perr[2], perr[3], perr[0], perr[1], perr[4]])
        y_fit = double_exponential(t, *popt)
        r2 = calc_r2(y, y_fit)
        A1, k1, A2, k2, c = popt
        eA1, ek1, eA2, ek2, ec = perr
        return {
            "y_fit": y_fit, "R2": r2,
            "A1": A1, "A1_err": eA1,
            "k1": k1, "k1_err": ek1,
            "t_half1": np.log(2) / k1, "t_half1_err": (np.log(2) / k1**2) * ek1,
            "A2": A2, "A2_err": eA2,
            "k2": k2, "k2_err": ek2,
            "t_half2": np.log(2) / k2, "t_half2_err": (np.log(2) / k2**2) * ek2,
            "c": c, "c_err": ec,
            "drift_m": np.nan, "drift_m_err": np.nan,
        }

    else:
        raise ValueError(f"Nieobsługiwany model dopasowania: {model_name}")


# ==============================================================================
# WCZYTYWANIE PLIKÓW POMOCNICZYCH (concentrations.txt, dead_times.txt)
# ==============================================================================
def read_concentrations(folder_path, default_conc=2.0):
    """Wczytuje concentrations.txt (2 kolumny: nazwa pliku i stężenie w uM,
    kropka jako separator dziesiętny, pomija 1 linię nagłówka)."""
    conc_map = {}
    conc_file = os.path.join(folder_path, "concentrations.txt")
    if not os.path.exists(conc_file):
        print(f"ℹ️ Brak pliku '{conc_file}'. Przyjęto domyślne stężenie {default_conc} uM.")
        return conc_map

    with open(conc_file, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p for p in re.split(r"[\s,;]+", stripped) if p]
        if len(parts) >= 2:
            raw_name = parts[0]
            try:
                val = float(parts[1])
                base_name = os.path.splitext(os.path.basename(raw_name))[0]
                conc_map[raw_name] = val
                conc_map[base_name] = val
            except ValueError:
                print(f"⚠️ Nieprawidłowy format stężenia w '{conc_file}': '{line.strip()}'")
    return conc_map


def read_dead_times(folder_path, default_dead_time=30.0):
    """Wczytuje dead_times.txt z danego folderu (gdy USE_DEAD_TIMES_FILE = True)."""
    dead_times = {}
    if not USE_DEAD_TIMES_FILE:
        return dead_times

    dead_file = os.path.join(folder_path, "dead_times.txt")
    if not os.path.exists(dead_file):
        print(f"ℹ️ Brak pliku '{dead_file}'. Przyjęto domyślny czas martwy {default_dead_time} s.")
        return dead_times

    with open(dead_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = [p for p in re.split(r"[\s,;]+", stripped) if p]
            if len(parts) >= 2:
                key = parts[0]
                match = re.match(r"^(\d+(?:\.\d+)?)[sS]?$", parts[1])
                if match:
                    val = float(match.group(1))
                    base_key = os.path.splitext(os.path.basename(key))[0]
                    dead_times[key] = val
                    dead_times[base_key] = val
    return dead_times


# ==============================================================================
# WCZYTYWANIE DANYCH KINETYCZNYCH CD I FL
# ==============================================================================
def read_cd_kinetics(filepath):
    """Wczytuje przebieg kinetyczny CD z pliku ProDataCSV lub standardowego CSV."""
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
            raise ValueError(f"Nie znaleziono bloku 'CircularDichroism' w pliku {filepath}")
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

    df_full = df_full.drop_duplicates(subset=["Time"], keep="first").reset_index(drop=True)
    return df_full


def read_fl_kinetics(filepath):
    """Wczytuje przebieg kinetyczny fluorescencji (nowy format Labels/ns oraz klasyczny CSV)."""
    with open(filepath, "r") as f:
        header_lines = [f.readline() for _ in range(15)]

    is_new = any(line.startswith("Labels,") for line in header_lines)
    if is_new:
        skip_rows = 0
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if re.match(r"^[\d]", line):
                    skip_rows = i
                    break
        df = pd.read_csv(filepath, skiprows=skip_rows, header=None).dropna(how="all", axis=1)
        df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
        df.iloc[:, 0] = df.iloc[:, 0] / 1e9  # ns -> s
        df.columns = ["Time", "Intensity"]
        return df.reset_index(drop=True)
    else:
        with open(filepath, "r") as f:
            l1, l2 = f.readline(), f.readline()
        is_min = ("Time (min)" in l1) or ("Time (min)" in l2)
        df_full = pd.read_csv(filepath, skiprows=2, sep=",").dropna(how="all", axis=1)
        df_full = df_full.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
        t = df_full.iloc[:, 0].copy()
        if is_min:
            t *= 60.0
        wrap = t[1:].sub(t.iloc[0]).abs().lt(1e-6)
        if not wrap.empty and wrap.any() and wrap.idxmax() > 1:
            df_full = df_full.iloc[:wrap.idxmax()]
            t = t.iloc[:wrap.idxmax()]
        df_full.iloc[:, 0] = t
        df_full.columns = ["Time", "Intensity"]
        return df_full.reset_index(drop=True)


# ==============================================================================
# ZAPIS WYNIKÓW DO CSV (IDEMPOTENTNY)
# ==============================================================================
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


def append_mean_summary(mean_records, construct_name, modality, fit_model, run_fits):
    if not run_fits:
        return
    k1_arr = np.array([r[3]["k1"] for r in run_fits], dtype=float)
    th1_arr = np.array([r[3]["t_half1"] for r in run_fits], dtype=float)
    k2_arr = np.array([r[3]["k2"] for r in run_fits], dtype=float)
    th2_arr = np.array([r[3]["t_half2"] for r in run_fits], dtype=float)
    m_arr = np.array([r[3]["drift_m"] for r in run_fits], dtype=float)

    ddof_val = 1 if len(run_fits) > 1 else 0
    mean_records.append({
        "Construct": construct_name,
        "Modality": modality,
        "Condition": "6.0 M GdmCl, 298 K",
        "Model": fit_model,
        "Fit_Range_s": run_fits[0][2],
        "N_runs": len(run_fits),
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


# ==============================================================================
# PRZETWARZANIE POJEDYNCZEGO PANELU (PANELE A, B, C)
# ==============================================================================
def process_single_axis_modality(ax, folder_path, construct_name, modality, individual_records, mean_records, table_guesses):
    cfg = CONFIG[construct_name]
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"⚠️ Brak plików CSV w folderze: {folder_path}")
        return

    is_cd = (modality == "MM_CD")
    default_dt = cfg["cd_default_dead_time"] if is_cd else cfg["fl_default_dead_time"]
    fit_model = cfg["cd_fit_model"] if is_cd else cfg["fl_fit_model"]
    fit_start = cfg["cd_fit_start"] if is_cd else cfg["fl_fit_start"]
    fit_end = cfg["cd_fit_end"] if is_cd else cfg["fl_fit_end"]

    dead_times_map = read_dead_times(folder_path, default_dt)
    conc_map = read_concentrations(folder_path, cfg["default_conc_uM"])

    t_guess = table_guesses.get((construct_name, modality)) if INIT_GUESS_SOURCE == "table" else None
    guess_mode_label = f"table ({t_guess})" if t_guess else "heuristic"
    print(f"\n>>> [{construct_name} | {modality}] Tryb punktów startowych: {guess_mode_label}")

    run_fits = []
    max_t_plotted = 0.0

    for idx, fp in enumerate(csv_files):
        fname = os.path.basename(fp)
        fbase = os.path.splitext(fname)[0]
        dt = dead_times_map.get(fname, dead_times_map.get(fbase, default_dt))
        conc_uM = conc_map.get(fname, conc_map.get(fbase, cfg["default_conc_uM"]))

        if is_cd:
            df = read_cd_kinetics(fp)
            t = df["Time"].values + dt
            # Przeliczenie mdeg -> [theta]_MRW (deg * cm^2 * dmol^-1)
            y = (df["CircularDichroism"].values * 1e6) / (conc_uM * cfg["pathlength_mm"] * cfg["n_res"])
            print(f"[{construct_name} CD] {fname}: c = {conc_uM} uM, dead_time = {dt} s")
        else:
            df = read_fl_kinetics(fp)
            t = df["Time"].values + dt
            y = df["Intensity"].values
            print(f"[{construct_name} FL] {fname}: c = {conc_uM} uM, dead_time = {dt} s")

        t0 = fit_start + dt
        t_end_eff = fit_end if fit_end is not None else float(np.max(t))
        mask = (t >= t0) & (t <= t_end_eff)
        t_m = t[mask]
        y_m = y[mask]

        if len(t_m) < 10:
            print(f"⚠️ Za mało punktów w {fname} dla zakresu {t0}-{t_end_eff} s.")
            continue

        max_t_plotted = max(max_t_plotted, float(np.max(t_m)))

        # Dla CD dzielimy przez 10^3 (w osi opisane jako * 10^-3), dla FL zostawiamy surowe wartości
        scale = 1e3 if is_cd else 1.0
        lbl_exp = f"Experimental ($n={len(csv_files)}$)" if idx == 0 else None
        ax.scatter(
            t_m, y_m / scale,
            s=POINT_SIZE, color=cfg["color_raw"], alpha=POINT_ALPHA,
            linewidths=0, rasterized=True, label=lbl_exp
        )

        try:
            res = fit_kinetic_trace(t_m, y_m, fit_model, table_guess=t_guess)
            lbl_fit = "Double-exp fit" if (idx == 0 and fit_model == "double_exponential") else (
                "Single-exp fit" if idx == 0 else None
            )
            ax.plot(
                t_m, res["y_fit"] / scale,
                color=cfg["color_fit"], ls="--", lw=FIT_LW, zorder=3, label=lbl_fit
            )

            range_str = f"{int(round(t_m[0]))}-{int(round(t_end_eff))}"
            run_fits.append((fname, conc_uM, range_str, res))

            individual_records.append({
                "Construct": construct_name,
                "Modality": modality,
                "Condition": "6.0 M GdmCl, 298 K",
                "Concentration_uM": conc_uM,
                "File": fname,
                "Model": fit_model,
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
        except Exception as e:
            print(f"❌ Dopasowanie nie powiodło się dla {fname}: {e}")

    append_mean_summary(mean_records, construct_name, modality, fit_model, run_fits)

    if is_cd:
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.5)
        ax.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$")
        if cfg.get("cd_ylim") is not None:
            ax.set_ylim(*cfg["cd_ylim"])
        style_legend(ax, loc="lower right", markerscale=2.6)
    else:
        ax.set_ylabel("Fluorescence Intensity (a.u.)")
        style_legend(ax, loc="upper right", markerscale=2.6)

    if max_t_plotted > 0:
        ax.set_xlim(0.0, max_t_plotted)
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle=":", alpha=0.5)


# ==============================================================================
# PRZETWARZANIE PANELU D (DWIE OSIE Y: 5.0 uM ORAZ 2.0 uM JAK W FIGURZE 5.1A)
# ==============================================================================
def process_dual_axis_fl_panel(ax_left, folder_path, construct_name, individual_records, mean_records, table_guesses):
    cfg = CONFIG[construct_name]
    modality = "MM_FL"
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"⚠️ Brak plików CSV w folderze: {folder_path}")
        return

    default_dt = cfg["fl_default_dead_time"]
    fit_model = cfg["fl_fit_model"]
    fit_start = cfg["fl_fit_start"]
    fit_end = cfg["fl_fit_end"]

    dead_times_map = read_dead_times(folder_path, default_dt)
    conc_map = read_concentrations(folder_path, cfg["default_conc_uM"])
    has_conc_file = os.path.exists(os.path.join(folder_path, "concentrations.txt"))

    t_guess = table_guesses.get((construct_name, modality)) if INIT_GUESS_SOURCE == "table" else None
    guess_mode_label = f"table ({t_guess})" if t_guess else "heuristic"
    print(f"\n>>> [{construct_name} | {modality} (2 osie: 5.0 uM i 2.0 uM)] Tryb punktów startowych: {guess_mode_label}")

    files_5uM = []
    files_2uM = []

    for fp in csv_files:
        fname = os.path.basename(fp)
        fbase = os.path.splitext(fname)[0]
        dt = dead_times_map.get(fname, dead_times_map.get(fbase, default_dt))
        df = read_fl_kinetics(fp)
        t = df["Time"].values + dt
        y = df["Intensity"].values

        if fname in conc_map or fbase in conc_map:
            conc_uM = conc_map.get(fname, conc_map.get(fbase))
        elif not has_conc_file:
            # Fallback gdyby brakowało concentrations.txt: wyższy sygnał (>15 a.u.) = 5.0 uM
            conc_uM = 5.0 if float(np.nanmax(y)) > 15.0 else 2.0
        else:
            conc_uM = cfg["default_conc_uM"]

        if np.isclose(conc_uM, 5.0, atol=0.5):
            files_5uM.append((fp, fname, conc_uM, dt, t, y))
        else:
            files_2uM.append((fp, fname, conc_uM, dt, t, y))

    ax_right = ax_left.twinx()
    run_fits = []
    max_t_plotted = 0.0

    # Seria 5.0 uM -> lewa oś Y (niebieska, COLOR_5UM)
    for idx, (fp, fname, conc_uM, dt, t, y) in enumerate(files_5uM):
        t0 = fit_start + dt
        t_end_eff = fit_end if fit_end is not None else float(np.max(t))
        mask = (t >= t0) & (t <= t_end_eff)
        t_m, y_m = t[mask], y[mask]
        if len(t_m) < 10:
            continue

        max_t_plotted = max(max_t_plotted, float(np.max(t_m)))

        lbl = rf"$5.0\ \mu\mathrm{{M}}$ ($n={len(files_5uM)}$)" if idx == 0 else None
        ax_left.scatter(
            t_m, y_m,
            s=POINT_SIZE, color=COLOR_5UM, alpha=POINT_ALPHA,
            linewidths=0, rasterized=True, label=lbl
        )

        try:
            res = fit_kinetic_trace(t_m, y_m, fit_model, table_guess=t_guess)
            ax_left.plot(
                t_m, res["y_fit"],
                color=darker(COLOR_5UM), ls="--", lw=FIT_LW, zorder=3
            )
            range_str = f"{int(round(t_m[0]))}-{int(round(t_end_eff))}"
            run_fits.append((fname, conc_uM, range_str, res))
            print(f"[{construct_name} FL | 5.0 uM] {fname}: dead_time = {dt:.0f} s")

            individual_records.append({
                "Construct": construct_name,
                "Modality": modality,
                "Condition": "6.0 M GdmCl, 298 K",
                "Concentration_uM": conc_uM,
                "File": fname,
                "Model": fit_model,
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
        except Exception as e:
            print(f"❌ Dopasowanie nie powiodło się dla {fname}: {e}")

    # Seria 2.0 uM -> prawa oś Y (pomarańczowa, COLOR_2UM)
    for idx, (fp, fname, conc_uM, dt, t, y) in enumerate(files_2uM):
        t0 = fit_start + dt
        t_end_eff = fit_end if fit_end is not None else float(np.max(t))
        mask = (t >= t0) & (t <= t_end_eff)
        t_m, y_m = t[mask], y[mask]
        if len(t_m) < 10:
            continue

        max_t_plotted = max(max_t_plotted, float(np.max(t_m)))

        lbl = rf"$2.0\ \mu\mathrm{{M}}$ ($n={len(files_2uM)}$)" if idx == 0 else None
        ax_right.scatter(
            t_m, y_m,
            s=POINT_SIZE, color=COLOR_2UM, alpha=POINT_ALPHA,
            linewidths=0, rasterized=True, label=lbl
        )

        try:
            res = fit_kinetic_trace(t_m, y_m, fit_model, table_guess=t_guess)
            ax_right.plot(
                t_m, res["y_fit"],
                color=darker(COLOR_2UM), ls="--", lw=FIT_LW, zorder=3
            )
            range_str = f"{int(round(t_m[0]))}-{int(round(t_end_eff))}"
            run_fits.append((fname, conc_uM, range_str, res))
            print(f"[{construct_name} FL | {conc_uM} uM] {fname}: dead_time = {dt:.0f} s")

            individual_records.append({
                "Construct": construct_name,
                "Modality": modality,
                "Condition": "6.0 M GdmCl, 298 K",
                "Concentration_uM": conc_uM,
                "File": fname,
                "Model": fit_model,
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
        except Exception as e:
            print(f"❌ Dopasowanie nie powiodło się dla {fname}: {e}")

    append_mean_summary(mean_records, construct_name, modality, fit_model, run_fits)

    if max_t_plotted > 0:
        ax_left.set_xlim(0.0, max_t_plotted)
    ax_left.set_xlabel("Time (s)")
    ax_left.set_ylabel(r"Fluorescence at $5.0\ \mu\mathrm{M}$ (a.u.)", color=COLOR_5UM)
    ax_right.set_ylabel(r"Fluorescence at $2.0\ \mu\mathrm{M}$ (a.u.)", color=COLOR_2UM)
    ax_left.tick_params(axis="y", colors=COLOR_5UM)
    ax_right.tick_params(axis="y", colors=COLOR_2UM)
    ax_left.grid(True, linestyle=":", alpha=0.5)

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    if h1 or h2:
        style_legend(ax_left, h1 + h2, l1 + l2, loc="upper right", markerscale=2.8)


# ==============================================================================
# GŁÓWNA FUNKCJA RYSUJĄCA FIGURĘ 5.4 (FsTrmD-Tm1570 oraz CdTrmD-Tm1570)
# ==============================================================================
def make_figure_fusions_fs_cd():
    os.makedirs(PATHS["fig_out_dir"], exist_ok=True)
    os.makedirs(PATHS["tables_out_dir"], exist_ok=True)

    table_guesses = load_table_initial_guesses(PATHS["tables_out_dir"]) if INIT_GUESS_SOURCE == "table" else {}

    plt.rcParams.update(FIG54_RC)

    fig, axes = plt.subplots(2, 2, figsize=FIG54_SIZE, dpi=300)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    individual_records = []
    mean_records = []

    # Rząd 1: FsTrmD-Tm1570 (Panel A: MM CD, Panel B: MM FL)
    process_single_axis_modality(axA, PATHS["fs_cd_dir"], "FsTrmD-Tm1570", "MM_CD", individual_records, mean_records, table_guesses)
    axA.set_title(r"$\mathsf{Fs}$TrmD–Tm1570 – Far-UV CD ($\mathbf{222\ nm}$)", fontweight="bold", fontsize=FS_TITLE, pad=8)

    process_single_axis_modality(axB, PATHS["fs_fl_dir"], "FsTrmD-Tm1570", "MM_FL", individual_records, mean_records, table_guesses)
    axB.set_title(r"$\mathsf{Fs}$TrmD–Tm1570 – Intrinsic Fluorescence", fontweight="bold", fontsize=FS_TITLE, pad=8)

    # Rząd 2: CdTrmD-Tm1570 (Panel C: MM CD, Panel D: MM FL z dwiema osiami Y dla 5.0 uM i 2.0 uM)
    process_single_axis_modality(axC, PATHS["cd_cd_dir"], "CdTrmD-Tm1570", "MM_CD", individual_records, mean_records, table_guesses)
    axC.set_title(r"$\mathsf{Cd}$TrmD–Tm1570 – Far-UV CD ($\mathbf{222\ nm}$)", fontweight="bold", fontsize=FS_TITLE, pad=8)

    process_dual_axis_fl_panel(axD, PATHS["cd_fl_dir"], "CdTrmD-Tm1570", individual_records, mean_records, table_guesses)
    axD.set_title(r"$\mathsf{Cd}$TrmD–Tm1570 – Intrinsic Fluorescence", fontweight="bold", fontsize=FS_TITLE, pad=8)

    # Etykiety paneli A, B, C, D
    for ax, label in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top")

    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    out_pdf = os.path.join(PATHS["fig_out_dir"], "Figura_5_4_Kinetyka_Fs_Cd.pdf")
    out_png = os.path.join(PATHS["fig_out_dir"], "Figura_5_4_Kinetyka_Fs_Cd.png")
    # BEZ bbox_inches="tight": PDF ma dokładnie 12.0 in szerokości (jak Fig. 4.3, 5.1, 5.2 i 5.3)
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    # plt.show()

    update_fits_tables(individual_records, mean_records, PATHS["tables_out_dir"])

    print("\n--- PODSUMOWANIE UŚREDNIONYCH DOPASOWAŃ ---")
    df_summary = pd.DataFrame(mean_records)
    if not df_summary.empty:
        cols_show = [
            "Construct", "Modality", "N_runs",
            "t_half1_mean_s", "t_half1_sd_s",
            "t_half2_mean_s", "t_half2_sd_s"
        ]
        print(df_summary[cols_show].to_string(index=False))

    print(f"\n✅ Zapisano figurę:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    make_figure_fusions_fs_cd()