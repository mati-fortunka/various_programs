import os
import re
from io import StringIO
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ==============================================================================
# 1. Konfiguracja Ścieżek, Parametrów i Flagi Startpoints
# ==============================================================================
USE_PREVIOUS_AVG_PARAMETERS = False

PATH_LENGTH_MM = 1.0
TARGET_WL = 217.0
RT = 0.592  # kcal/mol przy 298.15 K

BASE_CD_DIR_WARSAW = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD"
FS_CD_DATA_DIR = (
    "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/"
    "F8E4N/equilibrium/unfolding/2nd_set/CD/fuzja_F8_unfolding_22h"
)

OUTPUT_DIR = "/home/matifortunka/PycharmProjects/various_programs/PhD/tables"
ALL_PARAMS_CSV = os.path.join(OUTPUT_DIR, "three_state_dimer_all_parameters.csv")
DM_COMPARISON_CSV = os.path.join(OUTPUT_DIR, "three_state_dimer_dm_comparison.csv")
AVG_PARAMS_CSV = os.path.join(OUTPUT_DIR, "three_state_dimer_avg_parameters.csv")

MODEL_DIM_LABEL = "3-state_dimeric_int (N2<->I2<->2U)"
MODEL_MON_LABEL = "3-state_monomeric_int (N2<->2I<->2U)"

# Zbiory danych dopasowywane modelem 3-stanowym (bez duplikatów:
# usunięto Fig4_4_CnFusion_2uM_12h [duplikat Fig4_3] oraz Fig4_6_CnFusion_2uM_84h [duplikat Fig4_4])
THREE_STATE_DATASETS = [
    # CnTrmD-Tm1570, 2 uM, 12h (Fig4_3)
    {
        "dataset_id": "Fig4_3_CnFusion_2uM_12h",
        "source_script": "Fig4_3_dimer_model.py",
        "construct": "CnTrmD-Tm1570",
        "probe": "CD_217nm",
        "time_h": 12,
        "conc_uM": 2.0,
        "residues": 455,
        "hv_cutoff": 850.0,
        "savgol_win": 11,
        "folder_path": os.path.join(BASE_CD_DIR_WARSAW, "fuzja/3/2uM/28_07_26"),
        "p0_dim_int": [None, None, None, 5.0, 2.2, 1.0, 2.8],
        "bounds_dim_int": (
            [-np.inf, -np.inf, -np.inf, 0.1, 0.5, 0.1, 1.0],
            [np.inf, np.inf, np.inf, 15.0, 4.5, 8.0, 6.0],
        ),
        "p0_mon_int": [None, None, None, 3.0, 2.2, 1.0, 3.2],
        "bounds_mon_int": (
            [-np.inf, -np.inf, -np.inf, 0.1, 0.5, 0.1, 1.0],
            [np.inf, np.inf, np.inf, 15.0, 4.5, 8.0, 6.0],
        ),
    },
    # CnTrmD-Tm1570, 2 uM, 36h (Fig4_4)
    {
        "dataset_id": "Fig4_4_CnFusion_2uM_36h",
        "source_script": "Fig4_4_dimer.py",
        "construct": "CnTrmD-Tm1570",
        "probe": "CD_217nm",
        "time_h": 36,
        "conc_uM": 2.0,
        "residues": 455,
        "hv_cutoff": 900.0,
        "savgol_win": 11,
        "folder_path": os.path.join(BASE_CD_DIR_WARSAW, "fuzja/3/2uM/29_07_26"),
        "p0_dim_int": [None, None, None, 3.54, 2.10, 0.86, 3.28],
        "bounds_dim_int": (
            [-np.inf, -np.inf, -np.inf, 0.5, 1.2, 0.2, 2.2],
            [np.inf, np.inf, np.inf, 12.0, 2.8, 5.0, 4.5],
        ),
        "p0_mon_int": [None, None, None, 2.5, 2.10, 0.9, 3.4],
        "bounds_mon_int": (
            [-np.inf, -np.inf, -np.inf, 0.2, 1.0, 0.2, 1.8],
            [np.inf, np.inf, np.inf, 12.0, 3.5, 5.0, 5.0],
        ),
    },
    # CnTrmD-Tm1570, 2 uM, 84h (Fig4_4)
    {
        "dataset_id": "Fig4_4_CnFusion_2uM_84h",
        "source_script": "Fig4_4_dimer.py",
        "construct": "CnTrmD-Tm1570",
        "probe": "CD_217nm",
        "time_h": 84,
        "conc_uM": 2.0,
        "residues": 455,
        "hv_cutoff": 900.0,
        "savgol_win": 11,
        "folder_path": os.path.join(BASE_CD_DIR_WARSAW, "fuzja/3/2uM/31_07_26"),
        "p0_dim_int": [None, None, None, 5.09, 1.94, 1.25, 3.23],
        "bounds_dim_int": (
            [-np.inf, -np.inf, -np.inf, 0.5, 1.2, 0.2, 2.2],
            [np.inf, np.inf, np.inf, 12.0, 2.8, 5.0, 4.5],
        ),
        "p0_mon_int": [None, None, None, 3.0, 1.94, 1.2, 3.3],
        "bounds_mon_int": (
            [-np.inf, -np.inf, -np.inf, 0.2, 1.0, 0.2, 1.8],
            [np.inf, np.inf, np.inf, 12.0, 3.5, 5.0, 5.0],
        ),
    },
    # CnTrmD-Tm1570, 3 uM, 12h (dodana ścieżka fuzja/3/3uM/28_07_26)
    {
        "dataset_id": "CnFusion_3uM_12h",
        "source_script": "supplementary_3uM",
        "construct": "CnTrmD-Tm1570",
        "probe": "CD_217nm",
        "time_h": 12,
        "conc_uM": 3.0,
        "residues": 455,
        "hv_cutoff": 900.0,
        "savgol_win": 11,
        "folder_path": os.path.join(BASE_CD_DIR_WARSAW, "fuzja/3/3uM/28_07_26"),
        "p0_dim_int": [None, None, None, 4.50, 2.10, 1.10, 3.10],
        "bounds_dim_int": (
            [-np.inf, -np.inf, -np.inf, 0.5, 1.2, 0.2, 2.2],
            [np.inf, np.inf, np.inf, 12.0, 3.0, 5.0, 4.5],
        ),
        "p0_mon_int": [None, None, None, 3.0, 2.10, 1.1, 3.3],
        "bounds_mon_int": (
            [-np.inf, -np.inf, -np.inf, 0.2, 1.0, 0.2, 1.8],
            [np.inf, np.inf, np.inf, 12.0, 3.5, 5.0, 5.0],
        ),
    },
    # CnTrmD-Tm1570, 3 uM, 36h (dodana ścieżka fuzja/3/3uM/29_07_26)
    {
        "dataset_id": "CnFusion_3uM_36h",
        "source_script": "supplementary_3uM",
        "construct": "CnTrmD-Tm1570",
        "probe": "CD_217nm",
        "time_h": 36,
        "conc_uM": 3.0,
        "residues": 455,
        "hv_cutoff": 900.0,
        "savgol_win": 11,
        "folder_path": os.path.join(BASE_CD_DIR_WARSAW, "fuzja/3/3uM/29_07_26"),
        "p0_dim_int": [None, None, None, 4.50, 2.10, 1.10, 3.20],
        "bounds_dim_int": (
            [-np.inf, -np.inf, -np.inf, 0.5, 1.2, 0.2, 2.2],
            [np.inf, np.inf, np.inf, 12.0, 3.0, 5.0, 4.5],
        ),
        "p0_mon_int": [None, None, None, 3.0, 2.10, 1.1, 3.3],
        "bounds_mon_int": (
            [-np.inf, -np.inf, -np.inf, 0.2, 1.0, 0.2, 1.8],
            [np.inf, np.inf, np.inf, 12.0, 3.5, 5.0, 5.0],
        ),
    },
    # CnTrmD-Tm1570, 3 uM, 84h (Fig4_6)
    {
        "dataset_id": "Fig4_6_CnFusion_3uM_84h",
        "source_script": "Fig4_6_dimer.py",
        "construct": "CnTrmD-Tm1570",
        "probe": "CD_217nm",
        "time_h": 84,
        "conc_uM": 3.0,
        "residues": 455,
        "hv_cutoff": 900.0,
        "savgol_win": 11,
        "folder_path": os.path.join(BASE_CD_DIR_WARSAW, "fuzja/3/3uM/31_07_26"),
        "p0_dim_int": [None, None, None, 4.50, 2.10, 1.10, 3.25],
        "bounds_dim_int": (
            [-np.inf, -np.inf, -np.inf, 0.5, 1.2, 0.2, 2.2],
            [np.inf, np.inf, np.inf, 12.0, 3.0, 5.0, 4.5],
        ),
        "p0_mon_int": [None, None, None, 3.0, 2.10, 1.1, 3.3],
        "bounds_mon_int": (
            [-np.inf, -np.inf, -np.inf, 0.2, 1.0, 0.2, 1.8],
            [np.inf, np.inf, np.inf, 12.0, 3.5, 5.0, 5.0],
        ),
    },
    # CnTrmD-Tm1570, 5 uM, 84h (fuzja/seria2_1, Fig4_6)
    {
        "dataset_id": "Fig4_6_CnFusion_5uM_84h",
        "source_script": "Fig4_6_dimer.py",
        "construct": "CnTrmD-Tm1570",
        "probe": "CD_217nm",
        "time_h": 84,
        "conc_uM": 5.0,
        "residues": 455,
        "hv_cutoff": 900.0,
        "savgol_win": 11,
        "folder_path": os.path.join(BASE_CD_DIR_WARSAW, "fuzja/seria2_1"),
        "p0_dim_int": [None, None, None, 4.20, 2.25, 1.00, 3.35],
        "bounds_dim_int": (
            [-np.inf, -np.inf, -np.inf, 0.5, 1.2, 0.2, 2.2],
            [np.inf, np.inf, np.inf, 12.0, 3.0, 5.0, 4.5],
        ),
        "p0_mon_int": [None, None, None, 3.0, 2.25, 1.0, 3.4],
        "bounds_mon_int": (
            [-np.inf, -np.inf, -np.inf, 0.2, 1.0, 0.2, 1.8],
            [np.inf, np.inf, np.inf, 12.0, 3.5, 5.0, 5.0],
        ),
    },
    # FsTrmD-Tm1570, 1 uM, 22h (Fig4_7)
    {
        "dataset_id": "Fig4_7_FsFusion_1uM_22h",
        "source_script": "Fig4_7_dimer.py",
        "construct": "FsTrmD-Tm1570",
        "probe": "CD_217nm",
        "time_h": 22,
        "conc_uM": 1.0,
        "residues": 447,
        "hv_cutoff": 850.0,
        "savgol_win": 15,
        "folder_path": FS_CD_DATA_DIR,
        "p0_dim_int": [None, None, None, 3.98, 2.34, 0.93, 3.68],
        "bounds_dim_int": (
            [-np.inf, -np.inf, -np.inf, 1.5, 2.00, 0.3, 3.20],
            [np.inf, np.inf, np.inf, 7.0, 2.65, 2.5, 4.20],
        ),
        "p0_mon_int": [None, None, None, 3.0, 2.34, 0.93, 3.68],
        "bounds_mon_int": (
            [-np.inf, -np.inf, -np.inf, 0.3, 1.2, 0.2, 2.0],
            [np.inf, np.inf, np.inf, 12.0, 4.0, 5.0, 5.5],
        ),
    },
]


# ==============================================================================
# 2. Modele Trójstanowe dla Homodimeru
# ==============================================================================
def three_state_dimeric_intermediate_model(x, a_n, a_i, a_u, m1, d1, m2, d2, conc_uM):
    """
    Model trójstanowy z dimerycznym stanem pośrednim: N2 <-> I2 <-> 2U.
    Etap 1 (N2 <-> I2): K1(x) = [I2]/[N2] = exp(m1 * (x - d1) / RT)
    Etap 2 (I2 <-> 2U): K2(x) = [U]^2/[I2] = P_t * exp(m2 * (x - d2) / RT)
    """
    x = np.asarray(x, dtype=float)
    P_t = conc_uM * 1e-6
    K1 = np.exp(np.clip((m1 * (x - d1)) / RT, -250.0, 250.0))
    K2 = P_t * np.exp(np.clip((m2 * (x - d2)) / RT, -250.0, 250.0))
    K_eff = np.maximum((K1 * K2) / (1.0 + K1), 1e-300)
    f_u = 2.0 / (1.0 + np.sqrt(1.0 + (8.0 * P_t) / K_eff))
    f_n2 = (1.0 - f_u) / (1.0 + K1)
    f_i2 = (K1 * (1.0 - f_u)) / (1.0 + K1)
    return a_n * f_n2 + a_i * f_i2 + a_u * f_u


def three_state_monomeric_intermediate_model(x, a_n, a_i, a_u, m1, d1, m2, d2, conc_uM):
    """
    Model trójstanowy z monomerycznym stanem pośrednim: N2 <-> 2I <-> 2U.
    Etap 1 (N2 <-> 2I): K1(x) = [I]^2 / [N2] = P_t * exp(m1 * (x - d1) / RT)
    Etap 2 (I <-> U):   K2(x) = [U] / [I]    = exp(m2 * (x - d2) / RT)
    """
    x = np.asarray(x, dtype=float)
    exp1 = np.clip((m1 * (x - d1)) / RT, -500.0, 500.0)
    exp2 = np.clip((m2 * (x - d2)) / RT, -250.0, 250.0)
    K2 = np.exp(exp2)

    A = 2.0 * np.exp(-exp1)
    B = 1.0 + K2

    ratio = np.clip((4.0 * A) / (B * B), 0.0, 1e300)
    f_i = 2.0 / (B * (1.0 + np.sqrt(1.0 + ratio)))
    f_u = K2 * f_i

    f_n2_quad = np.clip(A * (f_i ** 2), 0.0, 1.0)
    f_n2_diff = np.clip(1.0 - f_i - f_u, 0.0, 1.0)
    f_n2 = np.where(f_i < 0.25, f_n2_quad, f_n2_diff)

    tot = f_n2 + f_i + f_u
    f_n2, f_i, f_u = f_n2 / tot, f_i / tot, f_u / tot
    return a_n * f_n2 + a_i * f_i + a_u * f_u


def goodness_of_fit(y_true, y_pred, k_params=7):
    n = len(y_true)
    residuals = y_true - y_pred
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else 0.0
    rmse = float(np.sqrt(rss / n)) if n > 0 else np.nan
    if n > k_params and rss > 0:
        aic = float(n * np.log(rss / n) + 2.0 * k_params)
        bic = float(n * np.log(rss / n) + k_params * np.log(n))
    else:
        aic, bic = np.nan, np.nan
    return r2, rss, rmse, aic, bic


# ==============================================================================
# 3. Wczytywanie Danych CD i Konwersja do [theta]_MRW
# ==============================================================================
def resolve_folder_and_conc_file(folder_path):
    candidates = [
        folder_path,
        folder_path.replace("_", "."),
        folder_path.replace(".", "_"),
    ]
    resolved_dir = next((c for c in candidates if os.path.isdir(c)), folder_path)

    conc_candidates = [
        os.path.join(resolved_dir, "concentrations.txt"),
        os.path.join(os.path.dirname(resolved_dir), "concentrations.txt"),
        os.path.join(os.path.dirname(os.path.dirname(resolved_dir)), "concentrations.txt"),
    ]
    conc_file = next((p for p in conc_candidates if os.path.exists(p)), None)
    return resolved_dir, conc_file


def load_concentrations_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace(",", ".")
    df = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")
    df["Sample_number"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df["den_concentration"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df.dropna(subset=["Sample_number", "den_concentration"], inplace=True)
    return dict(zip(df["Sample_number"].astype(int), df["den_concentration"].astype(float)))


def load_cd_217_series(cfg):
    folder_path, conc_file = resolve_folder_and_conc_file(cfg["folder_path"])
    if conc_file is None or not os.path.exists(conc_file):
        raise FileNotFoundError(f"Brak concentrations.txt dla {cfg['dataset_id']} ({folder_path})")

    conc_map = load_concentrations_file(conc_file)
    pattern = re.compile(r"(\d+)\.csv$", re.IGNORECASE)
    concs, y_vals = [], []

    for fname in sorted(os.listdir(folder_path)):
        m = pattern.search(fname)
        if not m:
            continue
        sn = int(m.group(1))
        conc = conc_map.get(sn, None)
        if conc is None or np.isnan(conc):
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            cd_idx = next(
                (i for i in range(len(lines) - 1)
                 if lines[i].strip().startswith("Wavelength")
                 and lines[i + 1].strip().startswith("CircularDichroism")),
                None,
            )
            if cd_idx is None:
                continue

            cd_lines = []
            for ln in lines[cd_idx + 2:]:
                if not ln.strip() or ln.strip().startswith("Wavelength") or ln.strip().startswith("Available"):
                    break
                cd_lines.append(list(map(float, ln.replace(",", " ").split())))
            if not cd_lines:
                continue
            cd_arr = np.array(cd_lines)
            wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

            hv_idx = next(
                (i for i in range(len(lines) - 1)
                 if lines[i].strip().startswith("Wavelength")
                 and lines[i + 1].strip().startswith("HV")),
                None,
            )
            if hv_idx is not None:
                hv_lines = []
                for ln in lines[hv_idx + 2:]:
                    if not ln.strip() or ln.strip().startswith("Wavelength") or ln.strip().startswith("Available"):
                        break
                    hv_lines.append(list(map(float, ln.replace(",", " ").split())))
                hv_arr = np.array(hv_lines)
                if len(hv_arr) == len(wls):
                    mask = hv_arr[:, 1] <= cfg["hv_cutoff"]
                    wls, mdeg = wls[mask], mdeg[mask]

            if len(wls) < 15:
                continue

            s_idx = np.argsort(wls)
            wls, mdeg = wls[s_idx], mdeg[s_idx]

            win = cfg["savgol_win"]
            if len(mdeg) >= win:
                mdeg = savgol_filter(mdeg, win, 3)

            mre = (mdeg * 1e6) / (cfg["conc_uM"] * PATH_LENGTH_MM * cfg["residues"])
            concs.append(float(conc))
            y_vals.append(float(np.interp(TARGET_WL, wls, mre)))
        except Exception:
            continue

    if not concs:
        raise ValueError(f"Brak poprawnych widm CD dla {cfg['dataset_id']} w {folder_path}")

    order = np.argsort(concs)
    return np.array(concs)[order], np.array(y_vals)[order]


# ==============================================================================
# 4. Punkty Startowe, Dopasowanie, Średnie i Eksport CSV
# ==============================================================================
def load_previous_avg_startpoints(avg_csv_path):
    """Wczytuje średnie parametry dla każdego modelu z pliku CSV, jeśli istnieje."""
    if not os.path.exists(avg_csv_path):
        print(f"[!] Flaga use_previous_avg_parameters=True, ale brak pliku: {avg_csv_path}. Użyto domyślnych p0.")
        return {}
    df_avg = pd.read_csv(avg_csv_path)
    startpoints = {}
    for _, row in df_avg.iterrows():
        m_name = row.get("Model")
        vals = []
        for col in ["a_N", "a_I", "a_U", "m1", "d1", "m2", "d2"]:
            v = row.get(col)
            if pd.isna(v):
                for alias in PARAM_COL_ALIASES.get(col, []):
                    if alias in row and pd.notna(row[alias]):
                        v = row[alias]
                        break
            vals.append(v)
        if m_name and all(pd.notna(v) and np.isfinite(float(v)) for v in vals):
            startpoints[m_name] = [float(v) for v in vals]
    if startpoints:
        print(f"[✓] Wczytano punkty startowe z: {avg_csv_path}")
    return startpoints


def build_initial_guess(p0_template, x, y, model_label, prev_avg_map, use_prev_avg):
    a_n_init = float(y[0])
    a_u_init = float(y[-1])
    a_i_init = float(0.5 * (a_n_init + a_u_init))

    if use_prev_avg and model_label in prev_avg_map:
        p0 = list(prev_avg_map[model_label])
    else:
        p0 = list(p0_template)

    if p0[0] is None or not np.isfinite(p0[0]):
        p0[0] = a_n_init
    if p0[1] is None or not np.isfinite(p0[1]):
        p0[1] = a_i_init
    if p0[2] is None or not np.isfinite(p0[2]):
        p0[2] = a_u_init
    return p0


def fit_single_model(model_func, x, y, p0, bounds, conc_uM):
    wrapped = lambda den, a_n, a_i, a_u, m1, d1, m2, d2: model_func(
        den, a_n, a_i, a_u, m1, d1, m2, d2, conc_uM
    )
    try:
        popt, pcov = curve_fit(wrapped, x, y, p0=p0, bounds=bounds, maxfev=30000)
        perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        y_pred = wrapped(x, *popt)
        r2, rss, rmse, aic, bic = goodness_of_fit(y, y_pred, k_params=7)
        return {
            "converged": True,
            "popt": popt,
            "perr": perr,
            "r2": r2,
            "rss": rss,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
        }
    except Exception as exc:
        return {
            "converged": False,
            "error": str(exc),
            "popt": [np.nan] * 7,
            "perr": [np.nan] * 7,
            "r2": np.nan,
            "rss": np.nan,
            "rmse": np.nan,
            "aic": np.nan,
            "bic": np.nan,
        }

PARAM_COL_ALIASES = {
    "a_N": ["a_N"],
    "a_I": ["a_I"],
    "a_U": ["a_U"],
    "m1": ["m1_kcal_mol_M", "m1"],
    "d1": ["d1_M", "d1"],
    "m2": ["m2_kcal_mol_M", "m2"],
    "d2": ["d2_M", "d2"],
}

def _extract_param_series(df, canonical_name):
    for cand in PARAM_COL_ALIASES.get(canonical_name, [canonical_name]):
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce").dropna()
            if len(s) > 0:
                return s
    return pd.Series(dtype=float)

def compute_and_save_avg_parameters(df_all, avg_csv_path):
    """Oblicza średnie (i odchylenia standardowe) parametrów dla każdego modelu i zapisuje do CSV."""
    param_cols = ["a_N", "a_I", "a_U", "m1", "d1", "m2", "d2"]
    out_cols = ["Model", "Pathway", "N_datasets"]
    for col in param_cols:
        out_cols.extend([col, f"{col}_SD"])

    avg_rows = []
    if df_all is not None and not df_all.empty and "Converged" in df_all.columns:
        conv_df = df_all[df_all["Converged"] == True]
        if not conv_df.empty and "Model" in conv_df.columns:
            for model_label, grp in conv_df.groupby("Model", sort=False):
                row = {
                    "Model": model_label,
                    "Pathway": grp["Pathway"].iloc[0] if "Pathway" in grp.columns and len(grp) > 0 else "",
                    "N_datasets": int(len(grp)),
                }
                for col in param_cols:
                    vals = _extract_param_series(grp, col)
                    row[col] = round(float(vals.mean()), 4) if len(vals) > 0 else np.nan
                    row[f"{col}_SD"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
                avg_rows.append(row)

    df_avg = pd.DataFrame(avg_rows, columns=out_cols)
    df_avg.to_csv(avg_csv_path, index=False)
    return df_avg


def run_comparison(use_previous_avg_parameters=USE_PREVIOUS_AVG_PARAMETERS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prev_avg_map = load_previous_avg_startpoints(AVG_PARAMS_CSV) if use_previous_avg_parameters else {}

    all_rows = []
    comp_rows = []

    for cfg in THREE_STATE_DATASETS:
        try:
            x, y = load_cd_217_series(cfg)
        except Exception as exc:
            print(f"[!] Pominięto {cfg['dataset_id']}: {exc}")
            continue

        p0_dim = build_initial_guess(
            cfg["p0_dim_int"], x, y, MODEL_DIM_LABEL, prev_avg_map, use_previous_avg_parameters
        )
        p0_mon = build_initial_guess(
            cfg["p0_mon_int"], x, y, MODEL_MON_LABEL, prev_avg_map, use_previous_avg_parameters
        )

        res_dim = fit_single_model(
            three_state_dimeric_intermediate_model, x, y, p0_dim, cfg["bounds_dim_int"], cfg["conc_uM"]
        )
        res_mon = fit_single_model(
            three_state_monomeric_intermediate_model, x, y, p0_mon, cfg["bounds_mon_int"], cfg["conc_uM"]
        )

        P_t = cfg["conc_uM"] * 1e-6

        for model_label, scheme, res, step2_is_dissoc in [
            (MODEL_DIM_LABEL, "N2 <-> I2 <-> 2U", res_dim, True),
            (MODEL_MON_LABEL, "N2 <-> 2I <-> 2U", res_mon, False),
        ]:
            a_n, a_i, a_u, m1, d1, m2, d2 = res["popt"]
            sa_n, sa_i, sa_u, sm1, sd1, sm2, sd2 = res["perr"]

            if res["converged"]:
                if step2_is_dissoc:
                    dg1 = m1 * d1
                    dg2 = m2 * d2 - RT * np.log(P_t)
                else:
                    dg1 = m1 * d1 - RT * np.log(P_t)
                    dg2 = m2 * d2
            else:
                dg1, dg2 = np.nan, np.nan

            all_rows.append({
                "Dataset_ID": cfg["dataset_id"],
                "Source_Script": cfg["source_script"],
                "Construct": cfg["construct"],
                "Probe": cfg["probe"],
                "Time_h": cfg["time_h"],
                "Conc_uM": cfg["conc_uM"],
                "N_points": len(x),
                "Model": model_label,
                "Pathway": scheme,
                "Converged": res["converged"],
                "a_N": round(a_n, 2) if np.isfinite(a_n) else np.nan,
                "a_N_SE": round(sa_n, 2) if np.isfinite(sa_n) else np.nan,
                "a_I": round(a_i, 2) if np.isfinite(a_i) else np.nan,
                "a_I_SE": round(sa_i, 2) if np.isfinite(sa_i) else np.nan,
                "a_U": round(a_u, 2) if np.isfinite(a_u) else np.nan,
                "a_U_SE": round(sa_u, 2) if np.isfinite(sa_u) else np.nan,
                "m1_kcal_mol_M": round(m1, 3) if np.isfinite(m1) else np.nan,
                "m1_SE": round(sm1, 3) if np.isfinite(sm1) else np.nan,
                "d1_M": round(d1, 3) if np.isfinite(d1) else np.nan,
                "d1_SE": round(sd1, 3) if np.isfinite(sd1) else np.nan,
                "m2_kcal_mol_M": round(m2, 3) if np.isfinite(m2) else np.nan,
                "m2_SE": round(sm2, 3) if np.isfinite(sm2) else np.nan,
                "d2_M": round(d2, 3) if np.isfinite(d2) else np.nan,
                "d2_SE": round(sd2, 3) if np.isfinite(sd2) else np.nan,
                "dG1_H2O_kcal_mol": round(dg1, 2) if np.isfinite(dg1) else np.nan,
                "dG2_H2O_kcal_mol": round(dg2, 2) if np.isfinite(dg2) else np.nan,
                "R_squared": round(res["r2"], 4) if np.isfinite(res["r2"]) else np.nan,
                "RSS": round(res["rss"], 2) if np.isfinite(res["rss"]) else np.nan,
                "RMSE": round(res["rmse"], 2) if np.isfinite(res["rmse"]) else np.nan,
                "AIC": round(res["aic"], 2) if np.isfinite(res["aic"]) else np.nan,
                "BIC": round(res["bic"], 2) if np.isfinite(res["bic"]) else np.nan,
            })

        _, _, _, m1_d, d1_d, m2_d, d2_d = res_dim["popt"]
        _, _, _, sm1_d, sd1_d, sm2_d, sd2_d = res_dim["perr"]

        _, _, _, m1_m, d1_m, m2_m, d2_m = res_mon["popt"]
        _, _, _, sm1_m, sd1_m, sm2_m, sd2_m = res_mon["perr"]

        delta_aic = (res_mon["aic"] - res_dim["aic"]) if (np.isfinite(res_mon["aic"]) and np.isfinite(res_dim["aic"])) else np.nan
        delta_bic = (res_mon["bic"] - res_dim["bic"]) if (np.isfinite(res_mon["bic"]) and np.isfinite(res_dim["bic"])) else np.nan

        if np.isfinite(delta_bic):
            if delta_bic > 2.0:
                better_model = "Dimeric_int (N2<->I2<->2U)"
            elif delta_bic < -2.0:
                better_model = "Monomeric_int (N2<->2I<->2U)"
            else:
                better_model = "Comparable (|dBIC|<=2)"
        else:
            better_model = "Check_convergence"

        comp_rows.append({
            "Dataset_ID": cfg["dataset_id"],
            "Construct": cfg["construct"],
            "Time_h": cfg["time_h"],
            "Conc_uM": cfg["conc_uM"],
            "Dim_d1_M": round(d1_d, 3) if np.isfinite(d1_d) else np.nan,
            "Dim_d1_SE": round(sd1_d, 3) if np.isfinite(sd1_d) else np.nan,
            "Dim_d2_M": round(d2_d, 3) if np.isfinite(d2_d) else np.nan,
            "Dim_d2_SE": round(sd2_d, 3) if np.isfinite(sd2_d) else np.nan,
            "Dim_m1": round(m1_d, 3) if np.isfinite(m1_d) else np.nan,
            "Dim_m1_SE": round(sm1_d, 3) if np.isfinite(sm1_d) else np.nan,
            "Dim_m2": round(m2_d, 3) if np.isfinite(m2_d) else np.nan,
            "Dim_m2_SE": round(sm2_d, 3) if np.isfinite(sm2_d) else np.nan,
            "Dim_R2": round(res_dim["r2"], 4) if np.isfinite(res_dim["r2"]) else np.nan,
            "Mon_d1_M": round(d1_m, 3) if np.isfinite(d1_m) else np.nan,
            "Mon_d1_SE": round(sd1_m, 3) if np.isfinite(sd1_m) else np.nan,
            "Mon_d2_M": round(d2_m, 3) if np.isfinite(d2_m) else np.nan,
            "Mon_d2_SE": round(sd2_m, 3) if np.isfinite(sd2_m) else np.nan,
            "Mon_m1": round(m1_m, 3) if np.isfinite(m1_m) else np.nan,
            "Mon_m1_SE": round(sm1_m, 3) if np.isfinite(sm1_m) else np.nan,
            "Mon_m2": round(m2_m, 3) if np.isfinite(m2_m) else np.nan,
            "Mon_m2_SE": round(sm2_m, 3) if np.isfinite(sm2_m) else np.nan,
            "Mon_R2": round(res_mon["r2"], 4) if np.isfinite(res_mon["r2"]) else np.nan,
            "Delta_AIC_Mon_minus_Dim": round(delta_aic, 2) if np.isfinite(delta_aic) else np.nan,
            "Delta_BIC_Mon_minus_Dim": round(delta_bic, 2) if np.isfinite(delta_bic) else np.nan,
            "Preferred_by_BIC": better_model,
        })

    all_cols = [
        "Dataset_ID", "Source_Script", "Construct", "Probe", "Time_h", "Conc_uM",
        "N_points", "Model", "Pathway", "Converged",
        "a_N", "a_N_SE", "a_I", "a_I_SE", "a_U", "a_U_SE",
        "m1_kcal_mol_M", "m1_SE", "d1_M", "d1_SE",
        "m2_kcal_mol_M", "m2_SE", "d2_M", "d2_SE",
        "dG1_H2O_kcal_mol", "dG2_H2O_kcal_mol",
        "R_squared", "RSS", "RMSE", "AIC", "BIC",
    ]
    comp_cols = [
        "Dataset_ID", "Construct", "Time_h", "Conc_uM",
        "Dim_d1_M", "Dim_d1_SE", "Dim_d2_M", "Dim_d2_SE",
        "Dim_m1", "Dim_m1_SE", "Dim_m2", "Dim_m2_SE", "Dim_R2",
        "Mon_d1_M", "Mon_d1_SE", "Mon_d2_M", "Mon_d2_SE",
        "Mon_m1", "Mon_m1_SE", "Mon_m2", "Mon_m2_SE", "Mon_R2",
        "Delta_AIC_Mon_minus_Dim", "Delta_BIC_Mon_minus_Dim", "Preferred_by_BIC",
    ]

    df_all = pd.DataFrame(all_rows, columns=all_cols)
    df_comp = pd.DataFrame(comp_rows, columns=comp_cols)

    df_all.to_csv(ALL_PARAMS_CSV, index=False)
    df_comp.to_csv(DM_COMPARISON_CSV, index=False)
    df_avg = compute_and_save_avg_parameters(df_all, AVG_PARAMS_CSV)

    print(f"\n[✓] Zapisano wszystkie parametry do:\n    {ALL_PARAMS_CSV}")
    print(f"[✓] Zapisano tabelę porównawczą m i d do:\n    {DM_COMPARISON_CSV}")
    print(f"[✓] Zapisano średnie parametry do:\n    {AVG_PARAMS_CSV}\n")

    if not df_avg.empty:
        print("--- Średnie parametry (do użycia przy use_previous_avg_parameters=True) ---")
        print(df_avg.to_string(index=False))
        print()

    if not df_comp.empty:
        print("--- Tabela porównawcza d i m ---")
        print(df_comp.to_string(index=True))


if __name__ == "__main__":
    run_comparison(use_previous_avg_parameters=USE_PREVIOUS_AVG_PARAMETERS)