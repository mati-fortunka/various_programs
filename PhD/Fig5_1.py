import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ==============================================================================
# KONFIGURACJA ŚCIEŻEK I PARAMETRÓW
# ==============================================================================
PATHS = {
    "mm_fl_dir": "/home/matifortunka/Downloads/PhD/rozdział_V/TrmD",
    "sk_fl_file": "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/fluo/TrmD/spectra_kin/MultipleEmScan1_TrmD_v1_4500.csv",
    "cd_native": "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/comparison/widma/folded/TrmD00000.csv",
    "cd_unfolded": "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/CD/TrmD_kinetyka/widma/TrmD_5uM_gucl_6M00001.csv",
    "fig_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch05",
    "tables_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/tables",
}

PARAMS = {
    "construct": "CnTrmD",
    "n_res": 262,            # CnTrmD (Mw = 29988.12, MRW = 114.46)
    "c_cd_native_uM": 5.0,
    "c_cd_unfolded_uM": 5.0,
    "pathlength_mm": 1.0,
    "hv_threshold": 900.0,   # Zaktualizowany próg HV <= 900 V
    "dead_time_mm": 30.0,    # Czas martwy ręcznego mieszania (s)
    "fit_end_mm": 2000.0,    # Okno dopasowania Panelu A (s)
    "sk_time_step": 32.0,    # Interwał między skanami w SK FL (s)
    "sk_target_wl": 340.0,   # Długość fali analizowana w Panelu C (nm)
    "sk_baseline_wl": 400.0,
    "sk_fit_end": 20000.0,   # Okno dopasowania Paneli B i C (s)
}

# Kolory spójne z Rozdziałem 4
COLOR_5UM = "#1F78B4"  # Jednolity niebieski dla serii 5.0 uM
COLOR_2UM = "#D95F02"  # Jednolity pomarańczowy dla serii 2.0 uM

plasma_full = plt.cm.plasma
plasma_trimmed = LinearSegmentedColormap.from_list(
    "plasma_trimmed",
    [plasma_full(x) for x in np.linspace(0.05, 0.85, 256)]
)


# ==============================================================================
# MODELE MATEMATYCZNE
# ==============================================================================
def double_exp(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def double_exp_drift(t, A1, k1, A2, k2, c, m):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c + m * t


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan


def fit_double_exp_robust(time, intensity):
    A0 = max(intensity) - min(intensity)
    C0 = min(intensity)
    p0 = [0.4 * A0, 0.015, 0.6 * A0, 0.0006, C0]
    bounds = ([0.0, 1e-4, 0.0, 1e-6, -np.inf], [np.inf, 1.0, np.inf, 0.1, np.inf])
    popt, pcov = curve_fit(double_exp, time, intensity, p0=p0, bounds=bounds, maxfev=30000)
    perr = np.sqrt(np.diag(pcov))
    if popt[1] < popt[3]:
        popt = np.array([popt[2], popt[3], popt[0], popt[1], popt[4]])
        perr = np.array([perr[2], perr[3], perr[0], perr[1], perr[4]])
    r2 = calc_r2(intensity, double_exp(time, *popt))
    return popt, perr, r2


def fit_double_exp_drift_robust(time, intensity):
    A0 = max(intensity) - min(intensity)
    C0 = min(intensity)
    p0 = [0.25 * A0, 0.015, 0.75 * A0, 0.0005, C0, -0.5]
    bounds = (
        [0.0, 1e-4, 0.0, 1e-6, -np.inf, -np.inf],
        [np.inf, 1.0, np.inf, 0.1, np.inf, np.inf]
    )
    popt, pcov = curve_fit(double_exp_drift, time, intensity, p0=p0, bounds=bounds, maxfev=40000)
    perr = np.sqrt(np.diag(pcov))
    if popt[1] < popt[3]:
        popt = np.array([popt[2], popt[3], popt[0], popt[1], popt[4], popt[5]])
        perr = np.array([perr[2], perr[3], perr[0], perr[1], perr[4], perr[5]])
    r2 = calc_r2(intensity, double_exp_drift(time, *popt))
    return popt, perr, r2


# ==============================================================================
# FUNKCJE WCZYTYWANIA DANYCH
# ==============================================================================
def read_mm_fl(filename):
    with open(filename, "r") as f:
        header_lines = [f.readline() for _ in range(15)]

    is_new = any(line.startswith("Labels,") for line in header_lines)
    if is_new:
        skip_rows = 0
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if re.match(r"^[\d]", line):
                    skip_rows = i
                    break
        df = pd.read_csv(filename, skiprows=skip_rows, header=None).dropna(how="all", axis=1)
        df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
        df.iloc[:, 0] = df.iloc[:, 0] / 1e9  # ns -> s
        df.columns = ["Time", "Intensity"]
        return df
    else:
        with open(filename, "r") as f:
            l1, l2 = f.readline(), f.readline()
        is_min = ("Time (min)" in l1) or ("Time (min)" in l2)
        df_full = pd.read_csv(filename, skiprows=2, sep=",").dropna(how="all", axis=1)
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
        return df_full


def load_sk_fl(filename):
    data_start = -1
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) > 1 and parts[0].replace(".", "", 1).isdigit():
                try:
                    float(parts[0])
                    data_start = i
                    break
                except ValueError:
                    continue
    df = pd.read_csv(filename, skiprows=data_start, header=None).dropna(axis=1, how="all")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
    df.rename(columns={df.columns[0]: "Wavelength"}, inplace=True)
    df.dropna(subset=["Wavelength"], inplace=True)
    return df


def parse_prodata_block(lines, keyword):
    """Znajduje drugie wystąpienie słowa kluczowego w pliku ProData CSV
    i automatycznie wykrywa pierwszy wiersz z danymi liczbowymi (bez pomijania 250 nm)."""
    count = 0
    kw_idx = -1
    for i, line in enumerate(lines):
        if keyword in line:
            count += 1
            if count == 2:
                kw_idx = i
                break
    if kw_idx == -1:
        raise ValueError(f"Drugie wystąpienie '{keyword}' nie zostało znalezione w pliku.")

    start_idx = None
    for i in range(kw_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        parts = [p for p in stripped.replace(",", " ").split() if p]
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
                start_idx = i
                break
            except ValueError:
                continue

    if start_idx is None:
        raise ValueError(f"Nie znaleziono danych liczbowych w bloku '{keyword}'.")

    rows = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            break
        parts = [p for p in stripped.replace(",", " ").split() if p]
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            break

    return np.array(rows)


def read_prodata_cd(file_path, hv_threshold, c_uM, l_mm, n_res, window=15, poly=3):
    """Parser plików ProData CSV z filtrowaniem HV, wygładzaniem, korektą linii bazowej
    (400 nm jeśli dostępne, w przeciwnym razie 250 nm) oraz przeliczeniem na [theta]_MRW."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = lines[15:]

    cd_data = parse_prodata_block(lines, "CircularDichroism")
    hv_data = parse_prodata_block(lines, "HV")

    wl = cd_data[:, 0]
    ellip_mdeg = cd_data[:, 1] if cd_data.shape[1] == 2 else cd_data[:, 1:].mean(axis=1)
    hv_vals = hv_data[:, 1] if hv_data.shape[1] == 2 else hv_data[:, 1:].mean(axis=1)

    valid = hv_vals <= hv_threshold
    wl = wl[valid]
    ellip_mdeg = ellip_mdeg[valid]

    sort_idx = np.argsort(wl)
    wl = wl[sort_idx]
    ellip_mdeg = ellip_mdeg[sort_idx]

    if len(ellip_mdeg) >= window:
        ellip_mdeg = savgol_filter(ellip_mdeg, window_length=window, polyorder=poly)

    mrw_ellip = (ellip_mdeg * 1e6) / (c_uM * l_mm * n_res)

    # Korekta linii bazowej: przy 400 nm (jeśli skan sięga 400 nm) lub przy 250 nm
    if wl.max() >= 395.0:
        bl_wl = 400.0
    else:
        bl_wl = 250.0
    baseline_val = np.interp(bl_wl, wl, mrw_ellip)
    mrw_ellip = mrw_ellip - baseline_val

    print(f"CD ({os.path.basename(file_path)}): zakres {wl.min():.1f}-{wl.max():.1f} nm, zerowanie przy {bl_wl:.0f} nm")
    return wl, mrw_ellip


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
        mask_keep = [
            (c, m) not in keys for c, m in zip(df_old["Construct"], df_old["Modality"])
        ]
        df_combined = pd.concat([df_old[mask_keep], df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(fits_path, index=False)

    df_mean_new = pd.DataFrame(mean_rows)
    if os.path.exists(mean_path):
        df_mean_old = pd.read_csv(mean_path)
        keys_m = set(zip(df_mean_new["Construct"], df_mean_new["Modality"]))
        mask_keep_m = [
            (c, m) not in keys_m for c, m in zip(df_mean_old["Construct"], df_mean_old["Modality"])
        ]
        df_mean_combined = pd.concat([df_mean_old[mask_keep_m], df_mean_new], ignore_index=True)
    else:
        df_mean_combined = df_mean_new
    df_mean_combined.to_csv(mean_path, index=False)

    print(f"Zaktualizowano tabele:\n  {fits_path}\n  {mean_path}")


# ==============================================================================
# GŁÓWNA FUNKCJA RYSUJĄCA FIGURĘ 5.1
# ==============================================================================
def make_figure_5_1():
    os.makedirs(PATHS["fig_out_dir"], exist_ok=True)
    os.makedirs(PATHS["tables_out_dir"], exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 10,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.0), dpi=300)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    individual_records = []
    mean_records = []

    # --------------------------------------------------------------------------
    # PANEL A: MM FL (wyłącznie 2 kolory: niebieski dla 5.0 uM, pomarańczowy dla 2.0 uM)
    # --------------------------------------------------------------------------
    all_csvs = sorted(glob.glob(os.path.join(PATHS["mm_fl_dir"], "**", "*.csv"), recursive=True))
    files_5uM = [f for f in all_csvs if re.search(r"TrmD[1-4]_4000s", os.path.basename(f))]
    files_2uM = [f for f in all_csvs if re.search(r"Sample1.*TrmD", os.path.basename(f))]

    axA_right = axA.twinx()
    mm_fl_runs = []

    for idx, fp in enumerate(files_5uM):
        df = read_mm_fl(fp)
        t = df["Time"].values + PARAMS["dead_time_mm"]
        y_raw = df["Intensity"].values
        mask = (t >= 0) & (t <= PARAMS["fit_end_mm"])
        t_m, y_m = t[mask], y_raw[mask]
        y_smooth = savgol_filter(y_m, 25, 3)

        lbl = r"$5.0~\mu\mathrm{M}$ ($n=4$)" if idx == 0 else None
        axA.plot(t_m, y_smooth / 1e5, color=COLOR_5UM, lw=1.4, alpha=0.85, label=lbl)

        popt, perr, r2 = fit_double_exp_robust(t_m, y_m)
        axA.plot(t_m, double_exp(t_m, *popt) / 1e5, color=COLOR_5UM, ls=":", lw=1.15)
        mm_fl_runs.append((fp, 5.0, popt, perr, r2))

    for idx, fp in enumerate(files_2uM):
        df = read_mm_fl(fp)
        t = df["Time"].values + PARAMS["dead_time_mm"]
        y_raw = df["Intensity"].values
        mask = (t >= 0) & (t <= PARAMS["fit_end_mm"])
        t_m, y_m = t[mask], y_raw[mask]
        y_smooth = savgol_filter(y_m, 25, 3)

        lbl = r"$2.0~\mu\mathrm{M}$ ($n=4$)" if idx == 0 else None
        axA_right.plot(t_m, y_smooth / 1e4, color=COLOR_2UM, lw=1.4, alpha=0.85, label=lbl)

        popt, perr, r2 = fit_double_exp_robust(t_m, y_m)
        axA_right.plot(t_m, double_exp(t_m, *popt) / 1e4, color=COLOR_2UM, ls=":", lw=1.15)
        mm_fl_runs.append((fp, 2.0, popt, perr, r2))

    for fp, conc_uM, popt, perr, r2 in mm_fl_runs:
        A1, k1, A2, k2, c = popt
        eA1, ek1, eA2, ek2, ec = perr
        th1 = np.log(2) / k1
        th1_err = (np.log(2) / (k1 ** 2)) * ek1
        th2 = np.log(2) / k2
        th2_err = (np.log(2) / (k2 ** 2)) * ek2

        individual_records.append({
            "Construct": PARAMS["construct"],
            "Modality": "MM_FL",
            "Condition": "6.0 M GdmCl, 298 K",
            "Concentration_uM": conc_uM,
            "File": os.path.basename(fp),
            "Model": "double_exponential",
            "Fit_Range_s": f"0-{int(PARAMS['fit_end_mm'])}",
            "A1": A1, "A1_err": eA1,
            "k1_s_inv": k1, "k1_err": ek1,
            "t_half1_s": th1, "t_half1_err": th1_err,
            "A2": A2, "A2_err": eA2,
            "k2_s_inv": k2, "k2_err": ek2,
            "t_half2_s": th2, "t_half2_err": th2_err,
            "c": c, "c_err": ec,
            "drift_m": np.nan, "drift_m_err": np.nan,
            "R2": r2,
        })

    if mm_fl_runs:
        k1_arr = np.array([r[2][1] for r in mm_fl_runs])
        k2_arr = np.array([r[2][3] for r in mm_fl_runs])
        th1_arr = np.log(2) / k1_arr
        th2_arr = np.log(2) / k2_arr
        mean_records.append({
            "Construct": PARAMS["construct"],
            "Modality": "MM_FL",
            "Condition": "6.0 M GdmCl, 298 K",
            "Model": "double_exponential",
            "Fit_Range_s": f"0-{int(PARAMS['fit_end_mm'])}",
            "N_runs": len(mm_fl_runs),
            "k1_mean_s_inv": np.mean(k1_arr), "k1_sd_s_inv": np.std(k1_arr, ddof=1),
            "t_half1_mean_s": np.mean(th1_arr), "t_half1_sd_s": np.std(th1_arr, ddof=1),
            "k2_mean_s_inv": np.mean(k2_arr), "k2_sd_s_inv": np.std(k2_arr, ddof=1),
            "t_half2_mean_s": np.mean(th2_arr), "t_half2_sd_s": np.std(th2_arr, ddof=1),
            "drift_m_mean": np.nan, "drift_m_sd": np.nan,
        })

    axA.set_xlim(0, PARAMS["fit_end_mm"])
    axA.set_xlabel("Time (s)")
    axA.set_ylabel(r"Fluorescence at $5.0~\mu\mathrm{M}$ ($\times 10^5$ a.u.)", color=COLOR_5UM)
    axA_right.set_ylabel(r"Fluorescence at $2.0~\mu\mathrm{M}$ ($\times 10^4$ a.u.)", color=COLOR_2UM)
    axA.tick_params(axis="y", colors=COLOR_5UM)
    axA_right.tick_params(axis="y", colors=COLOR_2UM)
    axA.grid(True, linestyle=":", alpha=0.5)

    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = axA_right.get_legend_handles_labels()
    if h1 or h2:
        axA.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, framealpha=0.92)

    # --------------------------------------------------------------------------
    # PANEL B: SK FL Widma (colorbar przesunięty bardziej w prawo: x0 = 0.81)
    # --------------------------------------------------------------------------
    df_sk = load_sk_fl(PATHS["sk_fl_file"])
    wls = df_sk["Wavelength"].values
    scan_cols = df_sk.columns[1:]
    num_scans = len(scan_cols)
    times_sk = np.arange(num_scans) * PARAMS["sk_time_step"]

    max_t = PARAMS["sk_fit_end"]
    valid_idx = np.where(times_sk <= max_t)[0]
    norm_b = Normalize(vmin=0.0, vmax=max_t)

    int_340 = np.zeros(num_scans)
    bl_idx = np.abs(wls - PARAMS["sk_baseline_wl"]).argmin()
    wl_idx = np.abs(wls - PARAMS["sk_target_wl"]).argmin()

    step = max(1, len(valid_idx) // 50)
    for i in range(num_scans):
        raw_y = df_sk[scan_cols[i]].values
        corr_y = raw_y - raw_y[bl_idx]
        smooth_y = savgol_filter(corr_y, 25, 3) if len(corr_y) >= 25 else corr_y
        int_340[i] = smooth_y[wl_idx]

        if i in valid_idx and (i % step == 0):
            axB.plot(wls, smooth_y / 1e4, color=plasma_trimmed(norm_b(times_sk[i])), lw=1.25, alpha=0.88)

    axB.set_xlim(300.0, 400.0)
    axB.set_xlabel("Wavelength (nm)")
    axB.set_ylabel(r"Fluorescence Intensity ($\times 10^4$ a.u.)")
    axB.grid(True, linestyle=":", alpha=0.5)

    # Inset colorbar przesunięty w prawo w prawym górnym rogu
    cax_b = axB.inset_axes([0.81, 0.50, 0.040, 0.42])
    sm_b = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_b)
    sm_b.set_array([])
    cbar_b = fig.colorbar(sm_b, cax=cax_b, orientation="vertical")
    cbar_b.solids.set_rasterized(True)
    cbar_b.set_ticks([0, 5000, 10000, 15000, 20000])
    cbar_b.set_ticklabels(["0", "5000", "10000", "15000", "20000"])
    cbar_b.ax.tick_params(labelsize=8.0)
    cbar_b.set_label("Time (s)", fontsize=9.0, labelpad=3)

    # --------------------------------------------------------------------------
    # PANEL C: SK FL przy 340 nm z marginesem przed t=0 (-300 s) + double_exp_drift
    # --------------------------------------------------------------------------
    mask_sk = times_sk <= PARAMS["sk_fit_end"]
    t_fit_sk = times_sk[mask_sk]
    y_fit_sk = int_340[mask_sk]

    popt_sk, perr_sk, r2_sk = fit_double_exp_drift_robust(t_fit_sk, y_fit_sk)
    A1_s, k1_s, A2_s, k2_s, c_s, m_s = popt_sk
    eA1_s, ek1_s, eA2_s, ek2_s, ec_s, em_s = perr_sk

    th1_s = np.log(2) / k1_s
    th1_s_err = (np.log(2) / (k1_s ** 2)) * ek1_s
    th2_s = np.log(2) / k2_s
    th2_s_err = (np.log(2) / (k2_s ** 2)) * ek2_s

    individual_records.append({
        "Construct": PARAMS["construct"],
        "Modality": "SK_FL",
        "Condition": "6.0 M GdmCl, 298 K",
        "Concentration_uM": 5.0,
        "File": os.path.basename(PATHS["sk_fl_file"]),
        "Model": "double_exponential_with_drift",
        "Fit_Range_s": f"0-{int(PARAMS['sk_fit_end'])}",
        "A1": A1_s, "A1_err": eA1_s,
        "k1_s_inv": k1_s, "k1_err": ek1_s,
        "t_half1_s": th1_s, "t_half1_err": th1_s_err,
        "A2": A2_s, "A2_err": eA2_s,
        "k2_s_inv": k2_s, "k2_err": ek2_s,
        "t_half2_s": th2_s, "t_half2_err": th2_s_err,
        "c": c_s, "c_err": ec_s,
        "drift_m": m_s, "drift_m_err": em_s,
        "R2": r2_sk,
    })

    mean_records.append({
        "Construct": PARAMS["construct"],
        "Modality": "SK_FL",
        "Condition": "6.0 M GdmCl, 298 K",
        "Model": "double_exponential_with_drift",
        "Fit_Range_s": f"0-{int(PARAMS['sk_fit_end'])}",
        "N_runs": 1,
        "k1_mean_s_inv": k1_s, "k1_sd_s_inv": ek1_s,
        "t_half1_mean_s": th1_s, "t_half1_sd_s": th1_s_err,
        "k2_mean_s_inv": k2_s, "k2_sd_s_inv": ek2_s,
        "t_half2_mean_s": th2_s, "t_half2_sd_s": th2_s_err,
        "drift_m_mean": m_s, "drift_m_sd": em_s,
    })

    axC.plot(t_fit_sk, y_fit_sk / 1e4, "o", color="#2B8CBE", ms=3.2, alpha=0.75, label="Experimental data")
    t_dense = np.linspace(0.0, PARAMS["sk_fit_end"], 600)
    axC.plot(t_dense, double_exp_drift(t_dense, *popt_sk) / 1e4, color="#E31A1C", lw=1.9, label="Double-exp + drift fit")

    # Przestrzeń przed t = 0 s, aby pierwsze punkty były dobrze widoczne
    axC.set_xlim(-350.0, PARAMS["sk_fit_end"] + 300.0)
    axC.set_xticks([0, 5000, 10000, 15000, 20000])
    axC.set_xlabel("Time (s)")
    axC.set_ylabel(r"Intensity at $340~\mathrm{nm}$ ($\times 10^4$ a.u.)")
    axC.grid(True, linestyle=":", alpha=0.5)
    axC.legend(loc="upper right", frameon=True, framealpha=0.92)

    # --------------------------------------------------------------------------
    # PANEL D: Far-UV CD (HV <= 900 V, pełny zakres do 250 nm, baseline zeroing)
    # --------------------------------------------------------------------------
    wl_nat, mrw_nat = read_prodata_cd(
        PATHS["cd_native"], PARAMS["hv_threshold"],
        PARAMS["c_cd_native_uM"], PARAMS["pathlength_mm"], PARAMS["n_res"]
    )
    wl_unf, mrw_unf = read_prodata_cd(
        PATHS["cd_unfolded"], PARAMS["hv_threshold"],
        PARAMS["c_cd_unfolded_uM"], PARAMS["pathlength_mm"], PARAMS["n_res"]
    )

    axD.plot(wl_nat, mrw_nat / 1000.0, color="#1B7837", lw=2.0, label="Native ($0~\\mathrm{M}$ GdmCl)")
    axD.plot(wl_unf, mrw_unf / 1000.0, color="#D95F02", lw=2.0, label="Post-burst (~170 s, $6~\\mathrm{M}$ GdmCl)")
    axD.axhline(0.0, color="black", lw=0.7, alpha=0.5)

    axD.set_xlim(200.0, 250.0)
    axD.set_xlabel("Wavelength (nm)")
    axD.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$")
    axD.grid(True, linestyle=":", alpha=0.5)
    axD.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Etykiety paneli A, B, C, D
    for ax, label in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=13.5, fontweight="bold", va="top")

    plt.tight_layout(pad=1.4)
    out_pdf = os.path.join(PATHS["fig_out_dir"], "Figura_5_1_Kinetyka_CnTrmD.pdf")
    out_png = os.path.join(PATHS["fig_out_dir"], "Figura_5_1_Kinetyka_CnTrmD.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    update_fits_tables(individual_records, mean_records, PATHS["tables_out_dir"])
    print(f"Zapisano figurę:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    make_figure_5_1()