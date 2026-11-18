import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ==============================================================================
# FONT & STYLE CONFIGURATION (CONSISTENT WITH FIGS. 5.1–5.5)
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"  # Bold-italic \mathsf{Cn}
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

FS_LABEL = 13.5
FS_RES_LABEL = 11.0
FS_TICK = 12.0
FS_RES_TICK = 10.0
FS_LEGEND = 10.5
FS_TITLE = 12.5
FS_PANEL = 14.0

FIG_RC = {
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.linewidth": 1.0,
}

# Exact 12.0 in width matching other Chapter 5 multi-panel figures
FIG_SIZE = (12.0, 12.8)

# ==============================================================================
# PATHS & CONSTRUCT CONFIGURATIONS (LONGEST SK_FL RUN PER CONSTRUCT)
# ==============================================================================
PATHS = {
    "fig_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch05",
    "tables_out_dir": "/home/matifortunka/PycharmProjects/various_programs/PhD/tables",
    "search_roots": [
        "/home/matifortunka/Downloads/PhD/rozdział_V",
        "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw",
    ],
}

OUT_BASENAME = "Figura_5_6_modelSK"

SK_FL_CONSTRUCTS = [
    {
        "construct": "CnTrmD",
        "title_prefix": r"$\mathsf{Cn}$TrmD",
        "filename": "MultipleEmScan1_TrmD_v1_4500.csv",
        "candidate_paths": [
            "/home/matifortunka/Downloads/PhD/rozdział_V/TrmD/FL_SK/MultipleEmScan1_TrmD_v1_4500.csv",
            "/home/matifortunka/Downloads/PhD/rozdział_V/TrmD/fluo/SK/MultipleEmScan1_TrmD_v1_4500.csv",
            "/home/matifortunka/Downloads/PhD/rozdział_V/CnTrmD/FL_SK/MultipleEmScan1_TrmD_v1_4500.csv",
            "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/fluo/TrmD/spectra_kinetics/MultipleEmScan1_TrmD_v1_4500.csv",
            "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/fluo/CnTrmD/spectra_kinetics/MultipleEmScan1_TrmD_v1_4500.csv",
        ],
        "dead_time": 30.0,
        "time_interval": 32.0,
        "fit_range": (0.0, 20000.0),
        "xlim": (-150.0, 20000.0),
        "color": "#1F78B4",
        "panel_left": "A",
        "panel_right": "B",
        "n_runs_note": "n = 1",
    },
    {
        "construct": "Tm1570",
        "title_prefix": r"$\mathsf{Cn}$Tm1570",
        "filename": "MultipleEmScan1.csv",
        "candidate_paths": [
            "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/fluo/Tm1570/spectra_kinetics/MultipleEmScan1.csv",
            "/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/FL_SK/MultipleEmScan1.csv",
        ],
        "dead_time": 30.0,
        "time_interval": 32.0,
        "fit_range": (0.0, 14500.0),
        "xlim": (-150.0, 14500.0),
        "color": "#1B9E77",
        "panel_left": "C",
        "panel_right": "D",
        "n_runs_note": "n = 1",
    },
    {
        "construct": "CnTrmD-Tm1570",
        "title_prefix": r"$\mathsf{Cn}$TrmD–Tm1570",
        "filename": "MultipleEmScan1_fusion.csv",
        "candidate_paths": [
            "/home/matifortunka/Downloads/PhD/rozdział_V/Cn_fusion/FL_SK/MultipleEmScan1_fusion.csv",
            "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/fluo/Cn_fusion/spectra_kinetics/MultipleEmScan1_fusion.csv",
        ],
        "dead_time": 45.0,
        "time_interval": 32.0,
        "fit_range": (0.0, 19000.0),
        "xlim": (-150.0, 19000.0),
        "color": "#7570B3",
        "panel_left": "E",
        "panel_right": "F",
        "n_runs_note": "longest run (n = 2 total)",
    },
]

SIGNAL_SETTINGS = {
    "target_wl": 340.0,
    "baseline_wl": 400.0,
    "smoothing": True,
    "window_size": 25,
    "poly_order": 3,
    "res_smooth_window_s": 350.0,
    "res_smooth_polyorder": 2,
}


# ==============================================================================
# HELPER FUNCTIONS: FILE RESOLUTION, LOADING & SMOOTHING
# ==============================================================================
def darker(color, factor=0.68):
    r, g, b = mpl.colors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def resolve_data_file(cfg):
    """Finds the target CSV file from explicit candidate paths or recursive search."""
    for p in cfg["candidate_paths"]:
        if os.path.isfile(p):
            return p
    fname = cfg["filename"]
    for root in PATHS["search_roots"]:
        if os.path.isdir(root):
            matches = glob.glob(os.path.join(root, "**", fname), recursive=True)
            if matches:
                return sorted(matches)[0]
    raise FileNotFoundError(
        f"Could not locate '{fname}'. Checked:\n  " + "\n  ".join(cfg["candidate_paths"])
    )


def load_fl_sk_dataset(filepath):
    """Loads multi-spectrum fluorescence kinetics CSV (Wavelength + consecutive scans)."""
    data_start_row = -1
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) > 1 and re.match(r"^-?\d+(?:\.\d+)?$", parts[0].strip()):
                data_start_row = i
                break
    if data_start_row == -1:
        raise ValueError(f"No numeric spectral data block found in: {filepath}")

    df = pd.read_csv(filepath, skiprows=data_start_row, header=None)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(axis=0, how="all", inplace=True)
    df.rename(columns={df.columns[0]: "Wavelength"}, inplace=True)
    df.dropna(subset=["Wavelength"], inplace=True)
    return df.reset_index(drop=True)


def extract_sk_fl_trace(cfg):
    """Extracts baseline-corrected and smoothed 340 nm kinetic trace for a construct."""
    fp = resolve_data_file(cfg)
    df = load_fl_sk_dataset(fp)
    wavelengths = df["Wavelength"].values
    scan_cols = df.columns[1:]

    t_start, t_end = cfg["fit_range"]
    target_idx = int(np.abs(wavelengths - SIGNAL_SETTINGS["target_wl"]).argmin())
    bl_idx = int(np.abs(wavelengths - SIGNAL_SETTINGS["baseline_wl"]).argmin())

    w_sg = SIGNAL_SETTINGS["window_size"]
    if w_sg % 2 == 0:
        w_sg += 1

    times_s = []
    int_340 = []

    for i, col in enumerate(scan_cols):
        t_s = cfg["dead_time"] + i * cfg["time_interval"]
        if t_s < t_start or t_s > t_end:
            continue

        y_raw = df[col].values.astype(float)
        y_corr = y_raw - y_raw[bl_idx] if SIGNAL_SETTINGS["baseline_wl"] else y_raw.copy()

        if SIGNAL_SETTINGS["smoothing"] and len(y_corr) >= w_sg:
            y_corr = savgol_filter(y_corr, w_sg, SIGNAL_SETTINGS["poly_order"])

        times_s.append(t_s)
        int_340.append(y_corr[target_idx])

    return np.array(times_s, dtype=float), np.array(int_340, dtype=float)


def load_table_seed(tables_dir, construct, filename):
    """Reads pre-computed fit parameters from kinetic_fits.csv as initial guesses if present."""
    csv_path = os.path.join(tables_dir, "kinetic_fits.csv")
    if not os.path.isfile(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        sub = df[
            (df["Construct"] == construct)
            & (df["Modality"] == "SK_FL")
            & (df["File"] == filename)
        ]
        if not sub.empty:
            return sub.iloc[0].to_dict()
    except Exception:
        pass
    return None


# ==============================================================================
# KINETIC MODELS, STATISTICAL METRICS (R2, |r|, MAE, RMSE, AIC, BIC) & FITTING
# ==============================================================================
def mono_exp_drift(t, A1, k1, c, m):
    return A1 * np.exp(-k1 * t) + c + m * t


def double_exp(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


def double_exp_drift(t, A1, k1, A2, k2, c, m):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c + m * t


def calc_fit_metrics(y_true, y_pred, n_params):
    """Computes goodness-of-fit statistics consistent with Fig. 5.5."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    if n > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r_pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        abs_r = abs(r_pearson)
    else:
        r_pearson, abs_r = np.nan, np.nan

    mae = float(np.mean(np.abs(residuals)))
    max_abs_res = float(np.max(np.abs(residuals)))
    rmse = float(np.sqrt(ss_res / max(n, 1)))

    if ss_res > 0 and n > n_params:
        aic = float(n * np.log(ss_res / n) + 2 * n_params)
        bic = float(n * np.log(ss_res / n) + n_params * np.log(n))
    else:
        aic, bic = np.nan, np.nan

    return {
        "n_points": n,
        "n_params": n_params,
        "residuals": residuals,
        "R2": r2,
        "Pearson_r": r_pearson,
        "Abs_r": abs_r,
        "RMSE": rmse,
        "AIC": aic,
        "BIC": bic,
        "MAE": mae,
        "Max_Abs_Res": max_abs_res,
    }


def fit_sk_fl_trace(t, y, model_name, seed_row=None):
    """Fits SK_FL 340 nm trace and returns kinetic parameters + statistical metrics."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(np.max(y) - np.min(y))

    if model_name == "mono_exp_drift":
        if seed_row is not None and pd.notna(seed_row.get("k1_s_inv")):
            p0 = [
                float(seed_row["A1"]),
                float(seed_row["k1_s_inv"]),
                float(seed_row["c"]),
                float(seed_row["drift_m"]),
            ]
        else:
            p0 = [span * 0.8, 5e-4, float(np.min(y)), -1.0]

        bounds = (
            [-np.inf, 1e-6, -np.inf, -np.inf],
            [np.inf, 1.0, np.inf, np.inf],
        )
        popt, pcov = curve_fit(mono_exp_drift, t, y, p0=p0, bounds=bounds, maxfev=40000)
        perr = np.sqrt(np.diag(pcov))
        y_fit = mono_exp_drift(t, *popt)

        A1, k1, c, m = popt
        A1_err, k1_err, c_err, m_err = perr
        t_half1 = np.log(2.0) / k1
        t_half1_err = t_half1 * (k1_err / k1)

        metrics = calc_fit_metrics(y, y_fit, n_params=4)

        return {
            "model": model_name,
            "A1": A1, "A1_err": A1_err,
            "k1": k1, "k1_err": k1_err,
            "t_half1": t_half1, "t_half1_err": t_half1_err,
            "c": c, "c_err": c_err,
            "m": m, "m_err": m_err,
            "y_fit": y_fit,
            **metrics,
        }

    elif model_name == "double_exponential":
        p0_candidates = [
            [span * 0.25, 6e-3, span * 0.65, 3e-4, float(np.min(y))],
        ]
        if seed_row is not None and pd.notna(seed_row.get("k1_s_inv")) and pd.notna(seed_row.get("k2_s_inv")):
            p0_candidates.append([
                float(seed_row["A1"]),
                float(seed_row["k1_s_inv"]),
                float(seed_row["A2"]),
                float(seed_row["k2_s_inv"]),
                float(np.min(y)),
            ])

        bounds = (
            [-np.inf, 1e-5, -np.inf, 1e-6, -np.inf],
            [np.inf, 1.0, np.inf, 1e-1, np.inf],
        )

        best_fit = None
        best_ss_res = np.inf
        for p0 in p0_candidates:
            try:
                popt_cand, pcov_cand = curve_fit(
                    double_exp, t, y, p0=p0, bounds=bounds, maxfev=60000
                )
                y_cand = double_exp(t, *popt_cand)
                ss_cand = float(np.sum((y - y_cand) ** 2))
                if ss_cand < best_ss_res:
                    best_ss_res = ss_cand
                    best_fit = (popt_cand, pcov_cand)
            except Exception:
                continue

        if best_fit is None:
            raise RuntimeError("curve_fit failed to converge for double_exponential model.")

        popt, pcov = best_fit
        perr = np.sqrt(np.diag(pcov))

        A1, k1, A2, k2, c = popt
        A1_err, k1_err, A2_err, k2_err, c_err = perr

        # Ensure component 1 is the faster phase (k1 > k2)
        if k2 > k1:
            A1, A2 = A2, A1
            A1_err, A2_err = A2_err, A1_err
            k1, k2 = k2, k1
            k1_err, k2_err = k2_err, k1_err

        y_fit = double_exp(t, A1, k1, A2, k2, c)
        t_half1 = np.log(2.0) / k1
        t_half1_err = t_half1 * (k1_err / k1)
        t_half2 = np.log(2.0) / k2
        t_half2_err = t_half2 * (k2_err / k2)

        metrics = calc_fit_metrics(y, y_fit, n_params=5)

        return {
            "model": model_name,
            "A1": A1, "A1_err": A1_err,
            "k1": k1, "k1_err": k1_err,
            "t_half1": t_half1, "t_half1_err": t_half1_err,
            "A2": A2, "A2_err": A2_err,
            "k2": k2, "k2_err": k2_err,
            "t_half2": t_half2, "t_half2_err": t_half2_err,
            "c": c, "c_err": c_err,
            "y_fit": y_fit,
            **metrics,
        }

    elif model_name == "double_exponential_with_drift":
        if seed_row is not None and pd.notna(seed_row.get("k1_s_inv")) and pd.notna(seed_row.get("k2_s_inv")):
            p0 = [
                float(seed_row["A1"]),
                float(seed_row["k1_s_inv"]),
                float(seed_row["A2"]),
                float(seed_row["k2_s_inv"]),
                float(seed_row["c"]),
                float(seed_row["drift_m"]) if pd.notna(seed_row.get("drift_m")) else -0.6,
            ]
        else:
            p0 = [span * 0.25, 6e-3, span * 0.65, 3e-4, float(np.min(y)), -0.6]

        bounds = (
            [-np.inf, 1e-5, -np.inf, 1e-6, -np.inf, -np.inf],
            [np.inf, 1.0, np.inf, 1e-1, np.inf, np.inf],
        )
        popt, pcov = curve_fit(double_exp_drift, t, y, p0=p0, bounds=bounds, maxfev=60000)
        perr = np.sqrt(np.diag(pcov))

        A1, k1, A2, k2, c, m = popt
        A1_err, k1_err, A2_err, k2_err, c_err, m_err = perr

        # Ensure component 1 is the faster phase (k1 > k2)
        if k2 > k1:
            A1, A2 = A2, A1
            A1_err, A2_err = A2_err, A1_err
            k1, k2 = k2, k1
            k1_err, k2_err = k2_err, k1_err

        y_fit = double_exp_drift(t, A1, k1, A2, k2, c, m)
        t_half1 = np.log(2.0) / k1
        t_half1_err = t_half1 * (k1_err / k1)
        t_half2 = np.log(2.0) / k2
        t_half2_err = t_half2 * (k2_err / k2)

        metrics = calc_fit_metrics(y, y_fit, n_params=6)

        return {
            "model": model_name,
            "A1": A1, "A1_err": A1_err,
            "k1": k1, "k1_err": k1_err,
            "t_half1": t_half1, "t_half1_err": t_half1_err,
            "A2": A2, "A2_err": A2_err,
            "k2": k2, "k2_err": k2_err,
            "t_half2": t_half2, "t_half2_err": t_half2_err,
            "c": c, "c_err": c_err,
            "m": m, "m_err": m_err,
            "y_fit": y_fit,
            **metrics,
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def smooth_residuals_sg(t, residuals):
    t = np.asarray(t, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n = len(t)
    poly = SIGNAL_SETTINGS["poly_order"]
    if n < poly + 3:
        return residuals
    dt = float(np.median(np.diff(t)))
    win = int(round(SIGNAL_SETTINGS["res_smooth_window_s"] / dt)) if dt > 0 else 15
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, n if n % 2 == 1 else n - 1))
    return savgol_filter(residuals, window_length=win, polyorder=SIGNAL_SETTINGS["res_smooth_polyorder"])


# ==============================================================================
# PANEL RENDERING (340 NM KINETIC FIT + RESIDUALS PANEL)
# ==============================================================================
def plot_kinetic_panel(ax_kin, ax_res, times_s, int_340, cfg, model_name, panel_label, seed_row=None):
    res = fit_sk_fl_trace(times_s, int_340, model_name, seed_row=seed_row)

    color = cfg["color"]
    fit_color = darker(color, 0.65)

    if res["model"] == "double_exponential_with_drift":
        fit_label = (
            f"Double-exp + drift ($R^2 = {res['R2']:.4f}$)\n"
            rf"$t_{{1/2,\mathrm{{A}}}} = {res['t_half1']:.0f} \pm {res['t_half1_err']:.0f}\ \mathrm{{s}}$, "
            rf"$t_{{1/2,\mathrm{{C}}}} = {res['t_half2']:.0f} \pm {res['t_half2_err']:.0f}\ \mathrm{{s}}$"
        )
    elif res["model"] == "double_exponential":
        fit_label = (
            f"Double-exp ($R^2 = {res['R2']:.4f}$)\n"
            rf"$t_{{1/2,\mathrm{{A}}}} = {res['t_half1']:.0f} \pm {res['t_half1_err']:.0f}\ \mathrm{{s}}$, "
            rf"$t_{{1/2,\mathrm{{C}}}} = {res['t_half2']:.0f} \pm {res['t_half2_err']:.0f}\ \mathrm{{s}}$"
        )
    else:
        fit_label = (
            f"Mono-exp + drift ($R^2 = {res['R2']:.4f}$)\n"
            rf"$t_{{1/2,\mathrm{{C}}}} = {res['t_half1']:.0f} \pm {res['t_half1_err']:.0f}\ \mathrm{{s}}$, "
            rf"$m = {res['m']:.2f}\ \mathrm{{a.u.\cdot s^{{-1}}}}$"
        )

    ax_kin.scatter(
        times_s, int_340 / 1e4,
        s=13, alpha=0.65, color=color, edgecolors="none",
        label=f"340 nm trace ({cfg['n_runs_note']})",
    )
    ax_kin.plot(
        times_s, res["y_fit"] / 1e4,
        color=fit_color, ls="--", lw=1.8, label=fit_label,
    )

    ax_kin.set_xlim(*cfg["xlim"])
    ax_kin.set_ylabel(r"$I_{340}$ ($\times 10^4$ a.u.)")
    ax_kin.grid(True, linestyle=":", alpha=0.5)
    ax_kin.tick_params(labelbottom=False)
    ax_kin.set_title(
        rf"{cfg['title_prefix']} – Spectra-Kinetics ($\mathbf{{340\ nm}}$)",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=8,
    )
    ax_kin.text(
        -0.14, 1.07, panel_label,
        transform=ax_kin.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top",
    )

    ax_kin.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.97),
        fontsize=FS_LEGEND,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        markerscale=2.0,
    )

    # Attached residual panel
    res_scaled = res["residuals"] / 1e4
    res_smooth = smooth_residuals_sg(times_s, res_scaled)

    ax_res.axhline(0.0, color="#333333", lw=0.9, ls="-", alpha=0.8)
    ax_res.scatter(times_s, res_scaled, s=6, alpha=0.35, color=color, edgecolors="none")
    ax_res.plot(times_s, res_smooth, color=fit_color, lw=1.6, alpha=0.95)

    max_abs_res = max(0.05, float(np.nanmax(np.abs(res_scaled))) * 1.25)
    ax_res.set_ylim(-max_abs_res, max_abs_res)
    ax_res.set_xlim(*cfg["xlim"])
    ax_res.set_xlabel("Time (s)")
    ax_res.set_ylabel("Residuals", fontsize=FS_RES_LABEL)
    ax_res.tick_params(axis="both", labelsize=FS_RES_TICK)
    ax_res.grid(True, linestyle=":", alpha=0.45)

    # Console summary per fit
    if res["model"] == "double_exponential_with_drift":
        print(
            f"[{cfg['construct']} | {cfg['filename']} | double-exp + drift] "
            f"t1/2_1 = {res['t_half1']:.1f} ± {res['t_half1_err']:.1f} s, "
            f"t1/2_2 = {res['t_half2']:.1f} ± {res['t_half2_err']:.1f} s, "
            f"m = {res['m']:.3f} a.u./s | "
            f"R2 = {res['R2']:.4f}, |r| = {res['Abs_r']:.4f}, "
            f"MAE = {res['MAE']:.1f}, RMSE = {res['RMSE']:.1f}, BIC = {res['BIC']:.1f}"
        )
    elif res["model"] == "double_exponential":
        print(
            f"[{cfg['construct']} | {cfg['filename']} | double-exp] "
            f"t1/2_1 = {res['t_half1']:.1f} ± {res['t_half1_err']:.1f} s, "
            f"t1/2_2 = {res['t_half2']:.1f} ± {res['t_half2_err']:.1f} s | "
            f"R2 = {res['R2']:.4f}, |r| = {res['Abs_r']:.4f}, "
            f"MAE = {res['MAE']:.1f}, RMSE = {res['RMSE']:.1f}, BIC = {res['BIC']:.1f}"
        )
    else:
        print(
            f"[{cfg['construct']} | {cfg['filename']} | mono-exp + drift] "
            f"t1/2 = {res['t_half1']:.1f} ± {res['t_half1_err']:.1f} s, "
            f"m = {res['m']:.3f} a.u./s | "
            f"R2 = {res['R2']:.4f}, |r| = {res['Abs_r']:.4f}, "
            f"MAE = {res['MAE']:.1f}, RMSE = {res['RMSE']:.1f}, BIC = {res['BIC']:.1f}"
        )

    return res


# ==============================================================================
# MAIN FIGURE BUILDER & STATISTICAL COMPARISON TABLE EXPORT
# ==============================================================================
def make_figure_sk_fl_cn_system():
    os.makedirs(PATHS["fig_out_dir"], exist_ok=True)
    os.makedirs(PATHS["tables_out_dir"], exist_ok=True)
    plt.rcParams.update(FIG_RC)

    fig = plt.figure(figsize=FIG_SIZE, dpi=300)
    outer_gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.32,
    )

    comparison_rows = []

    for row_idx, cfg in enumerate(SK_FL_CONSTRUCTS):
        times_s, int_340 = extract_sk_fl_trace(cfg)
        seed_row = load_table_seed(PATHS["tables_out_dir"], cfg["construct"], cfg["filename"])

        # Left column (A, C, E): double-exponential with drift
        left_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_gs[row_idx, 0],
            height_ratios=[3.2, 1.0],
            hspace=0.08,
        )
        ax_kin_left = fig.add_subplot(left_gs[0])
        ax_res_left = fig.add_subplot(left_gs[1], sharex=ax_kin_left)
        res_ded = plot_kinetic_panel(
            ax_kin_left, ax_res_left, times_s, int_340, cfg,
            model_name="double_exponential_with_drift",
            panel_label=cfg["panel_left"],
            seed_row=seed_row,
        )

        # Right column (B, D, F): double-exponential without drift
        right_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_gs[row_idx, 1],
            height_ratios=[3.2, 1.0],
            hspace=0.08,
        )
        ax_kin_right = fig.add_subplot(right_gs[0])
        ax_res_right = fig.add_subplot(right_gs[1], sharex=ax_kin_right)
        res_de = plot_kinetic_panel(
            ax_kin_right, ax_res_right, times_s, int_340, cfg,
            model_name="double_exponential",
            panel_label=cfg["panel_right"],
            seed_row=seed_row,
        )

        # Collect statistical metrics for table expansion (BIC, |r|, MAE, RMSE, AIC, R2)
        comparison_rows.append({
            "Construct": cfg["construct"],
            "Modality": "SK_FL",
            "File": cfg["filename"],
            "n_points": res_ded["n_points"],
            # Double-exponential + drift (Panels A, C, E)
            "R2_double_exp_drift": res_ded["R2"],
            "Abs_r_double_exp_drift": res_ded["Abs_r"],
            "RMSE_double_exp_drift": res_ded["RMSE"],
            "MAE_double_exp_drift": res_ded["MAE"],
            "MaxRes_double_exp_drift": res_ded["Max_Abs_Res"],
            "RMSE_1e4_double_exp_drift": res_ded["RMSE"] / 1e4,
            "MAE_1e4_double_exp_drift": res_ded["MAE"] / 1e4,
            "MaxRes_1e4_double_exp_drift": res_ded["Max_Abs_Res"] / 1e4,
            "AIC_double_exp_drift": res_ded["AIC"],
            "BIC_double_exp_drift": res_ded["BIC"],
            "t_half1_DED": res_ded["t_half1"],
            "t_half1_err_DED": res_ded["t_half1_err"],
            "t_half2_DED": res_ded["t_half2"],
            "t_half2_err_DED": res_ded["t_half2_err"],
            "drift_m_DED": res_ded["m"],
            "drift_m_err_DED": res_ded["m_err"],
            # Double-exponential without drift (Panels B, D, F)
            "R2_double_exp": res_de["R2"],
            "Abs_r_double_exp": res_de["Abs_r"],
            "RMSE_double_exp": res_de["RMSE"],
            "MAE_double_exp": res_de["MAE"],
            "MaxRes_double_exp": res_de["Max_Abs_Res"],
            "RMSE_1e4_double_exp": res_de["RMSE"] / 1e4,
            "MAE_1e4_double_exp": res_de["MAE"] / 1e4,
            "MaxRes_1e4_double_exp": res_de["Max_Abs_Res"] / 1e4,
            "AIC_double_exp": res_de["AIC"],
            "BIC_double_exp": res_de["BIC"],
            "t_half1_DE": res_de["t_half1"],
            "t_half1_err_DE": res_de["t_half1_err"],
            "t_half2_DE": res_de["t_half2"],
            "t_half2_err_DE": res_de["t_half2_err"],
            # Information criterion differences (positive => double-exp + drift preferred)
            "Delta_AIC_de_minus_ded": res_de["AIC"] - res_ded["AIC"],
            "Delta_BIC_de_minus_ded": res_de["BIC"] - res_ded["BIC"],
        })

        # Synchronize y-axis limits between left and right kinetic panels in each row
        y_min = min(ax_kin_left.get_ylim()[0], ax_kin_right.get_ylim()[0])
        y_max = max(ax_kin_left.get_ylim()[1], ax_kin_right.get_ylim()[1])
        ax_kin_left.set_ylim(y_min, y_max)
        ax_kin_right.set_ylim(y_min, y_max)

        fig.align_ylabels([ax_kin_left, ax_res_left])
        fig.align_ylabels([ax_kin_right, ax_res_right])

    fig.subplots_adjust(left=0.085, right=0.975, top=0.96, bottom=0.055)

    out_pdf = os.path.join(PATHS["fig_out_dir"], f"{OUT_BASENAME}.pdf")
    out_png = os.path.join(PATHS["fig_out_dir"], f"{OUT_BASENAME}.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"\nSaved figure:\n  {out_pdf}\n  {out_png}")

    if comparison_rows:
        df_comp = pd.DataFrame(comparison_rows)
        comp_csv = os.path.join(PATHS["tables_out_dir"], "sk_fl_model_selection_comparison.csv")
        df_comp.to_csv(comp_csv, index=False)

        print("\n--- STATYSTYKI DOPASOWANIA SK_FL (DOUBLE-EXP + DRIFT vs DOUBLE-EXP) ---")
        cols_show = [
            "Construct",
            "BIC_double_exp_drift", "BIC_double_exp", "Delta_BIC_de_minus_ded",
            "Abs_r_double_exp_drift", "Abs_r_double_exp",
            "MaxRes_1e4_double_exp_drift", "MaxRes_1e4_double_exp",
            "MAE_1e4_double_exp_drift", "MAE_1e4_double_exp",
            "RMSE_1e4_double_exp_drift", "RMSE_1e4_double_exp",
        ]
        print(df_comp[cols_show].to_string(index=False))
        print(f"\nSaved statistical comparison table:\n  {comp_csv}")


if __name__ == "__main__":
    make_figure_sk_fl_cn_system()