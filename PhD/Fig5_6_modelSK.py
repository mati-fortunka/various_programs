import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
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
FS_CBAR_LABEL = 11.0
FS_CBAR_TICK = 10.0

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

# Trimmed plasma colormap (avoids pale yellow at the top end)
plasma_trimmed = mpl.colors.LinearSegmentedColormap.from_list(
    "plasma_trimmed", plt.cm.plasma(np.linspace(0.0, 0.90, 256))
)

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

OUT_BASENAME = "Figura_A_SK_FL_Cn_System"

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
        "cbar_ticks": [0, 5000, 10000, 15000, 20000],
        "model": "double_exponential_with_drift",
        "color": "#1F78B4",
        "panel_spec": "A",
        "panel_kin": "B",
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
        "cbar_ticks": [0, 3500, 7000, 10500, 14500],
        "model": "double_exponential_with_drift",
        "color": "#1B9E77",
        "panel_spec": "C",
        "panel_kin": "D",
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
        "cbar_ticks": [0, 5000, 10000, 15000, 19000],
        "model": "double_exponential_with_drift",
        "color": "#7570B3",
        "panel_spec": "E",
        "panel_kin": "F",
        "n_runs_note": "longest run (n = 2 total)",
    },
]

SIGNAL_SETTINGS = {
    "target_wl": 340.0,
    "baseline_wl": 400.0,
    "smoothing": True,
    "window_size": 25,
    "poly_order": 3,
    "spec_wl_min": 300.0,
    "spec_wl_max": 400.0,
    "max_spectra_curves": 55,
    "res_smooth_window_s": 350.0,
    "res_smooth_polyorder": 2,
}


# ==============================================================================
# HELPER FUNCTIONS: FILE RESOLUTION, LOADING, SMOOTHING & FITTING
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


def mono_exp_drift(t, A1, k1, c, m):
    return A1 * np.exp(-k1 * t) + c + m * t


def double_exp_drift(t, A1, k1, A2, k2, c, m):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c + m * t


def fit_sk_fl_trace(t, y, model_name, seed_row=None):
    """Fits SK_FL 340 nm trace using mono_exp_drift or double_exponential_with_drift."""
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

        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot

        return {
            "model": model_name,
            "A1": A1, "A1_err": A1_err,
            "k1": k1, "k1_err": k1_err,
            "t_half1": t_half1, "t_half1_err": t_half1_err,
            "c": c, "c_err": c_err,
            "m": m, "m_err": m_err,
            "R2": r2,
            "y_fit": y_fit,
            "residuals": y - y_fit,
        }

    elif model_name == "double_exponential_with_drift":
        if seed_row is not None and pd.notna(seed_row.get("k1_s_inv")) and pd.notna(seed_row.get("k2_s_inv")):
            p0 = [
                float(seed_row["A1"]),
                float(seed_row["k1_s_inv"]),
                float(seed_row["A2"]),
                float(seed_row["k2_s_inv"]),
                float(seed_row["c"]),
                float(seed_row["drift_m"]),
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

        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot

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
            "R2": r2,
            "y_fit": y_fit,
            "residuals": y - y_fit,
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
# ROW RENDERING (SPECTRA PANEL + KINETIC FIT & RESIDUALS PANEL)
# ==============================================================================
def plot_construct_row(fig, ax_spec, ax_kin, ax_res, cfg):
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
    spectra_list = []

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
        spectra_list.append((t_s, y_corr))

    times_s = np.array(times_s, dtype=float)
    int_340 = np.array(int_340, dtype=float)

    # --- LEFT PANEL: Fluorescence Emission Spectra (300-400 nm) ---
    norm_t = Normalize(vmin=0.0, vmax=float(t_end))
    step = max(1, len(spectra_list) // SIGNAL_SETTINGS["max_spectra_curves"])
    wl_mask = (wavelengths >= SIGNAL_SETTINGS["spec_wl_min"]) & (wavelengths <= SIGNAL_SETTINGS["spec_wl_max"])

    max_spec_scaled = 0.0
    for idx_s, (t_s, spec_y) in enumerate(spectra_list):
        if np.any(wl_mask):
            max_spec_scaled = max(max_spec_scaled, float(np.nanmax(spec_y[wl_mask])) / 1e4)
        if idx_s % step == 0 or idx_s == len(spectra_list) - 1:
            ax_spec.plot(
                wavelengths[wl_mask],
                spec_y[wl_mask] / 1e4,
                color=plasma_trimmed(norm_t(t_s)),
                lw=1.25,
                alpha=0.88,
            )

    upper_y_spec = max(1.0, float(np.ceil(max_spec_scaled * 1.08)))
    ax_spec.axvline(340.0, color="#4D4D4D", ls=":", lw=1.1, alpha=0.75)
    ax_spec.set_xlim(SIGNAL_SETTINGS["spec_wl_min"], SIGNAL_SETTINGS["spec_wl_max"])
    ax_spec.set_ylim(0.0, upper_y_spec)
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel(r"Fluorescence Intensity ($\times 10^4$ a.u.)")
    ax_spec.grid(True, linestyle=":", alpha=0.5)
    ax_spec.set_title(
        f"{cfg['title_prefix']} – Fluorescence Spectra",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=8,
    )
    ax_spec.text(
        -0.14, 1.05, cfg["panel_spec"],
        transform=ax_spec.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top",
    )

    # Inset vertical colorbar for elapsed time
    cax = ax_spec.inset_axes([0.74, 0.46, 0.042, 0.44])
    sm = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_t)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.solids.set_rasterized(True)
    cbar.set_ticks(cfg["cbar_ticks"])
    cbar.set_ticklabels([str(v) for v in cfg["cbar_ticks"]])
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)
    cbar.set_label("Time (s)", fontsize=FS_CBAR_LABEL, labelpad=5)

    # --- RIGHT PANEL: 340 nm Kinetic Cross-Section + Fit + Residuals ---
    seed_row = load_table_seed(PATHS["tables_out_dir"], cfg["construct"], cfg["filename"])
    res = fit_sk_fl_trace(times_s, int_340, cfg["model"], seed_row=seed_row)

    color = cfg["color"]
    fit_color = darker(color, 0.65)

    if res["model"] == "double_exponential_with_drift":
        fit_label = (
            f"Double-exp + drift ($R^2 = {res['R2']:.4f}$)\n"
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
        -0.14, 1.07, cfg["panel_kin"],
        transform=ax_kin.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top",
    )

    leg = ax_kin.legend(
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
    ax_res.set_ylabel("Res.", fontsize=FS_RES_LABEL)
    ax_res.tick_params(axis="both", labelsize=FS_RES_TICK)
    ax_res.grid(True, linestyle=":", alpha=0.45)

    # Console summary
    if res["model"] == "double_exponential_with_drift":
        print(
            f"[{cfg['construct']} | {cfg['filename']}] "
            f"t1/2_1 = {res['t_half1']:.1f} ± {res['t_half1_err']:.1f} s, "
            f"t1/2_2 = {res['t_half2']:.1f} ± {res['t_half2_err']:.1f} s, "
            f"m = {res['m']:.3f} a.u./s, R2 = {res['R2']:.4f}"
        )
    else:
        print(
            f"[{cfg['construct']} | {cfg['filename']}] "
            f"t1/2 = {res['t_half1']:.1f} ± {res['t_half1_err']:.1f} s, "
            f"m = {res['m']:.3f} a.u./s, R2 = {res['R2']:.4f}"
        )


# ==============================================================================
# MAIN FIGURE BUILDER
# ==============================================================================
def make_figure_sk_fl_cn_system():
    os.makedirs(PATHS["fig_out_dir"], exist_ok=True)
    plt.rcParams.update(FIG_RC)

    fig = plt.figure(figsize=FIG_SIZE, dpi=300)
    outer_gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.0, 1.08],
        wspace=0.28,
        hspace=0.32,
    )

    for row_idx, cfg in enumerate(SK_FL_CONSTRUCTS):
        ax_spec = fig.add_subplot(outer_gs[row_idx, 0])

        right_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_gs[row_idx, 1],
            height_ratios=[3.2, 1.0],
            hspace=0.08,
        )
        ax_kin = fig.add_subplot(right_gs[0])
        ax_res = fig.add_subplot(right_gs[1], sharex=ax_kin)

        plot_construct_row(fig, ax_spec, ax_kin, ax_res, cfg)

    fig.subplots_adjust(left=0.085, right=0.975, top=0.96, bottom=0.055)

    out_pdf = os.path.join(PATHS["fig_out_dir"], f"{OUT_BASENAME}.pdf")
    out_png = os.path.join(PATHS["fig_out_dir"], f"{OUT_BASENAME}.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"\nSaved figure:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    make_figure_sk_fl_cn_system()