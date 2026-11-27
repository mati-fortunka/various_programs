import os
import glob
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==============================================================================
# KONFIGURACJA WIZUALNA (ZGODNA Z PANELAMI K2D Z FIG. 4.1)
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"

K2D_STYLE = {
    "alpha": {"label": r"$\alpha$-helix", "color": "#D62728", "marker": "o"},
    "beta":  {"label": r"$\beta$-strand", "color": "#1F78B4", "marker": "s"},
    "coil":  {"label": "Random coil",     "color": "#555555", "marker": "^"},
}

OUTPUT_DIR = "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch05"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "SK_CD_K2D_exploratory_2x3.png")

# ==============================================================================
# KONFIGURACJA DANYCH SK CD Z ROZDZIAŁU 5 (Fig. 5.2 i Fig. 5.3)
# ==============================================================================
COMMON_SETTINGS = {
    "nm_per_sec": 0.4,
    "sec_per_spectrum": 109.65,
    "time_mode": "Repeats",
    "hv_threshold": 950.0,
    "smoothing_window": 15,
    "smoothing_polyorder": 3,
    "baseline_correction": True,
    "baseline_wavelength": 250.0,
    "k2d_ref_wl": 215.0,  # środek pasma 208-222 nm do korekty czasu skanu
}

TM1570_OVERRIDES = {
    "Tm_30sp_12h00000":            {"max_time_s": 4500.0, "conc_uM": 5.0},
    "Tm1570_2h_spectra_2min00001": {"max_time_s": 3000.0, "conc_uM": 5.0},
    "Tm15_1200002_clean":          {"max_time_s": 4000.0, "conc_uM": 5.0},
    "Tm_1200003_clean":            {"max_time_s": 2000.0, "conc_uM": 8.0},
    "Tm15_12h0000000009":          {"max_time_s": None,   "conc_uM": 5.0},
    "Tm15_12h00000":               {"max_time_s": None,   "conc_uM": 5.0},
    "spectra_2h00000":             {"max_time_s": None,   "conc_uM": 4.0},
}

CONSTRUCTS = [
    {
        "name": r"$\mathsf{Cn}$Tm1570",
        "dir": "/home/matifortunka/Downloads/PhD/rozdział_V/Tm1570/CD/SK",
        "n_res": 224,
        "path_mm": 1.0,
        "dead_time": 30.0,
        "default_conc_uM": 5.0,
        "explicit_files": None,  # automatyczne wyszukanie *.csv jak w Fig5_2.py
    },
    {
        "name": r"$\mathsf{Cn}$TrmD–Tm1570",
        "dir": "/home/matifortunka/Downloads/PhD/rozdział_V/Cn_fusion/CD_SK",
        "n_res": 455,
        "path_mm": 1.0,
        "dead_time": 90.0,
        "default_conc_uM": 2.0,
        "explicit_files": [
            {
                "filename": "fusion_2h_spectra_2min00003.csv",
                "label": "Scan #1",
                "conc_uM": 2.0,
                "max_time_s": None,
                "remove_between": None,
                "transpose_data": False,
            },
            {
                "filename": "fuzja_12h_spectra_2min00000.csv",
                "label": "Scan #2",
                "conc_uM": 2.0,
                "max_time_s": 12000.0,
                "remove_between": (12000, 100000),
                "transpose_data": False,
            },
            {
                "filename": "spectra_2h00000.csv",
                "label": "Scan #3",
                "conc_uM": 2.0,
                "max_time_s": None,
                "remove_between": None,
                "transpose_data": True,
            },
        ],
    },
]

# ==============================================================================
# PRZELICZENIA JEDNOSTEK I ALGORYTM K2D (Z Fig4_1_urea_gucl.py)
# ==============================================================================
def mdeg_to_mre(mdeg, conc_uM, path_mm, residues):
    return (np.asarray(mdeg, dtype=float) * 1e6) / (conc_uM * path_mm * residues)


def mdeg_to_delta_epsilon(mdeg, conc_uM, path_mm, residues):
    return mdeg_to_mre(mdeg, conc_uM, path_mm, residues) / 3298.0


def solve_k2d(wavelengths, y_delta_eps):
    theta_208 = np.interp(208.0, wavelengths, y_delta_eps)
    theta_222 = np.interp(222.0, wavelengths, y_delta_eps)

    alpha = np.clip((-theta_222 - 0.5) / 9.5, 0.0, 1.0)
    beta = np.clip((-theta_208 * 0.45 + theta_222 * 0.3 + 1.2) / 6.0, 0.0, 1.0 - alpha)
    coil = np.clip(1.0 - (alpha + beta), 0.0, 1.0)

    s = alpha + beta + coil
    if s > 0:
        return alpha / s, beta / s, coil / s
    return 0.0, 0.0, 1.0


# ==============================================================================
# WCZYTYWANIE PRODATACSV (Z DETEKCJĄ TRANSPOZYCJI I FILTRACJĄ HV)
# ==============================================================================
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


def load_cd_sk_dataset(filepath, force_transpose=False, remove_between=None):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cd_df = pd.read_csv(StringIO("".join(extract_prodata_section(lines, "CircularDichroism"))), skipinitialspace=True)
    hv_df = pd.read_csv(StringIO("".join(extract_prodata_section(lines, "HV"))), skipinitialspace=True)

    cd_df.rename(columns={cd_df.columns[0]: "Wavelength"}, inplace=True)
    cd_df.dropna(axis=1, how="all", inplace=True)
    hv_df.rename(columns={hv_df.columns[0]: "Wavelength"}, inplace=True)
    hv_df.dropna(axis=1, how="all", inplace=True)

    row_axis = pd.to_numeric(cd_df.iloc[:, 0], errors="coerce").dropna()
    col_axis = pd.to_numeric(pd.Index(cd_df.columns[1:]), errors="coerce").dropna()
    rows_are_wl = (len(row_axis) > 5) and (row_axis.min() >= 170.0) and (row_axis.max() <= 320.0)
    cols_are_wl = (len(col_axis) > 5) and (col_axis.min() >= 170.0) and (col_axis.max() <= 320.0)

    if cols_are_wl and not rows_are_wl:
        need_transpose = True
    elif rows_are_wl and not cols_are_wl:
        need_transpose = False
    else:
        need_transpose = force_transpose

    if need_transpose:
        cd_times_raw = pd.to_numeric(cd_df.iloc[:, 0], errors="coerce")
        cd_wls = pd.to_numeric(cd_df.columns[1:], errors="coerce")
        cd_vals = cd_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
        cd_df = pd.DataFrame(cd_vals.T, index=cd_wls, columns=cd_times_raw).reset_index()
        cd_df.rename(columns={"index": "Wavelength"}, inplace=True)

        hv_times_raw = pd.to_numeric(hv_df.iloc[:, 0], errors="coerce")
        hv_wls = pd.to_numeric(hv_df.columns[1:], errors="coerce")
        hv_vals = hv_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
        hv_df = pd.DataFrame(hv_vals.T, index=hv_wls, columns=hv_times_raw).reset_index()
        hv_df.rename(columns={"index": "Wavelength"}, inplace=True)

    cd_df["Wavelength"] = pd.to_numeric(cd_df["Wavelength"], errors="coerce")
    hv_df["Wavelength"] = pd.to_numeric(hv_df["Wavelength"], errors="coerce")
    cd_df = cd_df.dropna(subset=["Wavelength"]).sort_values("Wavelength").reset_index(drop=True)
    hv_df = hv_df.dropna(subset=["Wavelength"]).sort_values("Wavelength").reset_index(drop=True)

    common_wl = np.intersect1d(cd_df["Wavelength"].values, hv_df["Wavelength"].values)
    cd_df = cd_df[cd_df["Wavelength"].isin(common_wl)].reset_index(drop=True)
    hv_df = hv_df[hv_df["Wavelength"].isin(common_wl)].reset_index(drop=True)
    wavelengths = cd_df["Wavelength"].values

    dt_spec = COMMON_SETTINGS["sec_per_spectrum"]
    if COMMON_SETTINGS["time_mode"] == "Repeats":
        old_cd_cols = [c for c in cd_df.columns if c != "Wavelength"]
        old_hv_cols = [c for c in hv_df.columns if c != "Wavelength"]
        cd_col_map = {i * dt_spec: col for i, col in enumerate(old_cd_cols)}
        hv_col_map = {i * dt_spec: col for i, col in enumerate(old_hv_cols)}
    else:
        cd_col_map = {float(col): col for col in cd_df.columns if col != "Wavelength"}
        hv_col_map = {float(col): col for col in hv_df.columns if col != "Wavelength"}

    if remove_between is not None and remove_between != (0, 0):
        t0, t1 = remove_between
        for t in [t for t in cd_col_map if t0 <= t <= t1]:
            cd_df.drop(columns=[cd_col_map[t]], inplace=True, errors="ignore")
            del cd_col_map[t]
        for t in [t for t in hv_col_map if t0 <= t <= t1]:
            hv_df.drop(columns=[hv_col_map[t]], inplace=True, errors="ignore")
            del hv_col_map[t]

    return wavelengths, cd_df, hv_df, cd_col_map, hv_col_map


def compute_k2d_trajectory(filepath, conc_uM, path_mm, n_res, dead_time,
                           max_time_s=None, remove_between=None, force_transpose=False):
    wavelengths, cd_df, hv_df, cd_col_map, hv_col_map = load_cd_sk_dataset(
        filepath, force_transpose=force_transpose, remove_between=remove_between
    )
    cd_times = sorted(cd_col_map.keys())
    hv_times = sorted(hv_col_map.keys())

    bl_wl = COMMON_SETTINGS["baseline_wavelength"]
    ref_wl = COMMON_SETTINGS["k2d_ref_wl"]
    hv_thresh = COMMON_SETTINGS["hv_threshold"]
    w_smooth = COMMON_SETTINGS["smoothing_window"]
    p_smooth = COMMON_SETTINGS["smoothing_polyorder"]

    time_shift = dead_time + (bl_wl - ref_wl) / COMMON_SETTINGS["nm_per_sec"]

    t_list, alpha_list, beta_list, coil_list, min_wls = [], [], [], [], []
    for cd_t in cd_times:
        t_eff = cd_t + time_shift
        if max_time_s is not None and t_eff > max_time_s:
            continue

        cd_col = cd_col_map[cd_t]
        hv_t = min(hv_times, key=lambda t: abs(t - cd_t))
        hv_col = hv_col_map[hv_t]

        cd_vals = cd_df[cd_col].values.astype(float)
        hv_vals = hv_df[hv_col].values.astype(float)

        mask = hv_vals <= hv_thresh
        x_wl = wavelengths[mask]
        y_mdeg = cd_vals[mask]
        if len(x_wl) < 10:
            continue

        if len(y_mdeg) >= w_smooth:
            y_mdeg = savgol_filter(y_mdeg, w_smooth, p_smooth)
        if COMMON_SETTINGS["baseline_correction"]:
            b_idx = int(np.argmin(np.abs(x_wl - bl_wl)))
            y_mdeg = y_mdeg - y_mdeg[b_idx]

        y_deps = mdeg_to_delta_epsilon(y_mdeg, conc_uM, path_mm, n_res)
        a, b, cl = solve_k2d(x_wl, y_deps)

        t_list.append(t_eff)
        alpha_list.append(a)
        beta_list.append(b)
        coil_list.append(cl)
        min_wls.append(float(x_wl.min()))

    return {
        "t": np.array(t_list),
        "alpha": np.array(alpha_list),
        "beta": np.array(beta_list),
        "coil": np.array(coil_list),
        "min_wl": np.median(min_wls) if min_wls else np.nan,
    }


# ==============================================================================
# GENEROWANIE WYKRESU PRZEGLĄDOWEGO (2 WIERSZE x 3 KOLUMNY)
# ==============================================================================
def run_exploratory_k2d():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True, dpi=300)
    panel_letters = [["A1", "A2", "A3"], ["B1", "B2", "B3"]]

    for row_idx, c_cfg in enumerate(CONSTRUCTS):
        if c_cfg["explicit_files"] is None:
            csv_paths = sorted(glob.glob(os.path.join(c_cfg["dir"], "*.csv")))
            scans = []
            for idx, fp in enumerate(csv_paths):
                stem = os.path.splitext(os.path.basename(fp))[0]
                ov = TM1570_OVERRIDES.get(stem, {})
                scans.append({
                    "filepath": fp,
                    "label": f"Scan #{idx + 1} ({stem})",
                    "short_label": f"Scan #{idx + 1}",
                    "conc_uM": ov.get("conc_uM", c_cfg["default_conc_uM"]),
                    "max_time_s": ov.get("max_time_s", None),
                    "remove_between": None,
                    "transpose_data": False,
                })
        else:
            scans = []
            for item in c_cfg["explicit_files"]:
                fp = os.path.join(c_cfg["dir"], item["filename"])
                stem = os.path.splitext(item["filename"])[0]
                scans.append({
                    "filepath": fp,
                    "label": f"{item['label']} ({stem})",
                    "short_label": item["label"],
                    "conc_uM": item["conc_uM"],
                    "max_time_s": item["max_time_s"],
                    "remove_between": item["remove_between"],
                    "transpose_data": item["transpose_data"],
                })

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(scans):
                ax.set_visible(False)
                continue

            sc = scans[col_idx]
            if not os.path.exists(sc["filepath"]):
                print(f"⚠️ Brak pliku: {sc['filepath']}")
                ax.set_title(f"{c_cfg['name']} | {sc['short_label']} (brak pliku)")
                continue

            res = compute_k2d_trajectory(
                sc["filepath"],
                conc_uM=sc["conc_uM"],
                path_mm=c_cfg["path_mm"],
                n_res=c_cfg["n_res"],
                dead_time=c_cfg["dead_time"],
                max_time_s=sc["max_time_s"],
                remove_between=sc["remove_between"],
                force_transpose=sc["transpose_data"],
            )

            print(
                f"[{c_cfg['name']}] {sc['label']}: n={len(res['t'])}, "
                f"t={res['t'][0]:.0f}–{res['t'][-1]:.0f} s, "
                f"min_wl(HV<=950)={res['min_wl']:.1f} nm, "
                f"alpha: {res['alpha'][0]:.2f}->{res['alpha'][-1]:.2f}"
            )

            for key in ["alpha", "beta", "coil"]:
                st = K2D_STYLE[key]
                ax.plot(
                    res["t"], res[key],
                    label=st["label"], color=st["color"], marker=st["marker"],
                    lw=1.8, markersize=4.5, alpha=0.9
                )

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Time (s)", fontsize=13)
            if col_idx == 0:
                ax.set_ylabel("Secondary Structure Fraction", fontsize=13)
            ax.tick_params(labelsize=11.5)
            ax.grid(True, linestyle=":", alpha=0.5)

            wl_note = f" (min {res['min_wl']:.0f} nm)" if np.isfinite(res["min_wl"]) else ""
            ax.set_title(
                f"{c_cfg['name']} — {sc['short_label']} ({sc['conc_uM']:g} µM){wl_note}",
                fontsize=12, fontweight="bold", pad=8
            )
            ax.text(
                0.03, 0.95, panel_letters[row_idx][col_idx],
                transform=ax.transAxes, fontsize=13.5, fontweight="bold", va="top"
            )

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="center right", fontsize=10.5, frameon=True, framealpha=0.88)

    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.show()
    print(f"✅ Zapisano podgląd K2D: {OUTPUT_FIG}")


if __name__ == "__main__":
    run_exploratory_k2d()