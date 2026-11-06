import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ==============================================================================
# 1. Configuration with Enhanced Smoothing
# ==============================================================================
EXPERIMENTS = [
    {
        "row_idx": 0,
        "label": "Tm1570 | Urea",
        "protein": "Tm1570",
        "denaturant": "Urea",
        "type": "bka_legacy",
        "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/Tm1570/equilibrium/CD/urea/tm_unfold_SAV",
        "conc_uM": 30.0,
        "path_mm": 0.1,
        "residues": 224,
        "model": "2-state",
        "target_wl": 217.0,
        "color": "#D95F02",
        "marker": "o",
        "linestyle": "--",
        "fill": "none",
        "smooth": False,
        "smooth_passes": 0,
        "smooth_window": 11,
        "smooth_poly": 3,
    },
    {
        "row_idx": 1,
        "label": "Tm1570 | GdmCl",
        "protein": "Tm1570",
        "denaturant": "GdmCl",
        "type": "bka_channels",
        "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/Tm1570/equilibrium/CD/GuCl/tm_unfold",
        "conc_uM": 25.0,
        "path_mm": 0.1,
        "residues": 224,
        "model": "2-state",
        "target_wl": 217.0,
        "color": "#FF7F0E",
        "marker": "o",
        "linestyle": "-",
        "fill": "full",
        "smooth": True,
        "smooth_passes": 1,
        "smooth_window": 21,
        "smooth_poly": 2,
    },
    {
        "row_idx": 2,
        "label": "TrmD–Tm1570 | Urea",
        "protein": "TrmD-Tm1570",
        "denaturant": "Urea",
        "type": "bka_legacy",
        "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/8b1n/equlibrium/CD/urea/fus_unfold_SAV",
        "conc_uM": 7.5,
        "path_mm": 0.1,
        "residues": 455,
        "model": "2-state",
        "target_wl": 217.0,
        "color": "#9E9AC8",
        "marker": "^",
        "linestyle": "--",
        "fill": "none",
        "smooth": True,
        "smooth_passes": 2,
        "smooth_window": 15,
        "smooth_poly": 3,
    },
    {
        "row_idx": 3,
        "label": "TrmD–Tm1570 | GdmCl",
        "protein": "TrmD-Tm1570",
        "denaturant": "GdmCl",
        "type": "prodata",
        "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/fuzja/3/2uM/28_07_26",
        "conc_file": "concentrations.txt",
        "conc_uM": 2.0,
        "path_mm": 1.0,
        "residues": 455,
        "model": "3-state",
        "target_wl": 217.0,
        "color": "#7570B3",
        "marker": "^",
        "linestyle": "-",
        "fill": "full",
        "smooth": True,
        "smooth_passes": 1,
        "smooth_window": 11,
        "smooth_poly": 3,
    },
]

# Ścieżki wyjściowe
OUTPUT_DIR = "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch04"
OUTPUT_FIGURE_PATH = os.path.join(OUTPUT_DIR, "Figura_4_1_CD_equilibrium_unfolding_summary_4x3.png")
OUTPUT_FIT_PATH = os.path.join("/home/matifortunka/PycharmProjects/various_programs/PhD/tables", "fit_fig4_1.txt")
CSV_PATH = "/home/matifortunka/PycharmProjects/various_programs/PhD/tables/thermo_fits_summary.csv"

RT = 0.592
HV_CUTOFF = 850

# Standardized X-axis parameters for Columns 1 and 3
COMMON_XLIM = (-0.3, 9.7)
COMMON_XTICKS = np.arange(0, 11, 2)


# ==============================================================================
# 2. Thermodynamic Models, K2D & Export
# ==============================================================================
def two_state_model(x, a_n, a_u, m, d):
    exp_term = np.exp(np.clip((m * (x - d)) / RT, -100, 100))
    return (a_n + a_u * exp_term) / (1.0 + exp_term)


def three_state_model(x, a_n, a_i, a_u, m1, d1, m2, d2):
    sig1 = 1.0 / (1.0 + np.exp(np.clip(-(m1 * (x - d1)) / RT, -100, 100)))
    sig2 = 1.0 / (1.0 + np.exp(np.clip(-(m2 * (x - d2)) / RT, -100, 100)))
    return a_n * (1.0 - sig1) + a_i * (sig1 - sig2) + a_u * sig2


def calc_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def export_fit_row(fig_id, construct, probe, time_cond, conc_uM, model_name,
                   cm_val, cm_se, m_val, m_se, r2_val):
    dG_val = (m_val * cm_val) if (m_val is not None and cm_val is not None) else None
    row = {
        "Figure": fig_id,
        "Construct": construct,
        "Probe": probe,
        "Condition_Time": str(time_cond),
        "Condition_Conc_uM": conc_uM,
        "Model": model_name,
        "Cm_M": round(cm_val, 3) if cm_val is not None else None,
        "Cm_SE": round(cm_se, 3) if cm_se is not None else None,
        "m_kcal_mol_M": round(m_val, 3) if m_val is not None else None,
        "m_SE": round(m_se, 3) if m_se is not None else None,
        "dG_H2O_kcal_mol": round(dG_val, 2) if dG_val is not None else None,
        "R_squared": round(r2_val, 4) if r2_val is not None else None
    }
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_row = pd.DataFrame([row])
    if not os.path.exists(CSV_PATH):
        df_row.to_csv(CSV_PATH, index=False)
    else:
        df_row.to_csv(CSV_PATH, mode="a", header=False, index=False)

    print(f"[{fig_id}] {construct} | {probe} ({model_name}): Cm={cm_val:.3f} ± {cm_se:.3f} M | m={m_val:.3f} ± {m_se:.3f} | R²={r2_val:.4f}")


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


def mdeg_to_mre(mdeg, conc_uM, path_mm, residues):
    return (mdeg * 1e6) / (conc_uM * path_mm * residues)


def mdeg_to_delta_epsilon(mdeg, conc_uM, path_mm, residues):
    mre = mdeg_to_mre(mdeg, conc_uM, path_mm, residues)
    return mre / 3298.0


def apply_custom_smoothing(y, cfg):
    if not cfg.get("smooth", False):
        return y

    passes = cfg.get("smooth_passes", 1)
    w = cfg.get("smooth_window", 11)
    p = cfg.get("smooth_poly", 3)

    y_proc = y.copy()
    for _ in range(passes):
        if len(y_proc) >= w:
            y_proc = savgol_filter(y_proc, w, p)
    return y_proc


def parse_bka_raw(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip().strip('"') == "_DATA") + 1
    except StopIteration:
        return None, None

    wls, vals = [], []
    for line in lines[start_idx:]:
        l_str = line.strip()
        if not l_str or not any(c.isdigit() for c in l_str):
            break
        parts = l_str.split()
        if len(parts) >= 2:
            try:
                wls.append(float(parts[0]))
                vals.append(float(parts[1]))
            except ValueError:
                continue
    return (np.array(wls), np.array(vals)) if wls else (None, None)


def parse_concentration_from_name(fname):
    name = os.path.splitext(fname)[0].lower()
    m_new = re.search(r"tm(\d+)(?:_ch\d+)?", name)
    if m_new:
        d = m_new.group(1)
        return float(d) if len(d) == 1 else float(f"{d[0]}.{d[1:]}")
    m_old = re.search(r"(\d+(?:\.\d+)?)\s*m", name)
    if m_old:
        return float(m_old.group(1))
    if any(k in name for k in ["native", "buffer", "tm0", "0m"]):
        return 0.0
    return None


def load_experiment_data(cfg):
    folder = cfg["path"]
    data_by_conc = {}

    if cfg["type"] == "prodata":
        conc_path = os.path.join(folder, cfg["conc_file"])
        with open(conc_path, "r", encoding="utf-8", errors="ignore") as f:
            mapping_text = f.read().replace(",", ".")
        map_df = pd.read_csv(StringIO(mapping_text), sep=r"\s+", engine="python")
        map_df["den_concentration"] = pd.to_numeric(map_df["den_concentration"], errors="coerce")
        c_map = map_df.set_index("Sample_number")["den_concentration"].to_dict()

        pattern = re.compile(r"(\d{5})\.csv$")
        for fname in sorted(os.listdir(folder)):
            m = pattern.search(fname)
            if not m:
                continue
            sample_num = int(m.group(1))
            conc = c_map.get(sample_num, None)
            if conc is None or np.isnan(conc):
                continue

            with open(os.path.join(folder, fname), "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            cd_idx = next((i for i in range(len(lines) - 1)
                           if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("CircularDichroism")), None)
            if cd_idx is None:
                continue

            cd_lines = []
            for ln in lines[cd_idx + 2:]:
                if not ln.strip() or ln.strip().startswith("Wavelength"):
                    break
                cd_lines.append(list(map(float, ln.replace(",", " ").split())))
            cd_arr = np.array(cd_lines)
            wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

            hv_idx = next((i for i in range(len(lines) - 1)
                           if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("HV")), None)
            if hv_idx is not None:
                hv_lines = []
                for ln in lines[hv_idx + 2:]:
                    if not ln.strip() or ln.strip().startswith("Wavelength"):
                        break
                    hv_lines.append(list(map(float, ln.replace(",", " ").split())))
                hv_arr = np.array(hv_lines)
                if len(hv_arr) == len(wls):
                    mask = hv_arr[:, 1] <= HV_CUTOFF
                    wls, mdeg = wls[mask], mdeg[mask]

            if len(wls) < 15:
                continue
            s_idx = np.argsort(wls)
            wls, mdeg = wls[s_idx], mdeg[s_idx]

            mdeg = apply_custom_smoothing(mdeg, cfg)
            mre = mdeg_to_mre(mdeg, cfg["conc_uM"], cfg["path_mm"], cfg["residues"])
            deps = mdeg_to_delta_epsilon(mdeg, cfg["conc_uM"], cfg["path_mm"], cfg["residues"])
            data_by_conc[float(conc)] = (wls, mre, deps)

    else:
        raw_grouped = {}
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".bka"):
                continue
            conc = parse_concentration_from_name(fname)
            if conc is None:
                continue
            wls, mdeg = parse_bka_raw(os.path.join(folder, fname))
            if wls is not None:
                raw_grouped.setdefault(conc, []).append((wls, mdeg))

        for conc, pairs in sorted(raw_grouped.items()):
            ref_wls = pairs[0][0]
            interp_mdeg = [np.interp(ref_wls, w, m) for w, m in pairs]
            avg_mdeg = np.mean(interp_mdeg, axis=0)

            avg_mdeg = apply_custom_smoothing(avg_mdeg, cfg)
            mre = mdeg_to_mre(avg_mdeg, cfg["conc_uM"], cfg["path_mm"], cfg["residues"])
            deps = mdeg_to_delta_epsilon(avg_mdeg, cfg["conc_uM"], cfg["path_mm"], cfg["residues"])
            data_by_conc[float(conc)] = (ref_wls, mre, deps)

    return data_by_conc


# ==============================================================================
# 3. Main Plotting Routine
# ==============================================================================
def generate_summary_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FIT_PATH), exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), constrained_layout=True, dpi=300)
    panel_letters = [
        ["A1", "A2", "A3"],
        ["B1", "B2", "B3"],
        ["C1", "C2", "C3"],
        ["D1", "D2", "D3"]
    ]

    fit_report_lines = [
        "================================================================================",
        "RAPORT DOPASOWANIA KRZYWYCH ROZWIJANIA CD (MODEL 2-STATE / 3-STATE)",
        f"Długość fali analitycznej: 217.0 nm | Stała RT: {RT} kcal/mol",
        "================================================================================\n"
    ]

    for row_idx, cfg in enumerate(EXPERIMENTS):
        ax_col1 = axes[row_idx, 0]
        ax_col2 = axes[row_idx, 1]
        ax_col3 = axes[row_idx, 2]

        print(f"Przetwarzanie wiersza {row_idx}: {cfg['label']} (Wygładzanie: {cfg['smooth']}, passes: {cfg.get('smooth_passes', 0)})...")
        data = load_experiment_data(cfg)
        concs = np.array(sorted(data.keys()))

        if len(concs) == 0:
            print(f"⚠️ Brak danych dla {cfg['label']}")
            continue

        # ----------------------------------------------------------------------
        # COLUMN 1: Transition Curves (Przeskalowanie * 10^-3)
        # ----------------------------------------------------------------------
        y_trans = [np.interp(cfg["target_wl"], data[c][0], data[c][1]) for c in concs]
        y_trans = np.array(y_trans)

        scatter_label = f"{cfg['protein']} ({cfg['denaturant']})"

        facecolor = cfg["color"] if cfg["fill"] == "full" else "none"
        ax_col1.scatter(concs, y_trans / 1000.0, color=cfg["color"], facecolors=facecolor,
                        marker=cfg["marker"], s=55, lw=1.6, zorder=3,
                        label=scatter_label)

        x_dense = np.linspace(concs.min(), concs.max(), 300)
        fit_success = False

        if cfg["model"] == "3-state":
            p0 = [y_trans[0], (y_trans[0] + y_trans[-1]) / 2, y_trans[-1], 2.0, 2.0, 2.0, 4.0]
            bounds = ([-np.inf, -np.inf, -np.inf, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
            try:
                popt, pcov = curve_fit(three_state_model, concs, y_trans, p0=p0, bounds=bounds, maxfev=15000)
                perr = np.sqrt(np.diag(pcov))
                y_fit = three_state_model(x_dense, *popt)
                r2_3s = calc_r_squared(y_trans, three_state_model(concs, *popt))

                fit_label = f"Fit 3-state ($R^2={r2_3s:.3f}$)"

                ax_col1.plot(x_dense, y_fit / 1000.0, color=cfg["color"], ls=cfg["linestyle"], lw=2.4,
                             label=fit_label, zorder=2)
                fit_success = True

                export_fit_row("Fig_4.1", cfg["protein"], f"CD_217nm_{cfg['denaturant']}_Step1",
                               cfg["denaturant"], cfg["conc_uM"], "3-state_trans1",
                               popt[4], perr[4], popt[3], perr[3], r2_3s)
                export_fit_row("Fig_4.1", cfg["protein"], f"CD_217nm_{cfg['denaturant']}_Step2",
                               cfg["denaturant"], cfg["conc_uM"], "3-state_trans2",
                               popt[6], perr[6], popt[5], perr[5], r2_3s)

                report = (
                    f"--- {cfg['label']} [3-State Model] ---\n"
                    f"a_n = {popt[0]:.2f} ± {perr[0]:.2f} deg*cm^2/dmol\n"
                    f"a_i = {popt[1]:.2f} ± {perr[1]:.2f} deg*cm^2/dmol\n"
                    f"a_u = {popt[2]:.2f} ± {perr[2]:.2f} deg*cm^2/dmol\n"
                    f"m1  = {popt[3]:.2f} ± {perr[3]:.2f} kcal/(mol*M)\n"
                    f"Cm1 = {popt[4]:.2f} ± {perr[4]:.2f} M\n"
                    f"dG1 = {popt[3]*popt[4]:.2f} kcal/mol\n"
                    f"m2  = {popt[5]:.2f} ± {perr[5]:.2f} kcal/(mol*M)\n"
                    f"Cm2 = {popt[6]:.2f} ± {perr[6]:.2f} M\n"
                    f"dG2 = {popt[5]*popt[6]:.2f} kcal/mol\n"
                    f"R^2 = {r2_3s:.4f}\n\n"
                )
                fit_report_lines.append(report)
            except Exception as e:
                print(f"3-state fit failed for {cfg['label']}: {e}")

        if not fit_success:
            p0 = [y_trans[0], y_trans[-1], 2.0, np.median(concs)]
            try:
                popt, pcov = curve_fit(two_state_model, concs, y_trans, p0=p0, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))
                y_fit = two_state_model(x_dense, *popt)
                r2_2s = calc_r_squared(y_trans, two_state_model(concs, *popt))

                fit_label = f"Fit 2-state ($R^2={r2_2s:.3f}$)"

                ax_col1.plot(x_dense, y_fit / 1000.0, color=cfg["color"], ls=cfg["linestyle"], lw=2.4,
                             label=fit_label, zorder=2)

                export_fit_row("Fig_4.1", cfg["protein"], f"CD_217nm_{cfg['denaturant']}",
                               cfg["denaturant"], cfg["conc_uM"], "2-state",
                               popt[3], perr[3], popt[2], perr[2], r2_2s)

                report = (
                    f"--- {cfg['label']} [2-State Model] ---\n"
                    f"a_n = {popt[0]:.2f} ± {perr[0]:.2f} deg*cm^2/dmol\n"
                    f"a_u = {popt[1]:.2f} ± {perr[1]:.2f} deg*cm^2/dmol\n"
                    f"m   = {popt[2]:.2f} ± {perr[2]:.2f} kcal/(mol*M)\n"
                    f"Cm  = {popt[3]:.2f} ± {perr[3]:.2f} M\n"
                    f"dG  = {popt[2]*popt[3]:.2f} kcal/mol\n"
                    f"R^2 = {r2_2s:.4f}\n\n"
                )
                fit_report_lines.append(report)
            except Exception as e:
                print(f"2-state fit failed for {cfg['label']}: {e}")

        # Skalowanie osi Y dla Col 1
        if row_idx in [0, 1]:
            ax_col1.set_ylim(-12.5, -2.75)
        else:
            ax_col1.set_ylim(-15.5, -1)

        ax_col1.set_xlim(COMMON_XLIM)
        ax_col1.set_xticks(COMMON_XTICKS)
        ax_col1.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=15)
        ax_col1.set_xlabel(f"[{cfg['denaturant']}] (M)", fontsize=15)

        # W panelach A1 (row 0) i C1 (row 2) symbole po lewej (markerfirst=True) i wyrównanie do lewej
        if row_idx in [0, 2]:
            leg_loc = "upper left"
            leg_bbox = (0.02, 0.87)
            m_first = True
        else:
            leg_loc = "lower right"
            leg_bbox = (1, 0.02)
            m_first = False

        leg1 = ax_col1.legend(
            loc=leg_loc,
            bbox_to_anchor=leg_bbox,
            fontsize=11.5,
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            edgecolor="none",
            markerfirst=m_first,
        )

        if row_idx in [0, 2]:
            for text_obj in leg1.get_texts():
                text_obj.set_ha("left")

        ax_col1.grid(True, linestyle=":", alpha=0.5)

        # ----------------------------------------------------------------------
        # COLUMN 2: Equilibrium CD Spectra (Przeskalowanie * 10^-3)
        # ----------------------------------------------------------------------
        norm = Normalize(vmin=0.0, vmax=max(concs.max(), 7.5))
        cmap = plt.cm.inferno

        for c in concs:
            wls, mre, _ = data[c]
            mask = (wls >= 200.0) & (wls <= 250.0)
            norm_val = 0.05 + 0.80 * norm(c)
            ax_col2.plot(wls[mask], mre[mask] / 1000.0, color=cmap(norm_val), lw=1.35)

        # Skalowanie osi Y dla Col 2
        if row_idx in [0, 1]:
            ax_col2.set_ylim(-12.5, 0.5)
        else:
            ax_col2.set_ylim(-15.5, 0.5)

        ax_col2.set_xlim(200.0, 250.0)
        ax_col2.set_xlabel("Wavelength (nm)", fontsize=15)
        ax_col2.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=15)
        ax_col2.grid(True, linestyle=":", alpha=0.5)

        # Proporcjonalnie powiększony colorbar z podziałkami w punktach 0, 2.5, 5, 7.5
        cax = ax_col2.inset_axes([0.74, 0.08, 0.048, 0.44])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.solids.set_rasterized(True)
        cbar.set_ticks([0, 2.5, 5, 7.5])
        cbar.set_ticklabels(["0", "2.5", "5", "7.5"])
        cbar.ax.tick_params(labelsize=10.5)
        cbar.set_label(f"[{cfg['denaturant']}] (M)", fontsize=11, labelpad=5)

        # ----------------------------------------------------------------------
        # COLUMN 3: K2D Secondary Structure
        # ----------------------------------------------------------------------
        alpha_list, beta_list, coil_list = [], [], []
        for c in concs:
            wls, _, deps = data[c]
            a, b, cl = solve_k2d(wls, deps)
            alpha_list.append(a)
            beta_list.append(b)
            coil_list.append(cl)

        ax_col3.plot(concs, alpha_list, label=r"$\alpha$-helix", color="#D62728", marker="o", lw=1.8, markersize=5)
        ax_col3.plot(concs, beta_list, label=r"$\beta$-strand", color="#1F78B4", marker="s", lw=1.8, markersize=5)
        ax_col3.plot(concs, coil_list, label="Random coil", color="#555555", marker="^", lw=1.8, markersize=5)

        ax_col3.set_xlim(COMMON_XLIM)
        ax_col3.set_xticks(COMMON_XTICKS)
        ax_col3.set_ylim(-0.05, 1.05)
        ax_col3.set_ylabel("Secondary Structure Fraction", fontsize=15)
        ax_col3.set_xlabel(f"[{cfg['denaturant']}] (M)", fontsize=15)
        ax_col3.grid(True, linestyle=":", alpha=0.5)

        # Przerzucenie legendy w A3 w prawy górny róg
        if row_idx == 0:
            ax_col3.legend(loc="upper right", fontsize=11.5, frameon=True, framealpha=0.85)

        # ----------------------------------------------------------------------
        # Panel Tags (A1-D3) & Row Titles
        # ----------------------------------------------------------------------
        for col_idx, ax in enumerate([ax_col1, ax_col2, ax_col3]):
            ax.tick_params(labelsize=13)
            ax.text(
                0.03,
                0.95,
                panel_letters[row_idx][col_idx],
                transform=ax.transAxes,
                fontsize=14.5,
                fontweight="bold",
                va="top",
                zorder=10,
            )

        if row_idx > 0:
            ax_col2.set_title(
                f"{cfg['label']} Equilibrium Unfolding",
                fontsize=13.5,
                fontweight="bold",
                pad=10,
            )

    # Column titles (including combined header for A2)
    axes[0, 0].set_title(r"$\bf{Transition\ Curve\ (217\ nm)}$", fontsize=13, pad=14)
    axes[0, 1].set_title(r"$\bf{Equilibrium\ CD\ Spectra}$" + "\n" + r"$\bf{Tm1570\ |\ Urea\ Equilibrium\ Unfolding}$", fontsize=13, pad=10)
    axes[0, 2].set_title(r"$\bf{K2D\ Secondary\ Structure}$", fontsize=13, pad=14)

    # Save outputs
    with open(OUTPUT_FIT_PATH, "w", encoding="utf-8") as f:
        f.writelines(fit_report_lines)
    print(f"[✓] Parametry dopasowań zapisano do: {OUTPUT_FIT_PATH}")
    print(f"[✓] Zbiorcze parametry dopasowań zaktualizowano w: {CSV_PATH}")

    fig.savefig(OUTPUT_FIGURE_PATH, dpi=300)
    pdf_path = os.path.splitext(OUTPUT_FIGURE_PATH)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf")
    print(f"[✓] Gotowa figura została zapisana do:\n  -> {OUTPUT_FIGURE_PATH}\n  -> {pdf_path}")


if __name__ == "__main__":
    generate_summary_figure()