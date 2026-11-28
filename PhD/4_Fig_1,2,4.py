import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ==============================================================================
# 0. Konfiguracja czcionek (zachowana z Fig4_2_dimer_model.py)
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"


# ==============================================================================
# 1. Konfiguracja Eksperymentów i Ścieżek Wyjściowych
# ==============================================================================
# Zasada doboru modeli:
#   - Izolowane domeny (CnTrmD, CnTm1570): 2-stanowy model dimeryczny (N2 <-> 2U)
#   - Fuzja CnTrmD-Tm1570 w GdmCl: 3-stanowy model monomeryczny (N <-> I <-> U)
#   - Fuzja CnTrmD-Tm1570 w moczniku: pozorny 2-stanowy model monomeryczny (brak rozdzielonego etapu pośredniego)

EXP_TRMD_GDMCL = {
    "label": "CnTrmD | GdmCl",
    "protein": "CnTrmD",
    "denaturant": "GdmCl",
    "time_cond": "12h",
    "type": "prodata",
    "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/trmd/3/2uM/28_07_26",
    "conc_file": "concentrations.txt",
    "conc_uM": 2.0,
    "path_mm": 1.0,
    "residues": 262,
    "hv_cutoff": 875,
    "model": "2-state-dimer",
    "target_wl": 217.0,
    "color": "#1F77B4",   # Classic Marine Blue (z Fig4_2_dimer_model.py)
    "marker": "o",
    "linestyle": "-",
    "fill": "full",
    "smooth": True,
    "smooth_passes": 1,
    "smooth_window": 11,
    "smooth_poly": 3,
}

EXPERIMENTS_TM1570 = [
    {
        "row_idx": 0,
        "label": "Tm1570 | Urea",
        "protein": "Tm1570",
        "denaturant": "Urea",
        "time_cond": "Urea",
        "type": "bka_legacy",
        "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/Tm1570/equilibrium/CD/urea/tm_unfold_SAV",
        "conc_uM": 30.0,
        "path_mm": 0.1,
        "residues": 224,
        "hv_cutoff": 850,
        "model": "2-state-dimer",
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
        "time_cond": "GdmCl",
        "type": "bka_channels",
        "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/Tm1570/equilibrium/CD/GuCl/tm_unfold",
        "conc_uM": 25.0,
        "path_mm": 0.1,
        "residues": 224,
        "hv_cutoff": 850,
        "model": "2-state-dimer",
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
]

EXPERIMENTS_FUSION = [
    {
        "row_idx": 0,
        "label": "TrmD–Tm1570 | Urea",
        "protein": "TrmD-Tm1570",
        "denaturant": "Urea",
        "time_cond": "Urea",
        "type": "bka_legacy",
        "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/8b1n/equlibrium/CD/urea/fus_unfold_SAV",
        "conc_uM": 7.5,
        "path_mm": 0.1,
        "residues": 455,
        "hv_cutoff": 850,
        "model": "2-state-dimer",
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
        "row_idx": 1,
        "label": "TrmD–Tm1570 | GdmCl",
        "protein": "TrmD-Tm1570",
        "denaturant": "GdmCl",
        "time_cond": "12h",
        "type": "prodata",
        "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/fuzja/3/2uM/28_07_26",
        "conc_file": "concentrations.txt",
        "conc_uM": 2.0,
        "path_mm": 1.0,
        "residues": 455,
        "hv_cutoff": 850,
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
TABLES_DIR = "/home/matifortunka/PycharmProjects/various_programs/PhD/tables"

OUT_FIG_4_1 = os.path.join(OUTPUT_DIR, "Figura_4_1_CnTrmD_GdmCl_1x2.png")
OUT_FIG_4_2 = os.path.join(OUTPUT_DIR, "Figura_4_2_CnTm1570_Urea_vs_GdmCl_2x3.png")
OUT_FIG_4_4 = os.path.join(OUTPUT_DIR, "Figura_4_4_CnTrmD_Tm1570_Urea_vs_GdmCl_2x3.png")

OUTPUT_FIT_PATH = os.path.join(TABLES_DIR, "fit_figs_4_1_4_2_4_4.txt")
CSV_PATH = os.path.join(TABLES_DIR, "thermo_fits_summary.csv")
CAPTIONS_TEX_PATH = os.path.join(OUTPUT_DIR, "captions_figs_4_1_4_2_4_4.tex")

RT = 0.592
DEFAULT_HV_CUTOFF = 850

# Standaryzowane osie X dla figur 4.2 i 4.4 (mocznik vs GdmCl)
COMMON_XLIM = (-0.3, 9.7)
COMMON_XTICKS = np.arange(0, 11, 2)

# Oś X dla Figury 4.1 (CnTrmD tylko w GdmCl)
TRMD_XLIM = (-0.2, 6.5)
TRMD_XTICKS = np.arange(0, 7, 1)


# ==============================================================================
# 2. Modele Termodynamiczne (Monomer & Dimer), K2D i Eksport
# ==============================================================================
def two_state_model(x, a_n, a_u, m, d):
    """Model 2-stanowy dla monomeru: N <-> U."""
    exp_term = np.exp(np.clip((m * (x - d)) / RT, -100, 100))
    return (a_n + a_u * exp_term) / (1.0 + exp_term)


def three_state_model(x, a_n, a_i, a_u, m1, d1, m2, d2):
    """Model 3-stanowy dla monomeru: N <-> I <-> U."""
    sig1 = 1.0 / (1.0 + np.exp(np.clip(-(m1 * (x - d1)) / RT, -100, 100)))
    sig2 = 1.0 / (1.0 + np.exp(np.clip(-(m2 * (x - d2)) / RT, -100, 100)))
    return a_n * (1.0 - sig1) + a_i * (sig1 - sig2) + a_u * sig2


def two_state_dimer_model(x, a_n, a_d, m, d, conc_uM):
    """
    Model 2-stanowy rozwijania z dysocjacją dla homodimeru: N2 <-> 2U (2D).
    P_t = conc_uM * 1e-6 [M] (całkowite stężenie w przeliczeniu na monomer).
    K_U(x) = [D]^2 / [N2] = P_t * exp(m * (x - d) / RT), gdzie d = Cm (f_D = 0.5).
    """
    P_t = conc_uM * 1e-6
    K_u = P_t * np.exp(np.clip((m * (x - d)) / RT, -500.0, 500.0))
    f_d = 2.0 / (1.0 + np.sqrt(1.0 + (8.0 * P_t) / K_u))
    f_n2 = 1.0 - f_d
    return a_n * f_n2 + a_d * f_d


def three_state_dimer_model(x, a_n, a_i, a_d, m1, d1, m2, d2, conc_uM):
    """
    Model 3-stanowy rozwijania dla homodimeru: N2 <-> I2 <-> 2D.
    Zachowany dla kompatybilności z poprzednimi skryptami.
    """
    P_t = conc_uM * 1e-6
    K1 = np.exp(np.clip((m1 * (x - d1)) / RT, -250.0, 250.0))
    K2 = P_t * np.exp(np.clip((m2 * (x - d2)) / RT, -250.0, 250.0))
    K_eff = (K1 * K2) / (1.0 + K1)
    f_d = 2.0 / (1.0 + np.sqrt(1.0 + (8.0 * P_t) / K_eff))
    f_n2 = (1.0 - f_d) / (1.0 + K1)
    f_i2 = (K1 * (1.0 - f_d)) / (1.0 + K1)
    return a_n * f_n2 + a_i * f_i2 + a_d * f_d


def calc_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def export_fit_row(fig_id, construct, probe, time_cond, conc_uM, model_name,
                   cm_val, cm_se, m_val, m_se, r2_val, is_dimer_dissociation=False):
    """Dopisuje pojedynczy wiersz parametrów dopasowania do zbiorczego pliku CSV."""
    if m_val is not None and cm_val is not None:
        if is_dimer_dissociation and conc_uM is not None:
            P_t = conc_uM * 1e-6
            dG_val = m_val * cm_val - RT * np.log(P_t)
        else:
            dG_val = m_val * cm_val
    else:
        dG_val = None

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
        "R_squared": round(r2_val, 4) if r2_val is not None else None,
    }
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_row = pd.DataFrame([row])
    if not os.path.exists(CSV_PATH):
        df_row.to_csv(CSV_PATH, index=False)
    else:
        df_row.to_csv(CSV_PATH, mode="a", header=False, index=False)

    print(
        f"[{fig_id}] {construct} | {probe} ({model_name}): "
        f"Cm={cm_val:.3f} ± {cm_se:.3f} M | m={m_val:.3f} ± {m_se:.3f} | "
        f"dG={dG_val:.2f} kcal/mol | R²={r2_val:.4f}"
    )


def solve_k2d(wavelengths, y_delta_eps):
    """Szacuje frakcje struktury drugorzędowej (alpha, beta, coil) z widma CD (Delta epsilon)."""
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


# ==============================================================================
# 3. Parsowanie Danych CD (ProData oraz BKA)
# ==============================================================================
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
    """Wczytuje widma CD (ProData CSV lub BKA) i zwraca słownik {stężenie: (wls, mre, deps)}."""
    folder = cfg["path"]
    hv_cutoff = cfg.get("hv_cutoff", DEFAULT_HV_CUTOFF)
    data_by_conc = {}

    if not os.path.isdir(folder):
        print(f"⚠️ Folder nie istnieje: {folder}")
        return data_by_conc

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

            cd_idx = next(
                (i for i in range(len(lines) - 1)
                 if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("CircularDichroism")),
                None,
            )
            if cd_idx is None:
                continue

            cd_lines = []
            for ln in lines[cd_idx + 2:]:
                if not ln.strip() or ln.strip().startswith("Wavelength"):
                    break
                cd_lines.append(list(map(float, ln.replace(",", " ").split())))
            cd_arr = np.array(cd_lines)
            wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

            hv_idx = next(
                (i for i in range(len(lines) - 1)
                 if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("HV")),
                None,
            )
            if hv_idx is not None:
                hv_lines = []
                for ln in lines[hv_idx + 2:]:
                    if not ln.strip() or ln.strip().startswith("Wavelength"):
                        break
                    hv_lines.append(list(map(float, ln.replace(",", " ").split())))
                hv_arr = np.array(hv_lines)
                if len(hv_arr) == len(wls):
                    mask = hv_arr[:, 1] <= hv_cutoff
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
# 4. Wspólne Funkcje Rysujące (zachowany styl, czcionki, kolory i legendy)
# ==============================================================================
def plot_transition_panel(ax, concs, data, cfg, fig_id, fit_report_lines,
                          xlim, xticks, ylim, leg_loc, leg_bbox, markerfirst,
                          align_legend_left=False):
    """Rysuje panel krzywej rozwijania (CD 217 nm) oraz wykonuje dopasowanie modelu."""
    y_trans = np.array([np.interp(cfg["target_wl"], data[c][0], data[c][1]) for c in concs])

    scatter_label = f"{cfg['protein']} ({cfg['denaturant']})"
    facecolor = cfg["color"] if cfg["fill"] == "full" else "none"

    ax.scatter(
        concs,
        y_trans / 1000.0,
        color=cfg["color"],
        facecolors=facecolor,
        marker=cfg["marker"],
        s=55,
        lw=1.6,
        zorder=3,
        label=scatter_label,
    )

    x_dense = np.linspace(concs.min(), concs.max(), 300)
    P_t = cfg["conc_uM"] * 1e-6
    fit_success = False

    # --- MODELE 3-STANOWE (MONOMER lub DIMER) ---
    if cfg["model"] in ("3-state", "3-state-dimer"):
        is_dimer = (cfg["model"] == "3-state-dimer")
        if is_dimer:
            fit_func_3s = lambda x, a_n, a_i, a_d, m1, d1, m2, d2: three_state_dimer_model(
                x, a_n, a_i, a_d, m1, d1, m2, d2, cfg["conc_uM"]
            )
        else:
            fit_func_3s = three_state_model

        p0 = [y_trans[0], (y_trans[0] + y_trans[-1]) / 2, y_trans[-1], 2.0, 2.0, 2.0, 4.0]
        bounds = (
            [-np.inf, -np.inf, -np.inf, 0, 0, 0, 0],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        )
        try:
            popt, pcov = curve_fit(fit_func_3s, concs, y_trans, p0=p0, bounds=bounds, maxfev=15000)
            perr = np.sqrt(np.diag(pcov))
            y_fit = fit_func_3s(x_dense, *popt)
            r2_3s = calc_r_squared(y_trans, fit_func_3s(concs, *popt))

            fit_label = f"Three-state fit" if is_dimer else f"Three-state fit"
            ax.plot(
                x_dense,
                y_fit / 1000.0,
                color=cfg["color"],
                ls=cfg["linestyle"],
                lw=2.4,
                label=fit_label,
                zorder=2,
            )
            fit_success = True

            dG1 = popt[3] * popt[4]
            dG2 = (popt[5] * popt[6] - RT * np.log(P_t)) if is_dimer else (popt[5] * popt[6])

            export_fit_row(
                fig_id, cfg["protein"], f"CD_217nm_{cfg['denaturant']}_Step1",
                cfg["time_cond"], cfg["conc_uM"], f"{cfg['model']}_trans1",
                popt[4], perr[4], popt[3], perr[3], r2_3s,
                is_dimer_dissociation=False,
            )
            export_fit_row(
                fig_id, cfg["protein"], f"CD_217nm_{cfg['denaturant']}_Step2",
                cfg["time_cond"], cfg["conc_uM"], f"{cfg['model']}_trans2",
                popt[6], perr[6], popt[5], perr[5], r2_3s,
                is_dimer_dissociation=is_dimer,
            )

            model_header = "3-State Dimer Model: N2 <-> I2 <-> 2D" if is_dimer else "3-State Monomer Model: N <-> I <-> U"
            report = (
                f"--- [{fig_id}] {cfg['label']} [{model_header} | P_t = {cfg['conc_uM']} uM] ---\n"
                f"a_n = {popt[0]:.2f} ± {perr[0]:.2f} deg*cm^2/dmol\n"
                f"a_i = {popt[1]:.2f} ± {perr[1]:.2f} deg*cm^2/dmol\n"
                f"a_u = {popt[2]:.2f} ± {perr[2]:.2f} deg*cm^2/dmol\n"
                f"m1  = {popt[3]:.2f} ± {perr[3]:.2f} kcal/(mol*M)\n"
                f"Cm1 = {popt[4]:.2f} ± {perr[4]:.2f} M\n"
                f"dG1 = {dG1:.2f} kcal/mol\n"
                f"m2  = {popt[5]:.2f} ± {perr[5]:.2f} kcal/(mol*M)\n"
                f"Cm2 = {popt[6]:.2f} ± {perr[6]:.2f} M\n"
                f"dG2 = {dG2:.2f} kcal/mol\n"
                f"R^2 = {r2_3s:.4f}\n\n"
            )
            fit_report_lines.append(report)
        except Exception as e:
            print(f"⚠️ {cfg['model']} fit failed for {cfg['label']}: {e}")

    # --- MODELE 2-STANOWE (MONOMER lub DIMER N2 <-> 2U) ---
    if not fit_success:
        is_dimer = (cfg["model"] == "2-state-dimer")
        if is_dimer:
            fit_func_2s = lambda x, a_n, a_d, m, d: two_state_dimer_model(
                x, a_n, a_d, m, d, cfg["conc_uM"]
            )
            model_tag = "2-state-dimer"
        else:
            fit_func_2s = two_state_model
            model_tag = "2-state"

        p0 = [y_trans[0], y_trans[-1], 2.0, np.median(concs)]
        bounds_2s = ([-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf])
        try:
            popt, pcov = curve_fit(fit_func_2s, concs, y_trans, p0=p0, bounds=bounds_2s, maxfev=15000)
            perr = np.sqrt(np.diag(pcov))
            y_fit = fit_func_2s(x_dense, *popt)
            r2_2s = calc_r_squared(y_trans, fit_func_2s(concs, *popt))

            fit_label = f"Two-state fit" if is_dimer else f"Two-state fit"
            ax.plot(
                x_dense,
                y_fit / 1000.0,
                color=cfg["color"],
                ls=cfg["linestyle"],
                lw=2.4,
                label=fit_label,
                zorder=2,
            )

            dG = (popt[2] * popt[3] - RT * np.log(P_t)) if is_dimer else (popt[2] * popt[3])

            export_fit_row(
                fig_id, cfg["protein"], f"CD_217nm_{cfg['denaturant']}",
                cfg["time_cond"], cfg["conc_uM"], model_tag,
                popt[3], perr[3], popt[2], perr[2], r2_2s,
                is_dimer_dissociation=is_dimer,
            )

            model_header = "2-State Dimer Model: N2 <-> 2U" if is_dimer else "2-State Monomer Model: N <-> U"
            report = (
                f"--- [{fig_id}] {cfg['label']} [{model_header} | P_t = {cfg['conc_uM']} uM] ---\n"
                f"a_n = {popt[0]:.2f} ± {perr[0]:.2f} deg*cm^2/dmol\n"
                f"a_d = {popt[1]:.2f} ± {perr[1]:.2f} deg*cm^2/dmol\n"
                f"m   = {popt[2]:.2f} ± {perr[2]:.2f} kcal/(mol*M)\n"
                f"Cm  = {popt[3]:.2f} ± {perr[3]:.2f} M\n"
                f"dG  = {dG:.2f} kcal/mol\n"
                f"R^2 = {r2_2s:.4f}\n\n"
            )
            fit_report_lines.append(report)
        except Exception as e:
            print(f"⚠️ {model_tag} fit failed for {cfg['label']}: {e}")

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=15)
    ax.set_xlabel(f"[{cfg['denaturant']}] (M)", fontsize=15)

    leg = ax.legend(
        loc=leg_loc,
        bbox_to_anchor=leg_bbox,
        fontsize=11.5,
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
        markerfirst=markerfirst,
    )
    if align_legend_left:
        for text_obj in leg.get_texts():
            text_obj.set_ha("left")

    ax.grid(True, linestyle=":", alpha=0.5)


def plot_spectra_panel(ax, fig, concs, data, cfg, ylim):
    """Rysuje panel widm CD z gradientem inferno oraz wbudowanym paskiem kolorów."""
    norm = Normalize(vmin=0.0, vmax=max(concs.max(), 7.5))
    cmap = plt.cm.inferno

    for c in concs:
        wls, mre, _ = data[c]
        mask = (wls >= 200.0) & (wls <= 250.0)
        norm_val = 0.05 + 0.80 * norm(c)
        ax.plot(wls[mask], mre[mask] / 1000.0, color=cmap(norm_val), lw=1.35)

    ax.set_ylim(ylim)
    ax.set_xlim(200.0, 250.0)
    ax.set_xlabel("Wavelength (nm)", fontsize=15)
    ax.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=15)
    ax.grid(True, linestyle=":", alpha=0.5)

    cax = ax.inset_axes([0.74, 0.08, 0.048, 0.44])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.solids.set_rasterized(True)
    cbar.set_ticks([0, 2.5, 5, 7.5])
    cbar.set_ticklabels(["0", "2.5", "5", "7.5"])
    cbar.ax.tick_params(labelsize=10.5)
    cbar.set_label(f"[{cfg['denaturant']}] (M)", fontsize=11, labelpad=5)


def plot_k2d_panel(ax, concs, data, cfg, xlim, xticks, show_legend=False, leg_loc = "upper right"):
    """Rysuje panel dekonwolucji struktury drugorzędowej K2D."""
    alpha_list, beta_list, coil_list = [], [], []
    for c in concs:
        wls, _, deps = data[c]
        a, b, cl = solve_k2d(wls, deps)
        alpha_list.append(a)
        beta_list.append(b)
        coil_list.append(cl)

    ax.plot(concs, alpha_list, label=r"$\alpha$-helix", color="#D62728", marker="o", lw=1.8, markersize=5)
    ax.plot(concs, beta_list, label=r"$\beta$-strand", color="#1F78B4", marker="s", lw=1.8, markersize=5)
    ax.plot(concs, coil_list, label="Random coil", color="#555555", marker="^", lw=1.8, markersize=5)

    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Secondary Structure Fraction", fontsize=15)
    ax.set_xlabel(f"[{cfg['denaturant']}] (M)", fontsize=15)
    ax.grid(True, linestyle=":", alpha=0.5)

    if show_legend:
        ax.legend(loc=leg_loc, fontsize=11.5, frameon=True, framealpha=0.85)


def save_figure(fig, png_path):
    """Zapisuje figurę w PNG (300 DPI) oraz wektorowym PDF."""
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print(f"[✓] Zapisano figurę:\n  -> {png_path}\n  -> {pdf_path}")


# ==============================================================================
# 5. Generatory Poszczególnych Figur (4.1, 4.2, 4.4)
# ==============================================================================
def generate_figure_4_1(fit_report_lines):
    """
    Nowa Figura 4.1 (1 x 2): izolowana domena CnTrmD w GdmCl (2 uM, 12 h).
      Panel A: Krzywa rozwijania CD przy 217 nm + fit 2-stanowy dimeryczny (N2 <-> 2U)
      Panel B: Dekonwolucja struktury drugorzędowej K2D
    """
    print("\n=== Generowanie Figury 4.1 (CnTrmD | GdmCl, 1x2) ===")
    cfg = EXP_TRMD_GDMCL
    data = load_experiment_data(cfg)
    concs = np.array(sorted(data.keys()))

    if len(concs) == 0:
        print(f"⚠️ Brak danych dla {cfg['label']} — pominięto Figurę 4.1.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True, dpi=300)
    ax_a, ax_b = axes[0], axes[1]

    # Panel A: Krzywa rozwijania (skala Y zgodna z Fig4_2_dimer_model.py: -14 do -1)
    plot_transition_panel(
        ax=ax_a,
        concs=concs,
        data=data,
        cfg=cfg,
        fig_id="Fig_4.1",
        fit_report_lines=fit_report_lines,
        xlim=TRMD_XLIM,
        xticks=TRMD_XTICKS,
        ylim=(-12.0, -1.0),
        leg_loc="lower right",
        leg_bbox=(0.98, 0.02),
        markerfirst=False,
        align_legend_left=False,
    )

    # Panel B: Analiza K2D
    plot_k2d_panel(
        ax=ax_b,
        concs=concs,
        data=data,
        cfg=cfg,
        xlim=TRMD_XLIM,
        xticks=TRMD_XTICKS,
        show_legend=True,
        leg_loc="center right"
    )

    for ax, tag in zip([ax_a, ax_b], ["A", "B"]):
        ax.tick_params(labelsize=13)
        ax.text(
            0.03,
            0.95,
            tag,
            transform=ax.transAxes,
            fontsize=14.5,
            fontweight="bold",
            va="top",
            zorder=10,
        )

    ax_a.set_title("Far-UV CD Unfolding Curve (217 nm)", fontsize=13, fontweight="bold", pad=10)
    ax_b.set_title(r"$\bf{K2D\ Secondary\ Structure}$", fontsize=13, pad=10)

    save_figure(fig, OUT_FIG_4_1)


def generate_two_row_comparison_figure(experiments, fig_id, out_png_path,
                                       col1_ylim, col2_ylim, fit_report_lines):
    """
    Wspólny generator figur 2 x 3 (mocznik vs GdmCl: krzywa 217 nm, widma CD, K2D)
    dla Figury 4.2 (CnTm1570) oraz Figury 4.4 (CnTrmD-Tm1570).
    """
    print(f"\n=== Generowanie {fig_id} (2x3) ===")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True, dpi=300)
    panel_letters = [
        ["A", "B", "C"],
        ["D", "E", "F"],
    ]

    for row_idx, cfg in enumerate(experiments):
        ax_col1, ax_col2, ax_col3 = axes[row_idx, 0], axes[row_idx, 1], axes[row_idx, 2]
        print(
            f"Przetwarzanie wiersza {row_idx}: {cfg['label']} "
            f"(Model: {cfg['model']}, c = {cfg['conc_uM']} uM, Wygładzanie: {cfg['smooth']})..."
        )

        data = load_experiment_data(cfg)
        concs = np.array(sorted(data.keys()))
        if len(concs) == 0:
            print(f"⚠️ Brak danych dla {cfg['label']}")
            continue

        # Kolumna 1: Krzywa rozwijania (wiersz 0 = mocznik -> legenda w lewym górnym rogu;
        # wiersz 1 = GdmCl -> legenda w prawym dolnym rogu, identycznie jak w Fig4_1)
        if row_idx == 0:
            leg_loc, leg_bbox, m_first, align_left = "upper left", (0.02, 0.87), True, True
        else:
            leg_loc, leg_bbox, m_first, align_left = "lower right", (1.0, 0.02), False, False

        plot_transition_panel(
            ax=ax_col1,
            concs=concs,
            data=data,
            cfg=cfg,
            fig_id=fig_id,
            fit_report_lines=fit_report_lines,
            xlim=COMMON_XLIM,
            xticks=COMMON_XTICKS,
            ylim=col1_ylim,
            leg_loc=leg_loc,
            leg_bbox=leg_bbox,
            markerfirst=m_first,
            align_legend_left=align_left,
        )

        # Kolumna 2: Widma CD
        plot_spectra_panel(ax_col2, fig, concs, data, cfg, ylim=col2_ylim)

        # Kolumna 3: K2D (legenda w prawym górnym rogu panelu C)
        plot_k2d_panel(
            ax=ax_col3,
            concs=concs,
            data=data,
            cfg=cfg,
            xlim=COMMON_XLIM,
            xticks=COMMON_XTICKS,
            show_legend=(row_idx == 0),
        )

        # Oznaczenia paneli A-F i tytuły wierszy
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

        ax_col2.set_title(
            f"Equilibrium Spectra in {cfg['denaturant']}",
            fontsize=13,
            fontweight="bold",
            pad=14 if row_idx == 0 else 10,
        )

    # Nagłówki kolumn
    axes[0, 0].set_title("Far-UV CD Unfolding Curve (217 nm)", fontsize=13, fontweight="bold", pad=14)
    axes[0, 2].set_title(r"$\bf{K2D\ Secondary\ Structure}$", fontsize=13, pad=14)

    save_figure(fig, out_png_path)


# ==============================================================================
# 6. Główna Funkcja Uruchomieniowa
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    fit_report_lines = [
        "================================================================================",
        "RAPORT DOPASOWANIA KRZYWYCH ROZWIJANIA CD (FIGURY 4.1, 4.2, 4.4)",
        "Domeny izolowane: 2-state dimer (N2 <-> 2U) | Fuzja: 2-state (mocznik) / 3-state (GdmCl)",
        f"Długość fali analitycznej: 217.0 nm | Stała RT: {RT} kcal/mol",
        "================================================================================\n",
    ]

    # 1. Nowa Figura 4.1: CnTrmD w GdmCl (1x2: krzywa + K2D)
    generate_figure_4_1(fit_report_lines)

    # 2. Figura 4.2: CnTm1570 mocznik vs GdmCl (2x3)
    generate_two_row_comparison_figure(
        experiments=EXPERIMENTS_TM1570,
        fig_id="Fig_4.2",
        out_png_path=OUT_FIG_4_2,
        col1_ylim=(-12.5, -2.75),
        col2_ylim=(-12.5, 0.5),
        fit_report_lines=fit_report_lines,
    )

    # 3. Figura 4.4: CnTrmD-Tm1570 mocznik vs GdmCl (2x3)
    generate_two_row_comparison_figure(
        experiments=EXPERIMENTS_FUSION,
        fig_id="Fig_4.4",
        out_png_path=OUT_FIG_4_4,
        col1_ylim=(-15.5, -1.0),
        col2_ylim=(-15.5, 0.5),
        fit_report_lines=fit_report_lines,
    )

    # Zapis raportu dopasowań i gotowych podpisów LaTeX
    with open(OUTPUT_FIT_PATH, "w", encoding="utf-8") as f:
        f.writelines(fit_report_lines)
    print(f"\n[✓] Raport dopasowań zapisano do: {OUTPUT_FIT_PATH}")
    print(f"[✓] Zbiorczy CSV zaktualizowano w: {CSV_PATH}")


if __name__ == "__main__":
    main()