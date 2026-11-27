import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ==============================================================================
# 1. Konfiguracja Matplotlib i Parametrów Doświadczenia (Nowy Układ: Fig. 4.9)
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"  # Pogrubiona kursywa (\mathsf)
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"             # Wycisza ostrzeżenie o 'cursive'
mpl.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

# Zasada doboru modeli:
#   - Modele 2-stanowe (CD ref., Fluo CSM, Fluo I310/I332): dimeryczne (N2 <-> 2U)
#   - Model 3-stanowy (CD 217 nm): monomeryczny (N <-> I <-> U)
FIGURE_ID = "Fig_4.9"
CONSTRUCT = "FsTrmD-Tm1570"
TIME_H = 22
CONC_UM = 1.0       # Stężenie białka [uM] (przeliczenie na monomer)
PATH_MM = 1.0       # Droga optyczna kuwety [mm]
N_RES = 447         # Liczba reszt dla FsTrmD-Tm1570 (UniProt: F8E4N3)
RT = 0.592          # kcal/mol przy 298.15 K
HV_CUTOFF = 850.0

# Zakres CSM/AEW dla fluorescencji
CSM_MIN_WL = 310.0
CSM_MAX_WL = 400.0

# Master Palette (spójna z pozostałymi figurami rozdziału 4)
COLOR_FS = "#17BECF"        # Teal / Stalowy cyjan
COLOR_FS_DARK = "#0E717B"   # Ciemny akcent / główne krzywe dopasowania
COLOR_2STATE = "#7F7F7F"    # Szary dla odniesienia 2-state dimer w Panelu A
COLOR_RATIO = "#7570B3"     # Fiolet dla panelu ratiometrycznego (I310/I332)

# Kolorystyka K2D zgodna z Figurami 4.1, 4.2 i 4.4
COLOR_ALPHA = "#D62728"     # K2D Alpha-helix (czerwony)
COLOR_BETA = "#1F78B4"      # K2D Beta-strand (błękitny)
COLOR_COIL = "#555555"      # K2D Random coil (grafitowy)

# Gradient plasma (obcięty 0.05 - 0.85 dla czytelności jasnych krzywych)
plasma_full = plt.cm.plasma
plasma_trimmed = LinearSegmentedColormap.from_list(
    "plasma_trimmed",
    [plasma_full(x) for x in np.linspace(0.05, 0.85, 256)]
)

# Ścieżki danych wejściowych
CD_DATA_DIR = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/CD/fuzja_F8_unfolding_22h"
FLUO_DATA_DIR = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/F8E4N/equilibrium/unfolding/2nd_set/fluo/spectra_F8_22h_26_04"

# Ścieżki wyjściowe (zaktualizowane do Figury 4.9)
OUTPUT_DIR = "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch04"
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Figura_4_9_FsTrmD_Tm1570_summary_2x3.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Figura_4_9_FsTrmD_Tm1570_summary_2x3.png")
CSV_PATH = "/home/matifortunka/PycharmProjects/various_programs/PhD/tables/thermo_fits_summary.csv"
CAPTIONS_TEX_PATH = os.path.join(OUTPUT_DIR, "caption_fig_4_9.tex")


# ==============================================================================
# 2. Równania Modeli Termodynamicznych, Dekonwolucja K2D i Eksport CSV
# ==============================================================================
def two_state_model(x, a_n, a_u, m, d):
    """Dwustanowy model równowagowy Santoro-Bolen dla monomeru: N <-> U."""
    exp_term = np.exp(np.clip((m * (x - d)) / RT, -60.0, 60.0))
    return (a_n + a_u * exp_term) / (1.0 + exp_term)


def three_state_model(x, a_n, a_i, a_u, m1, d1, m2, d2):
    """Trójstanowy model równowagowy dla monomeru: N <-> I <-> U."""
    sig1 = 1.0 / (1.0 + np.exp(np.clip(-(m1 * (x - d1)) / RT, -60.0, 60.0)))
    sig2 = 1.0 / (1.0 + np.exp(np.clip(-(m2 * (x - d2)) / RT, -60.0, 60.0)))
    return a_n * (1.0 - sig1) + a_i * (sig1 - sig2) + a_u * sig2


def two_state_dimer_model(x, a_n, a_d, m, d, conc_uM=CONC_UM):
    """
    Model dwustanowy rozwijania z dysocjacją dla homodimeru: N2 <-> 2U (2D).
    P_t = conc_uM * 1e-6 [M] (całkowite stężenie białka w przeliczeniu na monomer).
    K_U(x) = [D]^2 / [N2] = P_t * exp(m * (x - d) / RT), gdzie d = Cm (f_D = 0.5).
    """
    P_t = conc_uM * 1e-6
    K_u = P_t * np.exp(np.clip((m * (x - d)) / RT, -500.0, 500.0))
    f_d = 2.0 / (1.0 + np.sqrt(1.0 + (8.0 * P_t) / K_u))
    f_n2 = 1.0 - f_d
    return a_n * f_n2 + a_d * f_d


def calc_r_squared(y_true, y_pred):
    """Oblicza współczynnik determinacji R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def export_fit_row(fig_id, construct, probe, time_h, conc_uM, model_name,
                   cm_val, cm_se, m_val, m_se, r2_val,
                   is_dimer_dissociation=False, cov_m_cm=0.0):
    """
    Zapisuje wiersz parametrów dopasowania do zbiorczego pliku CSV
    oraz wyświetla w konsoli pełne parametry wraz z błędem propagowanym dG_H2O.
    """
    if m_val is not None and cm_val is not None:
        if is_dimer_dissociation and conc_uM is not None:
            P_t = conc_uM * 1e-6
            dG_val = m_val * cm_val - RT * np.log(P_t)
        else:
            dG_val = m_val * cm_val

        var_dG = (cm_val * m_se) ** 2 + (m_val * cm_se) ** 2 + 2.0 * cm_val * m_val * cov_m_cm
        dG_se = np.sqrt(max(var_dG, 0.0))
    else:
        dG_val = None
        dG_se = None

    row = {
        "Figure": fig_id,
        "Construct": construct,
        "Probe": probe,
        "Condition_Time": f"{time_h}h",
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

    print(
        f"[{fig_id}] {construct} | {probe} ({model_name}): "
        f"Cm={cm_val:.3f} ± {cm_se:.3f} M | m={m_val:.3f} ± {m_se:.3f} | "
        f"dG={dG_val:.2f} ± {dG_se:.2f} kcal/mol | R²={r2_val:.4f}"
    )


def estimate_k2d_secondary_structure(wls, mre):
    """
    Wyznacza frakcje struktury drugorzędowej (Alpha, Beta, Coil)
    z użyciem formalizmu K2D na bazie sygnału delta_epsilon przy 208 i 222 nm.
    Zwraca ułamki w skali 0.0 - 1.0 (spójne z Figurami 4.1, 4.2 i 4.4).
    """
    delta_eps_208 = np.interp(208.0, wls, mre) / 3298.0
    delta_eps_222 = np.interp(222.0, wls, mre) / 3298.0

    alpha = np.clip((-delta_eps_222 - 0.5) / 9.5, 0.0, 1.0)
    beta = np.clip((-0.45 * delta_eps_208 + 0.30 * delta_eps_222 + 1.2) / 6.0, 0.0, 1.0 - alpha)
    coil = np.clip(1.0 - (alpha + beta), 0.0, 1.0)

    total = alpha + beta + coil
    if total > 0:
        return alpha / total, beta / total, coil / total
    return 0.0, 0.3, 0.7


# ==============================================================================
# 3. Parsowanie concentrations.txt oraz Wczytywanie Danych CD i Fluo
# ==============================================================================
def mdeg_to_mre(mdeg, conc_uM=CONC_UM, path_mm=PATH_MM, residues=N_RES):
    """Przelicza eliptyczność [mdeg] na eliptyczność resztową [theta]_MRW."""
    return (mdeg * 1e6) / (conc_uM * path_mm * residues)


def load_concentrations(folder_path):
    """Wczytuje mapowanie Sample_number -> den_concentration z pliku concentrations.txt."""
    candidates = [
        os.path.join(folder_path, "concentrations.txt"),
        os.path.join(os.path.dirname(folder_path), "concentrations.txt"),
        os.path.join(os.path.dirname(os.path.dirname(folder_path)), "concentrations.txt")
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Nie znaleziono concentrations.txt dla {folder_path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace(",", ".")
    df = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")
    df["Sample_number"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df["den_concentration"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df.dropna(subset=["Sample_number", "den_concentration"], inplace=True)
    return dict(zip(df["Sample_number"].astype(int), df["den_concentration"].astype(float)))


def load_cd_prodata(folder_path):
    """Wczytuje widma CD (format ProData CSV), filtruje po HV i zwraca krzywą przy 217 nm."""
    conc_map = load_concentrations(folder_path)
    pattern = re.compile(r"(\d+)\.csv$", re.IGNORECASE)
    spectra_by_conc = {}

    for fname in sorted(os.listdir(folder_path)):
        m = pattern.search(fname)
        if not m:
            continue
        sn = int(m.group(1))
        conc = conc_map.get(sn, None)
        if conc is None or np.isnan(conc):
            continue

        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
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
            if not ln.strip() or ln.strip().startswith("Wavelength") or ln.strip().startswith("Available"):
                break
            cd_lines.append(list(map(float, ln.replace(",", " ").split())))

        cd_arr = np.array(cd_lines)
        wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

        # Filtracja HV
        hv_idx = next(
            (i for i in range(len(lines) - 1)
             if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("HV")),
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
                mask = hv_arr[:, 1] <= HV_CUTOFF
                wls, mdeg = wls[mask], mdeg[mask]

        if len(wls) < 15:
            continue

        s_idx = np.argsort(wls)
        wls, mdeg = wls[s_idx], mdeg[s_idx]

        if len(mdeg) >= 15:
            mdeg = savgol_filter(mdeg, 15, 3)

        mre = mdeg_to_mre(mdeg)
        spectra_by_conc[float(conc)] = (wls, mre)

    concs = np.array(sorted(spectra_by_conc.keys()))
    y_217 = np.array([np.interp(217.0, spectra_by_conc[c][0], spectra_by_conc[c][1]) for c in concs])

    return spectra_by_conc, concs, y_217


def load_fluo_data(folder_path):
    """
    Wczytuje widma emisji fluorescencji oraz wyznacza:
      1) stosunek intensywności I310 / I332 (bez odejmowania linii bazowej),
      2) średnią długość fali emisji <lambda> (CSM/AEW w zakresie 310-400 nm po korekcie na 400 nm).
    """
    conc_map = load_concentrations(folder_path)
    spectra_by_conc = {}
    csm_results = []
    ratio_results = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".csv"):
            continue
        matches = re.findall(r"(\d+)", fname)
        if not matches:
            continue
        sn = int(matches[-1])
        conc = conc_map.get(sn, None)
        if conc is None or np.isnan(conc):
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(fpath, header=1, usecols=[0, 1])
            df.columns = ["Wavelength", "Intensity"]
            df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors="coerce")
            df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
            df.dropna(inplace=True)
            if df.empty:
                continue

            step = df["Wavelength"].diff().median()
            w_pts = max(3, int(np.ceil(15.0 / step))) if step > 0 else 11
            w_pts = w_pts + 1 if w_pts % 2 == 0 else w_pts

            if len(df) >= w_pts:
                df["Intensity"] = savgol_filter(df["Intensity"], w_pts, 3)

            wls = df["Wavelength"].values
            intens_raw = df["Intensity"].values.copy()

            # 1. Ratio I310 / I332 liczone BEZ odejmowania baseline
            i310 = np.interp(310.0, wls, intens_raw)
            i332 = np.interp(332.0, wls, intens_raw)
            ratio_val = (i310 / i332) if i332 != 0 else np.nan
            ratio_results.append((float(conc), ratio_val))

            # 2. Korekta baseline na 400 nm dla widm i analizy CSM/AEW
            b_idx = (df["Wavelength"] - 400.0).abs().idxmin()
            intens_corr = intens_raw - intens_raw[b_idx]
            spectra_by_conc[float(conc)] = (wls, intens_corr)

            mask_csm = (wls >= CSM_MIN_WL) & (wls <= CSM_MAX_WL)
            wls_csm = wls[mask_csm]
            int_csm = np.maximum(intens_corr[mask_csm], 0.0)
            sum_int = np.sum(int_csm)
            csm_val = (np.sum(wls_csm * int_csm) / sum_int) if sum_int > 0 else np.nan
            csm_results.append((float(conc), csm_val))
        except Exception:
            continue

    csm_results.sort(key=lambda t: t[0])
    ratio_results.sort(key=lambda t: t[0])

    concs_csm = np.array([t[0] for t in csm_results if not np.isnan(t[1])])
    csm_vals = np.array([t[1] for t in csm_results if not np.isnan(t[1])])

    concs_ratio = np.array([t[0] for t in ratio_results if not np.isnan(t[1])])
    ratio_vals = np.array([t[1] for t in ratio_results if not np.isnan(t[1])])

    return spectra_by_conc, concs_csm, csm_vals, concs_ratio, ratio_vals


# ==============================================================================
# 4. Główny Generator Figury 4.9 (Siatka 2x3 w Nowym Układzie)
# ==============================================================================
def generate_figure_4_9():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    cd_spectra, concs_cd, y_217 = load_cd_prodata(CD_DATA_DIR)
    fluo_spectra, concs_csm, csm_vals, concs_ratio, ratio_vals = load_fluo_data(FLUO_DATA_DIR)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True, dpi=300)
    ax_a, ax_b, ax_c = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_d, ax_e, ax_f = axes[1, 0], axes[1, 1], axes[1, 2]

    bounds_2s_dimer = ([-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf])
    fit_func_2s_dimer = lambda x, a_n, a_d, m, d: two_state_dimer_model(x, a_n, a_d, m, d, CONC_UM)

    # --------------------------------------------------------------------------
    # PANEL A: Krzywa rozwijania CD przy 217 nm
    #   - Odniesienie (przerywana): 2-state dimer (N2 <-> 2U)
    #   - Dopasowanie główne (ciągła): 3-state monomer (N <-> I <-> U)
    # --------------------------------------------------------------------------
    ax_a.scatter(
        concs_cd, y_217 / 1000.0, color=COLOR_FS, marker="D", s=55,
        edgecolors="none", zorder=4, label=r"$\theta_{\mathrm{MRW}}$ (217 nm)"
    )

    x_dense_cd = np.linspace(concs_cd.min(), concs_cd.max(), 300)

    # 1) Fit 2-stanowy dimeryczny (N2 <-> 2U)
    p0_2s = [y_217[0], y_217[-1], 1.5, np.median(concs_cd)]
    popt_2s, pcov_2s = curve_fit(fit_func_2s_dimer, concs_cd, y_217, p0=p0_2s, bounds=bounds_2s_dimer, maxfev=15000)
    perr_2s = np.sqrt(np.diag(pcov_2s))
    r2_2s = calc_r_squared(y_217, fit_func_2s_dimer(concs_cd, *popt_2s))

    ax_a.plot(
        x_dense_cd, fit_func_2s_dimer(x_dense_cd, *popt_2s) / 1000.0,
        color=COLOR_2STATE, ls="--", dashes=(4.0, 2.5), lw=2.0,
        label=f"Two-state fit", zorder=2
    )
    export_fit_row(
        FIGURE_ID, CONSTRUCT, "CD_217nm", TIME_H, CONC_UM, "2-state-dimer",
        popt_2s[3], perr_2s[3], popt_2s[2], perr_2s[2], r2_2s,
        is_dimer_dissociation=True, cov_m_cm=pcov_2s[2, 3]
    )

    # 2) Fit 3-stanowy monomeryczny (N <-> I <-> U)
    p0_3s = [y_217[0], (y_217[0] + y_217[-1]) / 2.0, y_217[-1], 3.1, 2.3, 0.9, 3.7]
    bounds_3s = (
        [-np.inf, -np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    )
    popt_3s, pcov_3s = curve_fit(three_state_model, concs_cd, y_217, p0=p0_3s, bounds=bounds_3s, maxfev=20000)
    perr_3s = np.sqrt(np.diag(pcov_3s))
    r2_3s = calc_r_squared(y_217, three_state_model(concs_cd, *popt_3s))

    ax_a.plot(
        x_dense_cd, three_state_model(x_dense_cd, *popt_3s) / 1000.0,
        color=COLOR_FS_DARK, ls="-", lw=2.4,
        label=f"Three-state fit", zorder=3
    )
    export_fit_row(
        FIGURE_ID, CONSTRUCT, "CD_217nm_Step1", TIME_H, CONC_UM, "3-state_trans1",
        popt_3s[4], perr_3s[4], popt_3s[3], perr_3s[3], r2_3s,
        is_dimer_dissociation=False, cov_m_cm=pcov_3s[3, 4]
    )
    export_fit_row(
        FIGURE_ID, CONSTRUCT, "CD_217nm_Step2", TIME_H, CONC_UM, "3-state_trans2",
        popt_3s[6], perr_3s[6], popt_3s[5], perr_3s[5], r2_3s,
        is_dimer_dissociation=False, cov_m_cm=pcov_3s[5, 6]
    )

    ax_a.set_xlim(-0.2, 7.8)
    ax_a.set_xlabel("[GdmCl] (M)", fontsize=15)
    ax_a.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=15)
    ax_a.tick_params(labelsize=13)
    ax_a.grid(True, linestyle=":", alpha=0.5)

    leg_a = ax_a.legend(
        loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=11.0,
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="none",
        markerfirst=False, handlelength=2.6, handletextpad=0.8
    )
    for text_obj in leg_a.get_texts():
        text_obj.set_ha("right")

    ax_a.text(0.04, 0.95, "A", transform=ax_a.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax_a.set_title("Far-UV CD (217 nm)", fontsize=13, fontweight="bold", pad=8)

    # --------------------------------------------------------------------------
    # PANEL B: Średnia długość fali emisji <lambda> (CSM/AEW) - Model 2-state dimer
    # --------------------------------------------------------------------------
    ax_b.scatter(
        concs_csm, csm_vals, color=COLOR_FS, marker="D", s=55,
        edgecolors="none", zorder=4, label=r"$AEW \langle\lambda\rangle$ (310–400 nm)"
    )

    x_dense_csm = np.linspace(concs_csm.min(), concs_csm.max(), 300)
    p0_csm = [csm_vals[0], csm_vals[-1], 1.5, 2.0]
    popt_csm, pcov_csm = curve_fit(fit_func_2s_dimer, concs_csm, csm_vals, p0=p0_csm, bounds=bounds_2s_dimer, maxfev=15000)
    perr_csm = np.sqrt(np.diag(pcov_csm))
    r2_csm = calc_r_squared(csm_vals, fit_func_2s_dimer(concs_csm, *popt_csm))

    ax_b.plot(
        x_dense_csm, fit_func_2s_dimer(x_dense_csm, *popt_csm),
        color=COLOR_FS_DARK, ls="-", lw=2.4,
        label=f"Two-state fit", zorder=3
    )
    export_fit_row(
        FIGURE_ID, CONSTRUCT, "Fluo_CSM_310_400nm", TIME_H, CONC_UM, "2-state-dimer",
        popt_csm[3], perr_csm[3], popt_csm[2], perr_csm[2], r2_csm,
        is_dimer_dissociation=True, cov_m_cm=pcov_csm[2, 3]
    )

    ax_b.set_xlim(-0.2, 5.4)
    ax_b.set_xlabel("[GdmCl] (M)", fontsize=15)
    ax_b.set_ylabel(r"Avg. Emission Wavelength $\langle\lambda\rangle$ (nm)", fontsize=15)
    ax_b.tick_params(labelsize=13)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    leg_b = ax_b.legend(
        loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=11.0,
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="none",
        markerfirst=False, handlelength=2.6, handletextpad=0.8
    )
    for text_obj in leg_b.get_texts():
        text_obj.set_ha("right")

    ax_b.text(0.04, 0.95, "B", transform=ax_b.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax_b.set_title(r"Fluorescence AEW $\mathbf{\langle}\mathsf{\lambda}\mathbf{\rangle}$", fontsize=13, fontweight="bold", pad=8)

    # --------------------------------------------------------------------------
    # PANEL C: Stosunek intensywności I310 / I332 - Model 2-state dimer
    # --------------------------------------------------------------------------
    ax_c.scatter(
        concs_ratio, ratio_vals, color=COLOR_RATIO, marker="D", s=55,
        edgecolors="none", zorder=4, label=r"Ratio $I_{310}/I_{332}$"
    )

    x_dense_rat = np.linspace(concs_ratio.min(), concs_ratio.max(), 300)
    p0_rat = [ratio_vals[0], ratio_vals[-1], 1.5, 2.2]
    popt_rat, pcov_rat = curve_fit(fit_func_2s_dimer, concs_ratio, ratio_vals, p0=p0_rat, bounds=bounds_2s_dimer, maxfev=15000)
    perr_rat = np.sqrt(np.diag(pcov_rat))
    r2_rat = calc_r_squared(ratio_vals, fit_func_2s_dimer(concs_ratio, *popt_rat))

    ax_c.plot(
        x_dense_rat, fit_func_2s_dimer(x_dense_rat, *popt_rat),
        color=COLOR_RATIO, ls="-", lw=2.4,
        label=f"Two-state fit", zorder=3
    )
    export_fit_row(
        FIGURE_ID, CONSTRUCT, "Fluo_Ratio_I310_I332", TIME_H, CONC_UM, "2-state-dimer",
        popt_rat[3], perr_rat[3], popt_rat[2], perr_rat[2], r2_rat,
        is_dimer_dissociation=True, cov_m_cm=pcov_rat[2, 3]
    )

    ax_c.set_xlim(-0.2, 5.4)
    ax_c.set_xlabel("[GdmCl] (M)", fontsize=15)
    ax_c.set_ylabel(r"Emission Ratio $I_{310} / I_{332}$", fontsize=15)
    ax_c.tick_params(labelsize=13)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    leg_c = ax_c.legend(
        loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=11.0,
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="none",
        markerfirst=False, handlelength=2.6, handletextpad=0.8
    )
    for text_obj in leg_c.get_texts():
        text_obj.set_ha("right")

    ax_c.text(0.04, 0.95, "C", transform=ax_c.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax_c.set_title(
        r"Fluorescence Ratio ($\mathsf{I}_{\mathbf{310}}$/$\mathsf{I}_{\mathbf{332}}$)",
        fontsize=13, fontweight="bold", pad=8
    )

    # --------------------------------------------------------------------------
    # PANEL D: Widma CD (210 - 250 nm, gradient plasma)
    # --------------------------------------------------------------------------
    norm_cd = Normalize(vmin=0.0, vmax=max(concs_cd.max(), 7.5))
    for c in concs_cd:
        wls, mre = cd_spectra[c]
        mask = (wls >= 210.0) & (wls <= 250.0)
        ax_d.plot(wls[mask], mre[mask] / 1000.0, color=plasma_trimmed(norm_cd(c)), lw=1.35)

    ax_d.set_xlim(210.0, 250.0)
    ax_d.set_xlabel("Wavelength (nm)", fontsize=15)
    ax_d.set_ylabel(r"$[\theta]_{\mathrm{MRW}} \times 10^{-3}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=15)
    ax_d.tick_params(labelsize=13)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.text(0.04, 0.95, "D", transform=ax_d.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax_d.set_title("Far-UV CD Spectra", fontsize=13, fontweight="bold", pad=8)

    # Przesunięcie paska kolorów w lewo (x0=0.72), aby etykieta [GdmCl] (M) mieściła się wewnątrz ramki
    cax_d = ax_d.inset_axes([0.72, 0.08, 0.048, 0.44])
    sm_d = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_cd)
    sm_d.set_array([])
    cbar_d = fig.colorbar(sm_d, cax=cax_d, orientation="vertical")
    cbar_d.solids.set_rasterized(True)
    cbar_d.set_ticks([0, 2.5, 5, 7.5])
    cbar_d.set_ticklabels(["0", "2.5", "5", "7.5"])
    cbar_d.ax.tick_params(labelsize=10.5)
    cbar_d.set_label("[GdmCl] (M)", fontsize=11, labelpad=4)

    # --------------------------------------------------------------------------
    # PANEL E: Widma emisji fluorescencji (310 - 400 nm, gradient plasma)
    # --------------------------------------------------------------------------
    concs_fluo_sorted = np.array(sorted(fluo_spectra.keys()))
    norm_fluo = Normalize(vmin=0.0, vmax=max(concs_fluo_sorted.max(), 5.0))
    for c in concs_fluo_sorted:
        w_f, int_f = fluo_spectra[c]
        mask_f = (w_f >= 310.0) & (w_f <= 400.0)
        ax_e.plot(w_f[mask_f], int_f[mask_f], color=plasma_trimmed(norm_fluo(c)), lw=1.35)

    ax_e.set_xlim(310.0, 400.0)
    ax_e.set_xlabel("Wavelength (nm)", fontsize=15)
    ax_e.set_ylabel("Fluorescence Intensity (a.u.)", fontsize=15)
    ax_e.tick_params(labelsize=13)
    ax_e.grid(True, linestyle=":", alpha=0.5)
    ax_e.text(0.04, 0.95, "E", transform=ax_e.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax_e.set_title("Fluorescence Emission Spectra", fontsize=13, fontweight="bold", pad=8)

    # Przesunięcie paska kolorów w lewo (x0=0.73), aby etykieta [GdmCl] (M) nie nachodziła na prawą oś
    cax_e = ax_e.inset_axes([0.73, 0.50, 0.048, 0.44])
    sm_e = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_fluo)
    sm_e.set_array([])
    cbar_e = fig.colorbar(sm_e, cax=cax_e, orientation="vertical")
    cbar_e.solids.set_rasterized(True)
    cbar_e.set_ticks([0, 2.5, 5])
    cbar_e.set_ticklabels(["0", "2.5", "5"])
    cbar_e.ax.tick_params(labelsize=10.5)
    cbar_e.set_label("[GdmCl] (M)", fontsize=11, labelpad=4)

    # --------------------------------------------------------------------------
    # PANEL F: Dekonwolucja struktury drugorzędowej K2D
    # --------------------------------------------------------------------------
    alpha_list, beta_list, coil_list = [], [], []
    for c in concs_cd:
        wls, mre = cd_spectra[c]
        a, b, cl = estimate_k2d_secondary_structure(wls, mre)
        alpha_list.append(a)
        beta_list.append(b)
        coil_list.append(cl)

    ax_f.plot(concs_cd, alpha_list, label=r"$\alpha$-helix", color=COLOR_ALPHA, marker="o", lw=1.8, markersize=5)
    ax_f.plot(concs_cd, beta_list, label=r"$\beta$-strand", color=COLOR_BETA, marker="s", lw=1.8, markersize=5)
    ax_f.plot(concs_cd, coil_list, label="Random coil", color=COLOR_COIL, marker="^", lw=1.8, markersize=5)

    ax_f.set_xlim(-0.2, 7.8)
    ax_f.set_ylim(-0.05, 1.05)
    ax_f.set_xlabel("[GdmCl] (M)", fontsize=15)
    ax_f.set_ylabel("Secondary Structure Fraction", fontsize=15)
    ax_f.tick_params(labelsize=13)
    ax_f.grid(True, linestyle=":", alpha=0.5)
    ax_f.legend(
        loc="upper right", bbox_to_anchor=(0.99, 0.98), fontsize=11.5,
        frameon=True, facecolor="white", framealpha=0.85, edgecolor="none"
    )
    ax_f.text(0.04, 0.95, "F", transform=ax_f.transAxes, fontsize=14.5, fontweight="bold", va="top")
    ax_f.set_title("K2D Secondary Structure", fontsize=13, fontweight="bold", pad=8)

    # Zapis publikacyjnych plików PDF oraz PNG
    fig.savefig(OUTPUT_PDF, format="pdf")
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)

    print(f"\n[✓] Gotowa Figura 4.9 została zapisana do:\n  -> {OUTPUT_PNG}\n  -> {OUTPUT_PDF}")
    print(f"[✓] Zbiorczy CSV zaktualizowano w: {CSV_PATH}")


if __name__ == "__main__":
    generate_figure_4_9()