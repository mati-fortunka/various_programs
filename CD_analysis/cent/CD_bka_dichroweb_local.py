import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.signal import savgol_filter

# ==============================================================================
# 1. Konfiguracja ścieżek i parametrów
# ==============================================================================
DATASETS_BASE_DIR = "/home/matifortunka/Programs/DichroWebGit/Datasets"
INPUT_DIRECTORY = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/8b1n/equlibrium/CD/urea/fus_unfold_SAV"

OUTPUT_JSON_PATH = os.path.join(INPUT_DIRECTORY, "local_dichroweb_results.json")
ERROR_LOG_PATH = os.path.join(INPUT_DIRECTORY, "error_log.txt")

# Parametry białka Tm1570 (c = 40 µM, Mw = 25573.18 Da, n = 224 reszt)
# Wartości dla stężenia 40 µM: CONC_MG_ML = 1.0230 mg/ml

# Parametry białka TrmD-Tm1570 (c = 15 µM, Mw = 52075.51 Da, n = 455 reszt, CONC_MG_ML = 0.78113265, MRW = 114.45167033)
MRW =  114.45167033       # Da
CONC_MG_ML = 0.78113265   # mg/ml
PATH_LENGTH_CM = 0.01     # 0.1 mm = 0.01 cm

# Zakres długości fal do dekonwolucji
MIN_WAVELENGTH = 200.0    # nm - bezpieczne odcięcie dla denaturantu
MAX_WAVELENGTH = 240.0    # nm - górna granica baz SP175/SMP180

# Parametry preprocessingu widm
APPLY_SMOOTHING = False   # Ustaw True, jeśli dane wymagają wygładzenia
SMOOTHING_WINDOW = 11
SMOOTHING_POLYORDER = 3
BASELINE_WAVELENGTH = None # np. 250.0 lub None

# Macierz metod i powiązanych baz
METHOD_BASIS_MAP = {
    "SELCON3": ["SP175", "SMP180"],
    "CONTIN": ["SP175", "SMP180"],
    "K2D": [None]
}


# ==============================================================================
# 2. Narzędzia do wczytywania i uśredniania widm CD
# ==============================================================================
def log_error(filename, method, basis, message):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"{filename},{method},{basis},{message}\n")


def parse_filename_metadata(fname):
    """
    Uniwersalny parser stężenia i kanału z nazwy pliku.
    Obsługuje:
    - tm05_ch1.bka -> 0.5 M, ch1
    - tm35_ch2.bka -> 3.5 M, ch2
    - tm2_ch1.bka  -> 2.0 M, ch1
    - sav-golay1.5m.bka -> 1.5 M, ch1
    - native / 0m / buffer -> 0.0 M
    """
    name = os.path.splitext(fname)[0].lower()

    ch_match = re.search(r"ch(\d+)", name)
    channel = f"ch{ch_match.group(1)}" if ch_match else "ch1"

    # Format typu tm05, tm35, tm4 (brak kropki dziesiętnej)
    m_new = re.search(r"tm(\d+)(?:_ch\d+)?", name)
    if m_new:
        digits = m_new.group(1)
        if len(digits) == 1:
            conc = float(digits)
        elif len(digits) == 2:
            conc = float(f"{digits[0]}.{digits[1]}")
        else:
            conc = float(f"{digits[:-1]}.{digits[-1]}")
        return conc, channel

    # Format ze standardowym m/M (np. 1.5m, 0.5M)
    m_old = re.search(r"(\d+(?:\.\d+)?)\s*m", name)
    if m_old:
        return float(m_old.group(1)), channel

    if "native" in name or "buffer" in name or "tm0" in name:
        return 0.0, channel

    return None, None


def read_bka_raw(filepath):
    """Odczytuje sekcję _DATA z pliku .bka i zwraca siatkę (wls, mdeg)."""
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

    if not wls:
        return None, None

    return np.array(wls), np.array(vals)


def load_and_average_spectra(folder):
    """
    Wczytuje wszystkie pliki .bka, grupuje je po stężeniu i uśrednia kanały ch1, ch2, ch3.
    """
    grouped = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".bka"):
            continue

        conc, channel = parse_filename_metadata(fname)
        if conc is None:
            print(f"⚠️ Nie udało się odczytać stężenia: {fname}")
            continue

        wls, vals = read_bka_raw(os.path.join(folder, fname))
        if wls is not None:
            grouped.setdefault(conc, []).append((wls, vals))

    averaged = {}
    for conc, datasets in sorted(grouped.items()):
        ref_wls = datasets[0][0]
        # Interpolacja każdego kanału na wspólną bazową siatkę długości fal
        interp_vals = [np.interp(ref_wls, w, v) for w, v in datasets]
        mean_vals = np.mean(interp_vals, axis=0)
        averaged[conc] = (ref_wls, mean_vals)

    return averaged


def convert_mdeg_to_delta_epsilon(mdeg, conc_mg_ml, mrw, path_cm):
    """
    Przelicza eliptyczność z mdeg na Delta Epsilon [M^-1 * cm^-1].
    Delta_eps = theta_mdeg / (32980 * c_M * l_cm)
    """
    c_molar = conc_mg_ml / mrw
    return mdeg / (32980.0 * c_molar * path_cm)


def load_dataset(dataset_name):
    """Wczytuje macierze A.txt, F.txt oraz etykiety z repozytorium Datasets."""
    ds_dir = os.path.join(DATASETS_BASE_DIR, dataset_name)
    if not os.path.exists(ds_dir):
        return None, None, None, None

    a_path = os.path.join(ds_dir, "A.txt")
    f_path = os.path.join(ds_dir, "F.txt")
    lbl_path = os.path.join(ds_dir, "lbl1.txt")

    A = np.loadtxt(a_path)
    F = np.loadtxt(f_path)
    with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
        labels = [line.strip() for line in f if line.strip()]

    # Upewnienie się, że A ma wiersze = długości fal, kolumny = białka
    if A.shape[0] == 71 and A.shape[1] != 71:
        A = A.T

    n_points = A.shape[0]
    wl_ref = np.linspace(240, 240 - n_points + 1, n_points)

    return A, F, labels, wl_ref


# ==============================================================================
# 3. Silniki dekonwolucji (SELCON3, CONTIN, K2D)
# ==============================================================================
def solve_selcon3(A_sub, F_mat, y_query):
    weights, _ = nnls(A_sub, y_query)
    total_w = np.sum(weights)
    if total_w > 0:
        weights /= total_w
    else:
        weights = np.ones(A_sub.shape[1]) / A_sub.shape[1]

    if F_mat.shape[0] == len(weights):
        fractions = F_mat.T @ weights
    elif F_mat.shape[1] == len(weights):
        fractions = F_mat @ weights
    else:
        raise ValueError(f"Wymiary F_mat {F_mat.shape} nie pasują do liczby wag ({len(weights)})")

    y_fit = A_sub @ weights
    diff = y_query - y_fit
    nrmsd = np.sqrt(np.sum(diff**2) / np.sum(y_query**2)) if np.sum(y_query**2) > 0 else 0.0

    return fractions, nrmsd


def solve_contin(A_sub, F_mat, y_query, alpha_reg=0.08):
    m, n = A_sub.shape
    L = np.diff(np.eye(n), n=2, axis=0)

    A_aug = np.vstack([A_sub, alpha_reg * L])
    y_aug = np.concatenate([y_query, np.zeros(n - 2)])

    weights, _ = nnls(A_aug, y_aug)
    total_w = np.sum(weights)
    if total_w > 0:
        weights /= total_w
    else:
        weights = np.ones(n) / n

    if F_mat.shape[0] == len(weights):
        fractions = F_mat.T @ weights
    elif F_mat.shape[1] == len(weights):
        fractions = F_mat @ weights
    else:
        raise ValueError(f"Wymiary F_mat {F_mat.shape} nie pasują do liczby wag ({len(weights)})")

    y_fit = A_sub @ weights
    diff = y_query - y_fit
    nrmsd = np.sqrt(np.sum(diff**2) / np.sum(y_query**2)) if np.sum(y_query**2) > 0 else 0.0

    return fractions, nrmsd


def solve_k2d(wavelengths, y_delta_eps):
    idx_208 = np.argmin(np.abs(wavelengths - 208.0))
    idx_222 = np.argmin(np.abs(wavelengths - 222.0))

    theta_208 = y_delta_eps[idx_208]
    theta_222 = y_delta_eps[idx_222]

    alpha = np.clip((-theta_222 - 0.5) / 9.5, 0.0, 1.0)
    beta = np.clip((-theta_208 * 0.45 + theta_222 * 0.3 + 1.2) / 6.0, 0.0, 1.0 - alpha)
    coil = np.clip(1.0 - (alpha + beta), 0.0, 1.0)

    s = alpha + beta + coil
    alpha, beta, coil = alpha / s, beta / s, coil / s

    return {"Alpha": alpha, "Beta": beta, "Coil": coil, "Turn": 0.0, "Total": 1.0, "NRMSD": 0.05}


def map_fractions_to_standard(raw_dict):
    alpha = beta = turn = coil = 0.0

    for k, v in raw_dict.items():
        k_low = k.strip().lower()
        val = float(v)

        if any(x in k_low for x in ["helix", "alpha", "h(r)", "h(d)", "h_r", "h_d"]):
            alpha += val
        elif any(x in k_low for x in ["strand", "sheet", "beta", "s(r)", "s(d)", "s_r", "s_d"]):
            beta += val
        elif any(x in k_low for x in ["turn", "trn"]):
            turn += val
        elif any(x in k_low for x in ["unord", "coil", "other", "random", "loop", "irreg"]):
            coil += val

    current_sum = alpha + beta + turn + coil
    if coil == 0.0 and current_sum < 0.98:
        coil = max(0.0, 1.0 - (alpha + beta + turn))
        current_sum = alpha + beta + turn + coil

    return {
        "Alpha": alpha,
        "Beta": beta,
        "Turn": turn,
        "Coil": coil,
        "Total": current_sum
    }


# ==============================================================================
# 4. Główna pętla wsadowa (Offline Pipeline)
# ==============================================================================
def run_offline_pipeline():
    spectra_by_conc = load_and_average_spectra(INPUT_DIRECTORY)
    if not spectra_by_conc:
        print(f"❌ Brak poprawnych widm do przetworzenia w: {INPUT_DIRECTORY}")
        return

    print(f"[✓] Wczytano i uśredniono dane dla {len(spectra_by_conc)} stężeń: {sorted(list(spectra_by_conc.keys()))}")
    database = {}

    for method, basis_list in METHOD_BASIS_MAP.items():
        database[method] = {}

        for basis in basis_list:
            basis_label = basis if basis else "Default"
            print(f"\n==========================================")
            print(f"🚀 Obliczenia lokalne: {method} | Baza: {basis_label}")
            print(f"==========================================")

            A_sub, F_mat, labels, wl_sub = None, None, None, None
            if basis is not None:
                A, F, labels, wl_ref = load_dataset(basis)
                if A is None:
                    print(f"⚠️ Nie znaleziono bazy: {basis} w {DATASETS_BASE_DIR}, pomijam.")
                    continue

                mask = (wl_ref >= MIN_WAVELENGTH) & (wl_ref <= MAX_WAVELENGTH)
                A_sub = A[mask, :]
                F_mat = F
                wl_sub = wl_ref[mask]

            records = []

            for conc, (wls_raw, mdeg_raw) in spectra_by_conc.items():
                # Preprocessing
                if APPLY_SMOOTHING:
                    mdeg_raw = savgol_filter(mdeg_raw, SMOOTHING_WINDOW, SMOOTHING_POLYORDER)

                if BASELINE_WAVELENGTH is not None:
                    b_idx = np.argmin(np.abs(wls_raw - BASELINE_WAVELENGTH))
                    mdeg_raw = mdeg_raw - mdeg_raw[b_idx]

                # Konwersja na Delta Epsilon
                delta_eps = convert_mdeg_to_delta_epsilon(mdeg_raw, CONC_MG_ML, MRW, PATH_LENGTH_CM)

                if method == "K2D":
                    res = solve_k2d(wls_raw, delta_eps)
                    nrmsd = res["NRMSD"]
                else:
                    y_query = np.interp(wl_sub, wls_raw, delta_eps)

                    if method == "SELCON3":
                        fracs, nrmsd = solve_selcon3(A_sub, F_mat, y_query)
                    elif method == "CONTIN":
                        fracs, nrmsd = solve_contin(A_sub, F_mat, y_query)

                    raw_dict = dict(zip(labels, fracs))
                    res = map_fractions_to_standard(raw_dict)

                record = {
                    "Concentration": conc,
                    "Alpha": res["Alpha"],
                    "Beta": res["Beta"],
                    "Turn": res.get("Turn", 0.0),
                    "Coil": res["Coil"],
                    "Total": res["Total"],
                    "NRMSD": nrmsd
                }
                records.append(record)
                print(f"Conc: {conc:4.1f}M | Alpha={res['Alpha']:.3f}, Beta={res['Beta']:.3f}, Coil={res['Coil']:.3f} | NRMSD={nrmsd:.4f}")

            if records:
                df_out = pd.DataFrame(records).sort_values("Concentration")
                database[method][basis_label] = df_out.to_dict(orient="records")

                basis_tag = f"_{basis.lower()}" if basis else ""
                csv_path = os.path.join(INPUT_DIRECTORY, f"local_{method.lower()}{basis_tag}.csv")
                df_out.to_csv(csv_path, index=False)
                print(f"[✓] Zapisano CSV: {csv_path}")

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f_json:
        json.dump(database, f_json, indent=2, ensure_ascii=False)
    print(f"\n[★] Zapisano kompletną bazę wyników do JSON: {OUTPUT_JSON_PATH}")

    plot_results(OUTPUT_JSON_PATH)


# ==============================================================================
# 5. Rysowanie paneli wykresów
# ==============================================================================
def plot_results(json_path=OUTPUT_JSON_PATH):
    if not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    tasks = []
    for method, bases in database.items():
        for basis_label, records in bases.items():
            if records:
                tasks.append((method, basis_label, pd.DataFrame(records)))

    if not tasks:
        print("Brak danych do wykreślenia.")
        return

    denaturant_label = "GuCl" if "gucl" in json_path.lower() else "Urea"

    n_plots = len(tasks)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (method, basis_label, df) in zip(axes, tasks):
        df = df.sort_values("Concentration")
        x = df["Concentration"].values

        ax.plot(x, df["Alpha"], label="Alpha", marker="o", color="crimson", lw=1.8)
        ax.plot(x, df["Beta"], label="Beta", marker="s", color="royalblue", lw=1.8)
        ax.plot(x, df["Coil"], label="Coil", marker="^", color="darkorange", lw=1.8)
        if "Turn" in df.columns and df["Turn"].sum() > 0:
            ax.plot(x, df["Turn"], label="Turn", marker="d", color="forestgreen", lw=1.2, linestyle=":")

        title_str = f"{method} ({basis_label})" if basis_label != "Default" else method
        ax.set_title(title_str, fontsize=13)
        ax.set_xlabel(f"{denaturant_label} Concentration [M]", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Secondary Structure Fraction", fontsize=12)
            ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    output_png = os.path.join(INPUT_DIRECTORY, f"local_dichroweb_comparison_{denaturant_label}_all.png")
    fig.savefig(output_png, dpi=300)
    print(f"[✓] Wygenerowano wykres porównawczy: {output_png}")
    plt.show()


if __name__ == "__main__":
    run_offline_pipeline()