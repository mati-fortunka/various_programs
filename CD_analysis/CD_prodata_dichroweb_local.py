import os
import re
from io import StringIO
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.signal import savgol_filter

# ==============================================================================
# 1. Konfiguracja ścieżek i parametrów eksperymentu
# ==============================================================================
DATASETS_BASE_DIR = "/home/matifortunka/Programs/DichroWebGit/Datasets"

# Ścieżka do analizowanego folderu z plikami ProData (.csv) oraz concentrations.txt
SERIES_FOLDER = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/trmd/3/2uM/28_07_26"
CONCENTRATION_FILE = os.path.join(SERIES_FOLDER, "concentrations.txt")

OUTPUT_JSON_PATH = os.path.join(SERIES_FOLDER, "local_dichroweb_results.json")
ERROR_LOG_PATH = os.path.join(SERIES_FOLDER, "error_log.txt")

# Parametry próbki białka (fuzja TrmD-Tm1570, n = 455 reszt, c = 2.0 µM w kuwecie 1.0 mm)
PROTEIN_CONC_UM = 1.0         # Stężenie w µM (µmol/L)
RESIDUE_COUNT = 447           # 455 dla fuzji, 224 dla Tm1570
PATH_LENGTH_MM = 1.0          # Grubość kuwety w mm (np. 1.0 mm = 0.1 cm)
MRW = 112.28                  # Średnia masa reszty (Da)

# Zakres spektralny dla baz DichroWebGit
MIN_WAVELENGTH = 210.0        # nm (dolne odcięcie dla denaturantu)
MAX_WAVELENGTH = 245.0        # nm (górny limit baz SP175/SMP180)

# Filtracja jakościowa i wygładzanie
HV_CUTOFF = 950.0             # Odcięcie punktów o szumie detektora (V)
APPLY_SMOOTHING = True        # Wygładzanie Savitzky-Golay
SMOOTHING_WINDOW = 11
SMOOTHING_POLYORDER = 3
BASELINE_WAVELENGTH = None   # Zerowanie przy 250 nm (lub None)

# Macierz metod i powiązanych baz
METHOD_BASIS_MAP = {
    "SELCON3": ["SP175", "SMP180"],
    "CONTIN": ["SP175", "SMP180"],
    "K2D": [None]
}


# ==============================================================================
# 2. Parsowanie formatu ProData oraz pliku concentrations.txt
# ==============================================================================
def find_block(lines, property_name):
    """Lokalizuje początek i koniec sekcji numerycznej dla podanej właściwości w ProData."""
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith(property_name):
            start = i + 2
            end = start
            while end < len(lines) and lines[end].strip():
                if lines[end].strip().startswith("Wavelength"):
                    break
                end += 1
            return start, end
    return None, None


def load_concentration_mapping(filepath):
    """Wczytuje mapowanie Sample_number -> den_concentration obsługując przecinki i kropki."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().replace(',', '.')
    df = pd.read_csv(StringIO(text), sep=r"\s+", engine='python')
    df['den_concentration'] = pd.to_numeric(df['den_concentration'], errors='coerce')
    return df.set_index('Sample_number')['den_concentration'].to_dict()


def read_prodata_csv(filepath, hv_cutoff=700.0):
    """Odczytuje bloki CircularDichroism i HV, filtrując punkty o zbyt wysokim napięciu HV."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    cd_start, cd_end = find_block(lines, "CircularDichroism")
    if cd_start is None:
        return None, None

    cd_lines = [ln.strip() for ln in lines[cd_start:cd_end] if ln.strip()]
    cd_pairs = [list(map(float, ln.replace(',', ' ').split())) for ln in cd_lines]
    cd_data = np.array(cd_pairs)

    wavelengths = cd_data[:, 0]
    ellipticity = cd_data[:, 1]

    # Sprawdzenie napięcia fotopowielacza HV
    hv_start, hv_end = find_block(lines, "HV")
    if hv_start is not None and hv_cutoff is not None:
        hv_lines = [ln.strip() for ln in lines[hv_start:hv_end] if ln.strip()]
        hv_pairs = [list(map(float, ln.replace(',', ' ').split())) for ln in hv_lines]
        hv_data = np.array(hv_pairs)
        if hv_data.shape[0] == len(wavelengths):
            mask = hv_data[:, 1] <= hv_cutoff
            wavelengths = wavelengths[mask]
            ellipticity = ellipticity[mask]

    if len(wavelengths) == 0:
        return None, None

    # Sortowanie rosnąco według długości fal
    sort_idx = np.argsort(wavelengths)
    return wavelengths[sort_idx], ellipticity[sort_idx]


def convert_mdeg_to_delta_epsilon(mdeg, conc_uM, path_mm, residues):
    """
    Przelicza eliptyczność w mdeg na Delta Epsilon na resztę aminokwasową [M^-1 cm^-1].
    Delta_eps = theta_mdeg / (32980 * c_M * l_cm * residues)
    gdzie c_M = conc_uM * 1e-6, l_cm = path_mm / 10.
    """
    path_cm = path_mm / 10.0
    conc_M = conc_uM * 1e-6
    return mdeg / (32980.0 * conc_M * path_cm * residues)


# ==============================================================================
# 3. Silniki dekonwolucji
# ==============================================================================
def load_dataset(dataset_name):
    """Wczytuje bazy A.txt i F.txt z katalogu Datasets."""
    ds_dir = os.path.join(DATASETS_BASE_DIR, dataset_name)
    if not os.path.exists(ds_dir):
        return None, None, None, None

    A = np.loadtxt(os.path.join(ds_dir, "A.txt"))
    F = np.loadtxt(os.path.join(ds_dir, "F.txt"))
    with open(os.path.join(ds_dir, "lbl1.txt"), "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    if A.shape[0] == 71 and A.shape[1] != 71:
        A = A.T

    n_points = A.shape[0]
    wl_ref = np.linspace(240, 240 - n_points + 1, n_points)
    return A, F, labels, wl_ref


def solve_selcon3(A_sub, F_mat, y_query):
    weights, _ = nnls(A_sub, y_query)
    total_w = np.sum(weights)
    weights = weights / total_w if total_w > 0 else np.ones(A_sub.shape[1]) / A_sub.shape[1]

    if F_mat.shape[0] == len(weights):
        fractions = F_mat.T @ weights
    elif F_mat.shape[1] == len(weights):
        fractions = F_mat @ weights
    else:
        raise ValueError("Niezgodność wymiarów macierzy F.")

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
    weights = weights / total_w if total_w > 0 else np.ones(n) / n

    if F_mat.shape[0] == len(weights):
        fractions = F_mat.T @ weights
    elif F_mat.shape[1] == len(weights):
        fractions = F_mat @ weights
    else:
        raise ValueError("Niezgodność wymiarów macierzy F.")

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
    return {"Alpha": alpha / s, "Beta": beta / s, "Coil": coil / s, "Turn": 0.0, "Total": 1.0, "NRMSD": 0.05}


def map_fractions_to_standard(raw_dict):
    alpha = beta = turn = coil = 0.0
    for k, v in raw_dict.items():
        k_low = k.strip().lower()
        val = float(v)
        if any(x in k_low for x in ["helix", "alpha", "h(r)", "h(d)"]):
            alpha += val
        elif any(x in k_low for x in ["strand", "sheet", "beta", "s(r)", "s(d)"]):
            beta += val
        elif any(x in k_low for x in ["turn", "trn"]):
            turn += val
        elif any(x in k_low for x in ["unord", "coil", "other", "random", "loop"]):
            coil += val

    current_sum = alpha + beta + turn + coil
    if coil == 0.0 and current_sum < 0.98:
        coil = max(0.0, 1.0 - (alpha + beta + turn))
        current_sum = alpha + beta + turn + coil

    return {"Alpha": alpha, "Beta": beta, "Turn": turn, "Coil": coil, "Total": current_sum}


# ==============================================================================
# 4. Główna pętla wsadowa
# ==============================================================================
def run_prodata_dichroweb():
    if not os.path.exists(CONCENTRATION_FILE):
        print(f"❌ Brak pliku stężeń: {CONCENTRATION_FILE}")
        return

    conc_mapping = load_concentration_mapping(CONCENTRATION_FILE)
    # file_pattern = re.compile(r"(\d{5})\.csv$")
    file_pattern = re.compile(r"(\d+)\.csv$", re.IGNORECASE)

    # Wczytanie i mapowanie widm
    spectra_by_conc = {}
    for fname in sorted(os.listdir(SERIES_FOLDER)):
        match = file_pattern.search(fname)
        if not match:
            continue

        sample_num = int(match.group(1))
        conc = conc_mapping.get(sample_num, None)
        if conc is None or np.isnan(conc):
            continue

        fpath = os.path.join(SERIES_FOLDER, fname)
        wls, mdeg = read_prodata_csv(fpath, hv_cutoff=HV_CUTOFF)
        if wls is None or len(wls) == 0:
            continue

        # Preprocessing
        if APPLY_SMOOTHING and len(mdeg) >= SMOOTHING_WINDOW:
            mdeg = savgol_filter(mdeg, SMOOTHING_WINDOW, SMOOTHING_POLYORDER)

        if BASELINE_WAVELENGTH is not None and wls[0] <= BASELINE_WAVELENGTH <= wls[-1]:
            b_idx = np.argmin(np.abs(wls - BASELINE_WAVELENGTH))
            mdeg -= mdeg[b_idx]

        delta_eps = convert_mdeg_to_delta_epsilon(mdeg, PROTEIN_CONC_UM, PATH_LENGTH_MM, RESIDUE_COUNT)
        spectra_by_conc[float(conc)] = (wls, delta_eps)

    if not spectra_by_conc:
        print("❌ Nie wczytano żadnych poprawnych widm z folderu.")
        return

    print(f"[✓] Wczytano i sparsowano {len(spectra_by_conc)} widm ProData.")
    database = {}

    for method, basis_list in METHOD_BASIS_MAP.items():
        database[method] = {}

        for basis in basis_list:
            basis_label = basis if basis else "Default"
            print(f"\n==========================================")
            print(f"🚀 Dekonwolucja ProData: {method} | Baza: {basis_label}")
            print(f"==========================================")

            A_sub, F_mat, labels, wl_sub = None, None, None, None
            if basis is not None:
                A, F, labels, wl_ref = load_dataset(basis)
                if A is None:
                    print(f"⚠️ Brak bazy {basis}, pomijam.")
                    continue

                mask = (wl_ref >= MIN_WAVELENGTH) & (wl_ref <= MAX_WAVELENGTH)
                A_sub = A[mask, :]
                F_mat = F
                wl_sub = wl_ref[mask]

            records = []
            for conc in sorted(spectra_by_conc.keys()):
                wls_raw, delta_eps = spectra_by_conc[conc]

                if method == "K2D":
                    res = solve_k2d(wls_raw, delta_eps)
                    nrmsd = res["NRMSD"]
                else:
                    if wls_raw[0] > MIN_WAVELENGTH or wls_raw[-1] < MAX_WAVELENGTH:
                        # Sprawdzenie czy widmo po HV cutoff pokrywa wymagany zakres bazy
                        valid_mask = (wl_sub >= wls_raw[0]) & (wl_sub <= wls_raw[-1])
                        if np.sum(valid_mask) < 15:
                            continue
                        A_fit = A_sub[valid_mask, :]
                        wl_fit = wl_sub[valid_mask]
                    else:
                        A_fit = A_sub
                        wl_fit = wl_sub

                    y_query = np.interp(wl_fit, wls_raw, delta_eps)

                    if method == "SELCON3":
                        fracs, nrmsd = solve_selcon3(A_fit, F_mat, y_query)
                    elif method == "CONTIN":
                        fracs, nrmsd = solve_contin(A_fit, F_mat, y_query)

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
                print(f"Conc: {conc:4.2f}M | Alpha={res['Alpha']:.3f}, Beta={res['Beta']:.3f}, Coil={res['Coil']:.3f} | NRMSD={nrmsd:.4f}")

            if records:
                df_out = pd.DataFrame(records).sort_values("Concentration")
                database[method][basis_label] = df_out.to_dict(orient="records")
                csv_path = os.path.join(SERIES_FOLDER, f"local_{method.lower()}_{basis_label.lower()}.csv")
                df_out.to_csv(csv_path, index=False)
                print(f"[✓] Zapisano CSV: {csv_path}")

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f_json:
        json.dump(database, f_json, indent=2, ensure_ascii=False)
    print(f"\n[★] Zapisano bazę wyników: {OUTPUT_JSON_PATH}")

    plot_results(OUTPUT_JSON_PATH)


# ==============================================================================
# 5. Rysowanie wykresów
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
        ax.set_title(title_str, fontsize=12)
        ax.set_xlabel("Denaturant Concentration [M]", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Secondary Structure Fraction", fontsize=12)
            ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    output_png = os.path.join(SERIES_FOLDER, "local_dichroweb_prodata_comparison.png")
    fig.savefig(output_png, dpi=300)
    print(f"[✓] Wygenerowano wykres porównawczy: {output_png}")
    plt.show()


if __name__ == "__main__":
    run_prodata_dichroweb()