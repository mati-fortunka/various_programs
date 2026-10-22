import os
import re
import shutil
import subprocess
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==============================================================================
# 1. Konfiguracja ścieżek i parametrów eksperymentu
# ==============================================================================
sesca_script = "/home/matifortunka/Programs/SESCA/scripts/SESCA_deconv.py"
libs_dir = "/home/matifortunka/Programs/SESCA/libs"

# Folder z plikami ProData (.csv) oraz concentrations.txt
series_folder = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD/fuzja/3/2uM/28_07_26"
concentration_file = os.path.join(series_folder, "concentrations.txt")
denaturant_label = "GuCl"

# Zestaw map bazowych do przetestowania
maps_to_test = [
    "Map_DS-dT.dat",
    "Map_DSSP-T.dat",
    "Set_Perczel-1.dat",
    "Map_BB_DS-dTSC3.dat",
    "Map_DS3-1.dat"
]

# Parametry spektroskopowe i jakość widma
smoothing_window = 11
smoothing_polyorder = 3
baseline_wavelength = None     # None lub np. 250.0
min_wavelength = 210.0         # Dolne odcięcie widma [nm]
max_wavelength = 245.0         # Górne odcięcie widma [nm]
hv_cutoff = 950.0              # Maksymalne dopuszczalne napięcie HV [V]

# Parametry białka i kuwety (fuzja TrmD-Tm1570, n = 455 reszt, c = 2.0 µM w kuwecie 1.0 mm)
protein_conc = 2.0             # µmol/L (µM)
path_length_mm = 1.0           # mm (dla kuwety 1 mm; jeśli 0.1 mm, zmień na 0.1)
residue_count = 455            # 455 dla fuzji, 224 dla Tm1570

# Przelicznik do kdeg*cm^2/dmol (10^3 deg*cm^2/dmol)
# [theta] = (theta_mdeg * 100) / (c_uM * l_mm * n_res) = (theta_mdeg * 1000) / (c_uM * l_mm * n_res * 10)
conversion_factor = 1000.0 / (protein_conc * path_length_mm * residue_count)


# ==============================================================================
# 2. Parsowanie formatu ProData oraz pliku concentrations.txt
# ==============================================================================
def find_block(lines, property_name):
    """Wyszukuje zakres wierszy dla bloku danych (np. CircularDichroism lub HV)."""
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
    """Wczytuje mapowanie Sample_number -> den_concentration z obsługą przecinków i kropek."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().replace(',', '.')
    df = pd.read_csv(StringIO(text), sep=r"\s+", engine='python')
    df['den_concentration'] = pd.to_numeric(df['den_concentration'], errors='coerce')
    return df.set_index('Sample_number')['den_concentration'].to_dict()


def read_prodata_csv(filepath, hv_threshold=700.0):
    """Wczytuje widmo CD oraz napięcie HV z pliku CSV, filtrując zaszumione punkty."""
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

    # Filtracja po napięciu fotopowielacza
    hv_start, hv_end = find_block(lines, "HV")
    if hv_start is not None and hv_threshold is not None:
        hv_lines = [ln.strip() for ln in lines[hv_start:hv_end] if ln.strip()]
        hv_pairs = [list(map(float, ln.replace(',', ' ').split())) for ln in hv_lines]
        hv_data = np.array(hv_pairs)
        if hv_data.shape[0] == len(wavelengths):
            mask = hv_data[:, 1] <= hv_threshold
            wavelengths = wavelengths[mask]
            ellipticity = ellipticity[mask]

    if len(wavelengths) == 0:
        return None, None

    # Zapewnienie rosnącego porządku fal na etapie wczytania
    sort_idx = np.argsort(wavelengths)
    return wavelengths[sort_idx], ellipticity[sort_idx]


def load_prodata_series(folder, conc_file, hv_threshold=700.0):
    """Wczytuje wszystkie pliki serii ProData i przypisuje stężenia."""
    conc_map = load_concentration_mapping(conc_file)
    file_pattern = re.compile(r"(\d{5})\.csv$")
    spectra_by_conc = {}

    for fname in sorted(os.listdir(folder)):
        match = file_pattern.search(fname)
        if not match:
            continue

        sample_num = int(match.group(1))
        conc = conc_map.get(sample_num, None)
        if conc is None or np.isnan(conc):
            continue

        fpath = os.path.join(folder, fname)
        wls, mdeg = read_prodata_csv(fpath, hv_threshold=hv_threshold)
        if wls is not None and len(wls) > 0:
            spectra_by_conc[float(conc)] = (wls, mdeg)

    return spectra_by_conc


def parse_sesca_output(filepath):
    """Wyciąga udziały struktur z pliku wynikowego SESCA BS_deconv.out."""
    alpha = beta = coil = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean = line.strip()
            if clean.startswith("#") and ":" in clean:
                parts = clean.lstrip("#").split(":", 1)
                tag = parts[0].strip().lower()
                val_str = parts[1].strip()

                try:
                    val = float(val_str.split()[0])
                except (ValueError, IndexError):
                    continue

                if any(k in tag for k in ["alpha", "helix", "helx", "alh", "3h"]):
                    alpha = val if alpha is None else (alpha + val)
                elif any(k in tag for k in ["beta", "sheet", "ebs", "nbs"]):
                    beta = val if beta is None else (beta + val)
                elif any(k in tag for k in ["coil", "other", "turn", "random", "disorder", "loop", "uc"]):
                    coil = val if coil is None else (coil + val)

    return alpha, beta, coil


# ==============================================================================
# 3. Główna pętla wsadowa SESCA
# ==============================================================================
if not os.path.exists(concentration_file):
    print(f"❌ Brak pliku stężeń: {concentration_file}")
    exit(1)

spectra_by_conc = load_prodata_series(series_folder, concentration_file, hv_threshold=hv_cutoff)
if not spectra_by_conc:
    print(f"❌ Nie znaleziono poprawnych widm w: {series_folder}")
    exit(1)

print(f"[✓] Wczytano i sparsowano {len(spectra_by_conc)} widm ProData.")
print(f"    Stężenia: {sorted(list(spectra_by_conc.keys()))}")

all_map_results = {}
# denaturant_label = "GuCl" if "gucl" in series_folder.lower() else "Urea"

for map_name in maps_to_test:
    basis_path = os.path.join(libs_dir, map_name)
    if not os.path.exists(basis_path):
        print(f"❌ Brak pliku bazy: {basis_path}, pomijam.")
        continue

    print(f"\n==========================================")
    print(f"🚀 Przetwarzanie dla mapy: {map_name}")
    print(f"==========================================")

    map_out_dir = os.path.join(series_folder, f"out_{os.path.splitext(map_name)[0]}")
    os.makedirs(map_out_dir, exist_ok=True)

    concentrations = []
    alpha_contents = []
    beta_contents = []
    coil_contents = []

    for conc in sorted(spectra_by_conc.keys()):
        wavelengths, ellipticity = spectra_by_conc[conc]

        # Wygładzanie Savitzky-Golay
        if len(ellipticity) >= smoothing_window:
            ellipticity_smooth = savgol_filter(ellipticity, smoothing_window, smoothing_polyorder)
        else:
            ellipticity_smooth = ellipticity.copy()

        # Zerowanie baseline
        if baseline_wavelength is not None and wavelengths[0] <= baseline_wavelength <= wavelengths[-1]:
            baseline_idx = np.argmin(np.abs(wavelengths - baseline_wavelength))
            ellipticity_corrected = ellipticity_smooth - ellipticity_smooth[baseline_idx]
        else:
            ellipticity_corrected = ellipticity_smooth

        # Przeliczenie jednostek na MRE (kdeg*cm^2/dmol)
        ellipticity_mre = ellipticity_corrected * conversion_factor

        # Przycięcie zakresu fal
        mask = (wavelengths >= min_wavelength) & (wavelengths <= max_wavelength)
        wl_filtered = wavelengths[mask]
        mre_filtered = ellipticity_mre[mask]

        if len(wl_filtered) < 15:
            print(f"⚠️ Za mało punktów w zakresie {min_wavelength}-{max_wavelength} nm dla {conc} M po filtracji HV.")
            continue

        # Sortowanie malejące (wymóg dekonwolucji SESCA)
        if wl_filtered[0] < wl_filtered[-1]:
            wl_export = wl_filtered[::-1]
            mre_export = mre_filtered[::-1]
        else:
            wl_export = wl_filtered
            mre_export = mre_filtered

        spectrum_txt = os.path.join(map_out_dir, f"prodata_{conc:.2f}M_temp.txt")
        final_output = os.path.join(map_out_dir, f"prodata_{conc:.2f}M_deconv.out")

        with open(spectrum_txt, "w", encoding="utf-8") as f:
            for wl, val in zip(wl_export, mre_export):
                f.write(f"{wl:.1f}\t{val:.6f}\n")

        # Wywołanie skryptu SESCA
        try:
            result = subprocess.run(
                ["python3", sesca_script, spectrum_txt, basis_path],
                capture_output=True,
                text=True,
                cwd=series_folder
            )

            sesca_out = os.path.join(series_folder, "BS_deconv.out")
            if os.path.exists(sesca_out):
                shutil.copy(sesca_out, final_output)
                alpha, beta, coil = parse_sesca_output(sesca_out)
                os.remove(sesca_out)

                if alpha is not None and beta is not None and coil is not None:
                    concentrations.append(conc)
                    alpha_contents.append(alpha)
                    beta_contents.append(beta)
                    coil_contents.append(coil)
                    print(f"Conc {conc:4.2f}M | Alpha={alpha:.3f}, Beta={beta:.3f}, Coil={coil:.3f}")
                else:
                    print(f"⚠️ Nie udało się rozpoznać frakcji dla stężenia: {conc:.2f}M")
            else:
                print(f"❌ Brak wyjścia BS_deconv.out dla: {conc:.2f}M")
        except Exception as e:
            print(f"❌ Błąd wykonania dla {conc:.2f}M: {e}")
        finally:
            if os.path.exists(spectrum_txt):
                os.remove(spectrum_txt)

    if concentrations:
        all_map_results[map_name] = {
            "conc": np.array(concentrations),
            "alpha": np.array(alpha_contents),
            "beta": np.array(beta_contents),
            "coil": np.array(coil_contents)
        }

        # Zapis do CSV dla danej mapy
        df_map = pd.DataFrame({
            "Concentration": concentrations,
            "Alpha": alpha_contents,
            "Beta": beta_contents,
            "Coil": coil_contents
        }).sort_values("Concentration")
        csv_path = os.path.join(series_folder, f"sesca_{os.path.splitext(map_name)[0].lower()}.csv")
        df_map.to_csv(csv_path, index=False)
        print(f"[✓] Zapisano CSV: {csv_path}")

# ==============================================================================
# 4. Rysowanie panelu porównawczego dla wszystkich map
# ==============================================================================
if all_map_results:
    n_maps = len(all_map_results)
    fig, axes = plt.subplots(1, n_maps, figsize=(4.8 * n_maps, 4.5), sharey=True)
    if n_maps == 1:
        axes = [axes]

    for ax, (map_name, res) in zip(axes, all_map_results.items()):
        sort_idx = np.argsort(res["conc"])
        x = res["conc"][sort_idx]

        ax.plot(x, res["alpha"][sort_idx], label="Alpha", marker="o", color="crimson", lw=1.8)
        ax.plot(x, res["beta"][sort_idx], label="Beta", marker="s", color="royalblue", lw=1.8)
        ax.plot(x, res["coil"][sort_idx], label="Coil", marker="^", color="darkorange", lw=1.8)

        ax.set_title(map_name.replace(".dat", ""), fontsize=12)
        ax.set_xlabel(f"{denaturant_label} Conc. [M]", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Secondary Structure Fraction", fontsize=12)
            ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    output_comparison_plot = os.path.join(series_folder, f"SESCA_comparison_{denaturant_label}_all_maps.png")
    fig.savefig(output_comparison_plot, dpi=300)
    print(f"\n✅ Zapisano porównanie map do: {output_comparison_plot}")
    plt.show()