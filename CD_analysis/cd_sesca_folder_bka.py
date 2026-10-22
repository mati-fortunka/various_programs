import os
import re
import shutil
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# === Configuration ===
sesca_script = "/home/matifortunka/Programs/SESCA/scripts/SESCA_deconv.py"
libs_dir = "/home/matifortunka/Programs/SESCA/libs"
input_folder = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/8b1n/equlibrium/CD/urea/fus_unfold_SAV"

# Lista 5 baz do przetestowania:
maps_to_test = [
    "Map_DS-dT.dat",
    "Map_DSSP-T.dat",
    "Set_Perczel-1.dat",
    "Map_BB_DS-dTSC3.dat",
    "Map_DS3-1.dat"
]

smoothing_window = 11
smoothing_polyorder = 3
baseline_wavelength = None  # None lub 250.0
min_wavelength = 200.0      # 200.0 nm - bezpieczne odcięcie dla denaturantu
max_wavelength = 245.0

# === CD unit conversion parameters ===
protein_conc = 15.0         # µmol/L (sprawdź dokładne stężenie w kuwecie!)
path_length_mm = 0.1        # mm (0.01 cm)
residue_count = 455         # 224 dla Tm1570, 455 dla fuzji

# Przelicznik do kdeg*cm^2/dmol (10^3 deg*cm^2/dmol)
conversion_factor = 1000.0 / (protein_conc * path_length_mm * residue_count)


# === Uniwersalny ekstraktor stężenia i kanału ===
def parse_filename_metadata(fname):
    """
    Rozpoznaje:
    - Nowe formaty: tm05_ch1.bka -> (0.5, 'ch1'), tm35_ch2.bka -> (3.5, 'ch2'), tm4_ch1.bka -> (4.0, 'ch1')
    - Klasyczne formaty: sav-golay1.5m.bka -> (1.5, 'ch1'), 2.0M.bka -> (2.0, 'ch1')
    """
    name = os.path.splitext(fname)[0].lower()

    ch_match = re.search(r"ch(\d+)", name)
    channel = f"ch{ch_match.group(1)}" if ch_match else "ch1"

    # Wzorzec typu tm05, tm35, tm4
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

    # Wzorzec z literą m/M
    m_old = re.search(r"(\d+(?:\.\d+)?)\s*m", name)
    if m_old:
        return float(m_old.group(1)), channel

    if "native" in name or "buffer" in name or "tm0" in name:
        return 0.0, channel

    return None, None


def read_bka_raw(filepath):
    """Odczytuje sekcję _DATA z pojedynczego pliku .bka."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    try:
        data_start_idx = next(i for i, line in enumerate(lines) if line.strip().strip('"') == "_DATA") + 1
    except StopIteration:
        return None, None

    wls, vals = [], []
    for line in lines[data_start_idx:]:
        line_clean = line.strip()
        if not line_clean or not any(c.isdigit() for c in line_clean):
            break
        try:
            w, v = map(float, line_clean.split())
            wls.append(w)
            vals.append(v)
        except ValueError:
            continue

    if not wls:
        return None, None

    return np.array(wls), np.array(vals)


def load_and_average_all_spectra(folder):
    """Grupuje pliki .bka według stężeń i uśrednia sygnał z kanałów ch1, ch2, ch3."""
    grouped = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".bka"):
            continue

        conc, channel = parse_filename_metadata(fname)
        if conc is None:
            continue

        wls, vals = read_bka_raw(os.path.join(folder, fname))
        if wls is not None:
            grouped.setdefault(conc, []).append((wls, vals))

    averaged = {}
    for conc, datasets in sorted(grouped.items()):
        ref_wls = datasets[0][0]
        # Interpolacja każdego powtórzenia na wspólną siatkę długości fal
        interp_vals = [np.interp(ref_wls, w, v) for w, v in datasets]
        mean_vals = np.mean(interp_vals, axis=0)
        averaged[conc] = (ref_wls, mean_vals)

    return averaged


def parse_sesca_output(filepath):
    """Wyciąga frakcje strukturalne z pliku BS_deconv.out."""
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


# === Główna pętla wsadowa SESCA ===
spectra_by_conc = load_and_average_all_spectra(input_folder)
if not spectra_by_conc:
    print(f"❌ Nie znaleziono poprawnych widm w: {input_folder}")
    exit(0)

print(f"[✓] Wczytano i uśredniono widma dla {len(spectra_by_conc)} stężeń: {list(spectra_by_conc.keys())}")

all_map_results = {}
denaturant_label = "GuCl" if "gucl" in input_folder.lower() else "Urea"

for map_name in maps_to_test:
    basis_path = os.path.join(libs_dir, map_name)
    if not os.path.exists(basis_path):
        print(f"❌ Brak pliku bazy: {basis_path}, pomijam.")
        continue

    print(f"\n==========================================")
    print(f"🚀 Przetwarzanie dla mapy: {map_name}")
    print(f"==========================================")

    map_out_dir = os.path.join(input_folder, f"out_{os.path.splitext(map_name)[0]}")
    os.makedirs(map_out_dir, exist_ok=True)

    concentrations = []
    alpha_contents = []
    beta_contents = []
    coil_contents = []

    for conc, (wavelengths, ellipticity) in spectra_by_conc.items():
        # Preprocessing: wygładzanie i baseline
        ellipticity_smooth = savgol_filter(ellipticity, smoothing_window, smoothing_polyorder)
        if baseline_wavelength is not None:
            baseline_idx = np.argmin(np.abs(wavelengths - baseline_wavelength))
            ellipticity_corrected = ellipticity_smooth - ellipticity_smooth[baseline_idx]
        else:
            ellipticity_corrected = ellipticity_smooth

        ellipticity_mre = ellipticity_corrected * conversion_factor

        # Przycinanie zakresu fal
        mask = (wavelengths >= min_wavelength) & (wavelengths <= max_wavelength)
        wl_filtered = wavelengths[mask]
        mre_filtered = ellipticity_mre[mask]

        if len(wl_filtered) == 0:
            continue

        # Sortowanie malejące (wymóg SESCA)
        if wl_filtered[0] < wl_filtered[-1]:
            wl_export = wl_filtered[::-1]
            mre_export = mre_filtered[::-1]
        else:
            wl_export = wl_filtered
            mre_export = mre_filtered

        spectrum_txt = os.path.join(map_out_dir, f"Tm1570_{conc:.1f}M_temp.txt")
        final_output = os.path.join(map_out_dir, f"Tm1570_{conc:.1f}M_deconv.out")

        with open(spectrum_txt, "w") as f:
            for wl, val in zip(wl_export, mre_export):
                f.write(f"{wl:.1f}\t{val:.6f}\n")

        # Uruchomienie dekonwolucji SESCA
        try:
            result = subprocess.run(
                ["python3", sesca_script, spectrum_txt, basis_path],
                capture_output=True,
                text=True,
                cwd=input_folder
            )

            sesca_out = os.path.join(input_folder, "BS_deconv.out")
            if os.path.exists(sesca_out):
                shutil.copy(sesca_out, final_output)
                alpha, beta, coil = parse_sesca_output(sesca_out)
                os.remove(sesca_out)

                if alpha is not None and beta is not None and coil is not None:
                    concentrations.append(conc)
                    alpha_contents.append(alpha)
                    beta_contents.append(beta)
                    coil_contents.append(coil)
                    print(f"Conc {conc:4.1f}M | Alpha={alpha:.3f}, Beta={beta:.3f}, Coil={coil:.3f}")
                else:
                    print(f"⚠️ Nie udało się rozpoznać frakcji dla stężenia: {conc}M")
            else:
                print(f"❌ Brak wyjścia BS_deconv.out dla: {conc}M")
        except Exception as e:
            print(f"❌ Błąd wykonania dla {conc}M: {e}")
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

# === Generowanie wykresu wielopanelowego ===
if all_map_results:
    n_maps = len(all_map_results)
    fig, axes = plt.subplots(1, n_maps, figsize=(5.0 * n_maps, 4.5), sharey=True)
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
    output_comparison_plot = os.path.join(input_folder, f"SESCA_comparison_{denaturant_label}_all_maps.png")
    fig.savefig(output_comparison_plot, dpi=300)
    print(f"\n✅ Zapisano porównanie map do: {output_comparison_plot}")
    plt.show()