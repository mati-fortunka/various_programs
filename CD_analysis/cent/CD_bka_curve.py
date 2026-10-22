import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# === Parametry wejściowe ===
input_folder = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/fusions/8b1n/equlibrium/CD/unfolding_SAV"
den = "Urea"
# Parametry analizy
target_wavelength = 217.0          # Długość fali do wyznaczenia krzywej rozwijania [nm]
apply_smoothing = True             # True = wygładzanie Savitzky-Golay
smoothing_window = 11              # Długość okna (nieparzysta)
smoothing_polyorder = 3

apply_baseline = False             # True = wyzerowanie na baseline_wavelength
baseline_wavelength = 250.0        # Długość fali dla zera [nm]

# Opcja kanałów: None = uśredniaj ch1, ch2, ch3; "ch1" = bierz tylko ch1
CHANNEL_FILTER = None

RT = 0.592                         # Stała RT w kcal/mol (dla ~298 K)

output_spectra_plot = os.path.join(input_folder, "CD_spectra_comparison.png")
output_unfolding_plot = os.path.join(input_folder, f"unfolding_curve_{target_wavelength:.0f}nm.png")
output_fit_txt = os.path.join(input_folder, "fit.txt")


# === Model dwustanowy ===
def two_state_model(x, a_n, a_u, m, d):
    """
    Krzywa rozwijania białka w modelu dwustanowym:
    a_n - sygnał stanu natywnego
    a_u - sygnał stanu zdenaturowanego
    m   - współczynnik m (kcal/mol/M)
    d   - C_m, stężenie w punkcie przejścia (M)
    """
    exp_term = np.exp((m * (x - d)) / RT)
    return (a_n + a_u * exp_term) / (1.0 + exp_term)


# === Uniwersalny ekstraktor stężenia i kanału ===
def parse_filename_metadata(fname):
    """
    Rozpoznaje formaty:
    - Nowy: tm05_ch1.bka -> (0.5, 'ch1'), tm35_ch2.bka -> (3.5, 'ch2'), tm4_ch1.bka -> (4.0, 'ch1')
    - Stary: sav-golay1.5m.bka -> (1.5, 'ch1'), 2.0M.bka -> (2.0, 'ch1')
    """
    name = os.path.splitext(fname)[0].lower()

    # Ekstrakcja kanału (ch1, ch2, ch3 itp.)
    ch_match = re.search(r"ch(\d+)", name)
    channel = f"ch{ch_match.group(1)}" if ch_match else "ch1"

    # Wzorzec 1: Nowy format typu tm05, tm35, tm4 (część dziesiętna bez kropki)
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

    # Wzorzec 2: Klasyczny format ze stężeniem jawnym (np. 1.5m, 0.5M, 2m)
    m_old = re.search(r"(\d+(?:\.\d+)?)\s*m", name)
    if m_old:
        return float(m_old.group(1)), channel

    # Wzorzec 3: Pliki native / 0M / buffer
    if "native" in name or "buffer" in name or "tm0" in name:
        return 0.0, channel

    return None, None


# === Wczytywanie i uśrednianie kanałów .bka ===
def load_bka_files(folder):
    raw_by_conc = {}  # {conc: [df_ch1, df_ch2, ...]}

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".bka"):
            continue

        conc, channel = parse_filename_metadata(fname)
        if conc is None:
            print(f"⚠️ Nie udało się odczytać stężenia z nazwy pliku: {fname}")
            continue

        if CHANNEL_FILTER is not None and channel != CHANNEL_FILTER:
            continue

        path = os.path.join(folder, fname)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        try:
            start_idx = next(i for i, line in enumerate(lines) if line.strip().strip('"') == "_DATA") + 1
        except StopIteration:
            print(f"⚠️ Brak sekcji _DATA w: {fname}")
            continue

        data = []
        for line in lines[start_idx:]:
            line_str = line.strip()
            if not line_str or not any(c.isdigit() for c in line_str):
                break
            try:
                wavelength, value = map(float, line_str.split())
                data.append((wavelength, value))
            except ValueError:
                continue

        if not data:
            print(f"⚠️ Pusty blok danych w: {fname}")
            continue

        df = pd.DataFrame(data, columns=["Wavelength", "Ellipticity"]).sort_values("Wavelength")
        raw_by_conc.setdefault(conc, []).append(df)

    # Uśrednianie kanałów dla każdego stężenia na wspólnej siatce długości fal
    averaged_spectra = {}
    for conc, dfs in sorted(raw_by_conc.items()):
        ref_wls = dfs[0]["Wavelength"].values
        interp_y = [np.interp(ref_wls, d["Wavelength"].values, d["Ellipticity"].values) for d in dfs]
        mean_y = np.mean(interp_y, axis=0)
        averaged_spectra[conc] = pd.DataFrame({"Wavelength": ref_wls, "Ellipticity": mean_y})

    return averaged_spectra


# === Przetwarzanie sygnału pojedynczego widma ===
def process_spectrum(df, smooth=True, baseline=True):
    wavelengths = df["Wavelength"].values
    y = df["Ellipticity"].values.astype(float)

    if smooth:
        if len(y) >= smoothing_window:
            y = savgol_filter(y, smoothing_window, smoothing_polyorder)
        else:
            print("⚠️ Zbyt mało punktów do wygładzenia Savitzky-Golay, pomijam.")

    if baseline:
        baseline_idx = np.argmin(np.abs(wavelengths - baseline_wavelength))
        y -= y[baseline_idx]

    return wavelengths, y


# === Wykres porównawczy widm CD ===
def plot_all_spectra(spectra_dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    concentrations = list(spectra_dict.keys())
    norm = Normalize(vmin=min(concentrations), vmax=max(concentrations))
    cmap = cm.viridis

    for conc, df in spectra_dict.items():
        wls, y = process_spectrum(df, smooth=apply_smoothing, baseline=apply_baseline)
        ax.plot(wls, y, label=f"{conc} M", color=cmap(norm(conc)), alpha=0.85)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Denaturant Concentration [M]", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel("Wavelength [nm]", fontsize=13)
    ax.set_ylabel("Ellipticity [mdeg]", fontsize=13)
    ax.set_title(f"Equilibrium CD Spectra ({den} Unfolding)", fontsize=15)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_spectra_plot, dpi=300)
    print(f"Zapisano wykres widm: {output_spectra_plot}")
    plt.show()


# === Konstrukcja krzywej rozwijania i dopasowanie modelu dwustanowego ===
def fit_unfolding_curve(spectra_dict, wavelength_target):
    concentrations = []
    ellipticities = []

    for conc, df in spectra_dict.items():
        wls, y = process_spectrum(df, smooth=apply_smoothing, baseline=apply_baseline)
        target_val = np.interp(wavelength_target, wls, y)
        concentrations.append(conc)
        ellipticities.append(target_val)

    x_data = np.array(concentrations, dtype=float)
    y_data = np.array(ellipticities, dtype=float)

    # Wartości startowe dla modelu
    a_n_init = y_data[0]
    a_u_init = y_data[-1]
    d_init = np.median(x_data)
    m_init = 2.0
    initial_guess = [a_n_init, a_u_init, m_init, d_init]

    popt, perr = None, None
    try:
        popt, pcov = curve_fit(two_state_model, x_data, y_data, p0=initial_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))

        fit_text = (
            f"=== 2-State Equilibrium Unfolding Fit ===\n"
            f"Wavelength: {wavelength_target} nm\n"
            f"Baseline corrected at {baseline_wavelength} nm: {apply_baseline}\n"
            f"Smoothing applied: {apply_smoothing}\n"
            f"-----------------------------------------\n"
            f"a_n (native signal)      = {popt[0]:.3f} ± {perr[0]:.3f} mdeg\n"
            f"a_u (unfolded signal)    = {popt[1]:.3f} ± {perr[1]:.3f} mdeg\n"
            f"m-value                  = {popt[2]:.3f} ± {perr[2]:.3f} kcal/(mol*M)\n"
            f"C_m (midpoint, d)        = {popt[3]:.3f} ± {perr[3]:.3f} M\n"
            f"Delta G_H2O (m * C_m)    = {popt[2] * popt[3]:.3f} kcal/mol\n"
        )
        print("\n" + fit_text)

        with open(output_fit_txt, "w", encoding="utf-8") as f:
            f.write(fit_text)
        print(f"Zapisano parametry do: {output_fit_txt}")

    except Exception as e:
        print(f"⚠️ Dopasowanie krzywej nie powiodło się: {e}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_data, y_data, color="navy", s=45, label="Data (mean channels)", zorder=3)

    if popt is not None:
        x_dense = np.linspace(x_data.min(), x_data.max(), 300)
        y_fit = two_state_model(x_dense, *popt)
        ax.plot(x_dense, y_fit, color="crimson", linewidth=2,
                label=f"2-state fit (Cm={popt[3]:.2f} M, m={popt[2]:.2f})", zorder=2)

    ax.set_title(f"Unfolding Transition at {wavelength_target:.1f} nm", fontsize=15)
    ax.set_xlabel(f"{den} Concentration [M]", fontsize=13)
    ax.set_ylabel("Ellipticity [mdeg]", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(output_unfolding_plot, dpi=300)
    print(f"Zapisano wykres rozwijania: {output_unfolding_plot}")
    plt.show()


# === Główna pętla ===
if __name__ == "__main__":
    spectra = load_bka_files(input_folder)
    if spectra:
        print(f"[✓] Wczytano i uśredniono {len(spectra)} stężeń: {sorted(list(spectra.keys()))}")
        plot_all_spectra(spectra)
        fit_unfolding_curve(spectra, target_wavelength)
    else:
        print("Nie znaleziono poprawnych plików .bka w wybranym katalogu.")