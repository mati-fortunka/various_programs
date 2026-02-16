import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import os

# --- KONFIGURACJA ---
CONFIG = {
    # Ścieżka do pliku
    'filename': '/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/kinetyka/fluo/Tm1570/spectra_kinetics/MultipleEmScan1.csv',

    'time_interval': 34,  # Czas w sekundach między skanami

    # Parametry analizy
    'target_wl': 345,  # Długość fali do analizy pojedynczej intensywności (to będzie też fitowane)
    'ratio_wl1': 345,  # Licznik do ratio
    'ratio_wl2': 320,  # Mianownik do ratio

    # Wygładzanie
    'smoothing': True,
    'window_size': 11,
    'poly_order': 3,

    # --- NOWE: KOREKTA TŁA ---
    # Podaj długość fali (nm), względem której zerujemy widmo.
    # Wpisz 0 lub None, aby wyłączyć.
    'baseline_wl': 400,

    # --- NOWE: DOPASOWANIE (FITTING) ---
    'fit_enable': True,
    # Dostępne modele: 'mono_exp', 'mono_exp_drift', 'double_exp'
    'fit_model': 'mono_exp',
    # Zakres czasu do fitowania [start, koniec]. Ustaw [0, None] dla całości.
    'fit_range': [0, 15000]
}


# --- MODELE MATEMATYCZNE (z Twoich poprzednich programów) ---

def mono_exp(t, A, k, c):
    return A * np.exp(-k * t) + c


def mono_exp_drift(t, A, k, c, m):
    return A * np.exp(-k * t) + c + m * t


def double_exp(t, A1, k1, A2, k2, c):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + c


# --- FUNKCJE POMOCNICZE ---

def smooth_spectrum(y, window, poly):
    n = len(y)
    if n < window: return y
    try:
        if window % 2 == 0: window += 1
        return savgol_filter(y, window_length=window, polyorder=poly)
    except:
        return y


def calculate_csm(wavelengths, intensities):
    weighted_sum = np.sum(intensities * wavelengths)
    total_intensity = np.sum(intensities)
    if total_intensity == 0: return 0
    return weighted_sum / total_intensity


def get_intensity_at_lambda(wavelengths, intensities, target_wl):
    if len(wavelengths) == 0: return 0
    idx = (np.abs(wavelengths - target_wl)).argmin()
    return intensities[idx]


def load_custom_format(filename):
    if not os.path.exists(filename):
        print(f"BŁĄD: Plik nie istnieje: {filename}")
        return None

    data_start_row = -1
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) > 1 and parts[0].replace('.', '', 1).isdigit():
                try:
                    float(parts[0])
                    data_start_row = i
                    break
                except ValueError:
                    continue

    if data_start_row == -1: return None

    try:
        df = pd.read_csv(filename, skiprows=data_start_row, header=None)
        df.dropna(axis=1, how='all', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(axis=0, how='all', inplace=True)
        if df.empty: return None
        df.rename(columns={df.columns[0]: 'Wavelength'}, inplace=True)
        df.dropna(subset=['Wavelength'], inplace=True)
        print(f"Wczytano: {df.shape[0]} pkt widma, {df.shape[1] - 1} pkt czasu.")
        return df
    except Exception as e:
        print(f"Błąd wczytywania: {e}")
        return None


def perform_fitting(x_data, y_data, config):
    """Logika dopasowania krzywych."""
    model_name = config['fit_model']

    # 1. Filtrowanie zakresu
    start_t, end_t = config['fit_range']
    if end_t is None: end_t = np.max(x_data)

    mask = (x_data >= start_t) & (x_data <= end_t)
    x_fit = x_data[mask]
    y_fit = y_data[mask]

    if len(x_fit) < 5:
        print("Za mało punktów do fitowania w zadanym zakresie.")
        return None, None, None

    # 2. Parametry startowe (Guess)
    A0 = np.max(y_fit) - np.min(y_fit)
    C0 = np.min(y_fit)
    k0 = 1.0 / (np.mean(x_fit) + 1e-5)  # prymitywne przybliżenie

    try:
        if model_name == 'mono_exp':
            p0 = [A0, k0, C0]
            popt, pcov = curve_fit(mono_exp, x_fit, y_fit, p0=p0, maxfev=5000)
            y_model = mono_exp(x_fit, *popt)
            print(f"\n--- FIT RESULTS ({model_name}) ---")
            print(f"A = {popt[0]:.2f}, k = {popt[1]:.4e}, C = {popt[2]:.2f}")
            print(f"t_half = {np.log(2) / popt[1]:.2f} s")

        elif model_name == 'mono_exp_drift':
            p0 = [A0, k0, C0, 0]
            popt, pcov = curve_fit(mono_exp_drift, x_fit, y_fit, p0=p0, maxfev=5000)
            y_model = mono_exp_drift(x_fit, *popt)
            print(f"\n--- FIT RESULTS ({model_name}) ---")
            print(f"A = {popt[0]:.2f}, k = {popt[1]:.4e}, C = {popt[2]:.2f}, m = {popt[3]:.4e}")
            print(f"t_half = {np.log(2) / popt[1]:.2f} s")

        elif model_name == 'double_exp':
            # Zgadujemy: jedna szybka faza (10x k0), jedna wolna (k0)
            p0 = [A0 * 0.5, k0 * 10, A0 * 0.5, k0, C0]
            popt, pcov = curve_fit(double_exp, x_fit, y_fit, p0=p0, maxfev=10000)
            y_model = double_exp(x_fit, *popt)
            print(f"\n--- FIT RESULTS ({model_name}) ---")
            print(f"A1={popt[0]:.2f}, k1={popt[1]:.4e} (t1/2={np.log(2) / popt[1]:.1f}s)")
            print(f"A2={popt[2]:.2f}, k2={popt[3]:.4e} (t2/2={np.log(2) / popt[3]:.1f}s)")
            print(f"C ={popt[4]:.2f}")

        else:
            print("Nieznany model fitowania.")
            return None, None, None

        return x_fit, y_model, popt

    except Exception as e:
        print(f"Fitowanie nieudane: {e}")
        return None, None, None


# --- GŁÓWNA ANALIZA ---

def main():
    print(f"Analiza pliku: {CONFIG['filename']}")

    df = load_custom_format(CONFIG['filename'])
    if df is None: return

    wavelengths = df['Wavelength'].values
    scan_columns = df.columns[1:]
    num_scans = len(scan_columns)

    # Obliczamy całkowity czas dla skanów
    total_time = (num_scans - 1) * CONFIG['time_interval']

    kinetics_data = {'Time': [], 'Single_Int': [], 'Ratio': [], 'CSM': []}

    # --- RYSUNEK 1: WIDMA ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_scans))

    print("Przetwarzanie skanów...")

    for i, col_name in enumerate(scan_columns):
        raw_intensities = df[col_name].values

        # --- 1. POPRAWIONA KOREKTA TŁA ---
        # Jeśli zdefiniowano baseline_wl, znajdź wartość w tym punkcie i odejmij
        if CONFIG['baseline_wl'] and CONFIG['baseline_wl'] > 0:
            bl_idx = (np.abs(wavelengths - CONFIG['baseline_wl'])).argmin()
            baseline_val = raw_intensities[bl_idx]
            intensities = raw_intensities - baseline_val
        else:
            intensities = raw_intensities

        # Wygładzanie
        if CONFIG['smoothing']:
            intensities = smooth_spectrum(intensities, CONFIG['window_size'], CONFIG['poly_order'])

        # Rysowanie
        step = 1 if num_scans < 50 else int(num_scans / 50)
        if i % step == 0:
            ax.plot(wavelengths, intensities, color=colors[i], alpha=0.8, linewidth=1)

        # Kinetyka
        time_point = i * CONFIG['time_interval']

        kinetics_data['Time'].append(time_point)
        kinetics_data['Single_Int'].append(get_intensity_at_lambda(wavelengths, intensities, CONFIG['target_wl']))

        i1 = get_intensity_at_lambda(wavelengths, intensities, CONFIG['ratio_wl1'])
        i2 = get_intensity_at_lambda(wavelengths, intensities, CONFIG['ratio_wl2'])
        kinetics_data['Ratio'].append(i1 / i2 if i2 != 0 else 0)

        kinetics_data['CSM'].append(calculate_csm(wavelengths, intensities))

    # Opisy wykresu widm
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"Spectra Evolution (Baseline @ {CONFIG['baseline_wl']}nm)")
    ax.grid(True, linestyle='--', alpha=0.8)

    # --- 2. POPRAWIONY COLORBAR (Czas rzeczywisty) ---
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=total_time))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time (s)')

    # Zapis widm
    base_dir = os.path.dirname(CONFIG['filename'])
    plt.savefig(os.path.join(base_dir, "analiza_widma.png"), dpi=300)
    plt.show()

    # --- RYSUNEK 2: KINETYKA ---
    times = np.array(kinetics_data['Time'])
    int_data = np.array(kinetics_data['Single_Int'])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Panel 1: Pojedyncza intensywność + FIT
    ax1.plot(times, int_data, 'o', color='blue', markersize=3, alpha=0.7, label='Data')

    # --- 3. LOGIKA FITOWANIA ---
    if CONFIG['fit_enable']:
        x_fit, y_fit, params = perform_fitting(times, int_data, CONFIG)
        if x_fit is not None:
            ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label=f"Fit: {CONFIG['fit_model']}")
            ax1.legend()

    ax1.set_ylabel(f"Intensity @ {CONFIG['target_wl']} nm")
    ax1.set_title("Kinetic Analysis")
    ax1.grid(True)

    # Panel 2: Ratio
    ax2.plot(times, kinetics_data['Ratio'], 's-', color='green', markersize=3, linewidth=1)
    ax2.set_ylabel(f"Ratio {CONFIG['ratio_wl1']}/{CONFIG['ratio_wl2']}")
    ax2.grid(True)

    # Panel 3: CSM
    ax3.plot(times, kinetics_data['CSM'], 'd-', color='red', markersize=3, linewidth=1)
    ax3.set_ylabel("CSM (nm)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "analiza_kinetyka.png"), dpi=300)
    plt.show()

    # Zapis CSV
    pd.DataFrame(kinetics_data).to_csv(os.path.join(base_dir, "wyniki_kinetyczne.csv"), index=False)
    print("Zapisano wyniki.")


if __name__ == "__main__":
    main()