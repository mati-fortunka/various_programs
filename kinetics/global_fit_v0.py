import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Minimizer, Parameters, report_fit

# --- 1. Definicja modelu (suma wykładników) ---
def multi_exp_model(t, amps, rates, offset):
    """
    Model: Sum(A_i * exp(-k_i * t)) + offset
    amps: lista amplitud [A_a, A_b, ...]
    rates: lista stałych szybkości [k_a, k_b, ...]
    """
    model = np.full_like(t, offset, dtype=float)
    for A, k in zip(amps, rates):
        model += A * np.exp(-k * t)
    return model


# --- 2. Funkcja licząca residua (różnicę między danymi a modelem) ---
def objective_function(params, t_sf_fast, data_sf_fast,
                       t_sf_slow, data_sf_slow,
                       t_cd, data_cd):
    # Pobieramy GLOBALNE stałe szybkości (wspólne dla wszystkich)
    # Używamy nazw k_A, k_B itp.
    rates = [params['k_A'], params['k_B'], params['k_C'],
             params['k_D'], params['k_E'], params['k_F']]

    # --- Obliczanie modelu dla SF FAST ---
    # SF widzi głównie fazy szybkie (A, B), może widzieć początek C.
    # Fazy bardzo wolne (E, F) dla SF są stałą linią (częścią offsetu).
    amps_sf = [params['amp_sf_A'], params['amp_sf_B'], params['amp_sf_C'],
               params['amp_sf_D'], params['amp_sf_E'], params['amp_sf_F']]

    model_sf_fast = multi_exp_model(t_sf_fast, amps_sf, rates, params['off_sf'])
    resid_sf_fast = data_sf_fast - model_sf_fast

    # --- Obliczanie modelu dla SF SLOW ---
    # Te same amplitudy co SF Fast (bo to ta sama metoda), ale inny czas
    model_sf_slow = multi_exp_model(t_sf_slow, amps_sf, rates, params['off_sf'])
    resid_sf_slow = data_sf_slow - model_sf_slow

    # --- Obliczanie modelu dla CD (długie czasy) ---
    # CD ma WŁASNE amplitudy i offset, ale TE SAME stałe szybkości (rates)
    amps_cd = [params['amp_cd_A'], params['amp_cd_B'], params['amp_cd_C'],
               params['amp_cd_D'], params['amp_cd_E'], params['amp_cd_F']]

    model_cd = multi_exp_model(t_cd, amps_cd, rates, params['off_cd'])
    resid_cd = data_cd - model_cd

    # --- Łączenie rezyduów w jedną długą tablicę ---
    # Opcjonalnie: można dodać wagi, np. jeśli SF ma dużo szumu
    return np.concatenate((resid_sf_fast, resid_sf_slow, resid_cd))


# --- 3. Wczytywanie i przygotowanie danych (ZARYS) ---
# Tu musisz wstawić swoje funkcje wczytujące dane
# Pamiętaj: CD long musi mieć czas w SEKUNDACH (h * 3600)

# Przykład (zastąp swoimi danymi):
# t_sf_fast, y_sf_fast = load_sf_data("phase_A.csv")
# t_sf_slow, y_sf_slow = load_sf_data("phase_BC.csv")
# t_cd_kin, y_cd_kin = load_cd_kinetics("cd_kinetics.csv") # 30s - 2000s
# t_cd_spec, y_cd_spec = load_cd_spectra_trace("spectra_folder") # 0h - 30h
# t_cd_spec_sec = t_cd_spec * 3600

# Połączenie danych CD w jeden zbiór (jeśli się nie pokrywają lub usunięto duplikaty)
# t_cd_total = np.concatenate([t_cd_kin, t_cd_spec_sec])
# y_cd_total = np.concatenate([y_cd_kin, y_cd_spec])
# Sortowanie po czasie jest ważne!
# sort_idx = np.argsort(t_cd_total)
# t_cd_total = t_cd_total[sort_idx]
# y_cd_total = y_cd_total[sort_idx]

# --- DLA TESTU GENERUJEMY SZTUCZNE DANE ---
# (Usuń ten blok, gdy podstawisz swoje dane)
t_sf_fast = np.linspace(0, 1, 500)
t_sf_slow = np.linspace(1, 2000, 1000)
t_cd = np.linspace(30, 100000, 500)  # do 30h
# Symulacja: k_A=2.0, k_B=0.1, k_E=0.0001
y_sf_fast = 5 * np.exp(-2.0 * t_sf_fast) + 10 + np.random.normal(0, 0.05, len(t_sf_fast))
y_sf_slow = 2 * np.exp(-0.1 * t_sf_slow) + 0.5 * np.exp(-0.001 * t_sf_slow) + 10 + np.random.normal(0, 0.05,
                                                                                                    len(t_sf_slow))
y_cd = 1000 * np.exp(-0.001 * t_cd) + 500 * np.exp(-0.00002 * t_cd) + np.random.normal(0, 10, len(t_cd))
# ------------------------------------------


# --- 4. Konfiguracja parametrów (Parameters) ---
params = Parameters()

# A. Stałe szybkości (Globalne) - inicjujemy wartościami z Twojej tabeli
# k = ln(2) / t_1/2
params.add('k_A', value=np.log(2) / 0.31, min=0)
params.add('k_B', value=np.log(2) / 8.8, min=0)
params.add('k_C', value=np.log(2) / 350, min=0)
params.add('k_D', value=np.log(2) / 1600, min=0)
params.add('k_E', value=np.log(2) / (1.3 * 3600), min=0)  # h na s
params.add('k_F', value=np.log(2) / (27 * 3600), min=0)  # h na s

# B. Amplitudy dla Stopped Flow
params.add('off_sf', value=10)  # Offset
params.add('amp_sf_A', value=5)
params.add('amp_sf_B', value=1)
params.add('amp_sf_C', value=0.5)
params.add('amp_sf_D', value=0.1)
# Fazy E i F są prawdopodobnie zbyt wolne/małe dla SF, można je zablokować na 0
# lub pozwolić im być bardzo małe.
params.add('amp_sf_E', value=0, vary=False)
params.add('amp_sf_F', value=0, vary=False)

# C. Amplitudy dla CD
params.add('off_cd', value=0)
# Faza A jest za szybka dla CD (martwy czas), więc amplituda 0
params.add('amp_cd_A', value=0, vary=False)
params.add('amp_cd_B', value=100)
params.add('amp_cd_C', value=500)
params.add('amp_cd_D', value=500)
params.add('amp_cd_E', value=1000)
params.add('amp_cd_F', value=2000)

# --- 5. Uruchomienie minimalizacji ---
minner = Minimizer(objective_function, params,
                  fcn_args=(t_sf_fast, y_sf_fast,
                            t_sf_slow, y_sf_slow,
                            t_cd, y_cd))

result = minner.minimize(method='leastsq')

# --- 6. Raport i Wykresy ---
report_fit(result)

# Przeliczenie k na t_1/2
print("\n--- Global t_1/2 results ---")
for phase in ['A', 'B', 'C', 'D', 'E', 'F']:
    k_val = result.params[f'k_{phase}'].value
    if k_val > 0:
        t_half = np.log(2) / k_val
        print(f"Phase {phase}: {t_half:.2f} s ({t_half / 3600:.2f} h)")

# Wizualizacja
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot SF (skala logarytmiczna X jest kluczowa przy takim zakresie)
# Łączymy SF fast i slow do wykresu
t_sf_all = np.concatenate([t_sf_fast, t_sf_slow])
y_sf_all = np.concatenate([y_sf_fast, y_sf_slow])
# Odtworzenie modelu z dopasowanych parametrów
# Musimy pobrać fitted values ręcznie, bo funkcja celu zwraca residuals
# Ale możemy użyć naszej funkcji modelu:
final_rates = [result.params[f'k_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
final_amps_sf = [result.params[f'amp_sf_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
model_sf_all = multi_exp_model(t_sf_all, final_amps_sf, final_rates, result.params['off_sf'].value)

ax1.plot(t_sf_all, y_sf_all, 'o', alpha=0.3, label='SF Raw Data', color='gray')
ax1.plot(t_sf_all, model_sf_all, 'r-', lw=2, label='Global Fit')
ax1.set_xscale('log')
ax1.set_title('Stopped Flow (Global Fit)')
ax1.set_ylabel('Voltage (V)')
ax1.legend()

# Plot CD
final_amps_cd = [result.params[f'amp_cd_{p}'].value for p in ['A', 'B', 'C', 'D', 'E', 'F']]
model_cd = multi_exp_model(t_cd, final_amps_cd, final_rates, result.params['off_cd'].value)

ax2.plot(t_cd, y_cd, 'bo', alpha=0.3, label='CD Raw Data')
ax2.plot(t_cd, model_cd, 'k-', lw=2, label='Global Fit')
ax2.set_xscale('log')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Ellipticity')
ax2.set_title('CD Kinetics (Global Fit)')
ax2.legend()

plt.tight_layout()
plt.show()