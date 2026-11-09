import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ==============================================================================
# 1. Konfiguracja Matplotlib i Parametrów
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"

HV_CUTOFF = 850.0
DEFAULT_INPUT_CSV = "/home/matifortunka/Downloads/PhD/rozdział_IV/denaturants_CD/bufor_hepes00001.csv"
DEFAULT_OUTPUT_PDF = "figures/appendix/Fig_A_4_CD_buffer_control.pdf"


# ==============================================================================
# 2. Parser formatu ProDataCSV (CircularDichroism + HV)
# ==============================================================================
def parse_prodata_cd_and_hv(filepath, hv_cutoff=HV_CUTOFF):
    """
    Parsuje plik ProDataCSV, wyciąga bloki CircularDichroism oraz HV
    i odcina punkty z nasyceniem detektora (HV > hv_cutoff).
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1. Odczyt bloku CircularDichroism
    cd_start = None
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("CircularDichroism"):
            cd_start = i + 2
            break

    if cd_start is None:
        raise ValueError(f"Nie znaleziono bloku CircularDichroism w pliku: {filepath}")

    cd_lines = []
    for line in lines[cd_start:]:
        l_str = line.strip()
        if not l_str or l_str.startswith("Wavelength") or l_str.startswith("Available"):
            break
        parts = l_str.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                cd_lines.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    cd_arr = np.array(cd_lines)
    wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

    # 2. Odczyt bloku HV i filtracja
    hv_start = None
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("HV"):
            hv_start = i + 2
            break

    if hv_start is not None:
        hv_lines = []
        for line in lines[hv_start:]:
            l_str = line.strip()
            if not l_str or l_str.startswith("Wavelength") or l_str.startswith("Available"):
                break
            parts = l_str.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    hv_lines.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        if hv_lines:
            hv_arr = np.array(hv_lines)
            if len(hv_arr) == len(wls):
                mask = hv_arr[:, 1] <= hv_cutoff
                wls, mdeg = wls[mask], mdeg[mask]

    # Posortowanie według rosnącej długości fali
    sort_idx = np.argsort(wls)
    return wls[sort_idx], mdeg[sort_idx]


# ==============================================================================
# 3. Generator Wykresu Kontrolnego
# ==============================================================================
def plot_buffer_cd_control(
    buffer_csv=DEFAULT_INPUT_CSV,
    output_pdf=DEFAULT_OUTPUT_PDF,
    hv_cutoff=HV_CUTOFF,
    scale_factor=1.0  # Współczynnik skalowania (np. 0.95 przy 95% udziale buforu)
):
    """
    Generuje wykres widma CD buforu w formacie PDF oraz PNG.
    """
    if not os.path.exists(buffer_csv):
        raise FileNotFoundError(f"Brak pliku wejściowego: {buffer_csv}")

    wls, mdeg = parse_prodata_cd_and_hv(buffer_csv, hv_cutoff=hv_cutoff)

    # 1. Wygładzanie filtrem Savitzky'ego-Golaya
    if len(mdeg) >= 11:
        mdeg = savgol_filter(mdeg, window_length=11, polyorder=3)

    # 2. Korekta linii bazowej (odjęcie wartości przy 250 nm)
    baseline_offset = np.interp(250.0, wls, mdeg)
    mdeg = (mdeg - baseline_offset) * scale_factor

    # 3. Przycięcie danych do zakresu 210 - 250 nm
    mask_range = (wls >= 210.0) & (wls <= 250.0)
    wls = wls[mask_range]
    mdeg = mdeg[mask_range]

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

    # Pozioma linia zerowa oraz pionowy znacznik długości analitycznej 217 nm
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
    ax.axvline(217, color="#D95F02", linestyle="--", linewidth=1.2, alpha=0.8, label="217 nm readout", zorder=2)

    # Etykieta legendy uwzględniająca ewentualne skalowanie
    label_buffer = "Buffer (20 mM HEPES, 0.5 M NaCl,\n1 mM TCEP, 5% glycerol)"

    # Skorygowane i przeskalowane widmo buforu
    ax.plot(
        wls,
        mdeg,
        color="#1F77B4",
        linewidth=2.0,
        label=label_buffer,
        zorder=3
    )

    ax.set_xlim(210, 250)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel(r"Ellipticity $\theta$ (mdeg)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
        fontsize=9.5
    )

    plt.tight_layout()
    fig.savefig(output_pdf, format="pdf")
    output_png = os.path.splitext(output_pdf)[0] + ".png"
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    print(
        f"[OK] Zapisano wykres kontrolny buforu, baseline={baseline_offset:.3f} mdeg):\n"
        f"  -> {output_pdf}\n"
        f"  -> {output_png}"
    )


if __name__ == "__main__":
    # Wywołanie z domyślnym scale_factor=1.0 (np. zmień na 0.90 dla 90% buforu)
    plot_buffer_cd_control(scale_factor=0.5)