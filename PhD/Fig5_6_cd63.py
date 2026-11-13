import os
import re
import glob
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter

# =========================================================
# 1. KONFIGURACJA TYPOGRAFII I STYLU ROZPRAWY (LaTeX / PhD)
# =========================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"
mpl.rcParams["axes.linewidth"] = 1.1

OUT_DIR = "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch05"
OUT_BASENAME = "Figura_5_4_6_3_CD_Kinetics"

# Ścieżki wejściowe
DIR_PANEL_A = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/paper/CD/cd_kin_2000s_final/simple"
DIR_PANEL_BC = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/paper/CD/spectra-kin"
PANEL_C_SPECTRA_FILE = None  # None = pierwszy plik z DIR_PANEL_BC, lub np. "nazwa_pliku.csv"

# Zakresy osi Y (łatwe do ręcznego dostrojenia)
Y_MIN_ALL = -7.2   # wspólne minimum osi Y dla paneli A, B i C
Y_MAX_AB = -1.2    # wspólne maksimum osi Y dla paneli A i B
Y_MAX_C = 0.8      # maksimum osi Y dla widm w panelu C


# =========================================================
# 2. FUNKCJE POMOCNICZE (PARSOWANIE PRODATACSV / CSV)
# =========================================================
def safe_make_odd(n: int) -> int:
    n = max(3, int(n))
    return n if (n % 2 == 1) else n + 1


def read_dead_times(file_path: str) -> dict:
    dead_times = {}
    if not os.path.exists(file_path):
        return dead_times
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                name, time_str = parts
                try:
                    dead_times[name] = int(time_str.rstrip("s"))
                except ValueError:
                    pass
    return dead_times


def read_single_wl_cd(filename: str) -> pd.DataFrame:
    """Wczytuje kinetykę CD przy pojedynczej długości fali (ProDataCSV lub zwykły CSV)."""
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if lines[0].strip() == "ProDataCSV":
        start_idx = next((i for i, l in enumerate(lines) if l.strip() == "CircularDichroism"), None)
        if start_idx is None:
            raise ValueError(f"Brak sekcji CircularDichroism w pliku: {filename}")

        data_start_idx = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip() == "":
                continue
            if "Time" in lines[j]:
                data_start_idx = j
                break
        if data_start_idx is None:
            raise ValueError(f"Brak nagłówka 'Time' w pliku: {filename}")

        df_full = pd.read_csv(StringIO("".join(lines[data_start_idx:])), sep=",").dropna(how="all", axis=1)
    else:
        first_line = lines[0].strip()
        skiprows = 1 if not re.match(r"^[\d\.\-]", first_line.split(",")[0]) else 0
        df_full = pd.read_csv(filename, skiprows=skiprows, sep=",").dropna(how="all", axis=1)

    time_col = pd.to_numeric(df_full.iloc[:, 0], errors="coerce").values
    wrap_index = next((i for i in range(1, len(time_col)) if time_col[i] <= time_col[i - 1]), None)
    df = df_full.iloc[:wrap_index].copy() if wrap_index is not None else df_full.copy()

    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df = df.dropna(subset=[df.columns[0], df.columns[1]])
    df = df.drop_duplicates(subset=df.columns[0], keep="first").reset_index(drop=True)
    return df


def extract_section_lines(lines: list, section_name: str) -> list:
    for i, line in enumerate(lines):
        if line.strip() == section_name:
            start = i + 2
            break
    else:
        raise ValueError(f"Brak sekcji {section_name}")

    section_lines = []
    for line in lines[start:]:
        if not line.strip():
            break
        section_lines.append(line)
    return section_lines


def read_transposed_spectra_and_hv(filepath: str):
    """Wczytuje bloki CircularDichroism i HV i transponuje je do macierzy (Wavelength x Time)."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cd_df = pd.read_csv(StringIO("".join(extract_section_lines(lines, "CircularDichroism"))), skipinitialspace=True)
    hv_df = pd.read_csv(StringIO("".join(extract_section_lines(lines, "HV"))), skipinitialspace=True)

    cd_df.dropna(axis=1, how="all", inplace=True)
    hv_df.dropna(axis=1, how="all", inplace=True)

    def transpose_block(df):
        times = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        wavelengths = pd.to_numeric(df.columns[1:], errors="coerce").values
        values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
        out = pd.DataFrame(values.T, index=wavelengths, columns=times).reset_index()
        out.rename(columns={"index": "Wavelength"}, inplace=True)
        out = out.dropna(subset=["Wavelength"]).sort_values("Wavelength").reset_index(drop=True)
        return out

    cd_t = transpose_block(cd_df)
    hv_t = transpose_block(hv_df)

    common_wl = np.intersect1d(cd_t["Wavelength"].values, hv_t["Wavelength"].values)
    cd_t = cd_t[cd_t["Wavelength"].isin(common_wl)].reset_index(drop=True)
    hv_t = hv_t[hv_t["Wavelength"].isin(common_wl)].reset_index(drop=True)
    return cd_t, hv_t


def process_spectral_kinetics(
    filepath: str,
    target_wl: float = 218.0,
    hv_threshold: float = 900.0,
    dead_time: float = 30.0,
    nm_per_sec: float = 0.4,
    baseline_wl: float = 250.0,
    smooth_window: int = 15,
    smooth_poly: int = 3,
):
    """Zwraca przebieg przy target_wl (w godzinach) oraz pełne wygładzone widma dla panelu C."""
    cd_df, hv_df = read_transposed_spectra_and_hv(filepath)
    wavelengths = pd.to_numeric(cd_df["Wavelength"]).values
    smooth_w = safe_make_odd(smooth_window)

    target_idx = np.argmin(np.abs(wavelengths - target_wl))
    actual_target_wl = wavelengths[target_idx]
    extra_dead = (baseline_wl - actual_target_wl) / nm_per_sec
    eff_dead_time = dead_time + extra_dead

    cd_cols = {float(c): c for c in cd_df.columns if c != "Wavelength" and not str(c).startswith("Unnamed")}
    hv_cols = {float(c): c for c in hv_df.columns if c != "Wavelength" and not str(c).startswith("Unnamed")}

    cd_times = sorted(cd_cols.keys())
    hv_times = sorted(hv_cols.keys())

    times_hr = []
    vals_218 = []
    spectra_list = []

    for t_sec in cd_times:
        hv_t = min(hv_times, key=lambda t: abs(t - t_sec))
        cd_vec = pd.to_numeric(cd_df[cd_cols[t_sec]], errors="coerce").values
        hv_vec = pd.to_numeric(hv_df[hv_cols[hv_t]], errors="coerce").values[: len(wavelengths)]

        hv_mask = hv_vec <= hv_threshold
        wl_valid = wavelengths[hv_mask]
        cd_valid = cd_vec[hv_mask]

        if len(wl_valid) == 0:
            continue

        if len(cd_valid) >= smooth_w:
            cd_smooth = savgol_filter(cd_valid, window_length=smooth_w, polyorder=smooth_poly)
        else:
            cd_smooth = cd_valid

        # Wartość przy 218 nm (Panel B)
        idx_local = np.argmin(np.abs(wl_valid - actual_target_wl))
        if np.abs(wl_valid[idx_local] - actual_target_wl) <= 1.5:
            val = cd_smooth[idx_local]
            if not np.isnan(val):
                times_hr.append((t_sec + eff_dead_time) / 3600.0)
                vals_218.append(val)

        # Pełne widmo do Panelu C
        t_hr_spec = (t_sec + dead_time) / 3600.0
        spectra_list.append((t_hr_spec, wl_valid, cd_smooth))

    return np.array(times_hr), np.array(vals_218), spectra_list


def style_axes(ax, panel_label, xlabel, ylabel, xlim=None, ylim=None, label_pos=(0.05, 0.95)):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=6)
    ax.tick_params(axis="both", labelsize=12.5, width=1.1, length=4.5)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.text(
        label_pos[0], label_pos[1], panel_label,
        transform=ax.transAxes,
        fontsize=15.5, fontweight="bold",
        va="top", ha="left"
    )


# =========================================================
# 3. GENEROWANIE ZBIORCZEJ FIGURY 3-PANELOWEJ (A, B, C)
# =========================================================
def build_figure_6_3_cd():
    fig = plt.figure(figsize=(12.2, 9.6))
    gs = GridSpec(
        2, 4,
        figure=fig,
        wspace=0.45,
        hspace=0.34,
        top=0.96, bottom=0.08, left=0.08, right=0.95
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 1:3])

    trace_color = "#3D48A4"

    # -----------------------------------------------------
    # PANEL A: Manual-Mixing CD przy 218 nm (0 - 2000 s)
    # -----------------------------------------------------
    dead_times_dict = read_dead_times(os.path.join(DIR_PANEL_A, "dead_times.txt"))
    files_a = sorted(glob.glob(os.path.join(DIR_PANEL_A, "*.csv")))

    for fpath in files_a:
        try:
            df = read_single_wl_cd(fpath)
            base = os.path.splitext(os.path.basename(fpath))[0]
            dt = dead_times_dict.get(base, 0)
            t = df.iloc[:, 0].values + dt
            y = df.iloc[:, 1].values
            if len(y) >= 5:
                y = savgol_filter(y, window_length=5, polyorder=3)
            ax_a.plot(t, y, color=trace_color, linewidth=1.35)
        except Exception as e:
            print(f"[Panel A] Pominięto {os.path.basename(fpath)}: {e}")

    style_axes(
        ax_a, panel_label="A",
        xlabel="Time (s)",
        ylabel="Ellipticity at 218 nm (mdeg)",
        xlim=(0, 2000),
        ylim=(Y_MIN_ALL, Y_MAX_AB),
        label_pos=(0.05, 0.95)
    )

    # -----------------------------------------------------
    # PANEL B i C: Spektralna kinetyka CD (0 - 4 h)
    # -----------------------------------------------------
    files_bc = sorted(glob.glob(os.path.join(DIR_PANEL_BC, "*.csv")))
    spectra_for_panel_c = None

    for fpath in files_bc:
        try:
            t_hr, vals_218, spec_list = process_spectral_kinetics(
                fpath,
                target_wl=218.0,
                hv_threshold=900.0,
                dead_time=30.0,
                nm_per_sec=0.4,
                baseline_wl=250.0,
                smooth_window=15,
                smooth_poly=3,
            )
            if len(t_hr) > 0:
                ax_b.plot(t_hr, vals_218, color=trace_color, linewidth=1.4)

            if PANEL_C_SPECTRA_FILE is not None:
                if os.path.basename(fpath) == PANEL_C_SPECTRA_FILE:
                    spectra_for_panel_c = spec_list
            elif spectra_for_panel_c is None and len(spec_list) > 0:
                spectra_for_panel_c = spec_list

        except Exception as e:
            print(f"[Panel B/C] Pominięto {os.path.basename(fpath)}: {e}")

    style_axes(
        ax_b, panel_label="B",
        xlabel="Time (h)",
        ylabel="Ellipticity at 218 nm (mdeg)",
        xlim=(-0.15, 4.1),
        ylim=(Y_MIN_ALL, Y_MAX_AB),
        label_pos=(0.05, 0.95)
    )
    ax_b.set_xticks([0, 1, 2, 3, 4])

    # -----------------------------------------------------
    # PANEL C: Ewolucja widm CD (214 - 250 nm, 0 - 4 h)
    # -----------------------------------------------------
    if spectra_for_panel_c:
        norm = Normalize(vmin=0.0, vmax=4.0)
        cmap = mpl.colormaps["jet"]

        for t_hr, wl, cd_vals in spectra_for_panel_c:
            mask = (wl >= 214.0) & (wl <= 250.0)
            ax_c.plot(wl[mask], cd_vals[mask], color=cmap(norm(np.clip(t_hr, 0.0, 4.0))), linewidth=1.25)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = ax_c.inset_axes([0.92, 0.08, 0.032, 0.34])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([0.0, 4.0])
        cbar.set_ticklabels(["0 h", "4 h"])
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.tick_params(labelsize=10.5, length=3, pad=3)
        cbar.outline.set_linewidth(0.8)

    style_axes(
        ax_c, panel_label="C",
        xlabel="Wavelength (nm)",
        ylabel="Ellipticity (mdeg)",
        xlim=(214, 250),
        ylim=(Y_MIN_ALL, Y_MAX_C),
        label_pos=(0.05, 0.95)
    )
    ax_c.set_xticks([215, 220, 225, 230, 235, 240, 245, 250])

    # -----------------------------------------------------
    # ZAPIS DO PLIKÓW PDF I PNG (300 DPI)
    # -----------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.pdf")
    png_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    print(f"Zapisano:\n  - {pdf_path}\n  - {png_path}")
    plt.show()


if __name__ == "__main__":
    build_figure_6_3_cd()