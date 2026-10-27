import os
import re
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.signal import savgol_filter

# ==============================================================================
# 1. Konfiguracja Typografii i Projektowej Palety Barw
# ==============================================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"

# Obcięta mapa plasma (0.05 - 0.85) eliminująca blade, żółte linie na białym tle
plasma_full = plt.cm.plasma
plasma_trimmed = LinearSegmentedColormap.from_list(
    "plasma_trimmed",
    [plasma_full(x) for x in np.linspace(0.05, 0.85, 256)]
)

# Ścieżki bazowe na dysku
BASE_CD_DIR = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/biofizyka_CD"
BASE_FLUO_DIR = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw/equilibrium/fluorimetry"
OUTPUT_DIR = "/home/matifortunka/Downloads/PhD/rozdział_IV/appendix"

HV_CUTOFF = 900.0
PATH_LENGTH_MM = 1.0

# ==============================================================================
# 2. Definicja Białek i Warunków Eksperymentalnych (Siatka 2 x 4)
# ==============================================================================
PROTEINS = [
    {
        "id": "TrmD",
        "name": r"$\boldsymbol{Cn}$" + r"$\bf{TrmD}$",
        "cd_subfolder": "trmd",
        "fluo_subfolder": "TrmD",
        "residues": 262,
        "fig_file": "Fig_S4_1_TrmD_spectra",
    },
    {
        "id": "Tm1570",
        "name": r"$\boldsymbol{Cn}$" + r"$\bf{Tm1570}$",
        "cd_subfolder": "tm1570",
        "fluo_subfolder": "Tm1570",
        "residues": 224,
        "fig_file": "Fig_S4_2_Tm1570_spectra",
    },
    {
        "id": "Fusion",
        "name": r"$\bf{Fuzja\ }$" + r"$\boldsymbol{Cn}$" + r"$\bf{TrmD–Tm1570}$",
        "cd_subfolder": "fuzja",
        "fluo_subfolder": "Fuzja",
        "residues": 455,
        "fig_file": "Fig_S4_3_Fusion_spectra",
    },
]

# 4 brakujące kolumny pomiarowe
SUPPLEMENTARY_CONDITIONS = [
    {
        "label": r"2 $\mathsf{\mu}$M, 36 h",
        "conc_uM": 2.0,
        "cd_token": "3/2uM/29_07_26",
        "fluo_token": "3/2uM/29.07.26/csv"
    },
    {
        "label": r"2 $\mathsf{\mu}$M, 84 h",
        "conc_uM": 2.0,
        "cd_token": "3/2uM/31_07_26",
        "fluo_token": "3/2uM/31.07.26/csv"
    },
    {
        "label": r"3 $\mathsf{\mu}$M, 84 h",
        "conc_uM": 3.0,
        "cd_token": "3/3uM/31_07_26",
        "fluo_token": "3/3uM/31.07.26/csv"
    },
    {
        "label": r"5 $\mathsf{\mu}$M, 84 h",
        "conc_uM": 5.0,
        "cd_token": "seria2_1",
        "fluo_token": "seria2_1/csv"
    },
]

# ==============================================================================
# 3. Klasa Wczytująca i Przeliczająca Dane Spektroskopowe
# ==============================================================================
class SupplementaryDataLoader:
    @staticmethod
    def _load_conc_file(folder_path):
        conc_path = os.path.join(folder_path, "concentrations.txt")
        if not os.path.exists(conc_path):
            return {}
        with open(conc_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().replace(",", ".")
        try:
            df = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")
            df["Sample_number"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            df["den_concentration"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            df.dropna(subset=["Sample_number", "den_concentration"], inplace=True)
            return dict(zip(df["Sample_number"].astype(int), df["den_concentration"].astype(float)))
        except Exception:
            return {}

    @staticmethod
    def _resolve_dir(base_dir, subfolder, token):
        # Sprawdza warianty z kropkami i podkreśleniami w datach
        candidates = [
            os.path.join(base_dir, subfolder, token),
            os.path.join(base_dir, subfolder, token.replace("_", ".")),
            os.path.join(base_dir, subfolder, token.replace(".", "_")),
            os.path.join(base_dir, subfolder, token.replace("/csv", "")),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def get_cd_spectra(self, protein_cfg, cond):
        """Zwraca słownik: {conc_Gdm: (wavelengths, mre_values)}."""
        folder = self._resolve_dir(BASE_CD_DIR, protein_cfg["cd_subfolder"], cond["cd_token"])
        conc_map = self._load_conc_file(folder)
        if not conc_map:
            print(f"  [CD] Brak stężeń w: {folder}")
            return {}

        pattern = re.compile(r"(\d{5})\.csv$")
        results = {}

        for fname in sorted(os.listdir(folder)):
            m = pattern.search(fname)
            if not m:
                continue
            sn = int(m.group(1))
            conc = conc_map.get(sn, None)
            if conc is None or np.isnan(conc):
                continue

            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                cd_idx = next((i for i in range(len(lines) - 1)
                               if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("CircularDichroism")), None)
                if cd_idx is None:
                    continue

                cd_lines = []
                for ln in lines[cd_idx + 2:]:
                    if not ln.strip() or ln.strip().startswith("Wavelength"):
                        break
                    cd_lines.append(list(map(float, ln.replace(",", " ").split())))
                cd_arr = np.array(cd_lines)
                wls, mdeg = cd_arr[:, 0], cd_arr[:, 1]

                # Filtr HV
                hv_idx = next((i for i in range(len(lines) - 1)
                               if lines[i].strip().startswith("Wavelength") and lines[i + 1].strip().startswith("HV")), None)
                if hv_idx is not None:
                    hv_lines = []
                    for ln in lines[hv_idx + 2:]:
                        if not ln.strip() or ln.strip().startswith("Wavelength"):
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

                if len(mdeg) >= 11:
                    mdeg = savgol_filter(mdeg, 11, 3)

                # Przeliczenie mdeg -> [theta]_MRW
                mre = (mdeg * 1e6) / (cond["conc_uM"] * PATH_LENGTH_MM * protein_cfg["residues"])
                results[float(conc)] = (wls, mre)
            except Exception:
                continue

        return results

    def get_fl_spectra(self, protein_cfg, cond):
        """Zwraca słownik: {conc_Gdm: (wavelengths, intensities)}."""
        folder = self._resolve_dir(BASE_FLUO_DIR, protein_cfg["fluo_subfolder"], cond["fluo_token"])
        conc_map = self._load_conc_file(folder)
        if not conc_map:
            # Próba odczytu z folderu nadrzędnego jeśli brak pliku w podfolderze /csv
            parent_folder = os.path.dirname(folder)
            conc_map = self._load_conc_file(parent_folder)

        if not conc_map:
            print(f"  [FLUO] Brak stężeń w: {folder}")
            return {}

        results = {}
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".csv"):
                continue
            matches = re.findall(r"(\d+)", fname)
            if not matches:
                continue
            sn = int(matches[-1])
            conc = conc_map.get(sn, None)
            if conc is None or np.isnan(conc):
                continue

            fpath = os.path.join(folder, fname)
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

                # Korekta tła przy 400 nm
                b_idx = (df["Wavelength"] - 400.0).abs().idxmin()
                df["Intensity"] -= df.loc[b_idx, "Intensity"]

                results[float(conc)] = (df["Wavelength"].values, df["Intensity"].values)
            except Exception:
                continue

        return results

# ==============================================================================
# 4. Generator Figur Suplementarnych (2 x 4)
# ==============================================================================
def generate_supplementary_figure(protein_cfg, loader):
    print(f"\nGenerowanie figury dla: {protein_cfg['id']}...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.0), constrained_layout=True, dpi=300)

    for col_idx, cond in enumerate(SUPPLEMENTARY_CONDITIONS):
        ax_cd = axes[0, col_idx]
        ax_fl = axes[1, col_idx]

        # ----------------------------------------------------------------------
        # WIERSZ 1: Widma Far-UV CD
        # ----------------------------------------------------------------------
        cd_series = loader.get_cd_spectra(protein_cfg, cond)
        concs_cd = sorted(cd_series.keys())

        if concs_cd:
            norm_cd = Normalize(vmin=min(concs_cd), vmax=max(concs_cd))
            for c in concs_cd:
                wl, theta = cd_series[c]
                mask = (wl >= 210.0) & (wl <= 250.0)
                ax_cd.plot(wl[mask], theta[mask], color=plasma_trimmed(norm_cd(c)), lw=1.35)

            # Miniaturowy, stabilny pionowy pasek barwny w prawej kolumnie
            if col_idx == 3:
                cax_cd = ax_cd.inset_axes([0.78, 0.08, 0.04, 0.40])
                sm_cd = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_cd)
                sm_cd.set_array([])
                cbar_cd = fig.colorbar(sm_cd, cax=cax_cd, orientation="vertical")
                cbar_cd.ax.tick_params(labelsize=8)
                cbar_cd.set_label("[GdmCl] (M)", fontsize=8, labelpad=4)

        ax_cd.set_xlim(210.0, 250.0)
        ax_cd.set_xlabel("Wavelength (nm)", fontsize=11)
        ax_cd.tick_params(labelsize=10)
        ax_cd.grid(True, linestyle=":", alpha=0.5)
        ax_cd.set_title(f"{cond['label']} – CD", fontsize=12, fontweight="bold", pad=8)

        if col_idx == 0:
            ax_cd.set_ylabel(r"$[\theta]_{\mathrm{MRW}}\ \mathrm{(deg\cdot cm^2\cdot dmol^{-1})}$", fontsize=11)

        # ----------------------------------------------------------------------
        # WIERSZ 2: Widma Emisji Fluorescencji
        # ----------------------------------------------------------------------
        fl_series = loader.get_fl_spectra(protein_cfg, cond)
        concs_fl = sorted(fl_series.keys())

        if concs_fl:
            norm_fl = Normalize(vmin=min(concs_fl), vmax=max(concs_fl))
            for c in concs_fl:
                wl, intens = fl_series[c]
                mask = (wl >= 300.0) & (wl <= 400.0)
                ax_fl.plot(wl[mask], intens[mask], color=plasma_trimmed(norm_fl(c)), lw=1.35)

            # Miniaturowy pasek barwny w prawej kolumnie (prawy górny róg)
            if col_idx == 3:
                cax_fl = ax_fl.inset_axes([0.78, 0.52, 0.04, 0.40])
                sm_fl = plt.cm.ScalarMappable(cmap=plasma_trimmed, norm=norm_fl)
                sm_fl.set_array([])
                cbar_fl = fig.colorbar(sm_fl, cax=cax_fl, orientation="vertical")
                cbar_fl.ax.tick_params(labelsize=8)
                cbar_fl.set_label("[GdmCl] (M)", fontsize=8, labelpad=4)

        ax_fl.set_xlim(300.0, 400.0)
        ax_fl.set_xlabel("Wavelength (nm)", fontsize=11)
        ax_fl.tick_params(labelsize=10)
        ax_fl.grid(True, linestyle=":", alpha=0.5)
        ax_fl.set_title(f"{cond['label']} – Fluo", fontsize=12, fontweight="bold", pad=8)

        if col_idx == 0:
            ax_fl.set_ylabel("Fluorescence Intensity (a.u.)", fontsize=11)

    # Nagłówek nadrzędny dla całej figury
    fig.suptitle(f"{protein_cfg['name']} – Supplementary Equilibrium Spectra", fontsize=14, fontweight="bold", y=1.02)

    # Zapis w formatach wektorowym PDF i rastrowym 300 DPI PNG
    out_pdf = os.path.join(OUTPUT_DIR, f"{protein_cfg['fig_file']}.pdf")
    out_png = os.path.join(OUTPUT_DIR, f"{protein_cfg['fig_file']}.png")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Zapisano:\n  -> {out_pdf}\n  -> {out_png}")

# ==============================================================================
# 5. Wykonanie Skryptu
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loader = SupplementaryDataLoader()

    for p_cfg in PROTEINS:
        generate_supplementary_figure(p_cfg, loader)

    print(f"\n[✓] Wszystkie figury do Appendixu zostały pomyślnie wygenerowane w folderze:\n    {OUTPUT_DIR}")