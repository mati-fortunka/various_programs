import os
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Inicjalizacja biblioteki PyMOL w trybie bezokienkowym (headless)
import pymol
pymol.finish_launching(["pymol", "-cq"])
from pymol import cmd

# ==============================================================================
# 1. Konfiguracja Ścieżek, Danych i Typografii
# ==============================================================================
OUTPUT_DIR = "figures/ch04"
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Figura_4_8_Structural_Basis_Trp_Tyr.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Figura_4_8_Structural_Basis_Trp_Tyr.png")

TEMP_IMG_A = "temp_panel_A.png"
TEMP_IMG_B = "temp_panel_B.png"
TEMP_IMG_C = "temp_panel_C.png"

# Ścieżka do lokalnego pliku modelu AlphaFold dla Fs (UniProt: F8E4N3)
# W przypadku braku pliku lokalnego skrypt podejmie próbę pobrania z AlphaFold DB
AF_LOCAL_CANDIDATES = [
    "data/AF-F8E4N3-F1-model_v6.pdb",
    "AF-F8E4N3-F1-model_v6.pdb",
    "models/F8E4N.pdb",
    "F8E4N.pdb"
]

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"
mpl.rcParams["mathtext.cal"] = "DejaVu Sans"


# ==============================================================================
# 2. Etap 1: Renderowanie Paneli w Silniku PyMOL
# ==============================================================================
def setup_pymol_environment():
    cmd.reinitialize()
    cmd.set("bg_rgb", [1, 1, 1])
    cmd.set("ortho", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ambient", 0.35)
    cmd.set("direct", 0.65)
    cmd.set("reflect", 0.15)
    cmd.set("stick_radius", 0.25)
    cmd.set("label_size", 14)
    cmd.set("label_color", "black")
    cmd.set("label_font_id", 7)
    cmd.set("label_position", [1.0, 1.0, 1.5])


def render_panel_a():
    """Panel A: Dimer krystaliczny Cn (PDB: 8B1N) z interfejsem dimeryzacji."""
    cmd.reinitialize()
    setup_pymol_environment()

    cmd.fetch("8B1N", type="pdb")
    cmd.remove("solvent")
    cmd.remove("inorganic")

    # Kolorystyka łańcucha B (tło)
    cmd.color("grey85", "chain B")

    # Kolorystyka domenowa łańcucha A
    cmd.color("marine", "chain A and resi 1-235")       # CnTrmD (#1F77B4)
    cmd.color("grey80", "chain A and resi 236-245")     # Łącznik
    cmd.color("warmpink", "chain A and resi 246-455")   # CnTm1570 (#FF7F0E)

    cmd.show("cartoon", "all")

    # Wyeksponowanie reszt Tyr20 i Tyr55 na interfejsie obu łańcuchów
    cmd.select("interface_tyr", "(chain A or chain B) and resi 20+55")
    cmd.show("sticks", "interface_tyr")
    cmd.color("brightorange", "interface_tyr")
    cmd.show("surface", "interface_tyr")
    cmd.set("transparency", 0.45, "interface_tyr")

    # Orientacja wzdłuż osi symetrii C2 dimeru
    cmd.orient("chain A or chain B")
    cmd.turn("y", 15)
    cmd.turn("x", 10)
    cmd.zoom("all", 2.0)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_A)
    print("  -> Wygenerowano:", TEMP_IMG_A)


def render_panel_b():
    """Panel B: Monomer Cn (łańcuch A z 8B1N) z mapowaniem Trp i Tyr."""
    cmd.reinitialize()
    setup_pymol_environment()

    cmd.fetch("8B1N", type="pdb")
    cmd.remove("solvent")
    cmd.remove("inorganic")
    cmd.remove("chain B")

    cmd.show("cartoon", "chain A")
    cmd.color("marine", "resi 1-235")       # TrmD
    cmd.color("lightpink", "resi 236-245")   # Łącznik
    cmd.color("warmpink", "resi 246-455")    # Tm1570

    # Tryptofany jako sticks (deepteal)
    cmd.select("trp_all", "chain A and resn TRP")
    cmd.show("sticks", "trp_all")
    cmd.color("deepteal", "trp_all")

    # Etykiety dla tryptofanów
    cmd.label("chain A and resn TRP and name CA", '"Trp" + resi')

    # Tyrozyny interfejsowe i rdzeniowe
    cmd.select("tyr_interface", "chain A and (resi 20 or resi 55)")
    cmd.show("sticks", "tyr_interface")
    cmd.color("brightorange", "tyr_interface")

    cmd.select("tyr_core", "chain A and (resi 4 or resi 42 or resi 250 or resi 290 or resi 420)")
    cmd.show("sticks", "tyr_core")
    cmd.color("yelloworange", "tyr_core")

    cmd.orient("chain A")
    cmd.turn("z", -20)
    cmd.zoom("chain A", 2.0)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_B)
    print("  -> Wygenerowano:", TEMP_IMG_B)


def render_panel_c():
    """Panel C: Monomer Fs (model AlphaFold) z zaznaczeniem podstawienia Glu237."""
    cmd.reinitialize()
    setup_pymol_environment()

    # Wczytanie modelu z pliku lokalnego lub AlphaFold DB
    af_path = next((p for p in AF_LOCAL_CANDIDATES if os.path.exists(p)), None)
    if af_path:
        cmd.load(af_path, "Fs_model")
    else:
        try:
            cmd.fetch("AF-F8E4N5-F1", type="pdb", name="Fs_model")
        except Exception:
            # Rezerwowe pobranie z bazy AlphaFold przez URL
            import urllib.request
            url = "https://alphafold.ebi.ac.uk/files/AF-F8E4N5-F1-model_v4.pdb"
            urllib.request.urlretrieve(url, "temp_AF_Fs.pdb")
            cmd.load("temp_AF_Fs.pdb", "Fs_model")

    cmd.remove("solvent")
    cmd.show("cartoon", "Fs_model")

    # Zdefiniowanie koloru teal dla Fs (#17BECF)
    cmd.set_color("teal_fs", [0.09, 0.74, 0.81])
    cmd.color("teal_fs", "Fs_model")

    # Tryptofany: Trp206, Trp310, Trp386
    cmd.select("fs_trp", "Fs_model and resn TRP and (resi 206 or resi 310 or resi 386)")
    cmd.show("sticks", "fs_trp")
    cmd.color("deepteal", "fs_trp")
    cmd.label("fs_trp and name CA", '"Trp" + resi')

    # Podstawienie w łączniku: Glu237 (ex-Trp)
    cmd.select("fs_glu237", "Fs_model and resi 237")
    cmd.show("sticks", "fs_glu237")
    cmd.color("firebrick", "fs_glu237")
    cmd.label("fs_glu237 and name CA", '"Glu237 (ex-Trp)"')

    cmd.orient("Fs_model")
    cmd.turn("z", -20)
    cmd.zoom("Fs_model", 2.0)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_C)
    print("  -> Wygenerowano:", TEMP_IMG_C)


# ==============================================================================
# 3. Etap 2: Montaż Figury 4.8 w Matplotlib (1 x 3)
# ==============================================================================
def assemble_figure():
    print("Składanie publikacyjnej Figury 4.8 w Matplotlib...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_a = mpimg.imread(TEMP_IMG_A)
    img_b = mpimg.imread(TEMP_IMG_B)
    img_c = mpimg.imread(TEMP_IMG_C)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True, dpi=300)

    panel_configs = [
        {
            "ax": axes[0],
            "img": img_a,
            "tag": "A",
            "title": r"$\mathsf{Cn}$" + "TrmD–Tm1570 Crystalline Dimer (8B1N)\nDimerization Interface (Tyr20/Tyr55)"
        },
        {
            "ax": axes[1],
            "img": img_b,
            "tag": "B",
            "title": r"$\mathsf{Cn}$" + "TrmD–Tm1570 Monomer (Chain A)\nTrp and Tyr Fluorophore Distribution"
        },
        {
            "ax": axes[2],
            "img": img_c,
            "tag": "C",
            "title": r"$\mathsf{Fs}$" + "TrmD–Tm1570 Monomer (AlphaFold)\nLinker Substitution Glu237 (ex-Trp)"
        }
    ]

    for p in panel_configs:
        ax = p["ax"]
        ax.imshow(p["img"])
        ax.axis("off")

        # Pogrubione oznaczenie literowe panelu
        ax.text(
            0.02, 0.98, p["tag"],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
        )

        # Tytuł panelu z pogrubioną typografią
        ax.set_title(p["title"], fontsize=11.5, pad=10)

    # Zapis w formacie wektorowym PDF oraz rastrowym PNG
    fig.savefig(OUTPUT_PDF, format="pdf")
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)

    print(f"\n[✓] Gotowa Figura 4.8 została zapisana do:\n  -> {OUTPUT_PDF}\n  -> {OUTPUT_PNG}")

    # Porządkowanie plików tymczasowych
    for tmp in [TEMP_IMG_A, TEMP_IMG_B, TEMP_IMG_C, "temp_AF_Fs.pdb"]:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    render_panel_a()
    render_panel_b()
    render_panel_c()
    assemble_figure()