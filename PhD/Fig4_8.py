import os
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import numpy as np

# Inicjalizacja biblioteki PyMOL w trybie headless
import pymol

pymol.finish_launching(['pymol', '-cq'])
from pymol import cmd

# ==============================================================================
# 1. Konfiguracja Ścieżek i Typografii
# ==============================================================================
OUTPUT_DIR = "figures/ch04"
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Figura_4_8_Structural_Basis_Trp_Tyr.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Figura_4_8_Structural_Basis_Trp_Tyr.png")

TEMP_IMG_A = "temp_panel_A.png"
TEMP_IMG_B = "temp_panel_B.png"
TEMP_IMG_C = "temp_panel_C.png"
TEMP_IMG_D = "temp_panel_D.png"

AF_LOCAL_CANDIDATES = [
    "data/AF-F8E4N3-F1-model_v6.pdb",
    "AF-F8E4N3-F1-model_v6.pdb",
    "models/F8E4N.pdb",
    "F8E4N.pdb",
]

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'DejaVu Sans'
mpl.rcParams['mathtext.it'] = 'DejaVu Sans:italic'
mpl.rcParams['mathtext.bf'] = 'DejaVu Sans:bold'
mpl.rcParams['mathtext.sf'] = 'DejaVu Sans:bold:italic'
mpl.rcParams['mathtext.cal'] = 'DejaVu Sans'


def setup_pymol_environment():
    cmd.reinitialize()
    cmd.set('bg_rgb', [1, 1, 1])
    cmd.set('ortho', 1)
    cmd.set('ray_shadows', 0)
    cmd.set('antialias', 2)
    cmd.set('ray_trace_mode', 1)
    cmd.set('ambient', 0.40)
    cmd.set('direct', 0.60)
    cmd.set('reflect', 0.15)
    cmd.set('stick_radius', 0.28)

    cmd.set('label_font_id', 7)
    cmd.set('label_size', -16)
    cmd.set('label_color', 'black')
    cmd.set('label_outline_color', 'white')
    cmd.set('label_shadow_mode', 0)


# ==============================================================================
# 2. Renderowanie Paneli w PyMOL
# ==============================================================================
def render_panel_a():
    """Panel A: Dimer krystaliczny Cn (PDB: 8B1N)."""
    setup_pymol_environment()

    cmd.fetch("8B1N", type="pdb")
    cmd.remove("not polymer")
    cmd.remove("resi -5-0")
    cmd.hide("everything")
    cmd.hide("lines")

    cmd.color("grey85", "chain B")
    cmd.color("marine", "chain A and resi 1-235")
    cmd.color("grey70", "chain A and resi 236-245")
    cmd.color("warmpink", "chain A and resi 246-455")

    cmd.show("cartoon", "all")

    cmd.select("iface_tyr", "(chain A or chain B) and resi 20+55")
    cmd.show("sticks", "iface_tyr")
    cmd.color("brightorange", "iface_tyr")
    cmd.show("surface", "iface_tyr")
    cmd.set("transparency", 0.45, "iface_tyr")

    cmd.orient("all")
    cmd.turn("y", 15)
    cmd.turn("x", 10)
    cmd.zoom("all", 0.95)
    cmd.move("y", 5)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_A)
    print("  -> Wygenerowano:", TEMP_IMG_A)


def render_panel_b():
    """Panel B: Monomer Cn - Podział domenowy z resztami Trp i Tyr jako licorice."""
    setup_pymol_environment()

    cmd.fetch("8B1N", type="pdb")
    cmd.remove("not polymer")
    cmd.remove("resi -5-0")
    cmd.remove("chain B")
    cmd.hide("everything")
    cmd.hide("lines")
    cmd.show("cartoon", "chain A")

    # Podział sekwencyjny: TrmD (1-235), Linker (236-245), Tm1570 (246-455)
    cmd.color("marine", "chain A and resi 1-235")
    cmd.color("grey70", "chain A and resi 236-245")
    cmd.color("warmpink", "chain A and resi 246-455")

    # Reszty aromatyczne (Trp i Tyr) jako licorice (sticks)
    cmd.select("arom_fl", "chain A and (resn TRP or resn TYR)")
    cmd.show("sticks", "arom_fl")
    cmd.set("stick_radius", 0.28, "arom_fl")

    # Pokolorowanie licorice według domen i linkera
    cmd.color("marine", "chain A and resi 1-235 and arom_fl")
    cmd.color("grey70", "chain A and resi 236-245 and arom_fl")
    cmd.color("warmpink", "chain A and resi 246-455 and arom_fl")

    cmd.orient("chain A")
    cmd.turn("z", -20)
    cmd.zoom("chain A", 0.95)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_B)
    print("  -> Wygenerowano Panel B (Domain architecture & Trp/Tyr):", TEMP_IMG_B)


def render_panel_c():
    """Panel C: Monomer Cn - Mapowanie Trp i Tyr z etykietami."""
    setup_pymol_environment()

    cmd.fetch("8B1N", type="pdb")
    cmd.remove("not polymer")
    cmd.remove("resi -5-0")
    cmd.remove("chain B")
    cmd.hide("everything")
    cmd.hide("lines")

    cmd.show("cartoon", "chain A")
    cmd.color("marine", "resi 1-235")
    cmd.color("grey70", "resi 236-245")
    cmd.color("warmpink", "resi 246-455")

    cmd.set_color("trp_forest_green", [0.10, 0.58, 0.20])

    cmd.select("trp_all", "chain A and resn TRP")
    cmd.show("sticks", "trp_all")
    cmd.color("trp_forest_green", "trp_all")

    cmd.select("tyr_all", "chain A and resn TYR and resi 4+20+42+55+69+114+250+290+420")
    cmd.show("sticks", "tyr_all")
    cmd.color("brightorange", "tyr_all")

    cmd.label("chain A and resn TRP and name NE1", '"Trp" + resi')
    cmd.set("label_position", [0, 0.75, 8.0], "chain A and resi 237 and name NE1")
    cmd.set("label_position", [0, -29, 8.0], "chain A and resi 310 and name NE1")
    cmd.set("label_position", [-12, 2.25, 8.0], "chain A and resi 206 and name NE1")
    cmd.set("label_position", [0.25, 17, 6.0], "chain A and resi 385 and name NE1")

    cmd.orient("chain A")
    cmd.turn("z", -20)
    cmd.zoom("chain A", 0.95)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_C)
    print("  -> Wygenerowano:", TEMP_IMG_C)


def render_panel_d():
    """Panel D: Monomer Fs (AlphaFold) - Mutacja Glu237."""
    setup_pymol_environment()

    cmd.fetch("8B1N", type="pdb", name="ref_cn")
    cmd.remove("ref_cn and (not polymer or chain B)")

    af_path = next((p for p in AF_LOCAL_CANDIDATES if os.path.exists(p)), None)
    if af_path:
        cmd.load(af_path, "Fs_model")
    else:
        try:
            cmd.fetch("AF-F8E4N5-F1", type="pdb", name="Fs_model")
        except Exception:
            import urllib.request
            url = "https://alphafold.ebi.ac.uk/files/AF-F8E4N3-F1-model_v6.pdb"
            urllib.request.urlretrieve(url, "temp_AF_Fs.pdb")
            cmd.load("temp_AF_Fs.pdb", "Fs_model")

    cmd.remove("not polymer")
    cmd.hide("everything")
    cmd.hide("lines")

    cmd.align("Fs_model and resi 1-230", "ref_cn and resi 1-230")
    cmd.delete("ref_cn")

    cmd.show("cartoon", "Fs_model")

    cmd.set_color("teal_trmd", [0.09, 0.74, 0.81])
    cmd.set_color("teal_linker", [0.75, 0.90, 0.92])
    cmd.set_color("teal_tm", [0.06, 0.52, 0.57])

    cmd.color("teal_trmd", "Fs_model and resi 1-235")
    cmd.color("teal_linker", "Fs_model and resi 236-246")
    cmd.color("teal_tm", "Fs_model and resi 247-434")

    cmd.set_color("trp_forest_green", [0.10, 0.58, 0.20])

    cmd.select("fs_trp", "Fs_model and resn TRP and resi 206+310+386")
    cmd.show("sticks", "fs_trp")
    cmd.color("trp_forest_green", "fs_trp")
    cmd.label("fs_trp and name NE1", '"Trp" + resi')

    cmd.select("fs_glu237", "Fs_model and resi 237")
    cmd.show("sticks", "fs_glu237")
    cmd.color("firebrick", "fs_glu237")
    cmd.label("fs_glu237 and name OE1", '"Glu237\\n(ex-Trp)"')

    cmd.set("label_position", [-20.5, 1, 9.0], "Fs_model and resi 206 and name NE1")
    cmd.set("label_position", [0.75, -19, 8.0], "fs_glu237 and name OE1")
    cmd.set("label_position", [-9.5, 19, 8.0], "Fs_model and resi 310 and name NE1")
    cmd.set("label_position", [0, 8, 6.0], "Fs_model and resi 386 and name NE1")

    cmd.orient("Fs_model")
    cmd.turn("z", -20)
    cmd.zoom("Fs_model", 0.95)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_D)
    print("  -> Wygenerowano:", TEMP_IMG_D)


# ==============================================================================
# 3. Montaż Figury
# ==============================================================================
def autocrop_border(img_array, threshold=252, pad=2):
    rgb = img_array[:, :, :3]
    if rgb.max() <= 1.0:
        mask = np.any(rgb < (threshold / 255.0), axis=2)
    else:
        mask = np.any(rgb < threshold, axis=2)

    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return img_array

    y_min = max(0, y_indices.min() - pad)
    y_max = min(img_array.shape[0], y_indices.max() + pad)
    x_min = max(0, x_indices.min() - pad)
    x_max = min(img_array.shape[1], x_indices.max() + pad)

    return img_array[y_min:y_max, x_min:x_max]


def assemble_figure():
    print("Składanie zacieśnionej Figury 4.8...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_a = autocrop_border(mpimg.imread(TEMP_IMG_A), pad=2)
    img_b = autocrop_border(mpimg.imread(TEMP_IMG_B), pad=2)
    img_c = autocrop_border(mpimg.imread(TEMP_IMG_C), pad=2)
    img_d = autocrop_border(mpimg.imread(TEMP_IMG_D), pad=2)

    h_row1 = max(img_a.shape[0], img_b.shape[0])
    h_row2 = max(img_c.shape[0], img_d.shape[0])
    w_col1 = max(img_a.shape[1], img_c.shape[1])
    w_col2 = max(img_b.shape[1], img_d.shape[1])

    total_w = w_col1 + w_col2
    total_h = h_row1 + h_row2

    fig_width = 12.0
    fig_height = fig_width * (total_h / total_w) + 0.8

    fig, axes = plt.subplots(
        2, 2,
        figsize=(fig_width, fig_height),
        gridspec_kw={
            'width_ratios': [w_col1, w_col2],
            'height_ratios': [h_row1, h_row2]
        },
        dpi=300
    )

    plt.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.94, wspace=-0.25, hspace=0.03)

    panel_configs = [
        {"ax": axes[0, 0], "img": img_a, "tag": "A",
         "title": r"$\mathsf{Cn}$" + "TrmD–Tm1570 Crystalline Dimer (8B1N)\nDimerization Interface (Tyr20/Tyr55)"},
        {"ax": axes[0, 1], "img": img_b, "tag": "B",
         "title": r"$\mathsf{Cn}$" + "TrmD–Tm1570 Monomer (Chain A)\nDomain Architecture (TrmD / Linker / Tm1570)"},
        {"ax": axes[1, 0], "img": img_c, "tag": "C",
         "title": r"$\mathsf{Cn}$" + "TrmD–Tm1570 Monomer (Chain A)\nTrp and Tyr Fluorophore Distribution"},
        {"ax": axes[1, 1], "img": img_d, "tag": "D",
         "title": r"$\mathsf{Fs}$" + "TrmD–Tm1570 Monomer (AlphaFold)\nLinker Substitution Glu237 (ex-Trp)"},
    ]

    for p in panel_configs:
        ax = p["ax"]
        ax.imshow(p["img"], aspect='equal')
        ax.set_anchor("N")
        ax.axis("off")

        ax.text(
            0.12, 0.98, p["tag"],
            transform=ax.transAxes,
            fontsize=16, fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.90)
        )
        ax.set_title(p["title"], fontsize=11, pad=6, fontweight='bold')

    # Legenda podziału domenowego w prawym górnym rogu Panelu B (axes[0, 1])
    domain_elements = [
        Line2D([0], [0], color='#1F77B4', lw=4, label=r'$\mathit{Cn}$TrmD (1–235)'),
        Line2D([0], [0], color='#B0B0B0', lw=4, label='Linker (236–245)'),
        Line2D([0], [0], color='#D92687', lw=4, label=r'$\mathit{Cn}$Tm1570 (246–455)')
    ]
    axes[0, 1].legend(
        handles=domain_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.97),
        frameon=True,
        facecolor="white",
        edgecolor="grey",
        framealpha=0.92,
        fontsize=8.5,
        handlelength=1.5
    )

    TEMP_ASSEMBLED = "temp_assembled_full.png"
    fig.savefig(TEMP_ASSEMBLED, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

    final_img = mpimg.imread(TEMP_ASSEMBLED)
    final_cropped = autocrop_border(final_img, threshold=252, pad=1)

    plt.imsave(OUTPUT_PNG, final_cropped)

    h_px, w_px = final_cropped.shape[:2]
    fig_pdf = plt.figure(figsize=(w_px / 300.0, h_px / 300.0), dpi=300)
    ax_pdf = fig_pdf.add_axes([0, 0, 1, 1])
    ax_pdf.imshow(final_cropped)
    ax_pdf.axis("off")
    fig_pdf.savefig(OUTPUT_PDF, format='pdf', pad_inches=0.0)
    plt.close(fig_pdf)

    print(f"\n[OK] Zapisano poprawioną Figurę 4.8:\n  -> {OUTPUT_PDF}\n  -> {OUTPUT_PNG}")

    for tmp in [TEMP_IMG_A, TEMP_IMG_B, TEMP_IMG_C, TEMP_IMG_D, TEMP_ASSEMBLED, "temp_AF_Fs.pdb"]:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    render_panel_a()
    render_panel_b()
    render_panel_c()
    render_panel_d()
    assemble_figure()