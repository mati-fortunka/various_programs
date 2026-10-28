import os
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Inicjalizacja biblioteki PyMOL w trybie headless
import pymol

pymol.finish_launching(["pymol", "-cq"])
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


def setup_pymol_environment():
    cmd.reinitialize()
    cmd.set("bg_rgb", [1, 1, 1])
    cmd.set("ortho", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ambient", 0.40)
    cmd.set("direct", 0.60)
    cmd.set("reflect", 0.15)
    cmd.set("stick_radius", 0.28)

    # Konfiguracja etykiet: wyrazny bialy kontur, brak cieni
    cmd.set("label_font_id", 7)  # Helvetica Bold
    cmd.set("label_size", -16)  # 16 punktow
    cmd.set("label_color", "black")
    cmd.set("label_outline_color", "white")  # Wyrazny bialy obrys tekstu
    cmd.set("label_shadow_mode", 0)


# ==============================================================================
# 2. Etap 1: Renderowanie Paneli w PyMOL
# ==============================================================================
def render_panel_a():
    """Panel A: Dimer krystaliczny Cn (PDB: 8B1N) z interfejsem dimeryzacji."""
    setup_pymol_environment()

    cmd.fetch("8B1N", type="pdb")
    cmd.remove("not polymer")
    cmd.remove("resi -5-0")
    cmd.hide("everything")
    cmd.hide("lines")

    cmd.color("grey85", "chain B")
    cmd.color("marine", "chain A and resi 1-235")  # TrmD (#1F77B4)
    cmd.color("grey70", "chain A and resi 236-245")  # Lacznik
    cmd.color("warmpink", "chain A and resi 246-455")  # Tm1570 (#FF7F0E)

    cmd.show("cartoon", "all")

    cmd.select("iface_tyr", "(chain A or chain B) and resi 20+55")
    cmd.show("sticks", "iface_tyr")
    cmd.color("brightorange", "iface_tyr")
    cmd.show("surface", "iface_tyr")
    cmd.set("transparency", 0.45, "iface_tyr")

    cmd.orient("all")
    cmd.turn("y", 15)
    cmd.turn("x", 10)
    cmd.zoom("all", 0.5)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_A)
    print("  -> Wygenerowano:", TEMP_IMG_A)


def render_panel_b():
    """Panel B: Monomer Cn (lancuch A z 8B1N) z mapowaniem Trp (ciemna zielen) i Tyr (pomarancz)."""
    setup_pymol_environment()

    cmd.fetch("8B1N", type="pdb")
    cmd.remove("not polymer")
    cmd.remove("resi -5-0")
    cmd.remove("chain B")
    cmd.hide("everything")
    cmd.hide("lines")

    cmd.show("cartoon", "chain A")
    cmd.color("marine", "resi 1-235")  # TrmD
    cmd.color("lightpink", "resi 236-245")  # Lacznik
    cmd.color("warmpink", "resi 246-455")  # Tm1570

    # Definicja ciemnej, nasyconej zieleni lesno-szmaragdowej dla Trp
    cmd.set_color("trp_forest_green", [0.10, 0.58, 0.20])

    # Tryptofany jako ciemnozielone sticks
    cmd.select("trp_all", "chain A and resn TRP")
    cmd.show("sticks", "trp_all")
    cmd.color("trp_forest_green", "trp_all")

    # Tyrozyny w jednolitym, wyrazistym pomaranczu
    cmd.select("tyr_all",
               "chain A and resn TYR and (resi 4 or resi 20 or resi 42 or resi 55 or resi 250 or resi 290 or resi 420)")
    cmd.show("sticks", "tyr_all")
    cmd.color("brightorange", "tyr_all")

    # Etykiety Trp przypisane do atomu azotu indolowego (NE1)
    cmd.label("chain A and resn TRP and name NE1", '"Trp" + resi')

    # Wysuniecie etykiet w biale, czyste tlo (duze Z zapobiega ucinaniu)
    cmd.set("label_position", [0, 0.75, 8.0], "chain A and resi 237 and name NE1")  # Mocno w gore-lewo nad lacznik
    cmd.set("label_position", [0, -29, 8.0], "chain A and resi 310 and name NE1")  # W dol pod domene Tm1570
    cmd.set("label_position", [-12, 2.25, 8.0], "chain A and resi 206 and name NE1")  # W lewo-dol
    cmd.set("label_position", [0.25, 17, 6.0], "chain A and resi 385 and name NE1")  # W prawo w puste tlo

    cmd.orient("chain A")
    cmd.turn("z", -20)
    cmd.zoom("chain A", 0.8)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_B)
    print("  -> Wygenerowano:", TEMP_IMG_B)


def render_panel_c():
    """Panel C: Monomer Fs (AlphaFold) wyrownany do Cn z wyraznymi etykietami."""
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

    # Wyrownanie strukturalne Fs do Cn
    cmd.align("Fs_model and resi 1-230", "ref_cn and resi 1-230")
    cmd.delete("ref_cn")

    cmd.show("cartoon", "Fs_model")

    # Kolorystyka domenowa Fs
    cmd.set_color("teal_trmd", [0.09, 0.74, 0.81])  # TrmD Fs
    cmd.set_color("teal_linker", [0.75, 0.90, 0.92])  # Lacznik Fs
    cmd.set_color("teal_tm", [0.06, 0.52, 0.57])  # Tm1570 Fs

    cmd.color("teal_trmd", "Fs_model and resi 1-235")
    cmd.color("teal_linker", "Fs_model and resi 236-246")
    cmd.color("teal_tm", "Fs_model and resi 247-434")

    # Ta sama ciemna, wyrazna zielen dla Trp w Fs
    cmd.set_color("trp_forest_green", [0.10, 0.58, 0.20])

    cmd.select("fs_trp", "Fs_model and resn TRP and (resi 206 or resi 310 or resi 386)")
    cmd.show("sticks", "fs_trp")
    cmd.color("trp_forest_green", "fs_trp")
    cmd.label("fs_trp and name NE1", '"Trp" + resi')

    # Podstawienie Glu237 w dwoch linijkach
    cmd.select("fs_glu237", "Fs_model and resi 237")
    cmd.show("sticks", "fs_glu237")
    cmd.color("firebrick", "fs_glu237")
    cmd.label("fs_glu237 and name OE1", '"Glu237\\n(ex-Trp)"')

    # Precyzyjne, calkowite wysuniecie etykiet w biale tlo:
    # Trp206 mocno w lewo-dol (Z=9.0 gwarantuje, ze zero nie zostanie uciete przez helise)
    cmd.set("label_position", [-20.5, 1, 9.0], "Fs_model and resi 206 and name NE1")
    # Glu237 wysoko nad lacznik w puste tlo
    cmd.set("label_position", [0.75, -19, 8.0], "fs_glu237 and name OE1")
    # Trp310 w dol pod domene Tm1570
    cmd.set("label_position", [-9.5, 19, 8.0], "Fs_model and resi 310 and name NE1")
    # Trp386 w prawo na zewnatrz
    cmd.set(
        "label_position",
        [0, 8, 6.0],
        "Fs_model and resi 386 and name NE1",
    )
    cmd.orient("Fs_model")
    cmd.turn("z", -20)
    cmd.zoom("Fs_model", 0.8)

    cmd.ray(2400, 2000)
    cmd.png(TEMP_IMG_C)
    print("  -> Wygenerowano:", TEMP_IMG_C)


def autocrop_border(img_array, threshold=252, pad=5):
  """Docina biale marginesy pojedynczych paneli z PyMOL-a."""
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


def pad_to_height(img_arr, target_h):
  """Wyrownuje wysokosc obrazow przed montazem."""
  h, w, c = img_arr.shape
  if h >= target_h:
    return img_arr
  pad_top = (target_h - h) // 2
  pad_bottom = target_h - h - pad_top
  pad_val = 1.0 if img_arr.max() <= 1.0 else 255
  padded = np.full((target_h, w, c), pad_val, dtype=img_arr.dtype)
  padded[pad_top : pad_top + h, :, :] = img_arr
  return padded


def assemble_figure():
  print(
      "Skladanie publikacyjnej Figury 4.8 (Maksymalne zaciesnienie krawedzi)..."
  )
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # 1. Wczytanie i dociecie pojedynczych paneli
  raw_a = autocrop_border(mpimg.imread(TEMP_IMG_A), pad=5)
  raw_b = autocrop_border(mpimg.imread(TEMP_IMG_B), pad=5)
  raw_c = autocrop_border(mpimg.imread(TEMP_IMG_C), pad=5)

  h_max = max(raw_a.shape[0], raw_b.shape[0], raw_c.shape[0])
  img_a = pad_to_height(raw_a, h_max)
  img_b = pad_to_height(raw_b, h_max)
  img_c = pad_to_height(raw_c, h_max)

  w_ratios = [img_a.shape[1], img_b.shape[1], img_c.shape[1]]

  # 2. Obnizona wysokosc figury (3.95 cala) likwiduje pionowy zapas
  fig, axes = plt.subplots(
      1,
      3,
      figsize=(16, 3.95),
      gridspec_kw={"width_ratios": w_ratios},
      dpi=300,
  )

  # UJEMNY wspace zsuwa panele na siebie i likwiduje biale pasy miedzy nimi
  plt.subplots_adjust(
      left=0.0, right=1.0, bottom=0.0, top=0.80, wspace=-0.55
  )

  panel_configs = [
      {
          "ax": axes[0],
          "img": img_a,
          "tag": "A",
          "title": (
              r"$\mathsf{Cn}$"
              "TrmD–Tm1570 Crystalline Dimer (8B1N)\nDimerization Interface"
              " (Tyr20/Tyr55)"
          ),
      },
      {
          "ax": axes[1],
          "img": img_b,
          "tag": "B",
          "title": (
              r"$\mathsf{Cn}$"
              "TrmD–Tm1570 Monomer (Chain A)\nTrp and Tyr Fluorophore"
              " Distribution"
          ),
      },
      {
          "ax": axes[2],
          "img": img_c,
          "tag": "C",
          "title": (
              r"$\mathsf{Fs}$"
              "TrmD–Tm1570 Monomer (AlphaFold)\nLinker Substitution Glu237"
              " (ex-Trp)"
          ),
      },
  ]

  for p in panel_configs:
    ax = p["ax"]
    ax.imshow(p["img"])
    ax.axis("off")

    ax.text(
        0.02,
        0.98,
        p["tag"],
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="square,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
        ),
    )
    ax.set_title(p["title"], fontsize=11.5, pad=10, fontweight="bold")

  # Zapis wstepny
  TEMP_ASSEMBLED = "temp_assembled_full.png"
  fig.savefig(TEMP_ASSEMBLED, dpi=300, bbox_inches="tight", pad_inches=0.0)
  plt.close(fig)

  # 3. KROK FINALNY: Bezwzgledny post-crop gotowego obrazu (scina dolny pas bieli do 0)
  final_img = mpimg.imread(TEMP_ASSEMBLED)
  final_cropped = autocrop_border(
      final_img, threshold=252, pad=2
  )  # pad=2 dla czystej krawedzi

  # Zapis ostateczny do PNG oraz wektorowego PDF
  plt.imsave(OUTPUT_PNG, final_cropped)

  # Zapis do PDF o scisle dopasowanych wymiarach
  h_px, w_px = final_cropped.shape[:2]
  fig_pdf = plt.figure(figsize=(w_px / 300.0, h_px / 300.0), dpi=300)
  ax_pdf = fig_pdf.add_axes([0, 0, 1, 1])
  ax_pdf.imshow(final_cropped)
  ax_pdf.axis("off")
  fig_pdf.savefig(OUTPUT_PDF, format="pdf", pad_inches=0.0)
  plt.close(fig_pdf)

  print(
      f"\n[OK] Zapisano maksymalnie zaciesniona Figure 4.8:\n  -> {OUTPUT_PDF}\n"
      f"  -> {OUTPUT_PNG}"
  )

  # Czyszczenie plikow tymczasowych
  for tmp in [
      TEMP_IMG_A,
      TEMP_IMG_B,
      TEMP_IMG_C,
      TEMP_ASSEMBLED,
      "temp_AF_Fs.pdb",
  ]:
    if os.path.exists(tmp):
      os.remove(tmp)

if __name__ == "__main__":
    render_panel_a()
    render_panel_b()
    render_panel_c()
    assemble_figure()