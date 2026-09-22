import ast
import os
import re
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ==============================================================================
# 1. DATA LOADING & PARSING
# ==============================================================================
DATA_URL = (
    "https://alphaknot.cent.uw.edu.pl/compute_static/b9bb675cadf65f/"
    "b9bb675cadf65f_1.txt"
)
LOCAL_FILE = "/home/matifortunka/Documents/JS/kinetics_stability/paper_fusion/figs/1/AK_output_fusion.txt"

if not os.path.exists(LOCAL_FILE):
    print("Downloading raw AlphaKnot data...")
    urllib.request.urlretrieve(DATA_URL, LOCAL_FILE)

print("Reading data...")
with open(LOCAL_FILE, "r") as f:
    raw_text = f.read().strip()

# Fast regex parser for the dictionary entries
# Matches (i, j): {'knot': prob, ...}
print("Parsing knot matrix...")
pattern = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*\{([^}]+)\}")
sub_pattern = re.compile(r"['\"]([^'\"]+)['\"]\s*:\s*([0-9.]+)")

N = 433
# Matrices for knot probabilities (1-indexed, size N+1 x N+1)
prob_31 = np.zeros((N + 1, N + 1))
prob_double = np.zeros((N + 1, N + 1))

for match in pattern.finditer(raw_text):
  i = int(match.group(1))
  j = int(match.group(2))
  knots = sub_pattern.findall(match.group(3))
  for ktype, pval in knots:
    p = float(pval)
    if ktype == "3_1":
      prob_31[i, j] = p
    elif "3_1#3_1" in ktype or "8_20" in ktype:
      prob_double[i, j] = p

print("Data parsing complete.")

# ==============================================================================
# 2. COLORMAPS & STYLING
# ==============================================================================
# Custom smooth colormaps matching manuscript palette:
# Blue for TrmD (Marine Blue: #1F77B4)
cmap_trmd = LinearSegmentedColormap.from_list(
    "TrmD_Blues", [(1, 1, 1, 0), "#9ecae1", "#1f77b4", "#084594"]
)

# Orange for Tm1570 (Warm Orange: #FF7F0E)
cmap_tm1570 = LinearSegmentedColormap.from_list(
    "Tm_Oranges", [(1, 1, 1, 0), "#fdd0a2", "#ff7f0e", "#d94801"]
)

# Charcoal for Composite 3_1#3_1
cmap_double = LinearSegmentedColormap.from_list(
    "Double_Knot", [(1, 1, 1, 0), "#bdbdbd", "#525252", "#1a1a1a"]
)

plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 0.8

# ==============================================================================
# 3. FIGURE GENERATION (3 SUBPLOTS)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=300)
CUTOFF = 0.40  # Probability threshold to display knot existence

# ------------------------------------------------------------------------------
# Subplot 1: CnTrmD (Residues 1 - 240)
# ------------------------------------------------------------------------------
ax1 = axes[0]
sub_trmd = prob_31[1:241, 1:241].copy()
sub_trmd[sub_trmd < CUTOFF] = np.nan

# Note: row index = i (N-cut), col index = j (C-cut)
# Transpose so x-axis is i (N-cut) and y-axis is j (C-cut) with origin='upper'
im1 = ax1.imshow(
    sub_trmd.T, cmap=cmap_trmd, vmin=0.0, vmax=1.0, origin="upper", extent=[1, 240, 240, 1]
)

# Knot core boundaries: 85 - 129
ax1.axvline(85, color="#084594", linestyle="--", linewidth=0.75, alpha=0.8)
ax1.axhline(129, color="#084594", linestyle="--", linewidth=0.75, alpha=0.8)
ax1.plot([1, 240], [1, 240], color="gray", linestyle=":", linewidth=0.6)

ax1.set_title(r"$\mathbf{Cn}$TrmD ($3_1$)", fontsize=10, pad=8)
ax1.set_xlabel("N-terminal cut (residue)", fontsize=8)
ax1.set_ylabel("C-terminal cut (residue)", fontsize=8)
ax1.set_xlim(1, 240)
ax1.set_ylim(240, 1)
ax1.tick_params(labelsize=7)
ax1.text(
    90,
    140,
    "knot core\n(85–129)",
    fontsize=7,
    color="#084594",
    weight="semibold",
)

# ------------------------------------------------------------------------------
# Subplot 2: CnTm1570 (Rescaled 1 - 193 from fusion 241 - 433)
# ------------------------------------------------------------------------------
ax2 = axes[1]
sub_tm = prob_31[241:434, 241:434].copy()
sub_tm[sub_tm < CUTOFF] = np.nan

# Rescaled: 241 -> 1, 433 -> 193 (isolated monomer numbering)
# Knot core in fusion: 352-397 -> rescaled: 112-157
im2 = ax2.imshow(
    sub_tm.T, cmap=cmap_tm1570, vmin=0.0, vmax=1.0, origin="upper", extent=[1, 193, 193, 1]
)

core_start = 352 - 240
core_end = 397 - 240

ax2.axvline(
    core_start, color="#d94801", linestyle="--", linewidth=0.75, alpha=0.8
)
ax2.axhline(core_end, color="#d94801", linestyle="--", linewidth=0.75, alpha=0.8)
ax2.plot([1, 193], [1, 193], color="gray", linestyle=":", linewidth=0.6)

ax2.set_title(r"$\mathbf{Cn}$Tm1570 ($3_1$)", fontsize=10, pad=8)
ax2.set_xlabel("N-terminal cut (residue)", fontsize=8)
ax2.set_ylabel("C-terminal cut (residue)", fontsize=8)
ax2.set_xlim(1, 193)
ax2.set_ylim(193, 1)
ax2.tick_params(labelsize=7)
ax2.text(
    core_start + 4,
    core_end + 12,
    f"knot core\n({core_start}–{core_end})",
    fontsize=7,
    color="#d94801",
    weight="semibold",
)

# ------------------------------------------------------------------------------
# Subplot 3: CnTrmD-Tm1570 Fusion (1 - 433)
# ------------------------------------------------------------------------------
ax3 = axes[2]

# Prepare combined RGB overlay for the composite matrix
rgb_fusion = np.ones((N + 1, N + 1, 3))  # White canvas

for i in range(1, N + 1):
  for j in range(i, N + 1):
    p_comp = prob_double[i, j]
    p_single = prob_31[i, j]

    if p_comp >= CUTOFF:
      # Charcoal/Black for 3_1#3_1
      gray_val = max(0.1, 1.0 - p_comp * 0.85)
      rgb_fusion[i, j] = [gray_val, gray_val, gray_val]
    elif p_single >= CUTOFF:
      if j <= 240 or (i <= 85 and j < 352):
        # TrmD single knot: Blue
        rgb_fusion[i, j] = [
            1.0 - p_single * 0.85,
            1.0 - p_single * 0.50,
            1.0 - p_single * 0.25,
        ]
      elif i >= 241 or (i > 129 and j >= 352):
        # Tm1570 single knot: Orange
        rgb_fusion[i, j] = [
            1.0 - p_single * 0.05,
            1.0 - p_single * 0.50,
            1.0 - p_single * 0.90,
        ]

ax3.imshow(
    np.transpose(rgb_fusion[1:, 1:], (1, 0, 2)),
    origin="upper",
    extent=[1, 433, 433, 1],
)
ax3.plot([1, 433], [1, 433], color="gray", linestyle=":", linewidth=0.6)

# Boundary line dividing the domains (residue 240)
ax3.axvline(240, color="gray", linestyle=":", linewidth=0.6)
ax3.axhline(240, color="gray", linestyle=":", linewidth=0.6)

ax3.set_title(r"Fusion $\mathbf{Cn}$TrmD-Tm1570 ($3_1\#3_1$)", fontsize=10, pad=8)
ax3.set_xlabel("N-terminal cut (residue)", fontsize=8)
ax3.set_ylabel("C-terminal cut (residue)", fontsize=8)
ax3.set_xlim(1, 433)
ax3.set_ylim(433, 1)
ax3.tick_params(labelsize=7)

# Legend for Subplot 3
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label=r"TrmD $3_1$",
        markerfacecolor="#1f77b4",
        markersize=6,
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label=r"Tm1570 $3_1$",
        markerfacecolor="#ff7f0e",
        markersize=6,
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label=r"Composite $3_1\#3_1$",
        markerfacecolor="#2b2b2b",
        markersize=6,
    ),
]
ax3.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=6.5,
    frameon=True,
    facecolor="white",
    edgecolor="none",
)

plt.tight_layout()
plt.savefig("panel_D_knot_matrices.pdf", bbox_inches="tight")
plt.savefig("panel_D_knot_matrices.png", dpi=300, bbox_inches="tight")
print(
    "Generated: 'panel_D_knot_matrices.pdf' and 'panel_D_knot_matrices.png'!"
)
plt.show()