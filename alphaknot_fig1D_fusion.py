import os
import re
import urllib.request
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np

# ==============================================================================
# 1. DATA LOADING & PARSING
# ==============================================================================
DATA_URL = (
    "https://alphaknot.cent.uw.edu.pl/compute_static/b9bb675cadf65f/"
    "b9bb675cadf65f_1.txt"
)
LOCAL_FILE = "/home/matifortunka/Documents/JS/kinetics_stability/paper_fusion/figs/1/AK_output_fusion.txt"

if not os.path.exists(LOCAL_FILE):
  LOCAL_FILE = "AK_output_fusion.txt"
  if not os.path.exists(LOCAL_FILE):
    print("Downloading raw AlphaKnot data...")
    urllib.request.urlretrieve(DATA_URL, LOCAL_FILE)

print(f"Reading data from {LOCAL_FILE}...")
with open(LOCAL_FILE, "r") as f:
  raw_text = f.read().strip()

pattern = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*\{([^}]+)\}")
sub_pattern = re.compile(r"['\"]([^'\"]+)['\"]\s*:\s*([0-9.]+)")

N = 433
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
# 2. COLORMAPS & TYPOGRAPHY
# ==============================================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
    "sans-serif",
]
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["mathtext.fontset"] = "dejavusans"

cmap_trmd = LinearSegmentedColormap.from_list(
    "TrmD_Blues", [(1, 1, 1, 0), "#9ecae1", "#1f77b4", "#084594"]
)
cmap_tm1570 = LinearSegmentedColormap.from_list(
    "Tm_Oranges", [(1, 1, 1, 0), "#fdd0a2", "#ff7f0e", "#d94801"]
)
cmap_double = LinearSegmentedColormap.from_list(
    "Double_Knot", [(1, 1, 1, 0), "#bdbdbd", "#525252", "#1a1a1a"]
)

# ==============================================================================
# 3. FIGURE LAYOUT
# ==============================================================================
fig = plt.figure(figsize=(11.0, 3.6), dpi=300)
gs = gridspec.GridSpec(
    1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.48, figure=fig
)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

for ax in (ax1, ax2, ax3):
  ax.set_box_aspect(1)

CUTOFF = 0.40

# ------------------------------------------------------------------------------
# Subplot 1: CnTrmD (1 - 240)
# ------------------------------------------------------------------------------
sub_trmd = prob_31[1:241, 1:241].copy()
sub_trmd[sub_trmd < CUTOFF] = np.nan

ax1.imshow(
    sub_trmd.T,
    cmap=cmap_trmd,
    vmin=0.0,
    vmax=1.0,
    origin="upper",
    extent=[1, 240, 240, 1],
)
ax1.plot([1, 240], [1, 240], color="#999999", linestyle=":", linewidth=0.7)

# Knot core boundaries: 85 - 129
c1_start, c1_end = 85, 129

# Axis projection lines (from Y-axis to core, and core down to X-axis)
ax1.plot(
    [1, c1_end],
    [c1_end, c1_end],
    color="#084594",
    linestyle=":",
    linewidth=0.75,
    alpha=0.75,
)
ax1.plot(
    [c1_start, c1_start],
    [c1_start, 240],
    color="#084594",
    linestyle=":",
    linewidth=0.75,
    alpha=0.75,
)

# Diagonal core line with endpoint markers
ax1.plot(
    [c1_start, c1_end],
    [c1_start, c1_end],
    color="#084594",
    lw=2.8,
    solid_capstyle="round",
    zorder=4,
)
ax1.scatter(
    [c1_start, c1_end], [c1_start, c1_end], color="#084594", s=22, zorder=5
)

# Endpoint residue numbers
ax1.text(
    c1_start + 4,
    c1_start - 2,
    str(c1_start),
    fontsize=7.2,
    color="#084594",
    fontweight="bold",
    va="bottom",
    ha="left",
    zorder=6,
)
ax1.text(
    c1_end + 4,
    c1_end,
    str(c1_end),
    fontsize=7.2,
    color="#084594",
    fontweight="bold",
    va="center",
    ha="left",
    zorder=6,
)

# Arrow pointing to the midpoint of the diagonal segment
mid1 = (c1_start + c1_end) / 2.0
ax1.annotate(
    "knot core",
    xy=(mid1, mid1),
    xytext=(155, 60),
    arrowprops=dict(
        arrowstyle="->",
        color="#084594",
        lw=0.85,
        shrinkA=0,
        shrinkB=3,
        connectionstyle="arc3,rad=-0.15",
    ),
    fontsize=7.2,
    color="#084594",
    fontweight="bold",
    ha="center",
    va="bottom",
)

ax1.set_title(r"$\mathit{Cn}$TrmD ($3_1$)", fontsize=9.5, pad=6)
ax1.set_xlabel("N-terminal cut (residue)", fontsize=8)
ax1.set_ylabel("C-terminal cut (residue)", fontsize=8)
ax1.set_xlim(1, 240)
ax1.set_ylim(240, 1)
ax1.set_xticks([1, 50, 100, 150, 200])
ax1.set_yticks([1, 50, 100, 150, 200])
ax1.tick_params(labelsize=7)

# ------------------------------------------------------------------------------
# Subplot 2: CnTm1570 (Rescaled 1 - 193)
# ------------------------------------------------------------------------------
sub_tm = prob_31[241:434, 241:434].copy()
sub_tm[sub_tm < CUTOFF] = np.nan

ax2.imshow(
    sub_tm.T,
    cmap=cmap_tm1570,
    vmin=0.0,
    vmax=1.0,
    origin="upper",
    extent=[1, 193, 193, 1],
)
ax2.plot([1, 193], [1, 193], color="#999999", linestyle=":", linewidth=0.7)

# Knot core boundaries: 112 - 157
c2_start = 352 - 240  # 112
c2_end = 397 - 240  # 157

# Axis projection lines (from Y-axis to core, and core down to X-axis)
ax2.plot(
    [1, c2_end],
    [c2_end, c2_end],
    color="#d94801",
    linestyle=":",
    linewidth=0.75,
    alpha=0.75,
)
ax2.plot(
    [c2_start, c2_start],
    [c2_start, 193],
    color="#d94801",
    linestyle=":",
    linewidth=0.75,
    alpha=0.75,
)

# Diagonal core line with endpoint markers
ax2.plot(
    [c2_start, c2_end],
    [c2_start, c2_end],
    color="#d94801",
    lw=2.8,
    solid_capstyle="round",
    zorder=4,
)
ax2.scatter(
    [c2_start, c2_end],
    [c2_start, c2_end],
    color="#d94801",
    s=22,
    zorder=5,
)

# Endpoint residue numbers
ax2.text(
    c2_start + 4,
    c2_start - 2,
    str(c2_start),
    fontsize=7.2,
    color="#d94801",
    fontweight="bold",
    va="bottom",
    ha="left",
    zorder=6,
)
ax2.text(
    c2_end + 4,
    c2_end,
    str(c2_end),
    fontsize=7,
    color="#d94801",
    fontweight="bold",
    va="center",
    ha="left",
    zorder=6,
)

# Arrow pointing to the midpoint of the diagonal segment
mid2 = (c2_start + c2_end) / 2.0
ax2.annotate(
    "knot core",
    xy=(mid2, mid2),
    xytext=(155, 75),
    arrowprops=dict(
        arrowstyle="->",
        color="#d94801",
        lw=0.85,
        shrinkA=0,
        shrinkB=3,
        connectionstyle="arc3,rad=-0.15",
    ),
    fontsize=7,
    color="#d94801",
    ha="center",
    va="bottom",
)

ax2.set_title(r"$\mathit{Cn}$Tm1570 ($3_1$)", fontsize=9.5, pad=6)
ax2.set_xlabel("N-terminal cut (residue)", fontsize=8)
ax2.set_xlim(1, 193)
ax2.set_ylim(193, 1)
ax2.set_xticks([1, 50, 100, 150])
ax2.set_yticks([1, 50, 100, 150])
ax2.tick_params(labelsize=7)

# ------------------------------------------------------------------------------
# Subplot 3: CnTrmD-Tm1570 Fusion (1 - 433)
# ------------------------------------------------------------------------------
rgb_fusion = np.ones((N + 1, N + 1, 3))

for i in range(1, N + 1):
  for j in range(i, N + 1):
    p_comp = prob_double[i, j]
    p_single = prob_31[i, j]

    if p_comp >= CUTOFF:
      gray_val = max(0.12, 1.0 - p_comp * 0.88)
      rgb_fusion[i, j] = [gray_val, gray_val, gray_val]
    elif p_single >= CUTOFF:
      if j <= 240 or (i <= 85 and j < 398):
        rgb_fusion[i, j] = [
            1.0 - p_single * 0.85,
            1.0 - p_single * 0.50,
            1.0 - p_single * 0.25,
        ]
      elif i >= 241 or (i > 84 and j >= 352):
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
ax3.plot([1, 433], [1, 433], color="#999999", linestyle=":", linewidth=0.7)

# Domain boundary at residue 240
ax3.axvline(240, color="#555555", linestyle="--", linewidth=0.75)
ax3.axhline(240, color="#555555", linestyle="--", linewidth=0.75)

# Composite knot core boundaries: 85 - 397
c3_start, c3_end = 85, 397

# --- AXIS PROJECTION LINES (Knot core boundary origins) ---
# 1. Horizontal: from left Y-axis (x=1) to diagonal endpoint (x=397, y=397)
ax3.plot(
    [1, c3_end],
    [c3_end, c3_end],
    color="#2b2b2b",
    linestyle=":",
    linewidth=0.75,
    alpha=0.85,
)

# 2. Vertical: from diagonal start point (x=85, y=85) down to bottom X-axis (y=433)
# (If you prefer connecting to the top border instead, change [c3_start, 433] to [1, c3_start])
ax3.plot(
    [c3_start, c3_start],
    [c3_start, 433],
    color="#2b2b2b",
    linestyle=":",
    linewidth=0.75,
    alpha=0.85,
)

# Diagonal core line with endpoint markers
ax3.plot(
    [c3_start, c3_end],
    [c3_start, c3_end],
    color="#2b2b2b",
    lw=1,
    solid_capstyle="round",
    zorder=4,
)
ax3.scatter(
    [c3_start, c3_end], [c3_start, c3_end], color="#2b2b2b", s=22, zorder=5
)

# Endpoint residue numbers
ax3.text(
    c3_start + 6,
    c3_start - 3,
    str(c3_start),
    fontsize=7.2,
    color="#2b2b2b",
    fontweight="bold",
    va="bottom",
    ha="left",
    zorder=6,
)
ax3.text(
    c3_end + 6,
    c3_end,
    str(c3_end),
    fontsize=7.2,
    color="#2b2b2b",
    fontweight="bold",
    va="center",
    ha="left",
    zorder=6,
)

# Arrow pointing to the midpoint of the composite diagonal segment
mid3 = (c3_start + c3_end) / 2.0  # 241.0
ax3.annotate(
    r"$3_1\#3_1$ core",
    xy=(mid3, mid3),
    xytext=(290, 150),
    arrowprops=dict(
        arrowstyle="->",
        color="#2b2b2b",
        lw=0.85,
        shrinkA=0,
        shrinkB=3,
        connectionstyle="arc3,rad=-0.12",
    ),
    fontsize=7,
    color="#2b2b2b",
    ha="center",
    va="bottom",
)

ax3.set_title(r"$\mathit{Cn}$TrmD-Tm1570 ($3_1\#3_1$)", fontsize=9.5, pad=6)
ax3.set_xlabel("N-terminal cut (residue)", fontsize=8)
ax3.set_xlim(1, 433)
ax3.set_ylim(433, 1)
ax3.set_xticks([1, 100, 200, 300, 400])
ax3.set_yticks([1, 100, 200, 300, 400])
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
        label=r"$3_1\#3_1$",
        markerfacecolor="#2b2b2b",
        markersize=6,
    ),
]
ax3.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=6.2,
    frameon=True,
    facecolor="#ffffff",
    edgecolor="#cccccc",
)

# ------------------------------------------------------------------------------
# Subplot 4: Compact Stacked Colorbars
# ------------------------------------------------------------------------------
gs_cbar = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs[3], hspace=0.35, height_ratios=[1, 1, 1]
)
norm = Normalize(vmin=0.0, vmax=1.0)

ax_cb1 = fig.add_subplot(gs_cbar[0])
cb1 = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap_double),
    cax=ax_cb1,
    orientation="vertical",
)
cb1.set_ticks([0.0, 1.0])
cb1.ax.tick_params(labelsize=6)
ax_cb1.set_ylabel(r"$3_1\#3_1$", fontsize=6.8, labelpad=2)

ax_cb2 = fig.add_subplot(gs_cbar[1])
cb2 = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap_trmd),
    cax=ax_cb2,
    orientation="vertical",
)
cb2.set_ticks([0.0, 1.0])
cb2.ax.tick_params(labelsize=6)
ax_cb2.set_ylabel(r"TrmD $3_1$", fontsize=6.8, labelpad=2)

ax_cb3 = fig.add_subplot(gs_cbar[2])
cb3 = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap_tm1570),
    cax=ax_cb3,
    orientation="vertical",
)
cb3.set_ticks([0.0, 1.0])
cb3.ax.tick_params(labelsize=6)
ax_cb3.set_ylabel(r"Tm1570 $3_1$", fontsize=6.8, labelpad=2)

# ==============================================================================
# 4. EXPORT
# ==============================================================================
output_dir = os.path.dirname(LOCAL_FILE)
pdf_path = os.path.join(output_dir, "panel_D_knot_matrices.pdf")
png_path = os.path.join(output_dir, "panel_D_knot_matrices.png")

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Successfully generated:\n -> {pdf_path}\n -> {png_path}")
plt.show()