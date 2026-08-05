import os
import re
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Target output directory
folder = "/home/matifortunka/Documents/JS/kinetics_stability/data_Warsaw"
os.makedirs(folder, exist_ok=True)

# 1. Raw Data
raw_data = """
Tm1570
Parameter	CD 12h 2µM	CD 12h 3µM	CD 12h 5µM	CD 36h 2µM	CD 36h 3µM	CD 84h 2µM	CD 84h 3µM	CD 90 h 5µM	CD 114 h 5µM
m	1.33 ± 0.17	1.30 ± 0.15	1.40 ± 0.17	1.69 ± 0.32	1.19 ± 0.12	1.86 ± 0.40	-	1.52 ± 0.15	1.81 ± 0.23
d	3.14 ± 0.06	3.17 ± 0.06	3.21 ± 0.06	3.11 ± 0.07	2.99 ± 0.06	3.00 ± 0.08	3.14 ± 0.05	-	3.22 ± 0.05

TrmD
Parameter	CD 12h 2µM	CD 12h 3µM	CD 36h 2µM	CD 36h 3µM	CD 84h 2µM	CD 84h 3µM	CD 90 h 5µM	CD 114 h 5µM
m	0.94 ± 0.12	1.08 ± 0.13	1.22 ± 0.17	1.08 ± 0.14	1.17 ± 0.24	0.96 ± 0.15	-	1.04 ± 0.11
d	1.98 ± 0.09	1.93 ± 0.07	2.03 ± 0.07	2.05 ± 0.08	2.07 ± 0.11	2.04 ± 0.11	2.14 ± 0.07	1.94 ± 0.08

fusion
Parameter	CD 12h 2µM	CD 12h 3µM	CD 36h 2µM	CD 36h 3µM	CD 84h 2µM	CD 84h 3µM	
m	1.27 ± 0.15	1.08 ± 0.16	1.30 ± 0.18	1.20 ± 0.17	1.45 ± 0.34	1.13 ± 0.16
d	2.40 ± 0.06	2.45 ± 0.09	2.35 ± 0.07	2.37 ± 0.08	2.25 ± 0.11	2.45 ± 0.08
"""


# 2. Robust Data Parser
def parse_table_data(text):
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    rows = []
    current_protein = ""
    columns = []

    for line in lines:
        parts = [p.strip() for p in re.split(r'\t|\s{2,}', line) if p.strip()]

        if len(parts) == 1:
            current_protein = parts[0]
        elif parts[0] == "Parameter":
            columns = parts[1:]
        else:
            param_name = f"{current_protein}_{parts[0]}"
            values = parts[1:]
            parsed_vals = []
            parsed_errs = []
            for v in values:
                v_clean = v.strip()
                if v_clean == "-" or v_clean == "" or "±" not in v_clean:
                    parsed_vals.append(np.nan)
                    parsed_errs.append(np.nan)
                else:
                    val_str, err_str = v_clean.split('±')
                    parsed_vals.append(float(val_str.strip()))
                    parsed_errs.append(float(err_str.strip()))

            rows.append({
                "row_label": param_name,
                "values": parsed_vals,
                "errors": parsed_errs,
                "columns": columns
            })

    return rows


parsed_rows = parse_table_data(raw_data)
all_cols = list(dict.fromkeys([col for r in parsed_rows for col in r["columns"]]))
row_labels = [r["row_label"] for r in parsed_rows]

# =============================================================================
# TABLE 1: Deviation matrix from mean
# =============================================================================

MULTIPLIER = 2.0
MANUAL_LIMITS = {}

val_matrix = np.full((len(parsed_rows), len(all_cols)), np.nan)
text_matrix = [["" for _ in all_cols] for _ in row_labels]
normalized_matrix = np.full((len(parsed_rows), len(all_cols)), np.nan)

for i, r in enumerate(parsed_rows):
    valid_vals = [v for v in r["values"] if not np.isnan(v)]
    valid_errs = [e for e in r["errors"] if not np.isnan(e)]

    if len(valid_vals) > 0:
        mean_val = np.mean(valid_vals)
        mean_err = np.mean(valid_errs)
        max_dev = MANUAL_LIMITS.get(r["row_label"], mean_err * MULTIPLIER)
    else:
        mean_val, mean_err, max_dev = 0, 0, 1

    for j, col_name in enumerate(all_cols):
        if col_name in r["columns"]:
            idx = r["columns"].index(col_name)
            v = r["values"][idx]
            err = r["errors"][idx]

            if np.isnan(v):
                text_matrix[i][j] = "N/A"
                normalized_matrix[i, j] = -2.0  # Black cell for missing/ignored values
            else:
                val_matrix[i, j] = v
                text_matrix[i][j] = f"{v}\n±{err}"
                dev = abs(v - mean_val)

                if dev > max_dev:
                    normalized_matrix[i, j] = -1.0  # Purple for outlier
                else:
                    normalized_matrix[i, j] = dev / max_dev if max_dev != 0 else 0.0
        else:
            normalized_matrix[i, j] = -3.0  # Neutral background for missing column

colors = ["#1a9850", "#ffffbf", "#d73027"]  # Green -> Yellow -> Red
outlier_color = "#880e4f"  # Deep Purple
missing_color = "#1a1a1a"  # Dark/Black for NaN/"-"
blank_col_color = "#e0e0e0"  # Light neutral background for missing columns

cmap = LinearSegmentedColormap.from_list("deviation_gyr", colors)
cmap.set_under(outlier_color)

fig, ax = plt.subplots(figsize=(15, 7.5))

# Draw base background array for missing values and blank columns
bg_matrix = np.ones((len(parsed_rows), len(all_cols), 3))
for i in range(len(parsed_rows)):
    for j in range(len(all_cols)):
        val = normalized_matrix[i, j]
        if val == -2.0:
            bg_matrix[i, j] = [0.1, 0.1, 0.1]  # Black/Dark
        elif val == -3.0:
            bg_matrix[i, j] = [0.88, 0.88, 0.88]  # Blank background

ax.imshow(bg_matrix, aspect="auto")

# Mask special values to let imshow overlay valid entries
masked_norm = np.ma.masked_where(normalized_matrix < -1.0, normalized_matrix)
im = ax.imshow(masked_norm, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

ax.set_xticks(np.arange(len(all_cols)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(all_cols, rotation=30, ha="right", fontsize=10, weight="bold")
ax.set_yticklabels(row_labels, fontsize=10, weight="bold")

for i in range(len(row_labels)):
    for j in range(len(all_cols)):
        txt = text_matrix[i][j]
        norm_val = normalized_matrix[i, j]
        if norm_val == -2.0:
            ax.text(j, i, "N/A", ha="center", va="center", color="white", fontsize=8.5, weight="bold")
        elif txt and norm_val >= -1.0:
            font_color = "white" if norm_val < 0 or norm_val > 0.75 or norm_val < 0.25 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=font_color, fontsize=8.5)

ax.set_xticks(np.arange(len(all_cols) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", size=0)

plt.title(f"Deviation from mean (Tolerance = ± {MULTIPLIER} × mean error)", fontsize=12, pad=15, weight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_ticks([0.0, 0.5, 1.0])
cbar.set_ticklabels(['0 (mean)', '0.5 (higher deviation)', '1.0 (limit)'])
cbar.ax.tick_params(labelsize=9)

outlier_patch = mpatches.Patch(color=outlier_color, label='outlier (> limit)')
na_patch = mpatches.Patch(color=missing_color, label='N/A / Ignored (-)')
plt.legend(handles=[outlier_patch, na_patch], loc='upper right', bbox_to_anchor=(1.25, 1.12), frameon=True)

plt.tight_layout()
plt.savefig(folder + "/deviation_matrix.png", dpi=300, bbox_inches="tight")
plt.savefig(folder + "/deviation_matrix.pdf", bbox_inches="tight")

print("Saved Table 1 to files: deviation_matrix.png and deviation_matrix.pdf")
plt.show()

# =============================================================================
# TABLE 2: Overlapping Error Ranges with Dynamic Red Gradient
# =============================================================================

rgb_matrix = np.zeros((len(parsed_rows), len(all_cols), 3))
text_matrix_2 = [["" for _ in all_cols] for _ in row_labels]

for i, r in enumerate(parsed_rows):
    valid_vals = [v for v in r["values"] if not np.isnan(v)]
    valid_errs = [e for e in r["errors"] if not np.isnan(e)]

    if len(valid_vals) > 0:
        mean_val = np.mean(valid_vals)
        mean_err = np.mean(valid_errs)
    else:
        mean_val, mean_err = 0, 1

    ref_min = mean_val - mean_err
    ref_max = mean_val + mean_err

    for j, col_name in enumerate(all_cols):
        if col_name in r["columns"]:
            idx = r["columns"].index(col_name)
            v = r["values"][idx]
            err = r["errors"][idx]

            if np.isnan(v):
                text_matrix_2[i][j] = "N/A"
                color = [0.1, 0.1, 0.1]  # Black for missing cells
            else:
                text_matrix_2[i][j] = f"{v}\n±{err}"
                val_min = v - err
                val_max = v + err

                # 1. GREEN: Value falls inside [Mean - Mean Error, Mean + Mean Error]
                if ref_min <= v <= ref_max:
                    dist_to_mean = abs(v - mean_val) / (mean_err if mean_err != 0 else 1)
                    intensity = 1.0 - 0.5 * min(dist_to_mean, 1.0)
                    color = [0.1 * intensity, 0.8 * intensity, 0.2 * intensity]

                else:
                    overlaps = (val_max >= ref_min) and (val_min <= ref_max)

                    # 2. YELLOW: Value outside range, but error bars overlap
                    if overlaps:
                        # Normalize yellow intensity by distance from mean
                        rel_dev = min(abs(v - mean_val) / (2.0 * mean_err if mean_err != 0 else 1), 1.0)
                        r_val = 1.0
                        g_val = 0.95 - 0.45 * rel_dev
                        b_val = 0.60 - 0.55 * rel_dev
                        color = [max(0, r_val), max(0, g_val), max(0, b_val)]

                    # 3. RED: Disjoint error ranges (Outliers) with dynamic distance scaling
                    else:
                        # Measure gap between error range and reference range
                        if v > mean_val:
                            gap = val_min - ref_max
                        else:
                            gap = ref_min - val_max

                        # Boundary ratio: 2 * mean_err as maximum gap saturation
                        gap_ratio = min(max(gap, 0.0) / (2.0 * mean_err if mean_err != 0 else 1.0), 1.0)

                        # Interpolate Light Red (#ff9999) -> Deep Red (#990000)
                        r_val = 1.0 - 0.4 * gap_ratio
                        g_val = 0.6 * (1.0 - gap_ratio)
                        b_val = 0.6 * (1.0 - gap_ratio)
                        color = [r_val, g_val, b_val]
        else:
            # Blank column background
            color = [0.88, 0.88, 0.88]

        rgb_matrix[i, j] = color

fig2, ax2 = plt.subplots(figsize=(15, 7.5))
ax2.imshow(rgb_matrix, aspect="auto")

ax2.set_xticks(np.arange(len(all_cols)))
ax2.set_yticks(np.arange(len(row_labels)))
ax2.set_xticklabels(all_cols, rotation=30, ha="right", fontsize=10, weight="bold")
ax2.set_yticklabels(row_labels, fontsize=10, weight="bold")

for i in range(len(row_labels)):
    for j in range(len(all_cols)):
        txt = text_matrix_2[i][j]
        if txt:
            r_c, g_c, b_c = rgb_matrix[i, j]
            luminance = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
            font_color = "white" if luminance < 0.5 else "black"
            ax2.text(j, i, txt, ha="center", va="center", color=font_color, fontsize=8.5, weight="bold")

ax2.set_xticks(np.arange(len(all_cols) + 1) - 0.5, minor=True)
ax2.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
ax2.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax2.tick_params(which="minor", size=0)

plt.title("Overlapping Error Ranges Analysis", fontsize=12, pad=15, weight="bold")

legend_patches = [
    mpatches.Patch(color=[0.08, 0.64, 0.16], label="Green: Value within [Mean ± Mean Error]"),
    mpatches.Patch(color=[0.95, 0.75, 0.20], label="Yellow: Value outside, but ERRORS OVERLAP"),
    mpatches.Patch(color=[1.0, 0.6, 0.6], label="Light Red: Disjoint error ranges (small gap)"),
    mpatches.Patch(color=[0.6, 0.0, 0.0], label="Deep Red: Disjoint error ranges (large gap >= 2 mean errors)"),
    mpatches.Patch(color=[0.1, 0.1, 0.1], label="Black: N/A / Ignored (-)")
]
plt.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.32, 1.15), frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig(folder + "/overlapping_errors_matrix.png", dpi=300, bbox_inches="tight")
plt.savefig(folder + "/overlapping_errors_matrix.pdf", bbox_inches="tight")

print("Saved Table 2 to files: overlapping_errors_matrix.png and overlapping_errors_matrix.pdf")
plt.show()