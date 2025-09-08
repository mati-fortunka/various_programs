import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyteomics.mass import calculate_mass
from collections import defaultdict

# --- File paths ---
filename = '/home/matifortunka/Documents/JS/Allan_MS/Exp_2_Elution_1_MS_mz.txt'
proteins = "/home/matifortunka/Documents/JS/Allan_MS/known_proteins.csv"

# --- Constants ---
PROTON_MASS = 1.007276
CHARGE_RANGE = range(10, 40)
MASS_MATCH_TOLERANCE = 1.0  # Da
INTENSITY_THRESHOLD_PERCENTILE = 30

# --- Step 1: Load spectrum ---
df = pd.read_csv(filename, sep=r"\s+", names=["m/z", "Intensity"])
df = df.dropna()
df = df[df["Intensity"] > 0]

# Optional filtering
threshold = np.percentile(df["Intensity"], INTENSITY_THRESHOLD_PERCENTILE)
df = df[df["Intensity"] >= threshold].reset_index(drop=True)

# Smoothing
from scipy.ndimage import gaussian_filter1d
df["Smoothed"] = gaussian_filter1d(df["Intensity"], sigma=2)

# --- Step 2–3: Neutral Mass Estimation with Envelope Scoring and Binning ---
ENVELOPE_TOLERANCE_MZ = 0.1  # Tolerance in m/z to detect charge state peaks
NEUTRAL_MASS_BIN_SIZE = 1.0  # Finer resolution
mass_dict = defaultdict(float)

from bisect import bisect_left, bisect_right

# Prepare sorted m/z and intensity arrays for fast searching
mz_array = df["m/z"].to_numpy()
intensity_array = df["Smoothed"].to_numpy()
sorted_indices = np.argsort(mz_array)
mz_sorted = mz_array[sorted_indices]
intensity_sorted = intensity_array[sorted_indices]

# Efficient lookup function
def find_intensity_near(mz_target, tolerance):
    left = bisect_left(mz_sorted, mz_target - tolerance)
    right = bisect_right(mz_sorted, mz_target + tolerance)
    return intensity_sorted[left:right].sum() if right > left else 0.0

# --- Improved Deconvolution ---
mass_dict = defaultdict(float)

for i in range(len(mz_array)):
    mz = mz_array[i]
    for z in CHARGE_RANGE:
        neutral_mass = mz * z - z * PROTON_MASS
        if 10000 < neutral_mass < 100000:
            envelope_score = 0.0
            contributing_charges = 0
            for dz in [-1, 0, 1]:
                z_adj = z + dz
                if z_adj in CHARGE_RANGE:
                    expected_mz = (neutral_mass + z_adj * PROTON_MASS) / z_adj
                    intensity = find_intensity_near(expected_mz, ENVELOPE_TOLERANCE_MZ)
                    if intensity > 0:
                        envelope_score += intensity
                        contributing_charges += 1
            if contributing_charges >= 2:
                binned_mass = round(neutral_mass / NEUTRAL_MASS_BIN_SIZE) * NEUTRAL_MASS_BIN_SIZE
                mass_dict[binned_mass] += envelope_score / contributing_charges


# Convert to DataFrame
mass_df = pd.DataFrame(mass_dict.items(), columns=["Mass", "Intensity"])
mass_df = mass_df.sort_values("Mass").reset_index(drop=True)


# --- Step 4: Match to known proteins ---
try:
    known_df = pd.read_csv(proteins)
    matches = []
    for _, row in mass_df.iterrows():
        for _, ref in known_df.iterrows():
            delta = abs(row["Mass"] - ref["Mass"])
            if delta <= MASS_MATCH_TOLERANCE:
                matches.append({
                    "Observed Mass": row["Mass"],
                    "Intensity": row["Intensity"],
                    "Matched Protein": ref["Protein"],
                    "Expected Mass": ref["Mass"],
                    "Mass Error": delta
                })

    match_df = pd.DataFrame(matches)
    if not match_df.empty:
        print("\n✅ Matched Peaks:")
        print(match_df.to_string(index=False))
    else:
        print("\n⚠️ No matches found within the given tolerance.")
except FileNotFoundError:
    print("\n⚠️ 'known_proteins.csv' not found. Skipping matching step.")

# --- Step 5: Plotting ---

# Raw spectrum
plt.figure(figsize=(12, 5))
plt.stem(df["m/z"], df["Intensity"], basefmt=" ", linefmt="b-", markerfmt=" ")
plt.title("Raw m/z Spectrum")
plt.xlabel("m/z")
plt.ylabel("Intensity")
plt.tight_layout()
plt.show()

# Normalize intensities for clearer comparison
mass_df["Relative Intensity"] = 100 * mass_df["Intensity"] / mass_df["Intensity"].max()

plt.figure(figsize=(10, 6))
plt.stem(mass_df["Mass"], mass_df["Relative Intensity"], basefmt=" ", linefmt="k-", markerfmt=" ")
plt.title("Improved Neutral Mass Spectrum (Envelope-Filtered)")
plt.xlabel("Neutral Mass (Da)")
plt.ylabel("Relative Intensity (%)")
plt.tight_layout()
plt.show()
