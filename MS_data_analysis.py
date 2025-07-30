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
NEUTRAL_MASS_BIN_SIZE = 2.0
INTENSITY_THRESHOLD_PERCENTILE = 30

# --- Step 1: Load spectrum ---
df = pd.read_csv(filename, sep=r"\s+", names=["m/z", "Intensity"])
df = df.dropna()
df = df[df["Intensity"] > 0]

# Optional filtering
threshold = np.percentile(df["Intensity"], INTENSITY_THRESHOLD_PERCENTILE)
df = df[df["Intensity"] >= threshold].reset_index(drop=True)

# --- Step 2: Estimate neutral masses by charge state ---
# Smoothing
from scipy.ndimage import gaussian_filter1d
df["Smoothed"] = gaussian_filter1d(df["Intensity"], sigma=2)

# Neutral mass estimation (remove break!)
mass_list = []
for _, row in df.iterrows():
    mz = row["m/z"]
    intensity = row["Smoothed"]  # Use smoothed intensity!
    for z in CHARGE_RANGE:
        neutral_mass = mz * z - z * PROTON_MASS
        if 10000 < neutral_mass < 100000:
            mass_list.append((round(neutral_mass, 1), intensity))


# --- Step 3: Bin and aggregate ---
mass_dict = defaultdict(float)
for mass, intensity in mass_list:
    binned = round(mass / NEUTRAL_MASS_BIN_SIZE) * NEUTRAL_MASS_BIN_SIZE
    mass_dict[binned] += intensity

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

# Deconvoluted neutral mass spectrum
plt.figure(figsize=(12, 5))
plt.stem(mass_df["Mass"], mass_df["Intensity"], basefmt=" ", linefmt="g-", markerfmt=" ")
plt.title("Neutral Mass Spectrum (manual deconvolution)")
plt.xlabel("Neutral Mass (Da)")
plt.ylabel("Intensity")
plt.tight_layout()
plt.show()
