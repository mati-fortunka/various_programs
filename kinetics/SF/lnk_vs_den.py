import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

# --- Settings ---
input_file = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/kinetics/SF/mean_k.txt"
output_plot = Path(input_file).with_suffix(".png")
output_fit = Path(input_file).with_suffix(".fits.txt")

# Physical constants
R = 1.987e-3  # kcal/(mol*K)
T = 298.0     # temperature in K (change to your experimental T)

# --- Read data ---
df = pd.read_csv(input_file, sep=r"\s+", index_col=0)

# Extract concentrations from mean columns
mean_cols = [col for col in df.columns if col.endswith("_mean")]
std_cols = [col for col in df.columns if col.endswith("_std")]
concentrations = [float(col.replace("M_mean", "")) for col in mean_cols]

# --- Plotting ---
plt.figure(figsize=(6, 5))

fit_results = []

for i, phase in enumerate(df.index):
    mean_vals = df.loc[phase, mean_cols].replace("-", np.nan).astype(float).values
    std_vals  = df.loc[phase, std_cols].replace("-", np.nan).astype(float).values

    mask = ~np.isnan(mean_vals)
    concs = np.array(concentrations)[mask]
    k_values = mean_vals[mask]
    k_errors = std_vals[mask]

    # ln(k) and propagated error
    ln_k = np.log(k_values)
    ln_err = k_errors / k_values

    # consistent color
    color = plt.cm.tab10(i % 10)

    # scatter with error bars
    plt.errorbar(
        concs, ln_k, yerr=ln_err, fmt="o", capsize=4,
        color=color, label=f"Phase {phase}", markersize=6, linewidth=1.5
    )

    # linear regression fit
    if len(concs) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(concs, ln_k)
        fit_line = slope * concs + intercept
        plt.plot(concs, fit_line, linestyle="--", color=color, linewidth=2)

        # calculate physical quantities
        k_H2O = np.exp(intercept)  # rate in water
        m_value = slope * R * T    # kcal/mol/M

        fit_results.append({
            "Phase": phase,
            "Slope_ln(k)": slope,
            "Intercept_ln(k)": intercept,
            "k_H2O": k_H2O,
            "m_value(kcal/mol/M)": m_value,
            "R^2": r_value**2,
            "Std_err": std_err
        })

        print(f"Phase {phase}: k_H2O={k_H2O:.3e}, m={m_value:.3f} kcal/mol/M, R²={r_value**2:.4f}")
    else:
        print(f"Phase {phase}: not enough points for fitting")

# --- Figure style ---
plt.xlabel("GdmCl concentration (M)", fontsize=16)
plt.ylabel("ln(k)", fontsize=16)
plt.legend(fontsize=14, frameon=False)
plt.xlim(3.45,5.05)
xticks = np.arange(3.5, 5.1, 0.5)  # from 3.5 to 5.0 every 0.1 M
plt.xticks(xticks, fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()

# --- Save plot ---
plt.savefig(output_plot.with_suffix(".svg"), format="svg", dpi=600, bbox_inches="tight")
plt.savefig(output_plot.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight")
plt.close()
print(f"Plot saved to {output_plot.with_suffix('.svg')} and {output_plot.with_suffix('.png')}")

# --- Save fit results ---
if fit_results:
    fit_df = pd.DataFrame(fit_results)
    fit_df.to_csv(output_fit, index=False, sep="\t")
    print(f"Fit parameters saved to {output_fit}")
