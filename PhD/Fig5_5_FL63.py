import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# =========================================================
# 1. KONFIGURACJA TYPOGRAFII I STYLU ROZPRAWY (LaTeX / PhD)
# =========================================================
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "DejaVu Sans"
mpl.rcParams["mathtext.it"] = "DejaVu Sans:italic"
mpl.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
mpl.rcParams["mathtext.sf"] = "DejaVu Sans:bold:italic"
mpl.rcParams["axes.linewidth"] = 1.1

OUT_DIR = "/home/matifortunka/PycharmProjects/various_programs/PhD/figures/ch05"
OUT_BASENAME = "Figura_5_3_6_3_Fluorescence_Kinetics"

# =========================================================
# 2. MODELE KINETYCZNE
# =========================================================
def exponential_model(t, a, k, c):
    return a * np.exp(-k * t) + c

def sigmoid_model(t, y0, a, k, t_half):
    return y0 + a / (1.0 + np.exp(-k * (t - t_half)))

def linear_model(t, k, b):
    return k * t + b

def double_exponential_model(t, a1, k1, a2, k2, c):
    return a1 * np.exp(-k1 * t) + a2 * np.exp(-k2 * t) + c

def exponential_with_linear_model(t, a, k, k2, c):
    exponent = np.clip(-k * t, -700, 700)
    return a * np.exp(exponent) + k2 * t + c

MODEL_REGISTRY = {
    "exponential": {
        "func": exponential_model,
        "initial_guess": lambda t, v: [v[0] - v[-1], 1.0, v[-1]],
        "param_names": ["a", "k", "c"],
    },
    "sigmoid": {
        "func": sigmoid_model,
        "initial_guess": lambda t, v: [v.min(), v.max() - v.min(), 1.0, (t[0] + t[-1]) / 2.0],
        "param_names": ["y0", "a", "k", "t_half"],
    },
    "linear": {
        "func": linear_model,
        "initial_guess": lambda t, v: [(v[-1] - v[0]) / (t[-1] - t[0]), v[0]],
        "param_names": ["k", "b"],
    },
    "double_exponential": {
        "func": double_exponential_model,
        "initial_guess": lambda t, v: [(v[0] - v[-1]) / 2.0, 1.0, (v[0] - v[-1]) / 2.0, 0.1, v[-1]],
        "param_names": ["a1", "k1", "a2", "k2", "c"],
    },
    "exponential_with_linear": {
        "func": exponential_with_linear_model,
        "initial_guess": lambda t, v: [(v[0] - v[-1]), 0.01, 0.001, v[-1]],
        "param_names": ["a", "k", "k2", "c"],
    },
}

# =========================================================
# 3. FUNKCJE WCZYTUJĄCE I PRZETWARZAJĄCE DANE
# =========================================================
def read_stoppedflow_csv(path, voltage_offset=0.0):
    """Wczytuje pojedynczy przebieg stopped-flow, odcina zapętlenia czasu i artefakty > 20 V."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku stopped-flow: {path}")

    df = pd.read_csv(path, header=None, usecols=[0, 1])
    df.columns = ["time", "voltage"]

    wrap_index = next((i for i in range(1, len(df)) if df["time"].iloc[i] <= df["time"].iloc[i - 1]), None)
    if wrap_index is not None:
        df = df.iloc[:wrap_index]

    df = df.drop_duplicates(subset="time")
    df = df[df["voltage"] < 20.0].reset_index(drop=True)
    df["voltage"] = df["voltage"] + voltage_offset
    return df


def read_fluorimetry_csv(filename, skiprows=2):
    """Wczytuje przebieg manual-mixing z fluorymetru i odcina ewentualne zapętlenie czasu."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Nie znaleziono pliku fluorymetrycznego: {filename}")

    df_full = pd.read_csv(filename, skiprows=skiprows, sep=",").dropna(how="all", axis=1)
    time_col = df_full.iloc[:, 0]
    initial_time = time_col.iloc[0]
    wrap_index = time_col[1:].sub(initial_time).abs().lt(1e-6).idxmax()

    if wrap_index > 1:
        df = df_full.iloc[:wrap_index].copy()
    else:
        df = df_full.copy()
    return df


def apply_savgol(y, window_length=15, polyorder=3):
    if window_length % 2 == 0:
        window_length += 1
    if len(y) >= window_length:
        return savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return y


def fit_and_plot_optional(ax, df, time_col, val_col, model, fit_start, fit_end, color, label_prefix):
    if not model or model not in MODEL_REGISTRY:
        return
    info = MODEL_REGISTRY[model]
    fit_func = info["func"]
    guess_func = info["initial_guess"]

    fit_df = df[(df[time_col] >= fit_start) & (df[time_col] <= fit_end)]
    if len(fit_df) < 3:
        return

    t_fit = fit_df[time_col].values
    v_fit = fit_df[val_col].values
    p0 = guess_func(t_fit, v_fit)
    params, _ = curve_fit(fit_func, t_fit, v_fit, p0=p0, maxfev=10000)

    t_smooth = np.linspace(t_fit[0], t_fit[-1], 1000)
    v_smooth = fit_func(t_smooth, *params)
    ax.plot(t_smooth, v_smooth, linestyle="--", linewidth=1.6, color=color, label=f"{label_prefix} fit")


def style_standard_axes(ax, panel_label, xlabel, ylabel, xlim=None, ylim=None, label_pos=(0.04, 0.94)):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=6)
    ax.tick_params(axis="both", labelsize=12.5, width=1.1, length=4.5)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.text(
        label_pos[0], label_pos[1], panel_label,
        transform=ax.transAxes,
        fontsize=15.5, fontweight="bold",
        va="top", ha="left"
    )


# =========================================================
# 4. GENEROWANIE ZBIORCZEJ FIGURY 3-PANELOWEJ (A, B, C)
# =========================================================
def build_figure_6_3_fluorescence(
    fit_panel_a=None,
    fit_panel_b=None,
    fit_panel_c_sf=None,
):
    fig = plt.figure(figsize=(12.2, 9.6))
    gs = GridSpec(
        2, 4,
        figure=fig,
        wspace=0.55,
        hspace=0.34,
        top=0.96, bottom=0.08, left=0.08, right=0.94
    )

    # Układ: 2 panele na górze (A, B) i 1 wyśrodkowany na dole (C)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 1:3])

    # -----------------------------------------------------
    # PANEL A: Stopped-Flow szybki (0 - 1 s)
    # -----------------------------------------------------
    traces_a = [
        {
            "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/paper/SF/3.5M/fast/6_3_2uM_1s00010.csv",
            "label": "3.5 M GdmCl",
            "color": "#75053b",
            "offset": 0.0,
        },
        {
            "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/paper/SF/4M/fast/6_3_2uM_1s00021.csv",
            "label": "4.0 M GdmCl",
            "color": "#3D48A4",
            "offset": 0.0,
        },
    ]

    for tr in traces_a:
        df = read_stoppedflow_csv(tr["path"], voltage_offset=tr["offset"])
        ax_a.plot(df["time"], df["voltage"], label=tr["label"], color=tr["color"], linewidth=1.1)
        fit_and_plot_optional(
            ax_a, df, "time", "voltage",
            model=fit_panel_a, fit_start=0.007, fit_end=1.0,
            color=tr["color"], label_prefix=tr["label"]
        )

    style_standard_axes(
        ax_a, panel_label="A",
        xlabel="Time (s)", ylabel="Voltage (V)",
        xlim=(0, 1.0), label_pos=(0.05, 0.95)
    )
    ax_a.legend(fontsize=11.5, frameon=False, loc="upper right")

    # -----------------------------------------------------
    # PANEL B: Stopped-Flow wolniejszy (0 - 2000 s)
    # -----------------------------------------------------
    traces_b = [
        {
            "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/kinetics/SF/unfolding/4M/6_3_2uM00011.csv",
            "label": "4.0 M GdmCl",
            "color": "#3D48A4",
            "offset": 1.0,
        },
        {
            "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/kinetics/SF/unfolding/4.5M/2000s/6_3_2uM_2000s00021.csv",
            "label": "4.5 M GdmCl",
            "color": "#136308",
            "offset": -1.2,
        },
        {
            "path": "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/kinetics/SF/unfolding/5M/2000s/6_3_2uM_2000s00012.csv",
            "label": "5.0 M GdmCl",
            "color": "#FB7305",
            "offset": 0.0,
        },
    ]

    for tr in traces_b:
        df = read_stoppedflow_csv(tr["path"], voltage_offset=tr["offset"])
        ax_b.plot(df["time"], df["voltage"], label=tr["label"], color=tr["color"], linewidth=1.4)
        fit_and_plot_optional(
            ax_b, df, "time", "voltage",
            model=fit_panel_b, fit_start=0.0, fit_end=1000.0,
            color=tr["color"], label_prefix=tr["label"]
        )

    style_standard_axes(
        ax_b, panel_label="B",
        xlabel="Time (s)", ylabel="Voltage (V)",
        xlim=(0, 2000), label_pos=(0.05, 0.95)
    )
    ax_b.legend(fontsize=11.5, frameon=False, loc="lower right")

    # -----------------------------------------------------
    # PANEL C: Manual-Mixing FL vs Stopped-Flow FL (0 - 2000 s)
    # -----------------------------------------------------
    fluo_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/paper/fluo/new fits/double/63_2h_3.csv"
    sf_comp_path = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/6_3/kinetics/SF/unfolding/4M/6_3_2uM00011.csv"

    fluor_df = read_fluorimetry_csv(fluo_path, skiprows=2)
    time_f = fluor_df.iloc[:, 0].values
    intensity = fluor_df.iloc[:, 1].values

    sf_df = read_stoppedflow_csv(sf_comp_path, voltage_offset=0.0)
    time_s = sf_df["time"].values
    voltage_s = apply_savgol(sf_df["voltage"].values, window_length=15, polyorder=3)

    color_mm = "#75053b"
    color_sf = "#3D48A4"

    l1, = ax_c.plot(time_f, intensity, color=color_mm, linewidth=1.3, label="Manual-mixing FL")
    ax_c.spines["top"].set_visible(False)
    ax_c.set_xlabel("Time (s)", fontsize=14, labelpad=6)
    ax_c.set_ylabel("Fluorescence Intensity (a.u.)", fontsize=14, color=color_mm, labelpad=6)
    ax_c.tick_params(axis="x", labelsize=12.5, width=1.1, length=4.5)
    ax_c.tick_params(axis="y", labelsize=12.5, colors=color_mm, width=1.1, length=4.5)
    ax_c.spines["left"].set_color(color_mm)
    ax_c.set_xlim(0, 2000)

    ax_c2 = ax_c.twinx()
    l2, = ax_c2.plot(time_s, voltage_s, color=color_sf, linewidth=1.4, label="Stopped-flow FL")
    ax_c2.spines["top"].set_visible(False)
    ax_c2.set_ylabel("Voltage (V)", fontsize=14, color=color_sf, labelpad=6)
    ax_c2.tick_params(axis="y", labelsize=12.5, colors=color_sf, width=1.1, length=4.5)
    ax_c2.spines["right"].set_color(color_sf)
    ax_c2.spines["left"].set_visible(False)

    if fit_panel_c_sf:
        fit_and_plot_optional(
            ax_c2, sf_df, "time", "voltage",
            model=fit_panel_c_sf, fit_start=0.0, fit_end=1000.0,
            color=color_sf, label_prefix="Stopped-flow"
        )

    ax_c.text(
        0.05, 0.95, "C",
        transform=ax_c.transAxes,
        fontsize=15.5, fontweight="bold",
        va="top", ha="left"
    )
    ax_c.legend(
        handles=[l1, l2],
        fontsize=11.0,
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
        loc="upper right",
        bbox_to_anchor=(0.76, 0.98)
    )

    # -----------------------------------------------------
    # ZAPIS DO PLIKÓW PDF I PNG (300 DPI)
    # -----------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.pdf")
    png_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    print(f"Zapisano:\n  - {pdf_path}\n  - {png_path}")
    plt.show()


if __name__ == "__main__":
    build_figure_6_3_fluorescence(
        fit_panel_a=None,      # np. "exponential"
        fit_panel_b=None,      # np. "sigmoid"
        fit_panel_c_sf=None,   # np. "exponential"
    )