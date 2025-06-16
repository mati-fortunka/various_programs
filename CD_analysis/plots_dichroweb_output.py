import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# --- Konfiguracja ---
INPUT_DIRECTORY = '/home/matifortunka/Documents/JS/data_Cambridge/8_3/Maciek_CD/unfolding_SAV'
OUTPUT_DIRECTORY = '/home/matifortunka/Documents/JS/data_Cambridge/8_3/Maciek_CD/unfolding_SAV'


def extract_concentration(filename: str) -> float | None:
    match = re.search(r'(\d+\.?\d*)m', filename)
    if match:
        return float(match.group(1))
    return None


def plot_multiple_denaturation_subplots(data_list, titles, method_name, output_filename):
    """
    Rysuje wiele subplotów na jednym wykresie (jeden na każdy plik wejściowy).
    """
    num_plots = len(data_list)
    cols = 2
    rows = (num_plots + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)
    plt.style.use('seaborn-v0_8-whitegrid')

    for idx, (df, title) in enumerate(zip(data_list, titles)):
        if df.empty:
            continue
        ax = axes[idx // cols][idx % cols]

        df_sorted = df.set_index('Concentration').sort_index()
        df_sorted.plot(ax=ax, marker='o', linestyle='-')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Denaturant concentration (M)", fontsize=11)
        ax.set_ylabel("Secondary structure fraction", fontsize=11)
        ax.legend(title="Structure")

    plt.tight_layout()
    full_output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
    plt.savefig(full_output_path, dpi=300)
    print(f"✅ Plot saved as: {full_output_path}")
    plt.close(fig)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to English for plotting.
    """
    return df.rename(columns={
        'α-helisa': 'Alpha helix',
        'β-kartka': 'Beta sheet',
        'Random Coil': 'Random Coil',
        'α-helisa (Helix1+2)': 'Alpha helix',
        'β-kartka (Strand1+2)': 'Beta sheet',
        'Skręty (Turns)': 'Turns',
        'Nieuporządkowana (Unordered)': 'Unordered'
    })


def process_k2d(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['Concentration'] = df['Filename'].apply(extract_concentration)
    df_processed = df[['Concentration', 'Alpha helix', 'Beta sheet', 'Random coil']].copy()

    return df_processed


def process_contin(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['Concentration'] = df['Filename'].apply(extract_concentration)
    numeric_cols = ['Helix1', 'Helix2', 'Strand1', 'Strand2', 'Turns', 'Unordered']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    df['Alpha'] = df['Helix1'] + df['Helix2']
    df['Beta'] = df['Strand1'] + df['Strand2']
    df_to_average = df[['Concentration', 'Alpha', 'Beta', 'Turns', 'Unordered']]
    df_final = df_to_average.groupby('Concentration').mean().reset_index()

    return df_final


def process_selcon3(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df_final = df[df['Result'] == 'final'].copy()
    if df_final.empty:
        print(f"⚠️ No 'final' rows in {os.path.basename(filepath)}.")
        return pd.DataFrame()
    df_final['Concentration'] = df_final['Filename'].apply(extract_concentration)
    numeric_cols = ['Helix1', 'Helix2', 'Strand1', 'Strand2', 'Turns', 'Unordered']
    for col in numeric_cols:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    df_final.dropna(subset=numeric_cols, inplace=True)
    df_final['Alpha'] = df_final['Helix1'] + df_final['Helix2']
    df_final['Beta'] = df_final['Strand1'] + df_final['Strand2']
    df_processed = df_final[['Concentration', 'Alpha', 'Beta', 'Turns', 'Unordered']]
    return df_processed


def main():
    processors = {
        'k2d': process_k2d,
        'contin': process_contin,
        'selcon3': process_selcon3
    }

    for method, processor_func in processors.items():
        search_pattern = os.path.join(INPUT_DIRECTORY, f"*{method}*.csv")
        files_found = glob.glob(search_pattern)

        if not files_found:
            print(f"❌ No files found for method: {method}")
            continue

        data_list = []
        titles = []

        for filepath in files_found:
            try:
                df = processor_func(filepath)
                data_list.append(df)
                titles.append(os.path.basename(filepath))
            except Exception as e:
                print(f"🚨 Error processing {filepath}: {e}")

        if data_list:
            plot_title = f"Denaturation profiles - {method.upper()}"
            output_file = f"denaturation_subplots_{method}.png"
            plot_multiple_denaturation_subplots(data_list, titles, method, output_file)

        print("-" * 40)


main()
