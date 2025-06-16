import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter
import os

path = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/simulations/u2/ss_summary.csv"
dir = os.path.dirname(path)

# Load your CSV file
df = pd.read_csv(path)

# Apply Savitzky-Golay smoothing (window size must be odd and < len(data))
smoothing = True
window = 41  # adjust if needed
poly = 3     # polynomial order

# Apply smoothing safely
def safe_savgol(series, window, poly):
    return savgol_filter(series, window_length=min(window, len(series) // 2 * 2 + 1), polyorder=min(poly, 2))
if smoothing:
    df['alpha'] = safe_savgol(df['alpha'], window, poly)
    df['beta'] = safe_savgol(df['beta'], window, poly)
    df['coil'] = safe_savgol(df['coil'], window, poly)

# Detect changes in topology
df['topology_change'] = df['topology'].ne(df['topology'].shift())
change_indices = df.index[df['topology_change']].tolist()
change_indices.append(len(df))  # Add the final row for the last span

# Colors for each topology
topo_colors = {
    "8_3": "#e0f7fa",
    "+6_1": "#f1f8e9",
    "-6_1": "#fce4ec",
    "4_1": "#fff3e0",
    "0_1": "#ede7f6"
}

# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))

# Plot smoothed alpha, beta, coil
ax.plot(df['frame'], df['alpha'], label='Alpha', color='blue')
ax.plot(df['frame'], df['beta'], label='Beta', color='green')
ax.plot(df['frame'], df['coil'], label='Coil', color='orange')

# Background shading for topologies
seen_topologies = set()
for i in range(len(change_indices) - 1):
    start = change_indices[i]
    end = change_indices[i + 1]
    topo = df.loc[start, 'topology']
    color = topo_colors.get(topo, "#ffffff")
    label = topo if topo not in seen_topologies else None
    ax.axvspan(df.loc[start, 'frame'], df.loc[end - 1, 'frame'], color=color, alpha=1, label=label)
    seen_topologies.add(topo)

# Topology background legend only
topo_patches = [mpatches.Patch(color=c, label=t) for t, c in topo_colors.items()]
# Filter out patches created by axvspan (they have no linestyle)
handles, labels = ax.get_legend_handles_labels()
filtered = [(h, l) for h, l in zip(handles, labels) if not isinstance(h, mpatches.Patch)]
legend1 = ax.legend([h for h, _ in filtered], [l for _, l in filtered], loc='upper right', title="Secondary Structure")
legend2 = ax.legend(handles=topo_patches, title='Topology', loc='lower right')
ax.add_artist(legend1)

# Labels and title
ax.set_xlabel('Frame')
ax.set_ylabel('Fraction')
ax.set_title('Smoothed Secondary Structure Elements and Topology Changes')

plt.tight_layout()
plt.savefig(dir+"/pulchra_dssp.png")
plt.show()
