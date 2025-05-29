import re
import csv

# Raw multiline input string (replace this with reading from a file or stdin if needed)
raw_data = """
8_3_zeta_10ul_2000s_00006.csv: Exponential fit: A=-0.7783371±0.0292904, k=0.0035387±0.0002032, c=-8.9604439±0.0059974, t_half=195.87s
8_3_zeta_10ul_2000s_00009.csv: Exponential fit: A=-0.1832832±0.0138443, k=0.0017762±0.0003563, c=-9.2996576±0.0091423, t_half=390.24s
8_3_zeta_10ul_2000s_00003.csv: Exponential fit: A=-0.3273797±0.0168352, k=0.0023213±0.0002445, c=-6.1889235±0.0067084, t_half=298.60s
8_3_zeta_10ul_2000s_00001.csv: Exponential fit: A=-1.0225772±0.0220000, k=0.0021030±0.0001015, c=-7.7104908±0.0105035, t_half=329.61s
"""

# Compile regex patterns
param_pattern = re.compile(r'(\w+)=([-+Ee0-9.]+)±([0-9.Ee+-]+)')
t_half_k_pattern = re.compile(r't_half_k1=([0-9.]+)s, t_half_k2=([0-9.]+)s')
t_half_pattern = re.compile(r't_half=([0-9.]+)s')
model_pattern = re.compile(r':\s*(.*?)\s*:')

# Initial required fields
base_columns = {"filename", "model", "t_half_k1", "t_half_k2"}
all_columns = set(base_columns)
data_rows = []

# Parse each line
for line in raw_data.strip().split('\n'):
    filename = line.split(':')[0].strip()
    params = {"filename": filename}

    # Extract model
    model_match = model_pattern.search(line)
    if model_match:
        raw_model = model_match.group(1).lower().replace(" ", "_")
        params["model"] = raw_model
    else:
        params["model"] = "unknown"

    # Extract parameters and their errors
    for match in param_pattern.finditer(line):
        name, val, err = match.groups()
        params[name] = float(val)
        params[f"{name}_err"] = float(err)
        all_columns.update([name, f"{name}_err"])

    # Extract t_half_k1 and t_half_k2
    t_half_k_match = t_half_k_pattern.search(line)
    if t_half_k_match:
        params["t_half_k1"] = float(t_half_k_match.group(1))
        params["t_half_k2"] = float(t_half_k_match.group(2))
        all_columns.update(["t_half_k1", "t_half_k2"])
    else:
        # Try single t_half fallback
        t_half_match = t_half_pattern.search(line)
        if t_half_match:
            params["t_half"] = float(t_half_match.group(1))
            all_columns.add("t_half")

    data_rows.append(params)

# Final column order
# Sort dynamic columns except for base ones
base_columns = {"filename", "model"}
dynamic_columns = sorted(col for col in all_columns if col not in base_columns)
final_columns = ["filename", "model"] + dynamic_columns


folderpath = "/home/matifortunka/Documents/JS/data_Cambridge/8_3"

# Write to CSV
with open(f"{folderpath}/cd_fits.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=final_columns)
    writer.writeheader()
    for row in data_rows:
        writer.writerow(row)
