import re
import csv
import math
from collections import defaultdict
from pathlib import Path

# Path to your input text file
input_file_path = "/home/matifortunka/Documents/JS/data_Cambridge/8_3/A/Alpha_SF"
folder = "/".join(input_file_path.split("/")[:-1])

# Read and parse the text file
with open(input_file_path, "r", encoding="utf-8") as f:
    lines = f.read().strip().splitlines()

# Extract protein name
protein_name = next(line for line in lines if line.strip())

# Patterns
urea_pattern = re.compile(r'^(\d+M)$')
fit_header_pattern = re.compile(r'^(.*\.csv) \((.*) fit\):$')
param_pattern = re.compile(r'^(\w+) = ([\d\.\-eE]+) ± ([\d\.\-eE]+)$')

# Data structure
parsed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

current_urea = None
current_file = None
current_fit_type = None

# Parse data
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue

    if (match := urea_pattern.match(line)):
        current_urea = match.group(1)
        continue

    if (match := fit_header_pattern.match(line)):
        current_file, current_fit_type = match.groups()
        continue

    if (match := param_pattern.match(line)) and current_urea and current_file and current_fit_type:
        param, value, error = match.groups()
        parsed_data[current_urea][current_fit_type][current_file][param] = (value, error)

# Collect all parameter names
all_params = set()
for fits in parsed_data.values():
    for files in fits.values():
        for params in files.values():
            all_params.update(params.keys())

# Check if t_half needs to be calculated from k, k1, k2
ln2 = math.log(2)
computed_thalves = ['k', 'k1', 'k2']
for urea, fits in parsed_data.items():
    for fit_type, files in fits.items():
        for file, params in files.items():
            for k_name in computed_thalves:
                if k_name in params:
                    th_name = 't_half' if k_name == 'k' else f't_half{k_name[-1]}'
                    if th_name not in params:
                        try:
                            k_val = float(params[k_name][0])
                            k_err = float(params[k_name][1])
                            if k_val != 0:
                                th_val = ln2 / k_val
                                th_err = abs(ln2 / (k_val ** 2) * k_err)
                                params[th_name] = (f"{th_val:.4f}", f"{th_err:.4f}")
                        except Exception as e:
                            # skip if conversion fails
                            continue
                        all_params.add(th_name)

# Ensure consistent order of columns
ordered_params = sorted(all_params)

# Prepare CSV header
header = ['Urea', 'Filename', 'Fit Type']
for p in ordered_params:
    header += [p, f'{p}_err']

# Collect rows
rows = []
for urea, fits in parsed_data.items():
    for fit_type, files in fits.items():
        for file, params in files.items():
            row = [urea, file, fit_type]
            for p in ordered_params:
                if p in params:
                    row += [params[p][0], params[p][1]]
                else:
                    row += ['', '']
            rows.append(row)

# Output CSV
output_filename = f"{folder}/{protein_name}_fits_summary.csv"
with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f"# {protein_name}"])
    writer.writerow(header)
    writer.writerows(rows)

print(f"CSV file created: {output_filename}")
