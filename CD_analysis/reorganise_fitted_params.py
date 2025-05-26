import re
import csv

# Raw multiline input string (replace this with reading from a file or stdin if needed)
raw_data = """
8_3_alpha_5uM_2000s00008.csv: Double exp fit: A1=-31.1592269±2.5112439, k1=0.0412863±0.0014325, A2=21.2165801±7.4125646, k2=0.0000761±0.0000290, c=-27.0421438±7.4320149, t_half_k1=16.79s, t_half_k2=9105.59s
8_3_alpha_5uM_2000s_nat_filt100019.csv: Double exp fit: A1=-6.2238147±0.1484326, k1=0.0054869±0.0002387, A2=-6.9281574±0.3795831, k2=0.0004800±0.0000742, c=-21.4575163±0.5510550, t_half_k1=126.33s, t_half_k2=1443.91s
8_3_alpha_5uM_2000s_nat_filt100025.csv: Double exp fit: A1=-8.0650126±0.1303482, k1=0.0081189±0.0002826, A2=2.0721864±0.0791282, k2=0.0010189±0.0001529, c=-17.6803345±0.0904764, t_half_k1=85.37s, t_half_k2=680.30s
8_3_alpha_5uM_2000s00001.csv: Double exp fit: A1=190.6758330±2385790.2969366, k1=0.0029543±0.1304458, A2=-191.8458593±2385790.3183954, k2=0.0029336±0.1287168, c=-4.9384080±0.0364277, t_half_k1=234.63s, t_half_k2=236.28s
8_3_alpha_5uM_2000s_nat_filter00015.csv: Double exp fit: A1=-10.0915735±0.1493102, k1=0.0087657±0.0002723, A2=3.7345440±0.0648262, k2=0.0008976±0.0000900, c=-18.6989991±0.1276725, t_half_k1=79.07s, t_half_k2=772.23s
8_3_alpha_5uM_2000s_nat_filt100022.csv: Double exp fit: A1=-7.6247436±0.1987103, k1=0.0128439±0.0004407, A2=5.7356238±0.0915886, k2=0.0006844±0.0000360, c=-14.9161813±0.1391522, t_half_k1=53.97s, t_half_k2=1012.73s
8_3_alpha_2000s_150ul00004.csv: Double exp fit: A1=-0.9112912±0.1166705, k1=0.0092020±0.0020956, A2=9.8998198±0.6999728, k2=0.0003111±0.0000351, c=-17.5095637±0.7606194, t_half_k1=75.33s, t_half_k2=2228.10s
8_3_alpha_2000s_150ul00006.csv: Double exp fit: A1=-15.1375771±0.5097526, k1=0.0213010±0.0005500, A2=11.8813560±0.3651772, k2=0.0003699±0.0000185, c=-20.2133272±0.3956832, t_half_k1=32.54s, t_half_k2=1873.98s
8_3_alpha_5uM_2000s_nat_filter_140ul.csv: Double exp fit: A1=-1.6786524±0.2350908, k1=0.0061128±0.0012648, A2=5.9494253±0.2063544, k2=0.0011463±0.0000813, c=-17.6984647±0.0942277, t_half_k1=113.39s, t_half_k2=604.67s
8_3_5uM_222nm_2000s00002.csv: Double exp fit: A1=-29.4248675±3.5225174, k1=0.0445520±0.0021967, A2=-1.7919200±0.0313482, k2=0.0010694±0.0000722, c=-11.4034571±0.0437664, t_half_k1=15.56s, t_half_k2=648.18s
8_3_alpha_5uM_2000s_nat_filt100020.csv: Double exp fit: A1=-2.6426314±0.1849472, k1=0.0143398±0.0017028, A2=-0.7141372±0.1171831, k2=0.0024384±0.0003961, c=-12.2310078±0.0122599, t_half_k1=48.34s, t_half_k2=284.27s
8_3_alpha_5uM_2000s00007.csv: Double exp fit: A1=-25.5405097±0.8071316, k1=0.0275770±0.0005304, A2=10.3057456±0.8749350, k2=0.0002243±0.0000249, c=-14.5644138±0.8978172, t_half_k1=25.13s, t_half_k2=3089.67s
8_3_alpha_2000s_oldurea00001.csv: Double exp fit: A1=-6.9472374±0.4180020, k1=0.0242742±0.0010283, A2=2.7779232±0.0331974, k2=0.0008223±0.0000341, c=-6.6481672±0.0493672, t_half_k1=28.55s, t_half_k2=842.96s
8_3_alpha_2000s_150ul00005.csv: Double exp fit: A1=-12.2077208±1.2410652, k1=0.0301033±0.0016500, A2=16822.1312859±1060943.2478069, k2=0.0000001±0.0000090, c=-16829.7344311±1060943.2499912, t_half_k1=23.03s, t_half_k2=4875972.58s
8_3_alpha_5uM_2000s_nat_filt100028.csv: Double exp fit: A1=-1.8524127±0.1017457, k1=0.0087619±0.0008798, A2=-7.5695393±1.7556358, k2=0.0001971±0.0000604, c=-13.7333631±1.8116933, t_half_k1=79.11s, t_half_k2=3517.56s
8_3_alpha_2000s_150ul00007.csv: Double exp fit: A1=-18.1800697±2.1891518, k1=0.0371415±0.0021095, A2=15.0130527±3.0528564, k2=0.0001307±0.0000309, c=-22.4350992±3.0773263, t_half_k1=18.66s, t_half_k2=5301.59s
8_3_alpha_5uM_2000s_nat_filt100026.csv: Double exp fit: A1=-4.2027034±0.0773758, k1=0.0068129±0.0002583, A2=-1078.6998647±360530.7821424, k2=0.0000004±0.0001474, c=1061.2401919±360530.8370435, t_half_k1=101.74s, t_half_k2=1572229.63s
8_3_alpha_5uM_2000s_nat_filter00011.csv: Double exp fit: A1=0.9517711±0.4839829, k1=0.0036166±0.0012648, A2=-1.6086943±0.3905246, k2=0.0009756±0.0003709, c=-50.7464302±0.1264259, t_half_k1=191.65s, t_half_k2=710.52s
8_3_alpha_2000s_150ul00008-poczatek _zly_dalej_dobrze.csv: Double exp fit: A1=1.6619485±0.1044706, k1=0.0112870±0.0008202, A2=-2137.1712531±321162.4590237, k2=0.0000003±0.0000429, c=2122.8561071±321162.4762495, t_half_k1=61.41s, t_half_k2=2427003.51s
8_3_5uM_6M00000.csv: Double exp fit: A1=-17.9480756±5.6297858, k1=0.0445329±0.0060260, A2=-1.2012650±0.2333804, k2=0.0012134±0.0006015, c=-7.3535886±0.3045517, t_half_k1=15.56s, t_half_k2=571.24s
"""

# Compile regex patterns
param_pattern = re.compile(r'(\w+)=([-+Ee0-9.]+)±([0-9.Ee+-]+)')
t_half_pattern = re.compile(r't_half_k1=([0-9.]+)s, t_half_k2=([0-9.]+)s')
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

    # Extract t_half values
    t_half_match = t_half_pattern.search(line)
    if t_half_match:
        params["t_half_k1"] = float(t_half_match.group(1))
        params["t_half_k2"] = float(t_half_match.group(2))

    data_rows.append(params)

# Final column order
# Sort dynamic columns except for base ones
dynamic_columns = sorted(col for col in all_columns if col not in base_columns)
final_columns = ["filename", "model"] + dynamic_columns + ["t_half_k1", "t_half_k2"]

folderpath = "/home/matifortunka/Documents/JS/data_Cambridge/8_3"

# Write to CSV
with open(f"{folderpath}/cd_fits.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=final_columns)
    writer.writeheader()
    for row in data_rows:
        writer.writerow(row)
