import os
import shutil
import subprocess
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--frames_dir', required=True)
args = parser.parse_args()

FRAMES_DIR = args.frames_dir

# CONFIG
PULCHRA_BIN = "/home/users/mfortunka/83-trajectories/pulchra/pulchra"
MKDSSP_BIN = os.path.expanduser("~/dssp/build/mkdssp")
LIBCIFPP_DATA_DIR=os.path.expanduser("~/libcifpp_data")
#FRAMES_DIR = "./A0A7X8EQ82/366_61/frames"  # where pdbs from traj are saved
REBUILT_DIR = f"{FRAMES_DIR}/rebuilt_fixed"  # full atom PDBs
DSSP_DIR = f"{FRAMES_DIR}/dssp"  # per-frame DSSP
TEST_MODE = False  # Set to True to test on subset
TEST_N = 200        # Use first 10 or every Nth if TEST_MODE is True
EVERY_NTH = 100
TOPOLOGY_INTERPOLATE = True

os.environ["LIBCIFPP_DATA_DIR"] = os.path.expanduser("~/libcifpp_data")

os.makedirs(REBUILT_DIR, exist_ok=True)
os.makedirs(DSSP_DIR, exist_ok=True)

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"ERROR in cmd: {cmd}\n{result.stderr.decode()}")
    return result

def get_test_subset(files):
    if TEST_MODE:
        return files[:TEST_N] if TEST_N else files[::EVERY_NTH]
    return files

def fix_line(line):
    if not line.startswith("ATOM"):
        return line
    line = line.rstrip("\n").ljust(80)
    atom_name = line[12:16].strip()
    element = ''.join([c for c in atom_name if c.isalpha()])[:2].rjust(2)
    fixed = line[:21] + "A" + line[22:]  # Ensure chain ID
    fixed = fixed[:76] + element.rjust(2) + fixed[78:]  # Fix element symbol
    return fixed + "\n"

def fix_pdb_file(in_path, out_path):
    with open(in_path) as f:
        lines = f.readlines()
    fixed = [fix_line(l) for l in lines]
    with open(out_path, "w") as f:
        f.writelines(fixed)

def pulchra_one(pdb_path):
    try:
        cmd = f"{PULCHRA_BIN} {pdb_path}"
        run_cmd(cmd)
        rebuilt = pdb_path.replace('.pdb', '.rebuilt.pdb')
        out_path = os.path.join(REBUILT_DIR, os.path.basename(rebuilt))
        shutil.move(rebuilt, out_path)
    except Exception as e:
        print(f"Failed PULCHRA: {pdb_path} - {e}")

def fix_all_rebuilt_pdbs():
    pdb_files = sorted(glob(os.path.join(REBUILT_DIR, "*.pdb")))
    pdb_files = get_test_subset(pdb_files)
    for pdb in tqdm(pdb_files, desc="Fixing PDBs"):
        fix_pdb_file(pdb, pdb)  # In-place fixing

def load_topology_map(topology_file="y.temp", max_frame=None):
    topo_raw = {}
    with open(topology_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                frame = int(parts[0])
                topo = parts[1]
                if topo == "TMC":
                    topo = "8_3"
                topo_raw[frame] = topo

    if not TOPOLOGY_INTERPOLATE:
        return topo_raw

    # Interpolate missing frames if requested
    if max_frame is None:
        max_frame = max(topo_raw.keys(), default=0)

    topo_map = {}
    sorted_frames = sorted(topo_raw.items())
    current_topo = "NA"
    j = 0
    for i in range(max_frame + 1):
        if j < len(sorted_frames) and i == sorted_frames[j][0]:
            current_topo = sorted_frames[j][1]
            j += 1
        topo_map[i] = current_topo
    return topo_map

def dssp_one(pdb_path):
    try:
        out_file = os.path.join(DSSP_DIR, os.path.basename(pdb_path).replace('.pdb', '.dssp'))
        cmd = f"{MKDSSP_BIN} {pdb_path} {out_file}"
        run_cmd(cmd)
    except Exception as e:
        print(f"Failed DSSP: {pdb_path} - {e}")

def pulchra_all():
    pdbs = sorted(glob(os.path.join(FRAMES_DIR, '*.pdb')))
    pdbs = get_test_subset(pdbs)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(pulchra_one, pdbs), total=len(pdbs), desc="Rebuilding with PULCHRA"))

def dssp_all():
    pdbs = sorted(glob(os.path.join(REBUILT_DIR, '*.pdb')))
    pdbs = get_test_subset(pdbs)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(dssp_one, pdbs), total=len(pdbs), desc="Running DSSP"))

def parse_dssp_file(dssp_path):
    with open(dssp_path) as f:
        lines = f.readlines()
    start = False
    counts = {"H": 0, "E": 0, "C": 0}
    helix_codes = {"H", "G", "I"}
    beta_codes = {"E", "B"}
    for line in lines:
        if line.startswith("  #"):
            start = True
            continue
        if start and len(line) > 16:
            code = line[16]
            if code in helix_codes:
                counts["H"] += 1
            elif code in beta_codes:
                counts["E"] += 1
            else:
                counts["C"] += 1
    total = sum(counts.values())
    if total == 0:
        return {"alpha": 0, "beta": 0, "coil": 0}
    return {
        "alpha": counts["H"] / total,
        "beta": counts["E"] / total,
        "coil": counts["C"] / total
    }

def analyze_all():
    dssp_files = sorted(glob(os.path.join(DSSP_DIR, '*.dssp')))
    dssp_files = get_test_subset(dssp_files)
    results = []
    for dssp in tqdm(dssp_files, desc="Analyzing DSSP"):
        ss = parse_dssp_file(dssp)
        results.append(ss)
    return results

def save_results(results, out_file="ss_summary.csv", topology_file="y.temp"):
    topo_map = load_topology_map(topology_file, max_frame=len(results) - 1)
    with open(out_file, "w") as f:
        f.write("frame,alpha,beta,coil,topology\n")
        for i, r in enumerate(results):
            topo = topo_map.get(i, "NA")
            f.write(f"{i},{r['alpha']:.3f},{r['beta']:.3f},{r['coil']:.3f},{topo}\n")

def main():
    pulchra_all()
    fix_all_rebuilt_pdbs()  # <-- Insert this step
    dssp_all()
    results = analyze_all()
    topology_file = os.path.join(os.path.dirname(FRAMES_DIR), "y.temp")
    save_results(results, out_file=os.path.join(os.path.dirname(FRAMES_DIR), "ss_summary.csv"), topology_file=topology_file)



if __name__ == "__main__":
    main()