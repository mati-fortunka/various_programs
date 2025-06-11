import os
import shutil
import subprocess
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# CONFIG
PULCHRA_BIN = "pulchra"
MKDSSP_BIN = "mkdssp"
FRAMES_DIR = "frames"  # where pdbs from traj are saved
REBUILT_DIR = "rebuilt"  # full atom PDBs
DSSP_DIR = "dssp"  # per-frame DSSP
TEST_MODE = False  # Set to True to test on subset
TEST_N = 10        # Use first 10 or every Nth if TEST_MODE is True
EVERY_NTH = 100

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

def pulchra_one(pdb_path):
    try:
        cmd = f"{PULCHRA_BIN} {pdb_path}"
        run_cmd(cmd)
        rebuilt = pdb_path.replace('.pdb', '.rebuilt.pdb')
        out_path = os.path.join(REBUILT_DIR, os.path.basename(rebuilt))
        shutil.move(rebuilt, out_path)
    except Exception as e:
        print(f"Failed PULCHRA: {pdb_path} - {e}")

def dssp_one(pdb_path):
    try:
        out_file = os.path.join(DSSP_DIR, os.path.basename(pdb_path).replace('.pdb', '.dssp'))
        cmd = f"{MKDSSP_BIN} -i {pdb_path} -o {out_file}"
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

def save_results(results, out_file="ss_summary.csv"):
    with open(out_file, "w") as f:
        f.write("frame,alpha,beta,coil\n")
        for i, r in enumerate(results):
            f.write(f"{i},{r['alpha']:.3f},{r['beta']:.3f},{r['coil']:.3f}\n")

def main():
    pulchra_all()
    dssp_all()
    results = analyze_all()
    save_results(results)

if __name__ == "__main__":
    main()
