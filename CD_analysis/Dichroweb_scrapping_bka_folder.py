import os
import time
import json
import re
import socket
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==========================================
# Konfiguracja i ścieżki
# ==========================================
DICHROWEB_FORM_URL = "http://dichroweb.cryst.bbk.ac.uk/html/process.shtml"
INPUT_DIRECTORY = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/Tm1570/equilibrium/CD/urea/tm_unfold_SAV"

CLEAN_FILES_DIR = os.path.join(INPUT_DIRECTORY, "dichroweb_clean_free")
RESULTS_JSON_PATH = os.path.join(INPUT_DIRECTORY, "dichroweb_results.json")
ERROR_LOG_PATH = os.path.join(INPUT_DIRECTORY, "error_log.txt")

# Konfiguracja pierwszego formularza DichroWeb
FORM_DATA_TEMPLATE_BASE = {
    "CD-USERNAME": "wKd8Z",
    "CD-PWD": "zhdnzNX",
    "NAME": "Tm1570",
    "FORMAT": "FREE",  # Format czystych kolumn numerycznych
    "units": "theta",  # Dane wejściowe w mdeg
    "START": "190",
    "END": "250",
    "WAVESTEP": "1.0",
    "QUALDATA": "200",  # Bezpieczny punkt odcięcia jakościowego dla denaturantu
    "scale_data": "1.0",
    "output_units": "theta"
}

# Wartości drugiego formularza dla Tm1570 (stężenie 20 µM = 0.5115 mg/ml)
SECOND_FORM_VALUES = {
    "MRW": "114.165982143",
    "CONC": "0.5115",
    "PATH": "0.01"  # 0.1 mm = 0.01 cm
}

# Macierz metod i powiązanych baz referencyjnych
METHOD_BASIS_MAP = {
    "CONTIN": ["SET 4", "SET 7"],
    "SELCON3": ["SET 4", "SET 7"],
    "K2D": [None]  # K2D używa wewnętrznej sieci Kohonena, brak bazy
}


def is_dichroweb_online(host="dichroweb.cryst.bbk.ac.uk", port=80, timeout=2.0):
    """Sprawdza, czy serwer DichroWeb żyje przed uruchomieniem przeglądarki."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, OSError):
        return False


def log_error(filename, method, step, message):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"{filename},{method},{step},{message}\n")


def extract_concentration(filename):
    match = re.search(r"(\d+(?:\.\d+)?)\s*[mM](?:\.[^.]+)?$", filename)
    if match:
        return float(match.group(1))
    name_lower = filename.lower()
    if "native" in name_lower or "0m" in name_lower or "buffer" in name_lower:
        return 0.0
    return None


def convert_and_cache_bka_files(source_dir, target_dir):
    """
    Konwertuje pliki .bka do formatu FREE (czyste kolumny numeryczne).
    Zapisuje je w podfolderze, przyspieszając kolejne uruchomienia.
    """
    os.makedirs(target_dir, exist_ok=True)
    clean_files = []

    for fname in sorted(os.listdir(source_dir)):
        if not fname.endswith(".bka"):
            continue

        base_name = os.path.splitext(fname)[0]
        target_path = os.path.join(target_dir, f"{base_name}.dat")

        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            clean_files.append(target_path)
            continue

        src_path = os.path.join(source_dir, fname)
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        try:
            start_idx = next(i for i, line in enumerate(lines) if line.strip().strip('"') == "_DATA") + 1
        except StopIteration:
            continue

        clean_lines = []
        for line in lines[start_idx:]:
            l_str = line.strip()
            if not l_str or not any(c.isdigit() for c in l_str):
                break
            parts = l_str.split()
            if len(parts) >= 2:
                clean_lines.append(f"{float(parts[0]):.1f}\t{float(parts[1]):.5f}\n")

        if clean_lines:
            with open(target_path, "w", encoding="utf-8") as f_out:
                f_out.writelines(clean_lines)
            clean_files.append(target_path)

    print(f"[✓] Przygotowano {len(clean_files)} czystych plików w formacie FREE: {target_dir}")
    return clean_files


def extract_table_dataframe(html: str, method: str) -> pd.DataFrame | None:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", border="2")
    if not tables:
        tables = soup.find_all("table")

    if method == "K2D":
        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue
            headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            if headers == ["Alpha helix", "Beta sheet", "Random coil"]:
                if len(rows) > 1:
                    vals = [c.get_text(strip=True) for c in rows[1].find_all("td")]
                    if len(vals) == 3:
                        return pd.DataFrame([vals], columns=headers)

    elif method in ["CONTIN", "SELCON3"]:
        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue
            headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            if len(headers) == 8 and headers[0].lower() == "result":
                data_rows = []
                for row in rows[1:]:
                    cells = row.find_all("td")
                    if len(cells) == 8:
                        vals = [c.get_text(strip=True) for c in cells]
                        if method == "CONTIN" and vals[0].isdigit():
                            data_rows.append(vals)
                        elif method == "SELCON3" and vals[0] in ["Guess", "SVD", "Convergent", "Stage2", "final"]:
                            data_rows.append(vals)
                if data_rows:
                    df_res = pd.DataFrame(data_rows, columns=headers)
                    return df_res.iloc[[-1]]

    return None


def submit_file_selenium(driver, filepath, method, basis, all_results):
    form_data = FORM_DATA_TEMPLATE_BASE.copy()
    form_data["prog"] = method
    if basis is not None:
        form_data["basis"] = basis

    filename = os.path.basename(filepath)
    conc = extract_concentration(filename)
    print(f"[→] Processing: {filename} ({conc}M) | {method} | Basis: {basis if basis else 'Default'}")

    loaded = False
    for attempt in range(3):
        try:
            driver.get(DICHROWEB_FORM_URL)
            loaded = True
            break
        except Exception as e:
            print(f"[!] Próba {attempt + 1}/3 nieudana dla {DICHROWEB_FORM_URL}: {e}")
            time.sleep(3)

    if not loaded:
        print(f"[✗] Serwer DichroWeb nie odpowiada. Przerywam zadanie dla {filename}.")
        log_error(filename, method, "connect", "Server timeout")
        return

    # 1. Wypełnianie pierwszego formularza
    for field, value in form_data.items():
        try:
            element = driver.find_element(By.NAME, field)
            tag = element.tag_name.lower()
            if tag == "input":
                if element.get_attribute("type") == "radio":
                    for btn in driver.find_elements(By.NAME, field):
                        if btn.get_attribute("value") == value:
                            btn.click()
                            break
                else:
                    element.clear()
                    element.send_keys(value)
            elif tag == "select":
                for option in element.find_elements(By.TAG_NAME, "option"):
                    if option.get_attribute("value") == value:
                        option.click()
                        break
        except Exception:
            pass

    # Wysłanie oczyszczonego pliku formatu FREE
    try:
        file_input = driver.find_element(By.NAME, "FILE")
        file_input.send_keys(filepath)
    except Exception as e:
        log_error(filename, method, "upload", str(e))
        return

    try:
        submit_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@type='IMAGE' and @name='submit data']"))
        )
        submit_btn.click()
    except Exception as e:
        log_error(filename, method, "step1", str(e))
        return

    # 2. Drugi formularz: parametry próbki (MRW, CONC, PATH)
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "MRW")))
        for field, value in SECOND_FORM_VALUES.items():
            try:
                el = driver.find_element(By.NAME, field)
                el.clear()
                el.send_keys(value)
            except Exception:
                pass

        submit_btn2 = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//form[@name='hide']//input[@type='image' and contains(@src, 'continue.jpg')]")
            )
        )
        driver.execute_script("arguments[0].click();", submit_btn2)
    except Exception as e:
        log_error(filename, method, "step2", str(e))
        return

    # 3. Potwierdzenie parametrów (databox confirmation)
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "databox")))
        confirm_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//form[@name='hide']//input[@type='image' and contains(@src, 'continue.jpg')]")
            )
        )
        driver.execute_script("arguments[0].click();", confirm_btn)
    except Exception as e:
        log_error(filename, method, "step3", str(e))
        return

    # 4. Przejście do wyników kompaktowych
    try:
        result_xpath = {
            "SELCON3": "//a[starts-with(@href, 'sel.pl')]",
            "K2D": "//a[starts-with(@href, 'k2d.pl')]",
            "CONTIN": "//a[starts-with(@href, 'contin.pl')]",
        }[method]

        compact_link = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, result_xpath))
        )
        driver.execute_script("arguments[0].click();", compact_link)

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//table[@border='2']"))
        )
        html = driver.page_source

        df = extract_table_dataframe(html, method)
        if df is None:
            raise Exception("Nie znaleziono tabeli wyników na stronie.")

        df.insert(0, "Filename", filename)
        df.insert(1, "Concentration", conc)
        all_results.append(df)
        print(f"[✓] Sparsowano: {filename}")

    except Exception as e:
        log_error(filename, method, "results", str(e))


def run_dichroweb_batch():
    clean_files = convert_and_cache_bka_files(INPUT_DIRECTORY, CLEAN_FILES_DIR)
    if not clean_files:
        print("Brak poprawnych plików do przetworzenia.")
        return

    print("[→] Sprawdzanie dostępności serwera DichroWeb...")
    if not is_dichroweb_online():
        print("❌ Serwer dichroweb.cryst.bbk.ac.uk nie odpowiada (connection timeout/refused).")
        print("   Spróbuj ponownie później, gdy administratorzy podniosą usługę.")
        return
    print("[✓] Połączenie z serwerem nawiązane pomyślnie.")

    if os.path.exists(ERROR_LOG_PATH):
        os.remove(ERROR_LOG_PATH)

    options = Options()
    options.binary_location = "/usr/bin/google-chrome"
    # options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument("--disable-features=HttpsFirstBalancedMode,HttpsUpgrades")
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2
    })
    options.page_load_strategy = "eager"

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)
    database = {}

    try:
        for method, basis_list in METHOD_BASIS_MAP.items():
            database[method] = {}
            for basis in basis_list:
                basis_label = basis if basis else "Default"
                print(f"\n==========================================")
                print(f"🚀 Uruchamianie: {method} | Baza: {basis_label}")
                print(f"==========================================")

                all_results = []
                for filepath in clean_files:
                    driver.delete_all_cookies()
                    submit_file_selenium(driver, filepath, method, basis, all_results)
                    time.sleep(1.0)

                if all_results:
                    final_df = pd.concat(all_results, ignore_index=True)
                    if "Concentration" in final_df.columns:
                        final_df.sort_values(by="Concentration", inplace=True)

                    basis_tag = f"_set{basis.split()[-1]}" if basis else ""
                    csv_name = f"{method.lower()}{basis_tag}.csv"
                    csv_path = os.path.join(INPUT_DIRECTORY, csv_name)
                    final_df.to_csv(csv_path, index=False)
                    print(f"[✓] Zapisano CSV: {csv_path}")

                    database[method][basis_label] = final_df.to_dict(orient="records")

        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f_json:
            json.dump(database, f_json, indent=2, ensure_ascii=False)
        print(f"\n[★] Zapisano całą bazę wyników do JSON: {RESULTS_JSON_PATH}")

    finally:
        driver.quit()

    plot_from_json(RESULTS_JSON_PATH)


def plot_from_json(json_path=RESULTS_JSON_PATH):
    """
    Rysuje panele porównawcze dla wszystkich metod bezpośrednio z pliku JSON.
    """
    if not os.path.exists(json_path):
        print(f"Plik JSON nie istnieje: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for method, bases in data.items():
        for basis_name, records in bases.items():
            if records:
                tasks.append((method, basis_name, pd.DataFrame(records)))

    if not tasks:
        print("Brak danych do wykreślenia w JSON.")
        return

    n_plots = len(tasks)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (method, basis_name, df) in zip(axes, tasks):
        if "Concentration" not in df.columns:
            continue

        df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
        df.dropna(subset=["Concentration"], inplace=True)
        df.sort_values(by="Concentration", inplace=True)
        x = df["Concentration"].values

        if method == "K2D":
            alpha = pd.to_numeric(df["Alpha helix"], errors="coerce")
            beta = pd.to_numeric(df["Beta sheet"], errors="coerce")
            coil = pd.to_numeric(df["Random coil"], errors="coerce")
        else:
            h_cols = [c for c in df.columns if any(k in c.lower() for k in ["helix", "alpha"])]
            b_cols = [c for c in df.columns if any(k in c.lower() for k in ["strand", "beta", "sheet"])]
            c_cols = [c for c in df.columns if any(k in c.lower() for k in ["unord", "coil", "turn", "other"])]

            alpha = df[h_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if h_cols else None
            beta = df[b_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if b_cols else None
            coil = df[c_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if c_cols else None

        if alpha is not None:
            ax.plot(x, alpha, label="Alpha", marker="o", color="crimson", lw=1.8)
        if beta is not None:
            ax.plot(x, beta, label="Beta", marker="s", color="royalblue", lw=1.8)
        if coil is not None:
            ax.plot(x, coil, label="Coil/Other", marker="^", color="darkorange", lw=1.8)

        ax.set_title(f"{method} ({basis_name})", fontsize=12)
        ax.set_xlabel("Urea Conc. [M]", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Structure Fraction", fontsize=12)
            ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    plot_path = os.path.join(INPUT_DIRECTORY, "dichroweb_comparison_all.png")
    fig.savefig(plot_path, dpi=300)
    print(f"[✓] Wygenerowano wykres zbiorczy: {plot_path}")
    plt.show()


if __name__ == "__main__":
    run_dichroweb_batch()
    # Gdy zechcesz tylko zmienić kolory/wykres bez uruchamiania przeglądarki,
    # zakomentuj run_dichroweb_batch() i wywołaj samo:
    # plot_from_json(RESULTS_JSON_PATH)
