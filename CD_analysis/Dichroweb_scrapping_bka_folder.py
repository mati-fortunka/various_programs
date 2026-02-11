import os
import time
import csv
import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
DICHROWEB_FORM_URL = "http://dichroweb.cryst.bbk.ac.uk/html/process.shtml"
CHROMEDRIVER_PATH = "/home/matifortunka/Programs/chromedriver-linux64/chromedriver"
INPUT_DIRECTORY = "/home/matifortunka/Documents/JS/kinetics_stability/data_Cambridge/8_3/Maciek_CD/unfolding_SAV"
ERROR_LOG_PATH = os.path.join(INPUT_DIRECTORY, "error_log.txt")

FORM_DATA_TEMPLATE_BASE = {
    "CD-USERNAME": "wKd8Z",
    "CD-PWD": "zhdnzNX",
    "NAME": "zeta",
    "FORMAT": "AP1",
    "units": "theta",
    "START": "190",
    "END": "250",
    "WAVESTEP": "1.0",
    "QUALDATA": "190",
    "scale_data": "1.0",
    "output_units": "theta"
}

SECOND_FORM_VALUES = {
    "MRW": "113.4811273",
    "CONC": "0.8153619",
    "PATH": "0.01"
}

METHODS = ["SELCON3"]
BASIS_SETS = ["SET 4"]
# METHODS = ["SELCON3", "CONTIN", "K2D"]
#  = ["SET 4", "SET 7", "SET 10"]


def log_error(filename, method, step, message):
    with open(ERROR_LOG_PATH, "a") as log:
        log.write(f"{filename},{method},{step},{message}\n")


def extract_table_dataframe(html: str, method: str) -> pd.DataFrame | None:
    """
    Przetwarza zawartość HTML w celu znalezienia i wyodrębnienia tabeli wyników
    dla konkretnej, podanej metody (CONTIN, SELCON3, lub K2D).
    """
    soup = BeautifulSoup(html, "html.parser")
    # Używamy selektora, który szuka tabel z atrybutem border="2" - to dobry trop na tej stronie
    tables = soup.find_all("table", border="2")
    if not tables:
        tables = soup.find_all("table")  # Awaryjnie przeszukaj wszystkie tabele

    # Logika dla metody K2D
    if method == "K2D":
        for table in tables:
            rows = table.find_all("tr")
            if not rows: continue

            header_cells = rows[0].find_all(["th", "td"])
            headers = [cell.get_text(strip=True) for cell in header_cells]

            # Szukamy unikalnych nagłówków tabeli K2D
            if headers == ["Alpha helix", "Beta sheet", "Random coil"]:
                if len(rows) > 1:
                    data_row_cells = rows[1].find_all("td")
                    if len(data_row_cells) == 3:
                        values = [cell.get_text(strip=True) for cell in data_row_cells]
                        return pd.DataFrame([values], columns=headers)

    # Logika dla metod CONTIN i SELCON3 (mają podobną strukturę)
    elif method in ["CONTIN", "SELCON3"]:
        for table in tables:
            rows = table.find_all("tr")
            if not rows: continue

            header_cells = rows[0].find_all(["th", "td"])
            headers = [cell.get_text(strip=True) for cell in header_cells]

            # Obie tabele mają 8 kolumn i pierwsza nazywa się 'Result'
            if len(headers) == 8 and headers[0].lower() == "result":
                data_rows_list = []
                for row in rows[1:]:
                    cells = row.find_all("td")
                    if len(cells) == 8:
                        values = [cell.get_text(strip=True) for cell in cells]

                        if method == "CONTIN" and values[0].isdigit():
                            data_rows_list.append(values)
                        elif method == "SELCON3" and values[0] in ["Guess", "SVD", "Convergent", "Stage2", "final"]:
                            data_rows_list.append(values)

                if data_rows_list:
                    return pd.DataFrame(data_rows_list, columns=headers)

    # Zwróć None, jeśli nie znaleziono odpowiedniej tabeli dla podanej metody
    return None


def submit_file_selenium(driver, filepath, method, basis, all_results):
    FORM_DATA_TEMPLATE = FORM_DATA_TEMPLATE_BASE.copy()
    FORM_DATA_TEMPLATE["prog"] = method
    FORM_DATA_TEMPLATE["basis"] = basis

    filename = os.path.basename(filepath)
    path = os.path.dirname(filepath)
    print(f"[→] Processing: {filename} with method {method}, basis {basis}")

    driver.get(DICHROWEB_FORM_URL)

    for field, value in FORM_DATA_TEMPLATE.items():
        try:
            element = driver.find_element(By.NAME, field)
            tag = element.tag_name.lower()
            if tag == "input":
                if element.get_attribute("type") == "radio":
                    radio_buttons = driver.find_elements(By.NAME, field)
                    for btn in radio_buttons:
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
        except Exception as e:
            print(f"[!] Skipped field {field}: {e}")

    try:
        file_input = driver.find_element(By.NAME, "FILE")
        file_input.send_keys(filepath)
    except Exception as e:
        print(f"[✗] File upload failed: {e}")
        log_error(filename, method, "upload", str(e))
        return

    print("[→] Submitting first form...")
    try:
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@type='IMAGE' and @name='submit data']"))
        )
        submit_button.click()
        print("[✓] Step 1 submitted")
    except Exception as e:
        log_error(filename, method, "step1", str(e))
        return

    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "MRW")))
        for field, value in SECOND_FORM_VALUES.items():
            try:
                input_element = driver.find_element(By.NAME, field)
                input_element.clear()
                input_element.send_keys(value)
            except Exception as e:
                print(f"[!] Could not fill field {field}: {e}")

        print("[→] Submitting second form...")
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//form[@name='hide']//input[@type='image' and contains(@src, 'continue.jpg')]"))
        )
        driver.execute_script("arguments[0].click();", submit_button)
        print("[✓] Step 2 submitted")
    except Exception as e:
        log_error(filename, method, "step2", str(e))
        return

    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "databox")))
        print("[→] Submitting final confirmation form...")
        confirm_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//form[@name='hide']//input[@type='image' and contains(@src, 'continue.jpg')]"))
        )
        driver.execute_script("arguments[0].click();", confirm_button)
        print("[✓] Final confirmation submitted")
    except Exception as e:
        log_error(filename, method, "step3", str(e))
        return

    try:
        print("[→] Clicking Compact Results 'SHOW' button...")
        result_xpath = {
            "SELCON3": "//a[starts-with(@href, 'sel.pl')]",
            "K2D": "//a[starts-with(@href, 'k2d.pl')]",
            "CONTIN": "//a[starts-with(@href, 'contin.pl')]",
        }[method]

        compact_result_link = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, result_xpath))
        )
        driver.execute_script("arguments[0].click();", compact_result_link)
        print("[✓] Navigated to final results page")

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//table[@border='2']"))
        )
        html = driver.page_source

        # Przekazujemy 'method' do funkcji, aby wiedziała, czego szukać
        df = extract_table_dataframe(html, method)
        if df is None:
            raise Exception(f"Table extraction failed for method {method}")

        df.insert(0, "Filename", filename)
        all_results.append(df)
        print(f"[✓] Parsed results for {filename}")

    except Exception as e:
        log_error(filename, method, "results", str(e))
        return


def batch_submit_files():
    all_files = sorted([
        os.path.join(INPUT_DIRECTORY, f)
        for f in os.listdir(INPUT_DIRECTORY)
        if f.endswith(".bka")
    ])

    if os.path.exists(ERROR_LOG_PATH):
        os.remove(ERROR_LOG_PATH)

    options = Options()
    options.binary_location = "/usr/bin/google-chrome"
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)

    try:
        for method in METHODS:
            for basis in BASIS_SETS:
                print(f"\n[***] Running method: {method} | Basis: {basis}")
                all_results = []
                for filepath in all_files:
                    if os.path.isfile(filepath):
                        submit_file_selenium(driver, filepath, method, basis, all_results)
                        time.sleep(2)

                if all_results:
                    final_df = pd.concat(all_results, ignore_index=True)
                    basis_number = basis.split()[-1]  # e.g., 'SET 4' -> '4'
                    output_file = os.path.join(INPUT_DIRECTORY, f"{method.lower()}_set{basis_number}.csv")
                    final_df.to_csv(output_file, index=False)
                    print(f"[✓] Results for {method} ({basis}) saved to {output_file}")

    finally:
        driver.quit()

# Run all methods and basis sets on all input files
batch_submit_files()
