import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import re

# --- KONFIGURACJA ---
URL_SEARCH = "https://spispracownikow.uw.edu.pl/"
INPUT_FILE = "/home/matifortunka/Downloads/lista_WB_promotorzy.txt"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
TARGET_INSTITUTE = "Instytut Mikrobiologii"


def parse_promoters_file(filepath):
    """Wczytuje listę i usuwa duplikaty."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    unique_promoters = []
    seen = set()
    titles_to_remove = r'(prof\.|dr|hab\.|inż\.)'

    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue

        name_part = re.sub(titles_to_remove, '', clean_line).strip()
        name_part = " ".join(name_part.split())
        words = name_part.split()

        if len(words) >= 2:
            first_name = words[0]
            last_name = " ".join(words[1:])
            full_identity = (first_name.lower(), last_name.lower())

            if full_identity not in seen:
                seen.add(full_identity)
                unique_promoters.append({"first_name": first_name, "last_name": last_name})

    return unique_promoters


def check_profile_page(link, session):
    """Wchodzi głębiej w link profilu, aby znaleźć jednostkę."""
    try:
        r = session.get(link, headers=HEADERS)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        # Pobieramy cały tekst ze strony profilowej
        full_text = soup.get_text(separator=' ', strip=True)
        # Czyścimy go podwójnych spacji
        full_text = " ".join(full_text.split())

        if TARGET_INSTITUTE.lower() in full_text.lower():
            return full_text  # Zwracamy treść potwierdzającą
        return None
    except Exception as e:
        print(f"   [!] Błąd przy sprawdzaniu profilu: {e}")
        return None


def scrape_employee(first_name, last_name, session):
    results = []
    data = {
        "imie": first_name,
        "nazwisko": last_name,
        "strona": "1"
    }

    try:
        r = session.post(URL_SEARCH, data=data, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")

        table = soup.find("table")
        if not table: return []

        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2:
                # Pobieramy podstawowe dane z tabeli
                full_name = " ".join(cols[0].get_text(separator=' ').split())
                link_tag = cols[0].find("a")

                # Sprawdzamy tekst w całym wierszu (wszystkie kolumny na raz)
                row_text = " ".join(row.get_text(separator=' ').split())

                is_match = False
                unit_source = "Tabela"

                # KROK 1: Sprawdzamy czy widać instytut w tabeli (szybkie sprawdzenie)
                if TARGET_INSTITUTE.lower() in row_text.lower():
                    is_match = True

                # KROK 2: Jeśli nie widać w tabeli, a jest link -> wchodzimy w profil (dokładne sprawdzenie)
                elif link_tag:
                    profile_url = URL_SEARCH + link_tag["href"]
                    # print(f"   ...wchodzę w profil: {profile_url}") # Odkomentuj jeśli chcesz widzieć
                    profile_text = check_profile_page(profile_url, session)

                    if profile_text:
                        is_match = True
                        unit_source = "Profil_Głęboki"

                if is_match:
                    print(f"   >>> TRAFIENIE ({unit_source})! {full_name}")
                    results.append({
                        "Szukane": f"{first_name} {last_name}",
                        "Znalezione_Nazwisko": full_name,
                        "Jednostka": TARGET_INSTITUTE,  # Wpisujemy na sztywno, bo potwierdziliśmy obecność
                        "Link": URL_SEARCH + link_tag["href"] if link_tag else ""
                    })
                else:
                    print(f"   [Info] {full_name} - brak afiliacji '{TARGET_INSTITUTE}'")

        return results
    except Exception as e:
        print(f"Błąd przy {first_name} {last_name}: {e}")
        return []


# --- URUCHOMIENIE ---

people = parse_promoters_file(INPUT_FILE)
print(f"Startujemy! Lista unikalna: {len(people)} osób.")
print(f"Szukam afiliacji: '{TARGET_INSTITUTE}' (w tabeli lub w profilu)\n")

session = requests.Session()
all_results = []

for i, person in enumerate(people):
    print(f"[{i + 1}/{len(people)}] Sprawdzam: {person['last_name']} {person['first_name']}")
    found = scrape_employee(person['first_name'], person['last_name'], session)
    if found:
        all_results.extend(found)
    # Odstęp jest ważny, żeby nie zbanowali przy wchodzeniu w profile
    time.sleep(0.5)

if all_results:
    df = pd.DataFrame(all_results)
    output_name = "promotorzy_mikrobiologia_fixed.csv"
    df.to_csv(output_name, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 30)
    print(f"SUKCES! Znaleziono {len(all_results)} osób.")
    print(f"Wyniki w pliku: {output_name}")
else:
    print(
        "\nNadal brak wyników. To oznacza, że albo nazwa instytutu jest inna w bazie, albo ci ludzie tam nie pracują.")