import pandas as pd
import re

# --- KONFIGURACJA ---
FILE_PROMOTERS = "/home/matifortunka/PycharmProjects/various_programs/promotorzy_infa_fixed.csv"  # Plik z poprzedniego etapu
FILE_STUDENTS = "/home/matifortunka/Downloads/doktoranci MIM"  # Plik z doktorantami
OUTPUT_FILE = "doktoranci_infa.csv"


def load_target_promoters(csv_path):
    """Wczytuje listę nazwisk promotorów (małe litery)."""
    try:
        df = pd.read_csv(csv_path)
        target_surnames = set()

        # Pobieramy ostatnie słowo z kolumny 'Szukane' lub 'Znalezione_Nazwisko'
        # Zakładamy, że w pliku CSV są kolumny z poprzedniego programu
        source_col = 'Szukane' if 'Szukane' in df.columns else df.columns[0]

        for val in df[source_col]:
            # Bierzemy ostatni człon (nazwisko)
            parts = str(val).strip().split()
            if parts:
                surname = parts[-1].lower()
                target_surnames.add(surname)
        return target_surnames
    except Exception as e:
        print(f"BŁĄD przy wczytywaniu promotorów: {e}")
        return set()


def parse_line_smart(line):
    """
    Dzieli linię używając adresu email jako punktu odniesienia (kotwicy).
    Zwraca: (imie, nazwisko, mail, tekst_promotora) lub None
    """
    # 1. Znajdź email (ciąg znaków z @ w środku)
    email_match = re.search(r'\s+([^\s]+@[^\s]+)\s+', line)

    if not email_match:
        return None, "Nie znaleziono emaila w linii"

    email = email_match.group(1)
    start_email, end_email = email_match.span()

    # 2. Tekst PRZED mailem to Student
    part_student = line[:start_email].strip()

    # 3. Tekst PO mailu to Promotor
    part_promoter = line[end_email:].strip()

    # 4. Rozbijamy dane studenta na Nazwisko i Imię
    # Próbujemy rozbić po dużej dziurze (2+ spacje lub tab)
    name_parts = re.split(r'\s{2,}|\t', part_student)

    if len(name_parts) >= 2:
        surname = name_parts[0].strip()
        first_name = " ".join(name_parts[1:]).strip()
    else:
        # Jeśli nie ma dużej dziury, tniemy po pierwszej spacji
        # (Ryzykowne przy nazwiskach dwuczłonowych bez myślnika, ale zazwyczaj działa)
        single_space_parts = part_student.split()
        if len(single_space_parts) >= 2:
            surname = single_space_parts[0]
            first_name = " ".join(single_space_parts[1:])
        else:
            return None, f"Nie udało się rozdzielić imienia i nazwiska w: '{part_student}'"

    return {
        "imie": first_name,
        "nazwisko": surname,
        "mail": email,
        "promotor_raw": part_promoter
    }, "OK"


def main():
    print("1. Wczytuję promotorów...")
    target_promoters = load_target_promoters(FILE_PROMOTERS)

    if not target_promoters:
        print("Brak promotorów do szukania. Sprawdź plik CSV.")
        return
    print(f"   Liczba unikalnych nazwisk promotorów: {len(target_promoters)}")
    print(f"   Przykłady z bazy: {list(target_promoters)[:3]}")

    print("\n2. Analizuję doktorantów...")
    matched_students = []

    try:
        with open(FILE_STUDENTS, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"   Wczytano {len(lines)} linii z pliku txt.")

        for i, line in enumerate(lines):
            if not line.strip(): continue

            parsed, status = parse_line_smart(line)

            # --- DEBUGOWANIE DLA PIERWSZYCH LINIJEK ---
            if i < 3:
                print(f"   [DEBUG linia {i + 1}] Status: {status}")
                if parsed:
                    print(f"      Student: {parsed['nazwisko']} {parsed['imie']}")
                    print(f"      Promotor w tekście: '{parsed['promotor_raw']}'")
            # ------------------------------------------

            if parsed:
                # Sprawdzamy czy nazwisko promotora jest w sekcji promotora
                promoter_text_lower = parsed['promotor_raw'].lower()

                for target in target_promoters:
                    # Szukamy całego słowa (żeby 'Kot' nie złapał 'Kotwicy')
                    # re.escape jest ważne, jeśli nazwiska mają kropki itp.
                    if re.search(r'\b' + re.escape(target) + r'\b', promoter_text_lower):
                        matched_students.append({
                            "Imie": parsed['imie'],
                            "Nazwisko": parsed['nazwisko'],
                            "Mail": parsed['mail']
                        })
                        # Znaleziono, przerywamy pętlę nazwisk dla tego studenta
                        break

    except FileNotFoundError:
        print(f"Brak pliku {FILE_STUDENTS}!")
        return

    print(f"\n3. Wyniki: Znaleziono {len(matched_students)} pasujących doktorantów.")

    if matched_students:
        df = pd.DataFrame(matched_students)
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"   Plik zapisany: {OUTPUT_FILE}")
        print("\n   Pierwsze 5 wyników:")
        print(df.head().to_string(index=False))
    else:
        print(
            "   Nadal 0 wyników. Sprawdź sekcję DEBUG powyżej - czy program dobrze widzi nazwiska promotorów w tekście?")


if __name__ == "__main__":
    main()