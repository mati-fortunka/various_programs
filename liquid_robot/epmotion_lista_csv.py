import os
import math

# --- KONFIGURACJA NAGŁÓWKA PLIKU CSV ---
CSV_HEADER = """Labware;SRC.Barcode;SRC.List Name;Dest.Barcode;Dest.List Name;;;;;;;;;;;
;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;
Barcode ID;Labware;Source;Labware;Destination;Volume;Tool;Name"""


def get_well_name(index):
    """Zwraca nazwę dołka na płytce 96-dołkowej (wypełnianie kolumnami: A1, B1... H1, A2...)"""
    row = chr(65 + (index % 8))  # 65 to 'A' w ASCII
    col = (index // 8) + 1
    return f"{row}{col}"


def parse_input_file(filepath):
    """Wczytuje plik i zwraca listę krotek: (vol_protein, vol_buffer, vol_gucl)"""
    data = []
    if not os.path.exists(filepath):
        print(f"BŁĄD: Nie znaleziono pliku {filepath}")
        return data

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Pomijamy pierwszy wiersz (nagłówek)
        for line in lines:
            if not line.strip():
                continue
            # Dzielimy po białych znakach (tabulatory/spacje)
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    # Zamiana przecinków na kropki, aby Python mógł je przeczytać jako float
                    prot_vol = float(parts[2].replace(',', '.'))
                    buf_vol = float(parts[3].replace(',', '.'))
                    gucl_vol = float(parts[4].replace(',', '.'))
                    data.append((prot_vol, buf_vol, gucl_vol))
                except ValueError:
                    print(f"Błąd parsowania wartości w linii: {line.strip()}")
    return data


def get_tool(volume):
    """Zwraca odpowiednie narzędzie w zależności od objętości"""
    return "TS_50" if volume <= 50.0 else "TS_1000"


def calculate_starts(len1, len2, len3):
    """Oblicza indeksy startowe dla poszczególnych białek na płytce"""
    total = len1 + len2 + len3

    # 1. Czy w ogóle zmieszczą się na płytce?
    if total > 96:
        print("\nUWAGA: Liczba próbek przekracza 96 dołków!")
        print(f"Próbki zostaną ucięte do 96 (ułożone ciągiem, bez pustych kolumn).\n")
        return [0, len1, len1 + len2], True  # True oznacza, że musimy ucinać

    # 2. Próbujemy ułożyć z przeskakiwaniem do nowej kolumny
    start1 = 0
    # math.ceil(len / 8) * 8 daje indeks początku pierwszej wolnej kolumny
    start2_skip = math.ceil(len1 / 8) * 8
    start3_skip = start2_skip + math.ceil(len2 / 8) * 8

    if start3_skip + len3 <= 96:
        print("\nUdało się rozmieścić białka tak, by każde zaczynało się w nowej kolumnie.")
        return [start1, start2_skip, start3_skip], False
    else:
        print("\nBrak miejsca na rozdzielenie białek do nowych kolumn. Próbki zostaną ułożone ciągiem.")
        return [0, len1, len1 + len2], False


def generate_csv(filename, operations):
    """Sortuje operacje i zapisuje do pliku CSV we wskazanym formacie"""
    # Sortowanie: 1. Nazwa źródła (alfabetycznie), 2. Objętość (malejąco)
    sorted_ops = sorted(operations, key=lambda x: (x['source'], -x['volume']))

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(CSV_HEADER + "\n")
        for op in sorted_ops:
            # Format: ;1;<Source>;1;<Destination>;<Volume>;<Tool>;
            # Zamieniamy kropki z powrotem na format zrozumiały dla systemu robota, jeśli to potrzebne.
            # W przykładzie były użyte kropki (np. 192.5), więc zostawiamy kropki.
            line = f";1;{op['source']};1;{op['dest']};{op['volume']};{op['tool']};"
            f.write(line + "\n")
    print(f"Wygenerowano plik: {filename}")


def main():
    print("--- Konfigurator Pipetowania ---")

    # Pobieranie nazw plików
    file1 = input("Podaj nazwę 1. pliku z białkiem (np. bialko1.txt): ")
    file2 = input("Podaj nazwę 2. pliku z białkiem (np. bialko2.txt): ")
    file3 = input("Podaj nazwę 3. pliku z białkiem (np. bialko3.txt): ")

    # Pobieranie źródeł
    src_prot1 = input("Podaj źródło dla 1. białka (np. 1_1_A1): ")
    src_prot2 = input("Podaj źródło dla 2. białka (np. 1_1_A2): ")
    src_prot3 = input("Podaj źródło dla 3. białka (np. 1_1_A3): ")

    src_buf = input("Podaj źródło dla buforu [domyślnie: 1_3_A1]: ") or "1_3_A1"
    src_gucl = input("Podaj źródło dla GuCl [domyślnie: 1_3_B1]: ") or "1_3_B1"

    # Wczytywanie danych
    data1 = parse_input_file(file1)
    data2 = parse_input_file(file2)
    data3 = parse_input_file(file3)

    len1, len2, len3 = len(data1), len(data2), len(data3)
    if len1 == 0 and len2 == 0 and len3 == 0:
        print("Brak danych do przetworzenia. Zamykam program.")
        return

    # Obliczanie pozycji na płytce
    starts, truncate = calculate_starts(len1, len2, len3)

    # Grupowanie list
    datasets = [
        (data1, starts[0], src_prot1),
        (data2, starts[1], src_prot2),
        (data3, starts[2], src_prot3)
    ]

    protein_ops = []
    buffer_ops = []
    gucl_ops = []

    # Generowanie operacji
    for data, start_idx, src_prot in datasets:
        for i, (p_vol, b_vol, g_vol) in enumerate(data):
            well_idx = start_idx + i

            # Jeżeli musimy uciąć listę, bo przekracza 96 dołków
            if well_idx >= 96:
                break

            dest_well = get_well_name(well_idx)

            # Dodajemy tylko operacje gdzie objętość jest większa od zera
            if p_vol > 0:
                protein_ops.append({'source': src_prot, 'dest': dest_well, 'volume': p_vol, 'tool': get_tool(p_vol)})
            if b_vol > 0:
                buffer_ops.append({'source': src_buf, 'dest': dest_well, 'volume': b_vol, 'tool': get_tool(b_vol)})
            if g_vol > 0:
                gucl_ops.append({'source': src_gucl, 'dest': dest_well, 'volume': g_vol, 'tool': get_tool(g_vol)})

    # Zapis do plików
    print("\nTrwa generowanie plików...")
    generate_csv("protein_pipetting.csv", protein_ops)
    generate_csv("buffer_pipetting.csv", buffer_ops)
    generate_csv("gucl_pipetting.csv", gucl_ops)
    print("\nGotowe! Pliki znajdują się w folderze ze skryptem.")


if __name__ == "__main__":
    main()