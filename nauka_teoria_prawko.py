import csv
import os
import time
import platform
import subprocess

# ==========================================
# 1. ZMIENNE KONFIGURACYJNE I USTAWIENIA
# ==========================================

CSV_FILE = './Pytania_B_122025.csv'          # Ścieżka do pliku z pytaniami
MEDIA_DIR = '.'                  # Folder z multimediami
CATEGORY = 'B'                   # Kategoria prawa jazdy
CHUNK_SIZE = 100                 # Liczba pytań w jednym bloku
TIME_LIMIT = 20                   # Czas na odpowiedź w sekundach
TRACK_POINTS = True              
SAVE_MISTAKES = True             
MISTAKES_FILE = 'bledy.txt'      # Plik na błędne odpowiedzi
TOO_LONG_FILE = 'too_long.txt'   # NOWOŚĆ: Plik na poprawne, ale zbyt wolne odpowiedzi

START_FROM_QUESTION = 1       

# --- TRYBY DZIAŁANIA ---
LOAD_MISTAKES_ONLY = True       
LOAD_CUSTOM_FILE = False         
CUSTOM_QUESTIONS_FILE = 'numery_pytan.txt' 

# ==========================================
# 2. FUNKCJE POMOCNICZE
# ==========================================

last_media_filename = None

def load_ids_from_file(filename):
    """Wczytuje listę numerów pytań z pliku do zbioru (set)."""
    if not os.path.exists(filename):
        return set()
    with open(filename, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def open_media(media_filename, force=False):
    global last_media_filename
    if not media_filename:
        last_media_filename = None
        return
    if media_filename == last_media_filename and not force:
        return
    filepath = os.path.join(MEDIA_DIR, media_filename)
    if not os.path.exists(filepath):
        print(f"\n[!] Błąd: Nie znaleziono pliku multimedialnego -> {filepath}")
        return
    if platform.system() == 'Windows':
        os.startfile(filepath)
    elif platform.system() == 'Darwin': 
        subprocess.call(('open', filepath))
    else: 
        subprocess.call(('xdg-open', filepath))
    last_media_filename = media_filename

def get_questions():
    questions = []
    filter_ids = set()
    if LOAD_MISTAKES_ONLY:
        filter_ids = load_ids_from_file(MISTAKES_FILE)
    elif LOAD_CUSTOM_FILE:
        filter_ids = load_ids_from_file(CUSTOM_QUESTIONS_FILE)
    try:
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                categories = [c.strip() for c in row['Kategorie'].split(',')]
                if CATEGORY in categories:
                    if (LOAD_MISTAKES_ONLY or LOAD_CUSTOM_FILE) and row['Numer pytania'] not in filter_ids:
                        continue
                    questions.append(row)
    except FileNotFoundError:
        print(f"Błąd krytyczny: Nie znaleziono pliku {CSV_FILE}.")
        exit()
    return questions

# ==========================================
# 3. GŁÓWNA LOGIKA PROGRAMU
# ==========================================

def ask_question(q, internal_index):
    print("\n" + "="*60)
    print(f"Pytanie nr {internal_index} (Baza ID: {q['Numer pytania']})")
    print(f"Pytanie: {q['Pytanie']}")
    print("-" * 60)
    
    is_basic = (q['Zakres struktury'].strip().upper() == 'PODSTAWOWY' or not q['Odpowiedź A'])
    if not is_basic:
        print(f"A: {q['Odpowiedź A']}\nB: {q['Odpowiedź B']}\nC: {q['Odpowiedź C']}")
        
    if q['Media']:
        open_media(q['Media'].strip())
        
    correct_ans = q['Poprawna odp'].strip().upper()
    valid_choices = ['T', 'N'] if is_basic else ['A', 'B', 'C']
    prompt_text = f"\nWybierz ({'/'.join(valid_choices)}) lub 'M' (media): "
    
    start_time = time.time()
    while True:
        user_ans = input(prompt_text).strip().upper()
        if user_ans == 'M':
            if q['Media']:
                open_media(q['Media'].strip(), force=True)
            continue
        if user_ans in valid_choices:
            break
        print(f"Nieprawidłowy klawisz.")
            
    elapsed_time = time.time() - start_time
    is_timeout = (TIME_LIMIT > 0 and elapsed_time > TIME_LIMIT)
    
    if is_timeout:
        print(f"\n[!] Czas minął! Odpowiedź po {elapsed_time:.1f}s.")
            
    is_correct = (user_ans == correct_ans)
    if is_correct:
        print(">>> DOBRZE! <<<")
    else:
        print(f">>> ŹLE! <<< Poprawna: {correct_ans}")
        
    return is_correct, is_timeout

def study_mode():
    print(f"--- ROZPOCZYNAMY NAUKĘ (Kategoria: {CATEGORY}) ---")
    questions = get_questions()
    if not questions:
        print("Brak pytań.")
        return

    points = 0
    mistakes_made = []
    timeouts_made = []        
    corrected_mistakes = []   
    questions_processed = 0
    
    for idx, q in enumerate(questions, start=1):
        if idx < START_FROM_QUESTION:
            continue
        if questions_processed > 0 and questions_processed % CHUNK_SIZE == 0:
            print("\n" + "*"*60)
            if input(f"Przerobiłeś {CHUNK_SIZE} pytań. Kontynuować? (T/N): ").strip().upper() != 'T':
                break
                
        is_correct, is_timeout = ask_question(q, idx)
        questions_processed += 1
        
        if is_correct:
            if not is_timeout:
                points += 1
                if LOAD_MISTAKES_ONLY:
                    corrected_mistakes.append(q['Numer pytania'])
            else:
                # Poprawnie, ale za długo
                timeouts_made.append(q['Numer pytania'])
        else:
            # Błąd (niezależnie od czasu)
            mistakes_made.append(q['Numer pytania'])
            
        time.sleep(0.5) 
        
    # PODSUMOWANIE
    print("\n" + "="*60)
    print(f"KONIEC SESJI. Wynik: {points}/{questions_processed}")

    # 1. OBSŁUGA PYTAŃ "ZBYT DŁUGICH" (NOWOŚĆ)
    if timeouts_made:
        print(f"\n[INFO] {len(timeouts_made)} pytań zajęło Ci zbyt dużo czasu.")
        existing_long = load_ids_from_file(TOO_LONG_FILE)
        new_long = [t for t in timeouts_made if t not in existing_long]
        
        if new_long:
            with open(TOO_LONG_FILE, 'a', encoding='utf-8') as f:
                for t in new_long:
                    f.write(t + "\n")
            print(f"Dopisano te numery do pliku: {TOO_LONG_FILE}")

    # 2. OBSŁUGA BŁĘDÓW
    if SAVE_MISTAKES and mistakes_made:
        existing_mistakes = load_ids_from_file(MISTAKES_FILE)
        new_mistakes = [m for m in mistakes_made if m not in existing_mistakes]
        if new_mistakes:
            with open(MISTAKES_FILE, 'a', encoding='utf-8') as f:
                for m in new_mistakes:
                    f.write(m + "\n")
            print(f"Zapisano {len(new_mistakes)} nowych błędów do: {MISTAKES_FILE}")

    # 3. CZYSZCZENIE BŁĘDÓW
    if LOAD_MISTAKES_ONLY and corrected_mistakes:
        if input(f"\nUsunąć {len(corrected_mistakes)} pytań z listy błędów? (T/N): ").strip().upper() == 'T':
            current_mistakes = load_ids_from_file(MISTAKES_FILE)
            updated = current_mistakes - set(corrected_mistakes)
            with open(MISTAKES_FILE, 'w', encoding='utf-8') as f:
                for m in updated: f.write(m + "\n")
            print("Lista błędów zaktualizowana.")

if __name__ == "__main__":
    study_mode()
