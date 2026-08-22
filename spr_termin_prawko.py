import time
import random
import requests
from datetime import datetime
# import json
# import re

# ==========================================
# 1. KONFIGURACJA TELEGRAM
# ==========================================
TELEGRAM_BOT_TOKEN = "8560988995:AAEqMnsdMsuQFpDiUoZN0BOtwt3bg_tn3So"
TELEGRAM_CHAT_ID = "2075949423"

# ==========================================
# 2. FILTR TERMINÓW
# ==========================================
# Maksymalna data, która Cię interesuje (format YYYY-MM-DD)
# Terminy równe lub wcześniejsze wywołają powiadomienie
MAX_TARGET_DATE = "2026-09-02"
# Słownik ID ośrodków w Twojej okolicy
CENTERS = {
    26: "Warszawa Bemowo",
    25: "Warszawa Odlewnicza",
    31: "WORD Siedlce",
    19: "WORD Skierniewice",
    31001: "WORD Garwolin",
    32004: "WORD Grójec",
    30001: "WORD Sochaczew"
}

# ==========================================
# 3. ENDPOINTY I POCZĄTKOWE DANE
# ==========================================
URL_SCHEDULES = "https://info-kierowca.pl/bknd/exam/api/v1/Schedules/user/MultipleCentersExams"
URL_REFRESH = "https://info-kierowca.pl/bknd/auth/api/v1/jwt/refresh"

# Początkowe ciasteczka z Twojego logowania
INITIAL_COOKIES = {
    'CookieScriptConsent': '{"googleconsentmap":{"ad_storage":"targeting","analytics_storage":"performance","ad_personalization":"targeting","ad_user_data":"targeting","functionality_storage":"functionality","personalization_storage":"functionality","security_storage":"functionality"},"bannershown":1,"action":"accept","consenttime":1756813420,"categories":"[\\"performance\\",\\"targeting\\",\\"functionality\\",\\"unclassified\\"]","key":"04236399-ce48-4152-a174-56f2d71cb850"}',
    '__Secure-PUDOJT': 'eyJhbGciOiJSUzI1NiIsImtpZCI6Ik16QXdPRVpFUmtFMU1VWTVSREkzUkVZeE9EYzJNekpDTVRrNFF6ZEdPVVV4TkRBelJFUkZSUSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJQT1JUQUwgRS1VU8WBVUcgUFdQVyIsImV4cCI6MTc4NzM1MzY0NCwiaWF0IjoxNzg3MzUyNzQ0LCJuYmYiOjE3ODczNTI3MzksInN1YiI6ImRiM2M5MDFkLThkNzEtNDFhMC04N2ZlLTUyZTAzMzA1MmVkNSIsInNpZCI6Ijk3MWY3N2VlLThmMjgtNDc0ZS1hYmY1LWIwNTRmZDkwNzliYiJ9.xwF_wrMac_Gg6mRlVKDqjLv8MaIAoDFJKzpQ9Cur-AHv8BNTFYjS1-lkZ03Pk-ECzg61evqvO-ruvUwyXkfjifoVdZMJsYj5-m7GSEFVyQ1Fopj7Su4jwwnIDCeUL5Ivl9g3jtGiwQ2HL4I5v11N5-h1ft1QMs7SJDyOoEdxGvjbjcLnXB7wsQmHhUX-2ApaELt5Y5X9SuCYEjMJJzSlh2TnTREV9U2f7AP7d32wl_cL0tfroAREkikC2_700wnJTgZqcyPjePyCC2UcWo227-r3-c4_nr0p26DasADhPXqNC3whXEyFY85qPFJmqVy7wNJuhV7qDuqa8M0T6UYsQ8xSTlGXaJX_0zJW1M1pqn6Y8BZG4bblZ5N9MkFrgPbXYVzY_Lf1bWVZMf5l5UXsmpW-Vtrc5KHk5EUnRhtG0siACjtjvNi1gGywhb-2Yaf0rI9zCX0ikf_Skw6oRmgvgAKxZ8sTB_PjkkCxIkq1LYW3hQEmobLYi4d6z_XwC9hXOpU6_Ln8uqzgPW_wXZYxJo-4y9NeL_Kt_7BVViNb3H8VsKBLeOW3tOA1VSiJwMB38BLm_r9cNyiPMFC6dn4a38cpVTFKTEJBQcS0qh467BzMVXx2ZxPJtSGoMUEa4DxLZ9DN2GOhtEKfBrk-fHpIuC1re5FuGiKudYLdNwHdpZY',
    '__Secure-PUDOJTMD': 'eyJtYXhBZ2UiOjkwMCwiZXhwaXJlcyI6MTc4NzM1MzY0NCwiaXNzdWVkQXQiOjE3ODczNTI3NDR9',
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:152.0) Gecko/20100101 Firefox/152.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'pl,en-US;q=0.9,en;q=0.8',
    'Referer': 'https://info-kierowca.pl/reservation',
    'Content-Type': 'application/json',
    'Origin': 'https://info-kierowca.pl',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'Priority': 'u=4',
}

JSON_DATA = {
    'startDate': '2026-08-25',
    'organizationId': [26, 25, 31, 32004, 30001],
    'category': 5,
    'profileNumber': '72501092495855042122',
    'profileType': 'Pkk',
}

# Tworzymy stałą sesję HTTP
session = requests.Session()
session.headers.update(HEADERS)
session.cookies.update(INITIAL_COOKIES)


# ==========================================
# 4. LOGIKA SKRYPTU
# ==========================================
def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[!] Błąd wysyłania na Telegram: {e}")


def refresh_session_tokens():
    """Wysyła żądanie do endpointu refresh, aby odnowić token na kolejne 15 minut."""
    try:
        res = session.get(URL_REFRESH, timeout=10)
        if res.status_code == 200:
            print("[🔄] Sesja PWPW pomyślnie odświeżona!")
            return True
        else:
            print(f"[!] Błąd odświeżania sesji: status {res.status_code}")
            return False
    except Exception as e:
        print(f"[!] Błąd sieci podczas refresh: {e}")
        return False


def parse_schedule(data):
    results = []
    max_date_obj = datetime.strptime(MAX_TARGET_DATE, "%Y-%m-%d").date()

    exams_to_process = []
    if isinstance(data, list):
        for center in data:
            exams_to_process.extend(center.get("examCollectionForDay", []))
    elif isinstance(data, dict):
        for day in data.get("examCollectionForDay", []):
            exams_to_process.extend(day.get("examCollections", []))

    for exam in exams_to_process:
        if exam.get("examType") == "Practice" and exam.get("placePracticeAmount", 0) > 0:
            raw_dt = exam.get("practiceDateTime")
            if raw_dt:
                try:
                    dt_obj = datetime.fromisoformat(raw_dt)
                    if dt_obj.date() <= max_date_obj:
                        org_id = exam.get("organizationId")
                        center_name = exam.get("organizationName") or CENTERS.get(org_id, f"WORD #{org_id}")
                        date_str = dt_obj.strftime("%Y-%m-%d")
                        time_str = dt_obj.strftime("%H:%M")
                        places = exam.get("placePracticeAmount")
                        info = exam.get("additionalInfo") or ""
                        info_str = f" [{info}]" if info else ""

                        entry = f"{center_name}: {date_str} {time_str} (miejsc: {places}){info_str}"
                        if entry not in results:
                            results.append(entry)
                except ValueError:
                    continue
    return results


def check_slots():
    try:
        res = session.post(URL_SCHEDULES, json=JSON_DATA, timeout=15)

        if res.status_code in (401, 403):
            print("[!] Sesja wygasła (401/403). Próba natychmiastowego odświeżenia...")
            if refresh_session_tokens():
                # Ponawiamy zapytanie po odświeżeniu
                res = session.post(URL_SCHEDULES, json=JSON_DATA, timeout=15)
            else:
                send_telegram_alert("⚠️ *Sesja Info-Kierowca wygasła!* Wymagane ponowne logowanie.")
                return None

        if res.status_code != 200:
            print(f"[!] Kod błędu: {res.status_code}")
            return []

        return parse_schedule(res.json())

    except Exception as e:
        print(f"[!] Błąd połączenia: {e}")
        return []


# ==========================================
# 5. GŁÓWNA PĘTLA
# ==========================================
def main():
    print("[*] Bot monitorujący PWPW uruchomiony...")
    seen_slots = set()
    last_refresh_time = time.time()

    while True:
        # Odświeżaj sesję co 8 minut (480 sekund), aby token nigdy nie dobił do 15 min limitu
        if time.time() - last_refresh_time > 480:
            refresh_session_tokens()
            last_refresh_time = time.time()

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Sprawdzam wolne terminy...")

        slots = check_slots()

        if slots is None:
            time.sleep(300)
            continue

        new_slots = [s for s in slots if s not in seen_slots]

        if new_slots:
            msg = "🚗 *Zwolnił się termin na egzamin!*\n\n"
            msg += "\n".join([f"• `{s}`" for s in new_slots])
            msg += "\n\n🔗 Rezerwuj: https://info-kierowca.pl/reservation"
            print(f"[+] Nowy termin: {new_slots}")
            send_telegram_alert(msg)
            seen_slots.update(new_slots)
        else:
            print("[-] Brak nowych terminów.")

        # Losowy interwał 60–100 sekund
        time.sleep(random.randint(60, 100))


if __name__ == "__main__":
    main()