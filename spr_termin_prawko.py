import time
import random
import requests
from datetime import datetime

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
MAX_TARGET_DATE = "2026-09-28"
# ==========================================
# 3. DANE ZAPYTANIA (INFO-KIEROWCA)
# ==========================================
API_URL = "https://info-kierowca.pl/bknd/exam/api/v1/Schedules/user/OneCenterExam"

COOKIES = {
    'CookieScriptConsent': '{"googleconsentmap":{"ad_storage":"targeting","analytics_storage":"performance","ad_personalization":"targeting","ad_user_data":"targeting","functionality_storage":"functionality","personalization_storage":"functionality","security_storage":"functionality"},"bannershown":1,"action":"accept","consenttime":1756813420,"categories":"[\\"performance\\",\\"targeting\\",\\"functionality\\",\\"unclassified\\"]","key":"04236399-ce48-4152-a174-56f2d71cb850"}',
    '__Secure-PUDOJT': 'eyJhbGciOiJSUzI1NiIsImtpZCI6Ik16QXdPRVpFUmtFMU1VWTVSREkzUkVZeE9EYzJNekpDTVRrNFF6ZEdPVVV4TkRBelJFUkZSUSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJQT1JUQUwgRS1VU8WBVUcgUFdQVyIsImV4cCI6MTc4NzMzMDkzMywiaWF0IjoxNzg3MzMwMDMzLCJuYmYiOjE3ODczMzAwMjgsInN1YiI6ImMzZjBjOTRlLTQ0NjEtNDFhOC1hN2NiLWRiMTc5NmNiOGY4YSIsInNpZCI6ImMzZjBjOTRlLTQ0NjEtNDFhOC1hN2NiLWRiMTc5NmNiOGY4YSJ9.ifPuqi38qabY2BKFJoXMPlbe9PsB_iIvt0ZHO1VaCUdgIbH6KMMKzWgz2fYa_YdHT5OR0ddrAqKX0Y6U5U-VztTUjC2GdkXrUvvNVvBFfnpNKefkNZA_6m-iDThaTh12bItPTtMOZp6_TrwRP-kqRO-RFEL2K8Kes51FmwbCQZKR3TrgDM_2MTf991lszfdmROvDwBLwy98ZD0GQihQSagdP3Uo0JN_34lL2fuGw9HIkQ1d65FGXkxwqh5m_BWVHHDG5W7ZcDhIWli_OEfOJLmVH0hamnmKjf6B17sMFx85UD-dI4yAAjbD_zha84AA26DOz3rgWRmNair80a4tqpAf1fhWZIRVWnGvQTWZxV2pmcNz00lp6b0D0_rNqgnbcM61lSAqjsDJZ3FlJkgbMIjrznbmoNIi7wa7THF8rhBkIU0IGFnWt1Z8JlyYtxRSzqiB9Moh3bERAWH6TyBCvEV38DC96Je9zCAGwmP1RkYMAoWyVv6AHZ2ioQ0jwYMU_dzqRb82ZA5Svjagt7tV6_1eRP--Yd8jgP7aiKvLXE_Z6qPVggp4WkZSS-8Wx39zeQ0hneXiK2FXuYjCbgsK_JhYG8v0jiJAsOUGar4QPr7jq85zFvTFnYD4sUEPp86s9XG8VFcNmXAyxETJ3EeWyl8QAdWnALFjPT59B0AZGHRM',
    '__Secure-PUDOJTMD': 'eyJtYXhBZ2UiOjkwMCwiZXhwaXJlcyI6MTc4NzMzMDkzMywiaXNzdWVkQXQiOjE3ODczMzAwMzN9',
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
    'startDate': '2026-08-23',
    'organizationId': [26],
    'category': 5,
    'profileNumber': '72501092495855042122',
    'profileType': 'Pkk',
}


# ==========================================
# 4. FUNKCJE POMOCNICZE
# ==========================================
def send_telegram_alert(message: str):
    """Wysyła powiadomienie push na Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[!] Błąd wysyłania na Telegram: {e}")


def extract_slots(response_json):
    """Parsuje strukturę JSON i wyciąga listę wolnych slotów spełniających warunek daty."""
    found_slots = []

    # API PWPW najczęściej zwraca listę dni w tablicy lub w obiekcie 'response' / 'schedule'
    schedule_data = response_json if isinstance(response_json, list) else response_json.get("schedule",
                                                                                            response_json.get(
                                                                                                "response", []))

    for item in schedule_data:
        # Wyciąganie daty (np. "2026-09-02" lub ISO "2026-09-02T08:00:00")
        date_str = item.get("date", item.get("day", ""))[:10]

        if date_str and date_str <= MAX_TARGET_DATE:
            # Sprawdzanie dostępnych godzin
            hours = item.get("hours", item.get("times", []))
            if hours:
                for h in hours:
                    time_label = h if isinstance(h, str) else h.get("time", "")
                    found_slots.append(f"{date_str} {time_label}".strip())
            else:
                found_slots.append(date_str)

    return found_slots


def check_word_slots():
    """Wysyła żądanie do API PWPW i przetwarza odpowiedź."""
    try:
        res = requests.post(API_URL, cookies=COOKIES, headers=HEADERS, json=JSON_DATA, timeout=15)

        # Obsługa wygaśnięcia ciasteczek sesyjnych
        if res.status_code in (401, 403):
            print("[!] Sesja wygasła (401/403). Wymagane odświeżenie ciasteczek __Secure-PUDOJT.")
            send_telegram_alert(
                "⚠️ *Sesja Info-Kierowca wygasła!* Zaloguj się w przeglądarce i podmień ciasteczka w kodzie.")
            return None

        if res.status_code != 200:
            print(f"[!] Nieoczekiwany kod błędu HTTP: {res.status_code}")
            return []

        data = res.json()
        return extract_slots(data)

    except requests.exceptions.RequestException as e:
        print(f"[!] Błąd połączenia sieciowego: {e}")
        return []


# ==========================================
# 5. GŁÓWNA PĘTLA
# ==========================================
def main():
    print("[*] Bot monitorujący WORD Bemowo uruchomiony...")
    send_telegram_alert("🚀 *Bot uruchomiony!* Monitoruję terminy na Bemowie przed " + MAX_TARGET_DATE)

    seen_slots = set()

    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Sprawdzam kalendarz...")

        slots = check_word_slots()

        # Jeśli sesja padła, odczekaj 5 minut przed kolejną próbą
        if slots is None:
            time.sleep(300)
            continue

        new_slots = [s for s in slots if s not in seen_slots]

        if new_slots:
            msg = f"🚗 *Zwolnił się termin na Bemowie!*\n\n"
            msg += "\n".join([f"• `{s}`" for s in new_slots])
            msg += "\n\n🔗 Rezerwuj od razu: https://info-kierowca.pl/reservation"

            print(f"[+] ZNALEZIONO NOWY TERMIN: {new_slots}")
            send_telegram_alert(msg)
            seen_slots.update(new_slots)
        else:
            print("[-] Brak nowych terminów.")

        # Odstęp 60–110 sekund, aby utrzymać sesję żywą (heartbeat) i uniknąć blokad
        sleep_time = random.randint(60, 110)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()