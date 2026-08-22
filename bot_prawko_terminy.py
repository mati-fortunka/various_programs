import time
import random
from datetime import datetime
import requests
from playwright.sync_api import sync_playwright

# ==========================================
# 1. KONFIGURACJA
# ==========================================
TELEGRAM_BOT_TOKEN = "8560988995:AAEqMnsdMsuQFpDiUoZN0BOtwt3bg_tn3So"
TELEGRAM_CHAT_ID = "2075949423"

# Maksymalna data egzaminu (YYYY-MM-DD)
MAX_TARGET_DATE = "2026-09-11"

CENTERS = {
    26: "Warszawa Bemowo",
    25: "Warszawa Odlewnicza",
    31: "WORD Siedlce",
    31001: "WORD Garwolin",
    32004: "WORD Grójec",
    30001: "WORD Sochaczew"
}

def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[!] Błąd Telegram: {e}")

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

def main():
    seen_slots = set()
    print("[*] Uruchamianie silnika Chromium z trwałym profilem...")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir="./pwpw_session_profile",
            headless=False
        )
        page = context.new_page()

        print("[*] Otwieram stronę info-kierowca.pl...")
        page.goto("https://info-kierowca.pl/reservation")

        print("\n" + "="*60)
        print(">>> ZALOGUJ SIĘ W PRZEGLĄDARCE PRZEZ mOBYWATEL <<<")
        print(">>> PRZEJDŹ DO KROKU WYBORU TERMINÓW <<<")
        print(">>> GDY ZOBACZYSZ LISTĘ TERMINÓW, WCISNIJ ENTER TUTAJ <<<")
        print("="*60 + "\n")
        input()

        print("[*] Rozpoczynam stały monitoring bezpośrednio przez kontekst przeglądarki...")

        payload = {
            'startDate': '2026-08-25',
            'organizationId': [26, 25, 31, 31001, 32004],  # Bemowo, Odlewnicza, Siedlce, Garwolin, Grójec
            'category': 5,
            'profileNumber': '72501092495855042122',
            'profileType': 'Pkk',
        }

        api_url = "https://info-kierowca.pl/bknd/exam/api/v1/Schedules/user/MultipleCentersExams"

        while True:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Odpytuję API PWPW przez sesję przeglądarki...")

            try:
                # context.request automatycznie używa świeżych ciasteczek sesyjnych z Chromium
                response = context.request.post(
                    api_url,
                    data=payload,
                    headers={
                        "Accept": "application/json, text/plain, */*",
                        "Content-Type": "application/json",
                        "Referer": "https://info-kierowca.pl/reservation",
                        "Origin": "https://info-kierowca.pl"
                    }
                )

                if response.status == 200:
                    data = response.json()
                    slots = parse_schedule(data)
                    new_slots = [s for s in slots if s not in seen_slots]

                    if new_slots:
                        msg = "🚗 *Zwolnił się termin na egzamin!*\n\n"
                        msg += "\n".join([f"• `{s}`" for s in new_slots])
                        msg += "\n\n🔗 Rezerwuj: https://info-kierowca.pl/reservation"
                        print(f"[+] NOWY TERMIN: {new_slots}")
                        send_telegram_alert(msg)
                        seen_slots.update(new_slots)
                    else:
                        print("[-] Brak nowych terminów przed " + MAX_TARGET_DATE)
                elif response.status in (401, 403):
                    print("[!] Sesja wygasła. Zaloguj się ponownie w otwartym oknie przeglądarki.")
                else:
                    print(f"[!] Kod błędu: {response.status}")

            except Exception as e:
                print(f"[!] Błąd zapytania: {e}")

            # Lekki ping na stronie co jakiś czas, aby utrzymać aktywność w karcie
            try:
                page.evaluate("() => window.scrollTo(0, 0)")
            except Exception:
                pass

            # Losowy interwał 60–100 sekund
            time.sleep(random.randint(60, 100))

if __name__ == "__main__":
    main()