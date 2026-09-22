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
MAX_TARGET_DATE = "2026-09-30"

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
    print("[*] Uruchamianie Chromium z trwałym profilem...")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir="./pwpw_session_profile",
            headless=False
        )
        page = context.new_page()

        print("[*] Otwieram stronę info-kierowca.pl...")
        page.goto("https://info-kierowca.pl/reservation")

        print("\n" + "="*60)
        print(">>> 1. ZALOGUJ SIĘ W PRZEGLĄDARCE PRZEZ mOBYWATEL <<<")
        print(">>> 2. WYBIERZ PROFIL, KATEGORIĘ I PRZEJDŹ DO WYBORU TERMINU <<<")
        print(">>> 3. GDY ZOBACZYSZ EKRAN Z TERMINAMI, WCISNIJ ENTER TUTAJ <<<")
        print("="*60 + "\n")
        input()

        print("[*] Rozpoczynam stały monitoring oparty na realnych akcjach UI...")

        payload = {
            'startDate': '2026-09-08',
            'organizationId': [26, 25, 31, 31001, 32004],
            'category': 5,
            'profileNumber': '72501092495855042122',
            'profileType': 'Pkk',
        }

        # Ten skrypt symuluje ruch myszką/kliknięcie wewnątrz DOM, co resetuje licznik bezczynności w Angularze
        heartbeat_js = """
        () => {
            document.body.dispatchEvent(new MouseEvent('mousemove', { bubbles: true }));
            document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Shift' }));
        }
        """

        # Bezpieczne zapytanie przez fetch wewnątrz zalogowanej sesji
        fetch_js = """
        async (payload) => {
            try {
                const res = await fetch('/bknd/exam/api/v1/Schedules/user/MultipleCentersExams', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json, text/plain, */*'
                    },
                    body: JSON.stringify(payload)
                });
                if (res.status === 200) {
                    return { status: 200, data: await res.json() };
                }
                return { status: res.status, data: null };
            } catch (e) {
                return { status: -1, data: null };
            }
        }
        """

        while True:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Sprawdzam terminy i podtrzymuję aktywność UI...")

            try:
                # 1. Reset licznika bezczynności w aplikacji
                page.evaluate(heartbeat_js)

                # 2. Wywołanie zapytania
                result = page.evaluate(fetch_js, payload)
                status = result.get("status")
                data = result.get("data")

                if status == 200 and data:
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

                elif status in (401, 403):
                    print("[!] Sesja wygasła po >1h. Wymagane ponowne potwierdzenie.")
                    if not alert_session_sent:
                        # Dźwięk systemowy w Linuksie (terminal beep)
                        print('\a')
                        send_telegram_alert(
                            "⚠️ *Sesja Info-Kierowca wygasła po godzinie!*\nZaloguj się ponownie w oknie Chromium, aby wznowić monitoring.")
                        alert_session_sent = True

                    # Czekamy chwilę dłużej, aby nie spamować serwera
                    time.sleep(30)

            except Exception as e:
                print(f"[!] Błąd w pętli: {e}")

            # Odpytuj co 50–80 sekund (przed limitem 10 minut)
            time.sleep(random.randint(50, 80))

if __name__ == "__main__":
    main()