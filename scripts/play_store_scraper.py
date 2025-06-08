from google_play_scraper import Sort, reviews
import csv, os, logging, schedule, time
from datetime import datetime

# Setup logging in scripts/
log_path = os.path.join(os.path.dirname(__file__), 'scraper.log')
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("📢 Script started.")

# Bank app IDs and names
APPS = {
    'com.combanketh.mobilebanking': 'CBE',
    'com.boa.boaMobileBanking': 'BOA',
    'com.dashen.dashensuperapp': 'Dashen'
}

def scrape_bank(app_id, bank_name):
    logging.info(f"🔄 Fetching {bank_name} reviews...")
    try:
        results, _ = reviews(app_id, lang='en', country='us',
                             sort=Sort.NEWEST, count=5000)
        if len(results) < 400:
            logging.warning(f"{bank_name}: only {len(results)} reviews fetched; expected ≥400")

        # Create data folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        os.makedirs(data_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{bank_name}_reviews_{ts}.csv"
        fpath = os.path.join(data_dir, fname)

        seen = set()
        rows = []
        for r in results:
            text = r['content'].strip()
            date = r['at'].strftime('%Y-%m-%d')
            key = (text, date, r['score'])
            if key not in seen:
                seen.add(key)
                rows.append({'review_text': text, 'rating': r['score'],
                             'date': date, 'bank_name': bank_name, 'source': 'Google Play'})

        with open(fpath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        logging.info(f"✅ {bank_name}: {len(rows)} unique reviews saved to {fpath}")
    except Exception as ex:
        logging.error(f"❌ {bank_name} scraping failed: {ex}")

def job():
    for app_id, name in APPS.items():
        scrape_bank(app_id, name)

# Schedule daily at 01:00 AM
schedule.every().day.at("01:00").do(job)

if __name__ == "__main__":
    logging.info("🕒 Scheduler started, waiting for jobs...")
    job()  # Run once at startup
    while True:
        schedule.run_pending()
        time.sleep(60)
