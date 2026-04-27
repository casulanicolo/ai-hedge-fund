import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_pipeline import load_tickers
from src.alerts.premarket_builder import build_context, build_html, build_text
from src.alerts.email_sender import send_premarket
from datetime import date

tickers = load_tickers()
ctx = build_context(tickers=tickers)
html = build_html(ctx)
text = build_text(ctx)
ok = send_premarket(html, text, date=str(date.today()))
sys.exit(0 if ok else 1)
