import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts.postmarket_builder import build_context, build_html, build_text
from src.alerts.email_sender import send_postmarket
from datetime import date

ctx = build_context()
html = build_html(ctx)
text = build_text(ctx)
ok = send_postmarket(html, text, date=str(date.today()))
sys.exit(0 if ok else 1)
