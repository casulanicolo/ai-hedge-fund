"""Scan all agent files for data keys that are missing from make_initial_state."""
import os, glob

agents = glob.glob("src/agents/*.py")
for f in sorted(agents):
    txt = open(f, encoding="utf-8").read()
    hits = []
    if "end_date" in txt:   hits.append("end_date")
    if "start_date" in txt: hits.append("start_date")
    if 'data["tickers"]' in txt or "data['tickers']" in txt: hits.append("tickers-from-data")
    if hits:
        print(os.path.basename(f), "->", hits)
