from pathlib import Path

pipeline = Path("src/run_pipeline.py")
text = pipeline.read_text(encoding="utf-8")

old = '"model_name": "claude-sonnet-4-5",'
new = '"model_name": "claude-sonnet-4-5-20251001",'

if "claude-sonnet-4-5-20251001" in text:
    print("SKIP: gia aggiornato.")
else:
    text = text.replace(old, new, 1)
    pipeline.write_text(text, encoding="utf-8")
    print("OK: model_name aggiornato.")
