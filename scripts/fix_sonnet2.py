from pathlib import Path

pipeline = Path("src/run_pipeline.py")
text = pipeline.read_text(encoding="utf-8")

old = '"model_name": "claude-sonnet-4-5-20251001",'
new = '"model_name": "claude-sonnet-4-6",'

if old not in text:
    print("SKIP: stringa non trovata, controlla manualmente.")
else:
    text = text.replace(old, new, 1)
    pipeline.write_text(text, encoding="utf-8")
    print("OK: fallback aggiornato a claude-sonnet-4-6.")
