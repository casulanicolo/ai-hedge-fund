from pathlib import Path

f = Path("src/agents/charlie_munger.py")
text = f.read_text(encoding="utf-8")

old = '''    prompt = template.invoke({
        "ticker": ticker,
        "facts": json.dumps(facts_bundle, separators=(",", ":"), ensure_ascii=False),
        "confidence": confidence_hint,
    })'''

new = '''    prompt = template.invoke({
        "ticker": ticker,
        "facts": json.dumps(facts_bundle, separators=(",", ":"), ensure_ascii=False),
        "confidence": confidence_hint,
    }).to_string()'''

if ".to_string()" in text:
    print("SKIP: gia presente.")
elif old not in text:
    print("ERRORE: stringa non trovata.")
else:
    text = text.replace(old, new, 1)
    f.write_text(text, encoding="utf-8")
    print("OK: fix applicata.")
