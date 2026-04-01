from pathlib import Path

f = Path("src/agents/warren_buffett.py")
text = f.read_text(encoding="utf-8")

old = '''    prompt = template.invoke({
        "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
        "ticker": ticker,
    })'''

new = '''    prompt = template.invoke({
        "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
        "ticker": ticker,
    }).to_string()'''

if ".to_string()" in text:
    print("SKIP: gia presente.")
elif old not in text:
    print("ERRORE: stringa non trovata.")
else:
    text = text.replace(old, new, 1)
    f.write_text(text, encoding="utf-8")
    print("OK: fix applicata.")
