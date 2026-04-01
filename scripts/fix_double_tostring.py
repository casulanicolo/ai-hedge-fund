from pathlib import Path

for name in ["src/agents/warren_buffett.py", "src/agents/charlie_munger.py"]:
    f = Path(name)
    text = f.read_text(encoding="utf-8")
    fixed = text.replace(".to_string().to_string()", ".to_string()")
    f.write_text(fixed, encoding="utf-8")
    print(f"OK: {name}")
