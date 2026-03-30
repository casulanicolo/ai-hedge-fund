"""
patch_api_shim_v2.py — gestisce BOM UTF-8
"""

path = "src/tools/api_shim.py"

# Leggi con utf-8-sig che rimuove automaticamente il BOM
txt = open(path, encoding="utf-8-sig").read()

fixes = [
    (
        "def get_insider_trades(ticker: str, end_date: str = None, limit: int = 1000,\n"
        "                       api_key: str = None) -> list:",
        "def get_insider_trades(ticker: str, end_date: str = None, start_date: str = None,\n"
        "                       limit: int = 1000, api_key: str = None) -> list:",
    ),
    (
        "def get_company_news(ticker: str, end_date: str = None, limit: int = 100,\n"
        "                     api_key: str = None) -> list:",
        "def get_company_news(ticker: str, end_date: str = None, start_date: str = None,\n"
        "                     limit: int = 100, api_key: str = None) -> list:",
    ),
]

changed = 0
for old, new in fixes:
    if old in txt:
        txt = txt.replace(old, new, 1)
        changed += 1
        print(f"✓ Fixed: {old.split('(')[0].strip()}")
    elif new in txt:
        print(f"✓ Already fixed: {old.split('(')[0].strip()}")
    else:
        print(f"WARNING: not found — trying partial match")
        fname = old.split("(")[0].strip()
        idx = txt.find(fname)
        if idx >= 0:
            print(f"  Found at {idx}: {repr(txt[idx:idx+120])}")

# Scrivi senza BOM (utf-8 puro)
if changed:
    open(path, "w", encoding="utf-8").write(txt)
    print(f"✓ api_shim.py aggiornato e BOM rimosso")

try:
    compile(open(path, encoding="utf-8").read(), path, "exec")
    print("✓ Syntax check OK")
except SyntaxError as e:
    print(f"ERROR: {e}")
