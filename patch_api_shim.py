"""
patch_api_shim.py
Aggiunge start_date a get_insider_trades e get_company_news in api_shim.py
"""

path = "src/tools/api_shim.py"
txt = open(path, encoding="utf-8").read()

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
        print(f"WARNING: not found: {old.split('(')[0].strip()}")
        print("  Current content around that function:")
        idx = txt.find(old.split("(")[0])
        if idx >= 0:
            print("  ", repr(txt[idx:idx+150]))

if changed:
    open(path, "w", encoding="utf-8").write(txt)
    print(f"✓ api_shim.py updated ({changed} fix/es applied)")

try:
    compile(open(path, encoding="utf-8").read(), path, "exec")
    print("✓ Syntax check OK")
except SyntaxError as e:
    print(f"ERROR: {e}")
