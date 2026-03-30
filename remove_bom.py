"""
remove_bom.py — rimuove il BOM da api_shim.py riscrivendo in utf-8 puro
"""
path = "src/tools/api_shim.py"
txt = open(path, encoding="utf-8-sig").read()  # legge e rimuove BOM
open(path, "w", encoding="utf-8").write(txt)   # scrive senza BOM
print("✓ BOM rimosso da", path)
try:
    compile(open(path, encoding="utf-8").read(), path, "exec")
    print("✓ Syntax check OK")
except SyntaxError as e:
    print(f"ERROR: {e}")
