"""
patch_graph_dotenv.py
Aggiunge load_dotenv() all'inizio del blocco __main__ di graph.py
e l'import in cima al file.
"""

path = "src/graph/graph.py"
txt = open(path, encoding="utf-8").read()

# 1. Aggiungi import dotenv dopo gli altri import
OLD_IMPORT = "from src.graph.state import AgentState"
NEW_IMPORT = "from src.graph.state import AgentState\n\nfrom dotenv import load_dotenv\nload_dotenv()  # carica .env per ANTHROPIC_API_KEY e altri segreti"

if "from dotenv import load_dotenv" not in txt:
    if OLD_IMPORT in txt:
        txt = txt.replace(OLD_IMPORT, NEW_IMPORT, 1)
        print("✓ load_dotenv() aggiunto")
    else:
        print("ERROR: riga import non trovata")
else:
    print("✓ load_dotenv già presente")

open(path, "w", encoding="utf-8").write(txt)

# Verifica sintassi
try:
    compile(open(path, encoding="utf-8").read(), path, "exec")
    print("✓ Syntax check OK")
except SyntaxError as e:
    print(f"ERROR: {e}")
