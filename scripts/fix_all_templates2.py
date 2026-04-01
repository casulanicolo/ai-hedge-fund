from pathlib import Path
import re

agents_dir = Path("src/agents")
fixed = []
errors = []

for py_file in agents_dir.glob("*.py"):
    text = py_file.read_text(encoding="utf-8")
    
    if "template.invoke" not in text:
        continue

    # Rimuovi .to_string() messo nel posto sbagliato (dentro json.dumps)
    bad = re.sub(
        r'(json\.dumps\([^)]+\))\.to_string\(\)',
        r'\1',
        text,
        flags=re.DOTALL
    )
    
    # Ora applica .to_string() nel posto giusto: dopo la parentesi chiusa di template.invoke(...)
    # Pattern: prompt = template.invoke({...}) seguito da newline
    fixed_text = re.sub(
        r'(prompt\s*=\s*template\.invoke\(\{[^}]+\}\))',
        r'\1.to_string()',
        bad,
        flags=re.DOTALL
    )
    
    if fixed_text == text:
        errors.append(py_file.name + " (nessuna modifica)")
    else:
        py_file.write_text(fixed_text, encoding="utf-8")
        fixed.append(py_file.name)

print(f"Fixed  : {fixed}")
print(f"Errors : {errors}")
