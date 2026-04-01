from pathlib import Path
import re

agents_dir = Path("src/agents")
fixed = []
skipped = []
errors = []

for py_file in agents_dir.glob("*.py"):
    text = py_file.read_text(encoding="utf-8")
    
    if "template.invoke" not in text:
        continue
    
    if ".to_string()" in text:
        skipped.append(py_file.name)
        continue
    
    new_text = re.sub(
        r'(prompt\s*=\s*template\.invoke\([^)]+\))',
        r'\1.to_string()',
        text,
        flags=re.DOTALL
    )
    
    if new_text == text:
        errors.append(py_file.name)
    else:
        py_file.write_text(new_text, encoding="utf-8")
        fixed.append(py_file.name)

print(f"Fixed  : {fixed}")
print(f"Skipped: {skipped}")
print(f"Errors : {errors}")
