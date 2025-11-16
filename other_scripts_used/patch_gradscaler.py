import os

venv_root = "/Users/candeladavilamoreno/Documents/GitHub/DACIU/venv310"
target_path = os.path.join(
    venv_root,
    "lib",
    "python3.10",
    "site-packages",
    "nnunetv2",
    "training",
    "nnUNetTrainer",
    "nnUNetTrainer.py"
)

print("Fichero a parchear:", target_path)

if not os.path.exists(target_path):
    raise FileNotFoundError(f"No se encontró el fichero: {target_path}")

with open(target_path, "r") as f:
    content = f.read()

old = "from torch import GradScaler"
new = "from torch.amp import GradScaler"

if old not in content and new in content:
    print("Parece que ya está parcheado. No se hace nada.")
elif old not in content and new not in content:
    raise RuntimeError("No he encontrado la línea de import de GradScaler. Revisa el archivo a mano.")
else:
    content = content.replace(old, new)
    with open(target_path, "w") as f:
        f.write(content)
    print("✅ Import de GradScaler parcheado correctamente.")