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

# Posibles variantes que puede tener ahora mismo tu fichero
candidates_old = [
    "from torch import GradScaler",
    "from torch.amp import GradScaler",
]

replacement = "from torch.cuda.amp import GradScaler"

if replacement in content:
    print("Parece que ya está parcheado con 'from torch.cuda.amp import GradScaler'. No se hace nada.")
else:
    found = False
    for old in candidates_old:
        if old in content:
            content = content.replace(old, replacement)
            found = True
            print(f"✅ Sustituido:\n  '{old}'\npor:\n  '{replacement}'")
            break

    if not found:
        raise RuntimeError(
            "No he encontrado ninguna línea de import de GradScaler que pueda parchear.\n"
            "Abre el archivo y mira qué 'from torch...' hay exactamente."
        )

    with open(target_path, "w") as f:
        f.write(content)
        print("✅ Archivo guardado correctamente.")