import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================
# CONFIGURACIÓN
# ============================
BASE_RAW = "nnUNet_raw"
DATASET1 = "Dataset001_MSLesSeg"
DATASET2 = "Dataset002_MSLesSeg"
NUM_SLICES = 10
# ============================


src_path = Path(BASE_RAW) / DATASET1
dst_path = Path(BASE_RAW) / DATASET2

# Crear carpetas destino
(dst_path / "imagesTr").mkdir(parents=True, exist_ok=True)
(dst_path / "labelsTr").mkdir(parents=True, exist_ok=True)
(dst_path / "imagesTs").mkdir(parents=True, exist_ok=True)
(dst_path / "labelsTs").mkdir(parents=True, exist_ok=True)

# Cargar dataset.json original
with open(src_path / "dataset.json", "r") as f:
    original_json = json.load(f)

file_ending = original_json.get("file_ending", ".nii.gz")

# nº de canales según channel_names
channel_keys = sorted(original_json["channel_names"].keys(), key=lambda x: int(x))
num_channels = len(channel_keys)


def extract_slices_case(
    base_id: str,
    channel_paths: list[Path],
    label_path: Path | None,
    subset: str,
):
    """
    base_id: por ejemplo 'P1_T1'
    channel_paths: lista con paths a ..._0000.nii.gz, ..._0001.nii.gz, ...
    label_path: path a labelsTr/labelsTs correspondiente o None
    subset: 'Tr' o 'Ts'
    """
    # Cargar todos los canales y apilarlos como 4D (X,Y,Z,C)
    vols = []
    affine = None
    for p in channel_paths:
        nii = nib.load(str(p))
        data = nii.get_fdata()
        if affine is None:
            affine = nii.affine
        vols.append(data)
    vols = np.stack(vols, axis=-1)  # (X, Y, Z, C)

    depth = vols.shape[2]
    slice_indices = random.sample(range(depth), NUM_SLICES)

    lbl_data = None
    lbl_affine = None
    if label_path is not None and label_path.exists():
        lbl_nii = nib.load(str(label_path))
        lbl_data = lbl_nii.get_fdata()
        lbl_affine = lbl_nii.affine

    entries = []

    for i, sl in enumerate(slice_indices):
        slice_number = f"{i + 1:03d}"
        slice_id = f"{base_id}_{slice_number}"  # p.ej. P1_T1_001

        # Guardar cada canal como imagen 2D
        for c_idx in range(num_channels):
            slice_img = vols[:, :, sl, c_idx]
            nii_slice = nib.Nifti1Image(slice_img.astype(np.float32), affine)
            img_name = f"{slice_id}_{int(c_idx):04d}{file_ending}"
            img_save_path = dst_path / f"images{subset}" / img_name
            nib.save(nii_slice, img_save_path)

        # Guardar máscara si existe
        if lbl_data is not None:
            slice_lbl = lbl_data[:, :, sl]
            nii_lbl = nib.Nifti1Image(slice_lbl.astype(np.uint8), lbl_affine)
            lbl_name = f"{slice_id}{file_ending}"
            lbl_save_path = dst_path / f"labels{subset}" / lbl_name
            nib.save(nii_lbl, lbl_save_path)

            entries.append(
                {
                    "image": f"./images{subset}/{slice_id}",
                    "label": f"./labels{subset}/{slice_id}{file_ending}",
                }
            )
        else:
            # solo imagen (test sin label en dataset.json)
            entries.append(f"./images{subset}/{slice_id}")

    return entries


# ============================
# PROCESAR TRAINING
# ============================
print("Procesando TRAIN...")
new_training = []

for item in original_json["training"]:
    # Base sin extensión ni canal: imagesTr/P1_T1
    img_base_rel = item["image"].replace("./", "")  # imagesTr/P1_T1
    base_id = Path(img_base_rel).name               # P1_T1

    # Construir paths a todos los canales de entrada
    channel_paths = []
    for ck in channel_keys:
        cidx = int(ck)
        channel_rel = f"{img_base_rel}_{cidx:04d}{file_ending}"  # imagesTr/P1_T1_0000.nii.gz
        channel_paths.append(src_path / channel_rel)

    # Label 3D
    lbl_rel = item["label"].replace("./", "")  # labelsTr/P1_T1.nii.gz
    label_path = src_path / lbl_rel

    new_training.extend(
        extract_slices_case(base_id, channel_paths, label_path, subset="Tr")
    )


# ============================
# PROCESAR TEST
# ============================
print("Procesando TEST...")
new_test = []

for img_entry in original_json["test"]:
    img_base_rel = img_entry.replace("./", "")  # imagesTs/P10_T1
    base_id = Path(img_base_rel).name           # P10_T1

    # Canales de test
    channel_paths = []
    for ck in channel_keys:
        cidx = int(ck)
        channel_rel = f"{img_base_rel}_{cidx:04d}{file_ending}"  # imagesTs/P10_T1_0000.nii.gz
        channel_paths.append(src_path / channel_rel)

    # LabelTs (si existe)
    label_filename = base_id + file_ending      # P10_T1.nii.gz
    label_path = src_path / "labelsTs" / label_filename
    if not label_path.exists():
        label_path = None

    new_test.extend(
        extract_slices_case(base_id, channel_paths, label_path, subset="Ts")
    )


# ============================
# NUEVO dataset.json 2D
# ============================
new_json = original_json.copy()
new_json["tensorImageSize"] = "2D"
new_json["numTraining"] = len(new_training)
new_json["numTest"] = len(new_test)
new_json["training"] = new_training
new_json["test"] = new_test

with open(dst_path / "dataset.json", "w") as f:
    json.dump(new_json, f, indent=4)

print("\n========================================")
print(" Dataset002_MSLesSeg generado correctamente")
print(" Slices 2D creados en imagesTr/labelsTr e imagesTs/labelsTs")
print(" dataset.json actualizado a 2D")
print("========================================")