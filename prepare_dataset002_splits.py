import json
from pathlib import Path
from collections import defaultdict

# ============================
# CONFIGURACIÓN
# ============================
BASE_RAW = Path("nnUNet_raw")
BASE_PREPROCESSED = Path("nnUNet_preprocessed")
BASE_RESULTS = Path("nnUNet_results")

DATASET1 = "Dataset001_MSLesSeg"
DATASET2 = "Dataset002_MSLesSeg"

SPLITS_FILENAME = "splits_final.json"
# ============================


def load_splits_dataset1():
    """
    Carga splits_final.json del Dataset001.
    Se intenta primero en nnUNet_preprocessed, si no existe se busca en nnUNet_raw.
    """
    src_pre = BASE_PREPROCESSED / DATASET1 / SPLITS_FILENAME
    src_raw = BASE_RAW / DATASET1 / SPLITS_FILENAME

    if src_pre.exists():
        splits_path = src_pre
    elif src_raw.exists():
        splits_path = src_raw
    else:
        raise FileNotFoundError(
            f"No se encontró {SPLITS_FILENAME} ni en {src_pre} ni en {src_raw}"
        )

    print(f"Usando splits de: {splits_path}")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    return splits


def build_slice_mapping_dataset2():
    """
    Explora nnUNet_raw/Dataset002_MSLesSeg/imagesTr y construye:
      base_id -> [case_ids_slice]
    donde:
      fichero: P1_T1_001_0000.nii.gz
      base_id: P1_T1
      case_id_slice: P1_T1_001   (lo que aparece en dataset.json)
    """
    images_tr = BASE_RAW / DATASET2 / "imagesTr"
    if not images_tr.exists():
        raise FileNotFoundError(f"No existe carpeta {images_tr}")

    mapping = defaultdict(set)

    for nii_path in images_tr.glob("*.nii.gz"):
        stem = nii_path.stem  # p.ej. P1_T1_001_0000
        parts = stem.split("_")
        if len(parts) < 3:
            # no es del formato esperado
            continue

        channel = parts[-1]        # 0000
        slice_idx = parts[-2]      # 001
        base_id = "_".join(parts[:-2])  # P1_T1
        case_id_slice = f"{base_id}_{slice_idx}"  # P1_T1_001

        mapping[base_id].add(case_id_slice)

    # ordenar los IDs de slices
    mapping = {k: sorted(v) for k, v in mapping.items()}
    print(f"Encontrados {len(mapping)} casos base en Dataset002 (imagesTr).")
    return mapping


def create_splits_dataset2(splits1, slice_mapping):
    """
    A partir de los splits del Dataset001 y del mapping base_id -> slices,
    crea nuevos splits para Dataset002 donde cada caso base se expande a todas
    sus slices.
    """
    new_splits = []

    for fold_idx, fold in enumerate(splits1):
        new_fold = {"train": [], "val": []}

        for subset_key in ["train", "val"]:
            original_ids = fold[subset_key]  # p.ej. ["P11_T1", "P11_T2", ...]
            expanded_ids = []

            for base_id in original_ids:
                if base_id not in slice_mapping:
                    print(
                        f"[AVISO] Caso {base_id} no encontrado en Dataset002, "
                        "¿seguro que lo convertiste a slices?"
                    )
                    continue
                expanded_ids.extend(slice_mapping[base_id])

            new_fold[subset_key] = expanded_ids

        print(
            f"Fold {fold_idx}: "
            f"{len(new_fold['train'])} train, {len(new_fold['val'])} val (slices)"
        )
        new_splits.append(new_fold)

    return new_splits


def main():
    # 1) Cargar splits del Dataset001
    splits1 = load_splits_dataset1()

    # 2) Construir mapping base_id -> slices en Dataset002
    slice_mapping = build_slice_mapping_dataset2()

    # 3) Crear nuevos splits
    splits2 = create_splits_dataset2(splits1, slice_mapping)

    # 4) Crear carpetas de Dataset002 en preprocessed y results
    dst_pre = BASE_PREPROCESSED / DATASET2
    dst_res = BASE_RESULTS / DATASET2

    dst_pre.mkdir(parents=True, exist_ok=True)
    dst_res.mkdir(parents=True, exist_ok=True)

    # 5) Guardar nuevo splits_final.json en preprocessed/Dataset002
    dst_splits_path = dst_pre / SPLITS_FILENAME
    with open(dst_splits_path, "w") as f:
        json.dump(splits2, f, indent=4)

    print("\n===================================")
    print(f"Nuevo {SPLITS_FILENAME} guardado en: {dst_splits_path}")
    print(f"Carpeta de resultados creada (si no existía): {dst_res}")
    print("Listo para plan_and_preprocess + entrenamiento.")
    print("===================================")


if __name__ == "__main__":
    main()