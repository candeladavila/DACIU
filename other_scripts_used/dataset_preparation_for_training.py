'''
This file explores the dataset and adapts the files to use it for training (to create de dataset.json file needed)
As we are working with different patients, special consideration needs to be taken when creatining the training 
and test files. No files from the same patient can be mixed in different folds.
 '''

import os
import shutil
import json
import random

# Ruta donde tienes tus datos recortados y renombrados tipo CASEID_0000.nii.gz
# con estructura algo como: MSLesSeg-Dataset_Original/train/...
SOURCE_ROOT = "/Users/candeladavilamoreno/Documents/GitHub/DACIU/MSLesSeg-Dataset_Original"

# Ruta donde quieres crear la estructura nnUNet_raw/Dataset001_MSLesSeg
NNUNET_RAW_ROOT = "/Users/candeladavilamoreno/Documents/GitHub/DACIU/nnUNet_raw"

DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_MSLesSeg"

# Configuración de canales (coinciden con los sufijos que usaste)
CHANNEL_MAP = {
    "0000": "FLAIR",
    "0001": "T1",
    "0002": "T2",
}

def get_case_id_from_filename(filename):
    """
    A partir de un filename tipo 'P1_T1_0000.nii.gz' o 'P1_T1.nii.gz'
    devuelve el CASEID ('P1_T1').
    """
    if filename.endswith(".nii.gz"):
        name = filename[:-7]  # quita '.nii.gz'
    else:
        name = os.path.splitext(filename)[0]
    # Si termina en '_000X', quitar el sufijo de canal
    if len(name) > 5 and name[-5] == "_" and name[-4:].isdigit():
        return name[:-5]
    return name

def get_patient_id(case_id):
    """
    A partir de 'P1_T1' devuelve 'P1'.
    Si no hay '_', devuelve el case_id entero.
    """
    return case_id.split("_")[0]

def collect_cases(source_train_dir):
    """
    Recorre SOURCE_ROOT/train y encuentra todos los CASEID
    que tienen al menos el canal 0000.
    Devuelve:
      - cases: dict case_id -> info (dir, channels, label_path, patient_id)
      - patients: dict patient_id -> [case_ids]
    """
    cases = {}
    patients = {}

    for root, dirs, files in os.walk(source_train_dir):
        for f in files:
            if f.endswith(".nii.gz") and "_0000.nii.gz" in f:
                case_id = get_case_id_from_filename(f)
                patient_id = get_patient_id(case_id)

                # Directorio donde están los archivos de este caso
                case_dir = root

                # Construimos paths esperados
                channels = {}
                for ch_id in CHANNEL_MAP.keys():
                    ch_filename = f"{case_id}_{ch_id}.nii.gz"
                    ch_path = os.path.join(case_dir, ch_filename)
                    if os.path.exists(ch_path):
                        channels[ch_id] = ch_path

                # Segmentación (MASK) esperada: CASEID.nii.gz
                label_filename = f"{case_id}.nii.gz"
                label_path = os.path.join(case_dir, label_filename)
                if not os.path.exists(label_path):
                    print(f"[AVISO] No se encontró máscara para {case_id} en {label_path}. Este caso se ignora.")
                    continue

                cases[case_id] = {
                    "dir": case_dir,
                    "channels": channels,
                    "label": label_path,
                    "patient_id": patient_id,
                }

                patients.setdefault(patient_id, []).append(case_id)

    print(f"\nSe han encontrado {len(cases)} casos con máscara en train.")
    print(f"Número de pacientes (ids únicos): {len(patients)}")
    return cases, patients

def split_patients(patients, num_test=13, seed=42):
    """
    Divide la lista de pacientes en:
      - test_patients: num_test pacientes
      - trainval_patients: el resto
    La división es aleatoria pero reproducible (semilla fija).
    """
    patient_ids = list(patients.keys())
    if len(patient_ids) < num_test:
        raise ValueError("Número de pacientes menor que num_test, revisa los datos.")

    random.seed(seed)
    random.shuffle(patient_ids)

    test_patients = patient_ids[:num_test]
    trainval_patients = patient_ids[num_test:]

    print("\nPacientes elegidos para TEST externo:")
    print(test_patients)
    print("\nPacientes para TRAIN/VAL (cross-validation):")
    print(trainval_patients)

    return trainval_patients, test_patients

def prepare_nnUNet_raw_structure():
    dataset_root = os.path.join(NNUNET_RAW_ROOT, DATASET_NAME)
    imagesTr = os.path.join(dataset_root, "imagesTr")
    labelsTr = os.path.join(dataset_root, "labelsTr")
    imagesTs = os.path.join(dataset_root, "imagesTs")
    labelsTs = os.path.join(dataset_root, "labelsTs")  # opcional, nnU-Net no lo usa

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    return dataset_root, imagesTr, labelsTr, imagesTs, labelsTs

def copy_case_to_nnUNet(case_id, case_info, images_dir, labels_dir=None):
    """
    Copia los canales de un caso (0000,0001,0002) a images_dir.
    Si labels_dir no es None, también copia la máscara CASEID.nii.gz ahí.
    """
    # Canales
    for ch_id, src_path in case_info["channels"].items():
        base_name = os.path.basename(src_path)  # P1_T1_0000.nii.gz
        dst_path = os.path.join(images_dir, base_name)
        shutil.copy2(src_path, dst_path)

    # Máscara
    if labels_dir is not None and case_info["label"] is not None:
        label_src = case_info["label"]
        label_name = os.path.basename(label_src)  # P1_T1.nii.gz
        label_dst = os.path.join(labels_dir, label_name)
        shutil.copy2(label_src, label_dst)

def build_dataset_json(dataset_root, train_case_ids, test_case_ids):
    """
    Crea dataset.json en dataset_root con la información de canales y labels.
    train_case_ids y test_case_ids son listas de CASEID (strings).
    """
    dataset_json_path = os.path.join(dataset_root, "dataset.json")

    # canal_names sigue la convención índice->nombre
    channel_names = {str(i): name for i, name in enumerate(CHANNEL_MAP.values())}

    # Labels: tú tienes 0=fondo, 1=lesión
    labels = {
        "0": "background",
        "1": "lesion"
    }

    training_entries = []
    for case_id in sorted(train_case_ids):
        training_entries.append({
            "image": f"./imagesTr/{case_id}",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })

    test_entries = [f"./imagesTs/{case_id}" for case_id in sorted(test_case_ids)]

    dataset_info = {
        "name": "MSLesSeg",
        "description": "Multiple sclerosis lesion segmentation (FLAIR, T1, T2)",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": len(train_case_ids),
        "numTest": len(test_case_ids),
        "training": training_entries,
        "test": test_entries
    }

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

    print(f"\nSe ha creado dataset.json en: {dataset_json_path}")

def main():
    source_train_dir = os.path.join(SOURCE_ROOT, "train")
    if not os.path.isdir(source_train_dir):
        raise FileNotFoundError(f"No se encontró el directorio train en {source_train_dir}")

    # 1) Recoger casos y pacientes
    cases, patients = collect_cases(source_train_dir)

    # 2) Dividir pacientes en train/val y test externo
    trainval_patients, test_patients = split_patients(patients, num_test=13, seed=42)

    # 3) Crear estructura nnUNet_raw/Dataset001_MSLesSeg
    dataset_root, imagesTr, labelsTr, imagesTs, labelsTs = prepare_nnUNet_raw_structure()
    print(f"\nEstructura nnU-Net creada en: {dataset_root}")

    # 4) Copiar casos de TRAIN/VAL (solo imágenes y labelsTr)
    train_case_ids = []
    for pid in trainval_patients:
        for case_id in patients[pid]:
            info = cases[case_id]
            copy_case_to_nnUNet(case_id, info, imagesTr, labelsTr)
            train_case_ids.append(case_id)

    # 5) Copiar casos de TEST externo (solo imágenes a imagesTs, labels a labelsTs para tus métricas)
    test_case_ids = []
    for pid in test_patients:
        for case_id in patients[pid]:
            info = cases[case_id]
            copy_case_to_nnUNet(case_id, info, imagesTs, labelsTs)
            test_case_ids.append(case_id)

    print(f"\nTotal casos TRAIN/VAL (con máscara): {len(train_case_ids)}")
    print(f"Total casos TEST externo (con máscara copiada a labelsTs): {len(test_case_ids)}")

    # 6) Construir dataset.json
    build_dataset_json(dataset_root, train_case_ids, test_case_ids)

    print("\n✅ Todo listo. Ahora puedes ejecutar:")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")

if __name__ == "__main__":
    main()