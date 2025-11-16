import os
import nibabel as nib
import numpy as np

# Directorio origen con las resonancias originales (.nii.gz)
SOURCE_ROOT = "/Volumes/MB_Candela/DACIU"  # train/test aquí

# Directorio destino donde se guardarán los volúmenes recortados
DEST_ROOT = "/Users/candeladavilamoreno/Documents/GitHub/DACIU/MSLesSeg-Dataset"

# Sufijos esperados para cada modalidad en tus datos originales
MODALITY_SUFFIXES = {
    "FLAIR": "_FLAIR.nii.gz",
    "MASK":  "_MASK.nii.gz",
    "T1":    "_T1.nii.gz",
    "T2":    "_T2.nii.gz",
}

# Identificadores de canal para nnU-Net (imágenes de entrada)
CHANNEL_IDS = {
    "FLAIR": "0000",
    "T1":    "0001",
    "T2":    "0002",
    # MASK no va aquí porque es la segmentación (label), no un canal de entrada
}

def crop_along_z(data, affine, z_start, z_end):
    """
    Recorta el volumen 'data' en el eje Z (última dimensión) entre
    z_start y z_end (tipo range: z_start incluido, z_end excluido)
    y ajusta la affine en consecuencia.
    """
    # Recorte de datos
    cropped = data[:, :, z_start:z_end]

    # Ajuste de la affine: mover el origen según cuánto hemos cortado en Z
    new_affine = affine.copy()
    # La tercera columna de la affine (índice 2) corresponde al eje Z
    # Multiplicamos ese vector por z_start y lo sumamos a la traslación
    new_affine[:3, 3] = affine[:3, 3] + affine[:3, 2] * z_start

    return cropped, new_affine

def process_case(case_id, modality_paths, dest_dir):
    """
    Recorta el 80% central en Z para todas las modalidades disponibles
    y guarda los ficheros siguiendo el formato de nnU-Net:

      - Imágenes (canales): CASEID_XXXX.nii.gz (XXXX = 0000, 0001, 0002, ...)
      - Segmentación (MASK): CASEID.nii.gz

    case_id: identificador del caso (por ejemplo 'P1_T1' o 'P54')
    modality_paths: dict modalidad -> ruta .nii.gz original
    dest_dir: carpeta de salida para este caso
    """
    print(f"\nProcesando caso '{case_id}' con modalidades:")
    for m, p in modality_paths.items():
        print(f"  {m}: {p}")

    # Usamos FLAIR como referencia para dimensiones y eje Z
    flair_img = nib.load(modality_paths["FLAIR"])
    flair_data = flair_img.get_fdata()
    affine_ref = flair_img.affine
    header_ref = flair_img.header

    nz = flair_data.shape[2]
    z_start = int(0.10 * nz)
    z_end   = int(0.90 * nz)
    print(f"  Recorte Z: z={z_start}..{z_end} (80% central de {nz} slices)")

    os.makedirs(dest_dir, exist_ok=True)

    # 1) Guardar canales de entrada (FLAIR, T1, T2, ... que existan)
    for modality, path in modality_paths.items():
        if modality == "MASK":
            continue  # la MASK se trata aparte

        if modality not in CHANNEL_IDS:
            print(f"  [INFO] Modalidad {modality} no tiene CHANNEL_ID definido, se omite.")
            continue

        img = nib.load(path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # Comprobamos que comparte nz con FLAIR
        if data.shape[2] != nz:
            print(f"  [AVISO] {path} tiene nz={data.shape[2]} diferente a FLAIR nz={nz}. Se omite modalidad {modality}.")
            continue

        cropped_data, new_affine = crop_along_z(data, affine, z_start, z_end)

        channel_id = CHANNEL_IDS[modality]
        out_name = f"{case_id}_{channel_id}.nii.gz"
        out_path = os.path.join(dest_dir, out_name)

        out_img = nib.Nifti1Image(cropped_data.astype(np.float32), new_affine, header=header)
        nib.save(out_img, out_path)
        print(f"    Guardado canal {modality} en: {out_path}")

    # 2) Guardar segmentación (MASK) si existe
    if "MASK" in modality_paths:
        mask_path = modality_paths["MASK"]
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        affine = mask_img.affine
        header = mask_img.header

        if mask_data.shape[2] != nz:
            print(f"  [AVISO] MASK {mask_path} tiene nz={mask_data.shape[2]} diferente a FLAIR nz={nz}. No se guarda MASK.")
        else:
            cropped_mask, new_affine_mask = crop_along_z(mask_data, affine, z_start, z_end)

            # Aseguramos que la segmentación sea un mapa entero (0,1,2,...) como exige nnU-Net
            cropped_mask_int = np.rint(cropped_mask).astype(np.uint8)

            out_name = f"{case_id}.nii.gz"  # sin sufijo de canal
            out_path = os.path.join(dest_dir, out_name)

            out_mask_img = nib.Nifti1Image(cropped_mask_int, new_affine_mask, header=header)
            nib.save(out_mask_img, out_path)
            print(f"    Guardada MASK (segmentación) en: {out_path}")
    else:
        print("  [INFO] No hay MASK para este caso, solo se han guardado las imágenes de entrada.")

def main():
    os.makedirs(DEST_ROOT, exist_ok=True)

    # Recorremos tanto train como test
    for split in ["train", "test"]:
        source_split_dir = os.path.join(SOURCE_ROOT, split)

        if not os.path.isdir(source_split_dir):
            print(f"[AVISO] No existe {source_split_dir}, se omite.")
            continue

        print(f"\n=== Procesando split: {split} ===")

        for root, dirs, files in os.walk(source_split_dir):
            flair_suffix = MODALITY_SUFFIXES["FLAIR"]
            flair_files = [f for f in files if f.endswith(flair_suffix)]

            for flair_file in flair_files:
                flair_path = os.path.join(root, flair_file)
                # CASE_IDENTIFIER: nombre base sin sufijo de modalidad
                case_id = flair_file[:-len(flair_suffix)]  # quita "_FLAIR.nii.gz"

                # Construimos rutas esperadas para cada modalidad
                modality_paths = {}
                for modality, suffix in MODALITY_SUFFIXES.items():
                    file_name = case_id + suffix
                    file_path = os.path.join(root, file_name)
                    if os.path.exists(file_path):
                        modality_paths[modality] = file_path

                # Necesitamos al menos FLAIR como imagen de entrada
                if "FLAIR" not in modality_paths:
                    print(f"  [AVISO] Falta FLAIR para CASE_ID={case_id}, se omite el caso.")
                    continue

                # Directorio de salida: copia la estructura del origen relativa a SOURCE_ROOT
                rel_root = os.path.relpath(root, SOURCE_ROOT)
                case_dest_dir = os.path.join(DEST_ROOT, rel_root)
                os.makedirs(case_dest_dir, exist_ok=True)

                process_case(case_id, modality_paths, case_dest_dir)

    print("\n Proceso completado. Volúmenes recortados y renombrados guardados en:", DEST_ROOT)

if __name__ == "__main__":
    main()