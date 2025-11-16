import nibabel as nib
import numpy as np
import os
from PIL import Image

# Directorios origen y destino
SOURCE_ROOT = "/Volumes/MB_Candela/DACIU"
DEST_ROOT   = "/Users/candeladavilamoreno/Documents/GitHub/DACIU/MSLesSeg-Dataset"

# Sufijos esperados para cada modalidad
MODALITY_SUFFIXES = {
    "FLAIR": "_FLAIR.nii.gz",
    "MASK":  "_MASK.nii.gz",
    "T1":    "_T1.nii.gz",
    "T2":    "_T2.nii.gz",
}

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data

def save_slice_png(slice_2d, filename):
    # Transponer para mantener el equivalente a origin="lower"
    arr = slice_2d.T.astype(np.float32)

    # Normalización min-max tipo imshow
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="L")
    img.save(filename)

def process_sample(modality_paths, output_base_dir):
    """
    Procesa todas las modalidades disponibles (FLAIR, MASK, T1, T2)
    y guarda los slices en subcarpetas.
    """
    print("Procesando muestra con modalidades:")
    for m, p in modality_paths.items():
        print(f"  {m}: {p}")

    # Usamos FLAIR como referencia para el número de slices
    flair_data = load_nifti(modality_paths["FLAIR"])
    nz = flair_data.shape[2]
    z_start = int(0.20 * nz)
    z_end   = int(0.80 * nz)
    print(f"  Slices z={z_start}..{z_end} de {nz}")

    # Cargar todas las modalidades disponibles en memoria
    data_by_modality = {"FLAIR": flair_data}
    for modality, path in modality_paths.items():
        if modality == "FLAIR":
            continue
        data_by_modality[modality] = load_nifti(path)

    # Crear subcarpetas por modalidad
    modality_output_dirs = {}
    for modality in data_by_modality.keys():
        m_dir = os.path.join(output_base_dir, modality)
        os.makedirs(m_dir, exist_ok=True)
        modality_output_dirs[modality] = m_dir

    # Guardar slices para cada modalidad
    for z in range(z_start, z_end):
        for modality, volume in data_by_modality.items():
            slice_2d = volume[:, :, z]
            png_path = os.path.join(
                modality_output_dirs[modality],
                f"{modality.lower()}_z{z}.png"
            )
            save_slice_png(slice_2d, png_path)

def main():
    # Crear carpeta raíz de destino (MSLesSeg-Dataset)
    os.makedirs(DEST_ROOT, exist_ok=True)

    # Recorremos tanto train como test
    for split in ["train", "test"]:
        source_split_dir = os.path.join(SOURCE_ROOT, split)
        dest_split_dir   = os.path.join(DEST_ROOT, split)

        if not os.path.isdir(source_split_dir):
            print(f"Aviso: no existe {source_split_dir}, se omite.")
            continue

        print(f"\n=== Procesando split: {split} ===")

        # Caminamos por todas las subcarpetas de train/test
        for root, dirs, files in os.walk(source_split_dir):
            flair_suffix = MODALITY_SUFFIXES["FLAIR"]
            flair_files = [f for f in files if f.endswith(flair_suffix)]

            for flair_file in flair_files:
                flair_path = os.path.join(root, flair_file)
                base_name = flair_file[:-len(flair_suffix)]  # quita "_FLAIR.nii.gz"

                # Construimos rutas esperadas para cada modalidad
                modality_paths = {}
                missing_modalities = []

                for modality, suffix in MODALITY_SUFFIXES.items():
                    file_name = base_name + suffix
                    file_path = os.path.join(root, file_name)
                    if os.path.exists(file_path):
                        modality_paths[modality] = file_path
                    else:
                        missing_modalities.append(modality)

                # Requerimos al menos FLAIR para procesar
                if "FLAIR" not in modality_paths:
                    print(f"  [AVISO] Falta FLAIR para base {base_name}, se omite.")
                    continue

                if missing_modalities:
                    print(f"  [INFO] Para {base_name} faltan modalidades: {missing_modalities} (se procesan solo las disponibles).")

                # Calculamos la ruta relativa desde el split para mantener estructura
                rel_root = os.path.relpath(root, source_split_dir)

                # Directorio base de salida:
                # - En train: p.ej. train/P1/T1/P1_T1
                # - En test:  p.ej. test/P54 (evitar test/P54/P54)
                if os.path.basename(rel_root) == base_name:
                    # Ya estamos en carpeta con el nombre base (caso test/P54)
                    output_base_dir = os.path.join(dest_split_dir, rel_root)
                else:
                    # Añadimos carpeta base_name (caso train/P1/T1/P1_T1)
                    output_base_dir = os.path.join(dest_split_dir, rel_root, base_name)

                os.makedirs(output_base_dir, exist_ok=True)

                process_sample(modality_paths, output_base_dir)

    print("\nProceso completado. PNG generados en:", DEST_ROOT)

if __name__ == "__main__":
    main()