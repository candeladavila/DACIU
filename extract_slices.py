import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Nombres de archivo (asumimos que están en el mismo directorio que el script)
FLAIR_FILE = "P1_T1_FLAIR.nii.gz"
MASK_FILE  = "P1_T1_MASK.nii.gz"

OUTPUT_DIR = "P1_T1"   # Carpeta principal de salida

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data

def save_slice_png(slice_2d, filename):
    plt.figure(figsize=(4,4))
    plt.axis("off")
    plt.imshow(slice_2d.T, cmap="gray", origin="lower")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # Cargar imágenes
    flair = load_nifti(FLAIR_FILE)
    mask  = load_nifti(MASK_FILE)

    # Número total de slices en eje axial Z
    nz = flair.shape[2]

    # Definir rango central (20%-80%)
    z_start = int(0.20 * nz)
    z_end   = int(0.80 * nz)

    print(f"Extrayendo slices desde z={z_start} hasta z={z_end} de un total de {nz}")

    # Crear carpeta principal (P1_T1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Crear subcarpetas FLAIR y MASK dentro de P1_T1
    flair_dir = os.path.join(OUTPUT_DIR, "FLAIR")
    mask_dir  = os.path.join(OUTPUT_DIR, "MASK")
    os.makedirs(flair_dir, exist_ok=True)
    os.makedirs(mask_dir,  exist_ok=True)

    # Generar las imágenes PNG
    for z in range(z_start, z_end):
        flair_slice = flair[:, :, z]
        mask_slice  = mask[:,  :, z]

        flair_path = os.path.join(flair_dir, f"flair_z{z}.png")
        mask_path  = os.path.join(mask_dir,  f"mask_z{z}.png")

        save_slice_png(flair_slice, flair_path)
        save_slice_png(mask_slice,  mask_path)

    print("Proceso completado. PNG generados en la carpeta P1_T1.")

if __name__ == "__main__":
    main()