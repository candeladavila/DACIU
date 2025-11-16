# Código para visualizar la imágen y la máscara

from nilearn import plotting

def strip_extension(name: str) -> str:
    """Elimina extensiones .nii o .nii.gz si existen."""
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name

def make_flair(base: str) -> str:
    """Devuelve el nombre que termina en _FLAIR.nii.gz."""
    name = strip_extension(base)
    for suf in ("_FLAIR", "_MASK"):
        if name.endswith(suf):
            name = name[:-len(suf)]
    return f"{name}_FLAIR.nii.gz"

def make_mask(base: str) -> str:
    """Devuelve el nombre que termina en _MASK.nii.gz."""
    name = strip_extension(base)
    for suf in ("_FLAIR", "_MASK"):
        if name.endswith(suf):
            name = name[:-len(suf)]
    return f"{name}_MASK.nii.gz"

def setPaths(original_path: str):
    flair_path = make_flair(original_path)
    mask_path = make_mask(original_path)
    return flair_path, mask_path

def show_image(image_path: str):
    print(image_path)
    v = plotting.view_img(image_path, threshold=None)
    v.open_in_browser()

def show_mask(mask_path: str):
    print(mask_path)
    v2 = plotting.view_img(mask_path, threshold=None)
    v2.open_in_browser()

def main():
    flair_path, mask_path = setPaths("P1_T1")
    show_image(flair_path)
    show_mask(mask_path)

if __name__ == "__main__":
    main()