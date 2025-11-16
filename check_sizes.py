import os
from PIL import Image

def check_all_image_sizes(dataset_dir):
    sizes = set()
    mismatches = []

    # Recorre todo el directorio
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(".png"):
                path = os.path.join(root, f)
                try:
                    with Image.open(path) as img:
                        sizes.add(img.size)  # (ancho, alto)
                except Exception as e:
                    print(f"Error leyendo {path}: {e}")
                    return False

    # Si hay más de un tamaño → error
    if len(sizes) == 1:
        print(f"Todas las imágenes tienen el mismo tamaño: {sizes.pop()}")
        return True
    else:
        print("Tamaños distintos encontrados:")
        for s in sizes:
            print(" -", s)
        return False


if __name__ == "__main__":
    dataset_dir = "/Users/candeladavilamoreno/Documents/GitHub/DACIU/MSLesSeg-Dataset"
    result = check_all_image_sizes(dataset_dir)
    print("\nResultado final:", result)