# -*- coding: utf-8 -*-
import os
from collections import Counter

BASE_DIR = "/Volumes/MB_Candela/DACIU"
TRAIN_DIR = os.path.join(BASE_DIR, "train")  # Pacientes P_i con carpetas T_i dentro
TEST_DIR = os.path.join(BASE_DIR, "test")    # Pacientes P_i con archivos Pi_FLAIR/T1/T2
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset_analysis.txt")


def listar_subdirectorios(path):
    return [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]


def analizar_train():
    """
    TRAIN:
    - Directorio con pacientes P1, P2, ... (P_i), y dentro de cada P_i carpetas T1, T2, T3, T4 (T_i).
    - Para cada paciente:
        * Contar combinación de T1, T2, T3, T4.
        * Para cada T_i, analizar presencia de archivos Pi_Ti_T1.nii(.gz) y Pi_Ti_T2.nii(.gz).
    """
    t_comb_counts = Counter()  # combinaciones de T_i por paciente
    seq_counts = Counter()     # combinaciones T1/T2 por resonancia (P>T)

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"No existe el directorio TRAIN: {TRAIN_DIR}")

    pacientes = listar_subdirectorios(TRAIN_DIR)

    for p in pacientes:
        p_path = os.path.join(TRAIN_DIR, p)

        # Carpetas T_i dentro del paciente (T1, T2, T3, T4, etc.)
        t_dirs = [
            d for d in listar_subdirectorios(p_path)
            if d.upper().startswith("T")
        ]
        t_set = tuple(sorted(t_dirs))

        # Clasificación de combinaciones de T_i
        if t_set == ("T1",):
            t_comb_counts["T1"] += 1
        elif t_set == ("T1", "T2"):
            t_comb_counts["T1_T2"] += 1
        elif t_set == ("T1", "T2", "T3"):
            t_comb_counts["T1_T2_T3"] += 1
        elif t_set == ("T1", "T2", "T3", "T4"):
            t_comb_counts["T1_T2_T3_T4"] += 1
        else:
            t_comb_counts["otros"] += 1

        # Analizar archivos dentro de cada T_i
        for t in t_dirs:
            t_path = os.path.join(p_path, t)
            fns = os.listdir(t_path)

            # Nombre base esperado del prefijo: P1_T1_..., P2_T2_..., etc.
            # para esta resonancia del paciente p y tiempo t
            # Ejemplo: P1_T1_T1.nii.gz, P1_T1_T2.nii.gz
            prefix = f"{p}_{t}_"

            has_T1 = any(
                (fn.startswith(prefix) and (fn.endswith("T1.nii.gz") or fn.endswith("T1.nii")))
                for fn in fns
            )
            has_T2 = any(
                (fn.startswith(prefix) and (fn.endswith("T2.nii.gz") or fn.endswith("T2.nii")))
                for fn in fns
            )

            # Clasificación de combinación T1/T2 (FLAIR y MASK se ignoran aquí)
            if has_T1 and not has_T2:
                seq_counts["solo_T1"] += 1
            elif has_T2 and not has_T1:
                seq_counts["solo_T2"] += 1
            elif has_T1 and has_T2:
                seq_counts["ambos"] += 1
            else:
                seq_counts["otros"] += 1

    return t_comb_counts, seq_counts


def analizar_test():
    """
    TEST:
    - Directorio con pacientes P1, P2, ... (P_i).
    - En cada P_i mirar:
        * Pi_FLAIR.nii(.gz)
        * Pi_T1.nii(.gz)
        * Pi_T2.nii(.gz)
      donde Pi es el nombre de la carpeta (P1, P2, etc.).
    """
    test_counts = Counter()

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"No existe el directorio TEST: {TEST_DIR}")

    pacientes = listar_subdirectorios(TEST_DIR)

    for p in pacientes:
        p_path = os.path.join(TEST_DIR, p)
        fns = os.listdir(p_path)

        # Archivos esperados: P1_FLAIR.nii.gz, P1_T1.nii.gz, P1_T2.nii.gz, etc.
        flair_names = [
            f"{p}_FLAIR.nii.gz",
            f"{p}_FLAIR.nii",
        ]
        t1_names = [
            f"{p}_T1.nii.gz",
            f"{p}_T1.nii",
        ]
        t2_names = [
            f"{p}_T2.nii.gz",
            f"{p}_T2.nii",
        ]

        has_FLAIR = any(name in fns for name in flair_names)
        has_T1 = any(name in fns for name in t1_names)
        has_T2 = any(name in fns for name in t2_names)

        # Clasificación según combinación
        if has_FLAIR and not has_T1 and not has_T2:
            test_counts["solo_FLAIR"] += 1
        elif has_T1 and not has_FLAIR and not has_T2:
            test_counts["solo_T1"] += 1
        elif has_T2 and not has_FLAIR and not has_T1:
            test_counts["solo_T2"] += 1
        elif has_FLAIR and has_T1 and has_T2:
            test_counts["tres"] += 1
        else:
            # Cualquier otro caso: combinaciones de dos, ninguno, archivos extra, etc.
            test_counts["otros"] += 1

    return test_counts


def escribir_resultados(t_comb_counts, seq_counts, test_counts):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("TRAIN:\n")
        f.write("Análisis número de resonancias por paciente (P):\n")
        f.write(f"- T1: {t_comb_counts.get('T1', 0)}\n")
        f.write(f"- T1 + T2: {t_comb_counts.get('T1_T2', 0)}\n")
        f.write(f"- T1 + T2 + T3: {t_comb_counts.get('T1_T2_T3', 0)}\n")
        f.write(f"- T1 + T2 + T3 + T4: {t_comb_counts.get('T1_T2_T3_T4', 0)}\n")
        f.write(
            f"- Presentan alguna combinación no recogida anteriormente: "
            f"{t_comb_counts.get('otros', 0)}\n"
        )
        f.write("\n")

        f.write("Análisis archivos presentes en cada resonancia (P>T):\n")
        f.write(
            f"- Solo archivo Pi_Ti_T1.nii.gz: "
            f"{seq_counts.get('solo_T1', 0)}\n"
        )
        f.write(
            f"- Solo archivo Pi_Ti_T2.nii.gz: "
            f"{seq_counts.get('solo_T2', 0)}\n"
        )
        f.write(
            f"- Presentan ambos archivos: "
            f"{seq_counts.get('ambos', 0)}\n"
        )
        f.write(
            f"- Presentan archivos no recogidos anteriormente: "
            f"{seq_counts.get('otros', 0)}\n"
        )
        f.write("\n\n")

        f.write("TEST:\n")
        f.write("Análisis archivos presentes para cada paciente (P):\n")
        f.write(
            f"- Solo archivo Pi_FLAIR.nii.gz: "
            f"{test_counts.get('solo_FLAIR', 0)}\n"
        )
        f.write(
            f"- Solo archivo Pi_T1.nii.gz: "
            f"{test_counts.get('solo_T1', 0)}\n"
        )
        f.write(
            f"- Solo archivo Pi_T2.nii.gz: "
            f"{test_counts.get('solo_T2', 0)}\n"
        )
        f.write(
            f"- Presentan los tres archivos: "
            f"{test_counts.get('tres', 0)}\n"
        )
        f.write(
            f"- Presentan archivos no mencionados anteriormente: "
            f"{test_counts.get('otros', 0)}\n"
        )


def main():
    t_comb_counts, seq_counts = analizar_train()
    test_counts = analizar_test()
    escribir_resultados(t_comb_counts, seq_counts, test_counts)
    print(f"Análisis completado. Resultado guardado en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()