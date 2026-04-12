"""
    python ILP_Thor.py --input ThorDataset.csv
"""

import argparse
import time
import logging
import sys
import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:
    print("ERROR: PuLP no instalado.  pip install pulp")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger()

ALPHA = 0.3     # peso de la energía en la función objetivo
GAMMA = 0.85    # fracción máxima de tareas asignadas a GPU

def load_and_normalize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename = {
        "Kernel":        "kernel",
        "Sparsity":      "sparsity_raw",
        "OI_CPU":        "op_intensity_cpu",
        "OI_GPU":        "op_intensity_gpu",
        "Memoria_MB":    "mem_mb",
        "Tiempo_CPU_s":  "cpu_time_s",
        "Energia_CPU_J": "cpu_energy_j",
        "GFLOPS_CPU":    "gflops_cpu",
        "BW_CPU":        "bw_gbps_cpu",
        "gflops_GPU":    "gflops_gpu",
        "BW_GPU":        "bw_gbps_gpu",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["sparsity"] = pd.to_numeric(
        df["sparsity_raw"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0.0)
    df.drop(columns=["sparsity_raw"], inplace=True, errors="ignore")

    required = ["kernel", "N", "nnz", "sparsity",
                "cpu_time_s", "cpu_energy_j",
                "gpu_time_s", "gpu_energy_j"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Columnas faltantes: {missing}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=required).copy().reset_index(drop=True)
    if len(df) < before:
        log.warning(f"Descartadas {before - len(df)} filas con NaN")

    log.info(f"Kernels       : {df['kernel'].value_counts().to_dict()}")
    return df

def solve_ilp(df: pd.DataFrame,
              time_limit: int = 300,
              verbose: bool = False) -> tuple:
    n = len(df)

    T_cpu = df["cpu_time_s"].values.astype(float)
    T_gpu = df["gpu_time_s"].values.astype(float)
    E_cpu = df["cpu_energy_j"].values.astype(float)
    E_gpu = df["gpu_energy_j"].values.astype(float)

    # Factor σ: escala energía (J) a la misma dimensión que el tiempo (s)
    # Normalización simétrica: evita sesgo hacia CPU o GPU
    sigma = (T_cpu.sum() + T_gpu.sum()) / (E_cpu.sum() + E_gpu.sum() + 1e-12)
    log.info(f"σ = {sigma:.6f}  (factor de escala energía→tiempo)")

    prob = pulp.LpProblem("Thor_CPU_GPU_Assignment", pulp.LpMinimize)

    # R3
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    ZL = pulp.lpSum(x[i] * T_gpu[i] + (1 - x[i]) * T_cpu[i] for i in range(n))
    ZE = pulp.lpSum(
        (x[i] * E_gpu[i] + (1 - x[i]) * E_cpu[i]) * sigma for i in range(n)
    )
    prob += ZL + ALPHA * ZE, "Obj_Latencia_Energia"

    # R1
    max_gpu = int(n * GAMMA)
    prob += (
        pulp.lpSum(x[i] for i in range(n)) <= max_gpu,
        "R3_Fraccion_Max_GPU",
    )
    log.info(f"R1: {max_gpu} tareas como máximo en GPU(γ={GAMMA}, {max_gpu/n*100:.0f}% de {n} tareas)")

    # R2
    n_r4 = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if (str(row["kernel"]).upper() == "GEMM"
                and int(row["N"]) >= 2048
                and T_gpu[i] < 0.5 * T_cpu[i]):
            prob += x[i] == 1, f"R4_GEMM_GPU_{i}"
            n_r4 += 1
    log.info(f"R2: {n_r4} tareas GEMM forzadas a GPU")

    solver = pulp.CPLEX_PY(
        timeLimit=time_limit,
        msg=int(verbose),
        gapRel=1e-6,
    )
    log.info("Solver: CPLEX")

    t0      = time.time()
    status  = prob.solve(solver)
    elapsed = time.time() - t0

    status_str = pulp.LpStatus[status]
    obj_val    = pulp.value(prob.objective)
    log.info(f"Estado : {status_str}")
    log.info(f"Tiempo : {elapsed:.3f}s")
    if obj_val is not None:
        log.info(f"Obj    : {obj_val:.6f}")
    else:
        log.info("Obj    : N/A (sin solución factible)")

    if status_str not in ("Optimal", "Feasible"):
        log.error(f"CPLEX sin solución ({status_str}). "
                  "Verifica que R3 y R4 no sean contradictorias "
                  f"(R4 fuerza {n_r4} tareas a GPU, R3 permite máx {max_gpu}).")
        sys.exit(1)

    labels = np.array([int(round(pulp.value(x[i]))) for i in range(n)])
    return labels, obj_val

def print_summary(df: pd.DataFrame):
    n_cpu = (df["ilp_label"] == 0).sum()
    n_gpu = (df["ilp_label"] == 1).sum()
    print(f"  CPU (0) : {n_cpu:3d}  ({n_cpu/len(df)*100:.1f}%)")
    print(f"  GPU (1) : {n_gpu:3d}  ({n_gpu/len(df)*100:.1f}%)")

def main():
    global ALPHA
    parser = argparse.ArgumentParser(
        description="ILP exacto (CPLEX) — restricciones R3, R4, R5"
    )
    parser.add_argument("--input",      required=True,
                        help="CSV con las métricas (ThorDataset.csv)")
    parser.add_argument("--output",     default=None,
                        help="CSV de salida (default: <input>_labeled.csv)")
    parser.add_argument("--time-limit", type=int, default=300,
                        help="Límite de tiempo CPLEX en segundos (default: 300)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Mostrar output interno de CPLEX")
    parser.add_argument("--alpha",      type=float, default=ALPHA,
                        help=f"Peso de la energía α en el objetivo (default: {ALPHA})")
    args = parser.parse_args()

    ALPHA = args.alpha
    out   = args.output or args.input.replace(".csv", "_labeled.csv")

    df = load_and_normalize(args.input)

    log.info("Resolviendo ILP con CPLEX...")
    labels, obj = solve_ilp(df, time_limit=args.time_limit, verbose=args.verbose)

    df["ilp_label"] = labels
    print_summary(df)

    df.to_csv(out, index=False, float_format="%.8g")
    log.info(f"Dataset etiquetado guardado en: {out}")


if __name__ == "__main__":
    main()