"""
    python ILP_proliant_dl580.py --input combined_results_1.csv
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

# Parámetros del modelo
ALPHA        = 0.3    # peso de la energía en la función objetivo
GAMMA        = 0.85   # fracción máxima de tareas asignadas a GPU (R3)
R4_THRESHOLD = 0.5    # CPU es 50% mas lenta que GPU, 0.8 para 8 cores (CPU 20% mas lenta) para compensar

# R1 y R2, se fijan dinámicamente desde el CSV en load_and_normalize
GPU_MEM_LIMIT = None
GPU_BW_LIMIT  = None

def load_and_normalize(path: str) -> pd.DataFrame:
    global GPU_MEM_LIMIT, GPU_BW_LIMIT

    df = pd.read_csv(path)

    rename = {
        "Kernel":         "kernel",
        "Sparsity":       "sparsity",
        "Tiempo_CPU_s":   "cpu_time_s",
        "Energia_CPU_J":  "cpu_energy_j",
        "GFLOPS_CPU":     "gflops_cpu",
        "BW_Medido_GBps": "bw_gbps_cpu",
        "Memoria_MB":     "mem_mb",
        "gflops_GPU":     "gflops_gpu",
        "BW_GPU":         "bw_gbps_gpu",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "gpu_mem_gb" not in df.columns or df["gpu_mem_gb"].isna().all():
        log.error("Columna 'gpu_mem_gb' no encontrada.")
        sys.exit(1)
    gpu_mem_gb    = float(df["gpu_mem_gb"].dropna().iloc[0])
    GPU_MEM_LIMIT = gpu_mem_gb * 0.90
    log.info(f"Hardware GPU — VRAM    : {gpu_mem_gb:.4f} GB  →  R1 límite = {GPU_MEM_LIMIT:.4f} GB")

    if "gpu_mem_bw_gbps" in df.columns and df["gpu_mem_bw_gbps"].notna().any():
        gpu_bw_gbps  = float(df["gpu_mem_bw_gbps"].dropna().iloc[0])
        GPU_BW_LIMIT = gpu_bw_gbps * 0.85
        log.info(f"Hardware GPU — BW pico : {gpu_bw_gbps:.4f} GB/s  →  R2 límite = {GPU_BW_LIMIT:.4f} GB/s")
    else:
        log.warning("'gpu_mem_bw_gbps' no disponible — R2 se omitirá.")

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
        log.warning(f"Descartadas {before - len(df)} filas con NaN en columnas críticas")

    log.info(f"Filas válidas : {len(df)}")
    log.info(f"Kernels       : {df['kernel'].value_counts().to_dict()}")
    return df

# Memoria GPU requerida por tarea (para R1)
def memory_gb(row) -> float:
    """Calcula la memoria GPU necesaria para ejecutar la tarea (bytes → GB)."""
    k = str(row["kernel"]).upper()
    N = int(row["N"])
    if k == "GEMM":
        return 3.0 * N * N * 8 / 1e9
    elif k == "SPMV":
        nnz = max(int(row.get("nnz", N)), N)
        return (nnz * 12 + (N + 1) * 4 + 2 * N * 8) / 1e9
    else:  # FFT
        return (N * 8 + (N // 2 + 1) * 16) / 1e9

# Solver ILP
def solve_ilp(df: pd.DataFrame,
              time_limit: int = 300,
              verbose: bool = False) -> tuple:
    n = len(df)

    T_cpu = df["cpu_time_s"].values.astype(float)
    T_gpu = df["gpu_time_s"].values.astype(float)
    E_cpu = df["cpu_energy_j"].values.astype(float)
    E_gpu = df["gpu_energy_j"].values.astype(float)

    # escala energía (J) a la misma dimensión que el tiempo (s)
    sigma = (T_cpu.sum() + T_gpu.sum()) / (E_cpu.sum() + E_gpu.sum() + 1e-12)
    log.info(f"σ = {sigma:.6f}  (factor de escala energía→tiempo)")

    # Memoria y BW requeridos por tarea (para R1 y R2)
    mem_req = np.array([memory_gb(row) for _, row in df.iterrows()])
    has_bw  = "bw_gbps_gpu" in df.columns and df["bw_gbps_gpu"].notna().any()
    bw_req  = df["bw_gbps_gpu"].values.astype(float) if has_bw else np.zeros(n)
    log.info(f"Σ mem si todo→GPU = {mem_req.sum():.2f} GB")

    prob = pulp.LpProblem("DL580_CPU_GPU_Assignment", pulp.LpMinimize)

    # R5: Integralidad binaria
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    ZL = pulp.lpSum(x[i] * T_gpu[i] + (1 - x[i]) * T_cpu[i] for i in range(n))
    ZE = pulp.lpSum(
        (x[i] * E_gpu[i] + (1 - x[i]) * E_cpu[i]) * sigma for i in range(n)
    )
    prob += ZL + ALPHA * ZE, "Obj_Latencia_Energia"

    # R1: Capacidad de memoria GPU
    prob += (
        pulp.lpSum(x[i] * mem_req[i] for i in range(n)) <= GPU_MEM_LIMIT,
        "R1_Capacidad_Memoria_GPU",
    )
    log.info(f"R1: Σ mem·x ≤ {GPU_MEM_LIMIT:.4f} GB")

    # R2: Ancho de banda de memoria GPU

    if has_bw and bw_req.sum() > 0 and GPU_BW_LIMIT is not None:
        prob += (
            pulp.lpSum(x[i] * bw_req[i] for i in range(n)) <= GPU_BW_LIMIT,
            "R2_Ancho_Banda_GPU",
        )
        log.info(f"R2: Σ bw·x ≤ {GPU_BW_LIMIT:.4f} GB/s")
    else:
        log.info("R2: omitida — bw_gbps_gpu no disponible")

    # R3: Fracción máxima de tareas en GPU
    max_gpu = int(n * GAMMA)
    prob += (
        pulp.lpSum(x[i] for i in range(n)) <= max_gpu,
        "R3_Fraccion_Max_GPU",
    )
    log.info(f"R3: Σ xi ≤ {max_gpu}  (γ={GAMMA}, {max_gpu/n*100:.0f}% de {n} tareas)")

    # R4: GEMM de gran tamaño a GPU
    r4_candidates = []
    for i, (_, row) in enumerate(df.iterrows()):
        if (str(row["kernel"]).upper() == "GEMM"
                and int(row["N"]) >= 2048
                and T_gpu[i] < R4_THRESHOLD * T_cpu[i]):
            time_saved = T_cpu[i] - T_gpu[i]
            efficiency = time_saved / mem_req[i] if mem_req[i] > 0 else 0.0
            r4_candidates.append((i, mem_req[i], efficiency))

    r4_candidates.sort(key=lambda t: t[2], reverse=True)

    n_r4 = 0
    n_r4_free = 0
    mem_r4 = 0.0
    r4_budget = GPU_MEM_LIMIT

    for i, mem_i, eff_i in r4_candidates:
        if mem_i <= r4_budget:
            prob += x[i] == 1, f"R4_GEMM_GPU_{i}"
            r4_budget -= mem_i
            mem_r4    += mem_i
            n_r4      += 1
        else:
            n_r4_free += 1

    log.info(
        f"R4: {n_r4} tareas GEMM forzadas a GPU  "
        f"(umbral T_gpu < {R4_THRESHOLD}·T_cpu, N≥2048, "
        f"mem acum = {mem_r4:.2f} GB / {GPU_MEM_LIMIT:.2f} GB)"
    )
    if n_r4_free:
        log.info(
            f"R4: {n_r4_free} GEMM elegibles no forzadas "
            f"(presupuesto R1 agotado) — solver decide libremente"
        )

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
        log.error(
            f"CPLEX sin solución ({status_str}). "
            "Verifica que R1–R4 no sean contradictorias. "
            f"R4 acumula {mem_r4:.2f} GB, R1 permite {GPU_MEM_LIMIT:.2f} GB."
        )
        sys.exit(1)

    labels = np.array([int(round(pulp.value(x[i]))) for i in range(n)])
    return labels, obj_val

def print_summary(df: pd.DataFrame):
    print()
    print("=" * 68)
    print("  RESUMEN ILP — HPE ProLiant DL580 G9")
    print("=" * 68)
    print(f"  Filas procesadas : {len(df)}")
    print(f"  α (peso energía) : {ALPHA}")
    print(f"  γ (frac. máx GPU): {GAMMA}")
    print(f"  R4 umbral ratio  : T_gpu < {R4_THRESHOLD}·T_cpu")
    print(f"  R1 límite VRAM   : {GPU_MEM_LIMIT:.4f} GB")
    if GPU_BW_LIMIT:
        print(f"  R2 límite BW     : {GPU_BW_LIMIT:.4f} GB/s")
    print()

    df_s = df.copy()
    df_s["speedup_calc"] = df_s["cpu_time_s"] / (df_s["gpu_time_s"] + 1e-12)

    g = df_s.groupby("kernel").agg(
        Total    = ("ilp_label",    "count"),
        CPU      = ("ilp_label",    lambda x: (x == 0).sum()),
        GPU      = ("ilp_label",    lambda x: (x == 1).sum()),
        T_cpu_s  = ("cpu_time_s",   "mean"),
        T_gpu_s  = ("gpu_time_s",   "mean"),
        E_cpu_J  = ("cpu_energy_j", "mean"),
        E_gpu_J  = ("gpu_energy_j", "mean"),
        speedup  = ("speedup_calc", "mean"),
    ).round(4)
    g["GPU_%"] = (g["GPU"] / g["Total"] * 100).round(1)

    print(g[["Total", "CPU", "GPU", "GPU_%",
             "T_cpu_s", "T_gpu_s", "speedup",
             "E_cpu_J", "E_gpu_J"]].to_string())

    optional = [c for c in ["gflops_cpu", "gflops_gpu",
                             "bw_gbps_cpu", "bw_gbps_gpu",
                             "op_intensity_cpu", "op_intensity_gpu"]
                if c in df.columns]
    if optional:
        print()
        print("  Métricas adicionales (medias por kernel):")
        print(df.groupby("kernel")[optional].mean().round(3).to_string())

    print()
    n_cpu = (df["ilp_label"] == 0).sum()
    n_gpu = (df["ilp_label"] == 1).sum()
    print(f"  CPU (0) : {n_cpu:3d}  ({n_cpu/len(df)*100:.1f}%)")
    print(f"  GPU (1) : {n_gpu:3d}  ({n_gpu/len(df)*100:.1f}%)")

    gpu_mask = df["ilp_label"] == 1
    mem_used = sum(memory_gb(row) for _, row in df[gpu_mask].iterrows())
    print()
    print(f"  R1 mem usada  : {mem_used:.3f} / {GPU_MEM_LIMIT:.4f} GB  "
          f"({mem_used/GPU_MEM_LIMIT*100:.1f}%)")
    if "bw_gbps_gpu" in df.columns and GPU_BW_LIMIT:
        bw_used = df.loc[gpu_mask, "bw_gbps_gpu"].sum()
        print(f"  R2 BW usado   : {bw_used:.2f} / {GPU_BW_LIMIT:.4f} GB/s  "
              f"({bw_used/GPU_BW_LIMIT*100:.1f}%)")
    print(f"  R3 frac GPU   : {n_gpu}/{len(df)} = {n_gpu/len(df)*100:.1f}%  "
          f"(límite={int(len(df)*GAMMA)})")
    print("=" * 68)

def main():
    global ALPHA, R4_THRESHOLD

    parser = argparse.ArgumentParser(
        description="ILP exacto (CPLEX) — HPE ProLiant DL580 G9"
    )
    parser.add_argument("--input",        required=True,
                        help="CSV de entrada (combined_results_1.csv)")
    parser.add_argument("--output",       default=None,
                        help="CSV de salida (default: <input>_labeled.csv)")
    parser.add_argument("--time-limit",   type=int,   default=300,
                        help="Límite de tiempo CPLEX en segundos (default: 300)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Mostrar output interno de CPLEX")
    parser.add_argument("--alpha",        type=float, default=ALPHA,
                        help=f"Peso energía α en el objetivo (default: {ALPHA})")
    parser.add_argument("--r4-threshold", type=float, default=R4_THRESHOLD,
                        help=f"Umbral T_gpu/T_cpu para R4 (default: {R4_THRESHOLD}). "
                             "Reducir si la GPU tiene speedup GEMM >2x.")
    args = parser.parse_args()

    ALPHA        = args.alpha
    R4_THRESHOLD = args.r4_threshold
    out = args.output or args.input.replace(".csv", "_labeled.csv")

    log.info(f"Dataset        : {args.input}")
    log.info(f"α              : {ALPHA}")
    log.info(f"γ              : {GAMMA}")
    log.info(f"R4 umbral      : T_gpu < {R4_THRESHOLD}·T_cpu")
    log.info(f"T. límite CPLEX: {args.time_limit}s")

    df = load_and_normalize(args.input)

    log.info("Resolviendo ILP con CPLEX...")
    labels, obj = solve_ilp(df, time_limit=args.time_limit, verbose=args.verbose)

    df["ilp_label"] = labels
    print_summary(df)

    df.to_csv(out, index=False, float_format="%.8g")
    log.info(f"Dataset etiquetado guardado en: {out}")


if __name__ == "__main__":
    main()