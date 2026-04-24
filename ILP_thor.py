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

ALPHA        = 0.3   #  peso de la energía en la función objetivo
GAMMA        = 0.85  # fracción máxima de tareas asignadas a GPU (R3)
R4_THRESHOLD = 0.5   # umbral T_gpu / T_cpu para R4

# Límites R1 y R2 - se leen del CSV en load_and_normalize
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
    log.info(f"Hardware GPU — VRAM    : {gpu_mem_gb:.4f} GB  →  R1 = {GPU_MEM_LIMIT:.4f} GB")

    if "gpu_mem_bw_gbps" in df.columns and df["gpu_mem_bw_gbps"].notna().any():
        gpu_bw_gbps  = float(df["gpu_mem_bw_gbps"].dropna().iloc[0])
        GPU_BW_LIMIT = gpu_bw_gbps * 0.85
        log.info(f"Hardware GPU — BW pico : {gpu_bw_gbps:.4f} GB/s  →  R2 = {GPU_BW_LIMIT:.4f} GB/s")
    else:
        log.warning("'gpu_mem_bw_gbps' no disponible — R2 omitida.")

    required = ["kernel", "N", "nnz", "sparsity",
                "cpu_time_s", "cpu_energy_j",
                "gpu_time_s", "gpu_energy_j", "CPU_Cores"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Columnas faltantes: {missing}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=required).copy().reset_index(drop=True)
    if len(df) < before:
        log.warning(f"Descartadas {before - len(df)} filas con NaN")

    cores_list = sorted(df["CPU_Cores"].unique())
    log.info(f"Filas totales  : {len(df)}")
    log.info(f"CPU_Cores      : {cores_list}")
    log.info(f"Kernels (total): {df['kernel'].value_counts().to_dict()}")
    return df

# Memoria GPU por tarea (para R1)
def memory_gb(row) -> float:
    k = str(row["kernel"]).upper()
    N = int(row["N"])
    if k == "GEMM":
        return 3.0 * N * N * 8 / 1e9
    elif k == "SPMV":
        nnz = max(int(row.get("nnz", N)), N)
        return (nnz * 12 + (N + 1) * 4 + 2 * N * 8) / 1e9
    else:  # FFT
        return (N * 8 + (N // 2 + 1) * 16) / 1e9

# Solver ILP - una ejecución por subconjunto de cores
def solve_ilp(df: pd.DataFrame,
              cores: int,
              time_limit: int = 300,
              verbose: bool = False) -> tuple:
    """
    Resuelve el ILP para las filas que corresponden a 'cores' núcleos CPU.
    Retorna (labels_array, obj_val).
    """
    n = len(df)
    T_cpu = df["cpu_time_s"].values.astype(float)
    T_gpu = df["gpu_time_s"].values.astype(float)
    E_cpu = df["cpu_energy_j"].values.astype(float)
    E_gpu = df["gpu_energy_j"].values.astype(float)

    # σ: escala energía (J) → tiempo (s)
    sigma = (T_cpu.sum() + T_gpu.sum()) / (E_cpu.sum() + E_gpu.sum() + 1e-12)
    log.info(f"  σ = {sigma:.6f}")

    mem_req = np.array([memory_gb(row) for _, row in df.iterrows()])
    has_bw  = "bw_gbps_gpu" in df.columns and df["bw_gbps_gpu"].notna().any()
    bw_req  = df["bw_gbps_gpu"].values.astype(float) if has_bw else np.zeros(n)

    prob = pulp.LpProblem(f"CPU_GPU_Assignment_{cores}c", pulp.LpMinimize)

    # R5: integralidad binaria
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    ZL = pulp.lpSum(x[i] * T_gpu[i] + (1 - x[i]) * T_cpu[i] for i in range(n))
    ZE = pulp.lpSum(
        (x[i] * E_gpu[i] + (1 - x[i]) * E_cpu[i]) * sigma for i in range(n)
    )
    prob += ZL + ALPHA * ZE, "Obj"

    # R1: memoria GPU
    prob += (
        pulp.lpSum(x[i] * mem_req[i] for i in range(n)) <= GPU_MEM_LIMIT,
        "R1",
    )

    # R2: ancho de banda GPU
    if has_bw and bw_req.sum() > 0 and GPU_BW_LIMIT is not None:
        prob += (
            pulp.lpSum(x[i] * bw_req[i] for i in range(n)) <= GPU_BW_LIMIT,
            "R2",
        )

    # R3: fracción máxima en GPU
    max_gpu = int(n * GAMMA)
    prob += (pulp.lpSum(x[i] for i in range(n)) <= max_gpu, "R3")

    # R4: GEMM grande
    r4_candidates = []
    for i, (_, row) in enumerate(df.iterrows()):
        if (str(row["kernel"]).upper() == "GEMM"
                and int(row["N"]) >= 2048
                and T_gpu[i] < R4_THRESHOLD * T_cpu[i]):
            time_saved = T_cpu[i] - T_gpu[i]
            eff = time_saved / mem_req[i] if mem_req[i] > 0 else 0.0
            r4_candidates.append((i, mem_req[i], eff))

    r4_candidates.sort(key=lambda t: t[2], reverse=True)
    n_r4 = 0
    mem_r4 = 0.0
    r4_budget = GPU_MEM_LIMIT
    for i, mem_i, _ in r4_candidates:
        if mem_i <= r4_budget:
            prob += x[i] == 1, f"R4_{i}"
            r4_budget -= mem_i
            mem_r4    += mem_i
            n_r4      += 1

    log.info(f"  R4: {n_r4} GEMM forzados (mem={mem_r4:.2f} GB / {GPU_MEM_LIMIT:.2f} GB)")

    solver = pulp.CPLEX_PY(timeLimit=time_limit, msg=int(verbose), gapRel=1e-6)
    t0     = time.time()
    status = prob.solve(solver)
    elapsed = time.time() - t0

    status_str = pulp.LpStatus[status]
    obj_val    = pulp.value(prob.objective)
    log.info(f"  Estado: {status_str}  |  Tiempo: {elapsed:.3f}s  |  Obj: {obj_val}")

    if status_str not in ("Optimal", "Feasible"):
        log.error(f"  CPLEX sin solución ({status_str}).")
        sys.exit(1)

    labels = np.array([int(round(pulp.value(x[i]))) for i in range(n)])
    return labels, obj_val

def print_summary(df_all: pd.DataFrame):
    print()
    print("=" * 72)
    print("  RESUMEN ILP — Por configuración de CPU_Cores")
    print("=" * 72)
    print(f"  α = {ALPHA}   γ = {GAMMA}   R4 umbral = {R4_THRESHOLD}")
    print(f"  R1 = {GPU_MEM_LIMIT:.4f} GB   R2 = {GPU_BW_LIMIT:.4f} GB/s" if GPU_BW_LIMIT else
          f"  R1 = {GPU_MEM_LIMIT:.4f} GB   R2 = omitida")
    print()

    for cores in sorted(df_all["CPU_Cores"].unique()):
        df = df_all[df_all["CPU_Cores"] == cores].copy()
        df["speedup_c"] = df["cpu_time_s"] / (df["gpu_time_s"] + 1e-12)

        n_gpu = (df["ilp_label"] == 1).sum()
        n_cpu = (df["ilp_label"] == 0).sum()

        g = df.groupby("kernel").agg(
            Total   = ("ilp_label", "count"),
            CPU     = ("ilp_label", lambda x: (x == 0).sum()),
            GPU     = ("ilp_label", lambda x: (x == 1).sum()),
            Speedup = ("speedup_c", "mean"),
        ).round(2)
        g["GPU_%"] = (g["GPU"] / g["Total"] * 100).round(1)

        gpu_mask = df["ilp_label"] == 1
        mem_used = sum(memory_gb(row) for _, row in df[gpu_mask].iterrows())

        print(f"  ── {cores} cores {'─'*52}")
        print(g[["Total", "CPU", "GPU", "GPU_%", "Speedup"]].to_string())
        print(f"  Total:  {n_gpu} GPU ({n_gpu/len(df)*100:.1f}%)  /  {n_cpu} CPU ({n_cpu/len(df)*100:.1f}%)")
        print(f"  R1 uso: {mem_used:.3f} GB / {GPU_MEM_LIMIT:.4f} GB  ({mem_used/GPU_MEM_LIMIT*100:.1f}%)")
        obj = df["ilp_obj"].iloc[0] if "ilp_obj" in df.columns else "—"
        print(f"  Z obj : {obj}")
        print()

    print("=" * 72)

def main():
    global ALPHA, R4_THRESHOLD

    parser = argparse.ArgumentParser(
        description="ILP multi-core: ejecuta un ILP por cada valor único de CPU_Cores"
    )
    parser.add_argument("--input",        required=True,
                        help="CSV de entrada con múltiples CPU_Cores (results_thor.csv)")
    parser.add_argument("--output",       default=None,
                        help="CSV de salida con columna ilp_label (default: <input>_labeled.csv)")
    parser.add_argument("--time-limit",   type=int,   default=300)
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--alpha",        type=float, default=ALPHA)
    parser.add_argument("--r4-threshold", type=float, default=R4_THRESHOLD)
    args = parser.parse_args()

    ALPHA        = args.alpha
    R4_THRESHOLD = args.r4_threshold
    out = args.output or args.input.replace(".csv", "_labeled.csv")

    log.info(f"Dataset        : {args.input}")
    log.info(f"α={ALPHA}  γ={GAMMA}  R4_threshold={R4_THRESHOLD}")

    df_all = load_and_normalize(args.input)
    cores_list = sorted(df_all["CPU_Cores"].unique())
    log.info(f"Configuraciones: {cores_list}")

    # Columnas de resultado
    df_all["ilp_label"] = -1
    df_all["ilp_obj"]   = np.nan

    # Un ILP por cada configuración de cores
    for cores in cores_list:
        log.info(f"\n{'─'*60}")
        log.info(f"Resolviendo ILP para {cores} cores ({len(df_all[df_all['CPU_Cores']==cores])} tareas)...")

        mask = df_all["CPU_Cores"] == cores
        df_sub = df_all[mask].copy().reset_index(drop=True)

        labels, obj = solve_ilp(df_sub, cores,
                                time_limit=args.time_limit,
                                verbose=args.verbose)

        df_all.loc[mask, "ilp_label"] = labels
        df_all.loc[mask, "ilp_obj"]   = round(obj, 6) if obj is not None else np.nan

    print_summary(df_all)

    df_all.to_csv(out, index=False, float_format="%.8g")
    log.info(f"Dataset etiquetado guardado en: {out}")


if __name__ == "__main__":
    main()