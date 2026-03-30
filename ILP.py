"""
Columnas requeridas en el CSV:
    kernel, 
    N,
    cpu_exec_time_ms, 
    gpu_exec_time_ms,
    cpu_energy_j,     
    gpu_energy_j,
    cpu_cores, 
    cpu_freq_ghz, 
    cpu_mem_bw_gbps,
    gpu_sm_count, 
    gpu_freq_ghz, 
    gpu_mem_bw_gbps, 
    gpu_mem_gb,
    sparsity (solo relevante para SpMV; 0.01 para el resto)

Columnas opcionales (enriquecen el análisis pero no son obligatorias):
    gflops_cpu, 
    gflops_gpu, 
    bw_gbps_cpu, 
    bw_gbps_gpu

Uso:
    python ILP.py --input dataset.csv
    python ILP.py --input dataset.csv --output labeled.csv
    python ILP.py --input dataset.csv --exact --solver CPLEX
"""

import argparse
import time
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger()

ALPHA = 0.3   # peso energético (escenario balance típico)

# Validación del dataset
REQUIRED = [
    "kernel", 
    "N",
    "cpu_exec_time_ms", 
    "gpu_exec_time_ms",
    "cpu_energy_j", 
    "gpu_energy_j",
    "cpu_cores", 
    "cpu_freq_ghz", 
    "cpu_mem_bw_gbps",
    "gpu_sm_count", 
    "gpu_freq_ghz", 
    "gpu_mem_bw_gbps", 
    "gpu_mem_gb",
    "sparsity",
]

def validate(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        log.error(f"Columnas faltantes en el CSV: {missing}")
        raise SystemExit(1)
    nan_counts = {c: df[c].isna().sum() for c in REQUIRED if df[c].isna().any()}
    if nan_counts:
        log.warning(f"Filas con NaN en columnas clave: {nan_counts}")
        log.warning("Esas filas serán descartadas del ILP.")
    return df.dropna(subset=["cpu_exec_time_ms", "gpu_exec_time_ms",
                              "cpu_energy_j", "gpu_energy_j"]).copy()

# Estimación de memoria requerida por tarea (para restricción R1)
def memory_gb(row) -> float:
    N = int(row["N"])
    k = str(row["kernel"]).upper()
    prec = 8   # float64 (mediciones con doble precisión)

    if k == "GEMM":
        return 3 * N * N * prec / 1e9 # matrices A, B, C
    elif k == "SPMV":
        sp  = float(row.get("sparsity", 0.01))
        nnz = int(N * N * sp)
        return (nnz * (prec + 4) + (N + 1) * 4) / 1e9 # CSR
    else:  # FFT
        return N * prec * 2 / 1e9 # vector complejo

# Solver ILP
def solve_ilp(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Restricciones aplicadas:
        R1: Memoria GPU total ≤ 90% de gpu_mem_gb del nodo
        R2: BW GPU agregado ≤ 85% de gpu_mem_bw_gbps del nodo
        R3: Fracción máx GPU ≤ 85% de las tareas
        R4: SpMV con sparsity < 0.5% se asigna directamente a CPU (acceso irregular)
        R5: GEMM con N ≥ 2048 se asigna directamente a GPU
        R6: Integralidad binaria (CPU=0, GPU=1)
    """
    n = len(df)

    T_cpu = df["cpu_exec_time_ms"].values.astype(float)
    T_gpu = df["gpu_exec_time_ms"].values.astype(float)
    E_cpu = df["cpu_energy_j"].values.astype(float)
    E_gpu = df["gpu_energy_j"].values.astype(float)

    # σ: factor de escala global simétrico (CPU + GPU juntos)
    sigma = (T_cpu.sum() + T_gpu.sum()) / (E_cpu.sum() + E_gpu.sum() + 1e-12)

    # Score de cada tarea en cada dispositivo
    sc_cpu  = T_cpu + ALPHA * E_cpu * sigma
    sc_gpu  = T_gpu + ALPHA * E_gpu * sigma
    benefit = sc_cpu - sc_gpu   # Si es positivo, se asume que GPU es mejor para esa tarea

    # Límites del nodo - se toman de la primera fila (mismo nodo por lote)
    gpu_mem_limit = float(df["gpu_mem_gb"].iloc[0])    * 0.90
    gpu_bw_limit  = float(df["gpu_mem_bw_gbps"].iloc[0]) * 0.85

    # BW requerido por tarea (de la columna medida, si existe)
    if "bw_gbps_gpu" in df.columns:
        bw_req = df["bw_gbps_gpu"].values.astype(float)
    else:
        bw_req = np.zeros(n)

    # Memoria requerida por tarea
    mem_req = np.array([memory_gb(row) for _, row in df.iterrows()])

    # Restricciones R4 y R5
    fixed_cpu = np.zeros(n, dtype=bool)
    fixed_gpu = np.zeros(n, dtype=bool)

    for i, (_, row) in enumerate(df.iterrows()):
        k  = str(row["kernel"]).upper()
        sp = float(row.get("sparsity", 1.0))

        # R4
        if k == "SPMV" and sp < 0.005:
            fixed_cpu[i] = True

        # R5
        elif (k == "GEMM"
              and int(row["N"]) >= 2048
              and T_gpu[i] < T_cpu[i] * 0.5):
            fixed_gpu[i] = True

    # Inicializar con tareas fijas
    labels = np.zeros(n, dtype=int)
    labels[fixed_gpu] = 1

    mem_used = (mem_req * labels).sum()
    bw_used = (bw_req * labels).sum()
    gpu_count = int(labels.sum())
    max_gpu = int(n * 0.85) # R3

    # Ordenar tareas libres por beneficio descendente
    free = np.where(~fixed_cpu & ~fixed_gpu)[0]
    free = free[np.argsort(-benefit[free])]

    for i in free:
        r1_ok = (mem_used  + mem_req[i]) <= gpu_mem_limit
        r2_ok = (bw_req.sum() == 0) or \
                (bw_used + bw_req[i]) <= gpu_bw_limit
        r3_ok = (gpu_count + 1) <= max_gpu
        worth = benefit[i] > 0   # Solo si GPU reduce la función objetivo

        if r1_ok and r2_ok and r3_ok and worth:
            labels[i]  = 1
            mem_used  += mem_req[i]
            bw_used   += bw_req[i]
            gpu_count += 1

    # Valor de la función objetivo
    obj = (
        (labels * T_gpu + (1 - labels) * T_cpu).sum()
        + ALPHA * sigma * (labels * E_gpu + (1 - labels) * E_cpu).sum()
    )
    return labels, obj

# Solver exacto con PuLP
def solve_ilp_exact(df: pd.DataFrame,
                    solver_name: str = "CBC",
                    time_limit: int  = 300) -> tuple[np.ndarray, float]:
    """
    Resolución exacta con PuLP (CBC o CPLEX).
    Requiere: pip install pulp
    Activar con --exact en la línea de comandos.
    """
    import pulp

    n = len(df)
    T_cpu = df["cpu_exec_time_ms"].values.astype(float)
    T_gpu = df["gpu_exec_time_ms"].values.astype(float)
    E_cpu = df["cpu_energy_j"].values.astype(float)
    E_gpu = df["gpu_energy_j"].values.astype(float)

    sigma = (T_cpu.sum() + T_gpu.sum()) / (E_cpu.sum() + E_gpu.sum() + 1e-12)

    gpu_mem_limit = float(df["gpu_mem_gb"].iloc[0])      * 0.90
    gpu_bw_limit = float(df["gpu_mem_bw_gbps"].iloc[0]) * 0.85
    bw_req = df["bw_gbps_gpu"].values.astype(float) \
                    if "bw_gbps_gpu" in df.columns else np.zeros(n)
    mem_req = np.array([memory_gb(r) for _, r in df.iterrows()])

    prob = pulp.LpProblem("CPU_GPU_ILP", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]

    # Función objetivo
    lat = pulp.lpSum(x[i]*T_gpu[i] + (1-x[i])*T_cpu[i] for i in range(n))
    eng = pulp.lpSum((x[i]*E_gpu[i] + (1-x[i])*E_cpu[i])*sigma for i in range(n))
    prob += lat + ALPHA * eng

    # Restricciones
    prob += pulp.lpSum(x[i]*mem_req[i] for i in range(n)) <= gpu_mem_limit, "R1"
    if bw_req.sum() > 0:
        prob += pulp.lpSum(x[i]*bw_req[i] for i in range(n)) <= gpu_bw_limit, "R2"
    prob += pulp.lpSum(x[i] for i in range(n)) <= int(n*0.85), "R3"

    for i, (_, row) in enumerate(df.iterrows()):
        k  = str(row["kernel"]).upper()
        sp = float(row.get("sparsity", 1.0))
        if k == "SPMV" and sp < 0.005:
            prob += x[i] == 0, f"R4_{i}"
        elif (k == "GEMM" and int(row["N"]) >= 2048
              and T_gpu[i] < T_cpu[i] * 0.5):
            prob += x[i] == 1, f"R5_{i}"

    # Solver
    if solver_name.upper() == "CPLEX":
        try:
            slv = pulp.CPLEX_PY(timeLimit=time_limit, msg=0, gapRel=1e-6)
        except Exception:
            log.warning("CPLEX no disponible → CBC")
            slv = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0,
                                    gapRel=1e-6, threads=4)
    else:
        slv = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0,
                                gapRel=1e-6, threads=4)

    prob.solve(slv)
    status = pulp.LpStatus[prob.status]
    log.info(f"PuLP status: {status} | Obj: {pulp.value(prob.objective):.4f}")

    if status in ("Optimal", "Feasible"):
        labels  = np.array([int(round(pulp.value(x[i]))) for i in range(n)])
        obj_val = pulp.value(prob.objective)
    else:
        log.warning("PuLP sin solución óptima → fallback greedy")
        labels, obj_val = solve_ilp(df)

    return labels, obj_val

# Resumen
def print_summary(df: pd.DataFrame):
    print()
    print("=" * 60)
    print("  RESUMEN DE ASIGNACIÓN ILP")
    print("=" * 60)
    print(f"  Filas procesadas : {len(df)}")
    print(f"  α (peso energía) : {ALPHA}")
    print()

    g = df.groupby("kernel").agg(
        Total   = ("ilp_label", "count"),
        CPU     = ("ilp_label", lambda x: (x==0).sum()),
        GPU     = ("ilp_label", lambda x: (x==1).sum()),
        T_cpu   = ("cpu_exec_time_ms", "mean"),
        T_gpu   = ("gpu_exec_time_ms", "mean"),
        E_cpu   = ("cpu_energy_j",     "mean"),
        E_gpu   = ("gpu_energy_j",     "mean"),
    ).round(4)
    g["speedup"] = (g["T_cpu"] / g["T_gpu"]).round(2)
    g["GPU_%"]   = (g["GPU"] / g["Total"] * 100).round(1)
    print(g[["Total","CPU","GPU","GPU_%","T_cpu","T_gpu",
             "speedup","E_cpu","E_gpu"]].to_string())

    # Columnas opcionales de GFLOPS y BW si existen
    optional = [c for c in
                ["gflops_cpu","gflops_gpu","bw_gbps_cpu","bw_gbps_gpu"]
                if c in df.columns]
    if optional:
        print()
        print("  Métricas adicionales (medias por kernel):")
        print(df.groupby("kernel")[optional].mean().round(3).to_string())
    print("=" * 60)

# Main
def main():
    parser = argparse.ArgumentParser(
        description="Aplica el ILP sobre el dataset completo del clúster"
    )
    parser.add_argument("--input",   required=True,
                        help="CSV con todas las métricas medidas")
    parser.add_argument("--output",  default=None,
                        help="CSV de salida (default: input_labeled.csv)")
    parser.add_argument("--exact",   action="store_true",
                        help="Usar solver exacto PuLP/CBC en lugar del greedy")
    parser.add_argument("--solver",  default="CBC", choices=["CBC","CPLEX"],
                        help="Solver para --exact (default: CBC)")
    parser.add_argument("--time-limit", type=int, default=300,
                        help="Límite de tiempo del solver en segundos")
    args = parser.parse_args()

    out = args.output or args.input.replace(".csv", "_labeled.csv")

    log.info(f"Dataset  : {args.input}")
    log.info(f"α ILP    : {ALPHA}  (balance típico)")
    log.info(f"Solver   : {'PuLP/' + args.solver if args.exact else 'greedy+restricciones'}")

    # Carga y validación
    df_raw = pd.read_csv(args.input)
    log.info(f"Filas leídas: {len(df_raw)} | "
             f"Kernels: {df_raw['kernel'].value_counts().to_dict()}")

    df = validate(df_raw)
    if len(df) < len(df_raw):
        log.warning(f"Descartadas {len(df_raw)-len(df)} filas con NaN")

    # Aplicar ILP
    t0 = time.time()
    if args.exact:
        try:
            labels, obj = solve_ilp_exact(df, args.solver, args.time_limit)
        except ImportError:
            log.warning("PuLP no instalado (pip install pulp) → usando greedy")
            labels, obj = solve_ilp(df)
    else:
        labels, obj = solve_ilp(df)

    elapsed = time.time() - t0
    log.info(f"ILP resuelto en {elapsed:.3f}s | Obj = {obj:.6f}")

    n_gpu = int(labels.sum())
    n_cpu = len(labels) - n_gpu
    log.info(f"CPU (0): {n_cpu} ({n_cpu/len(labels)*100:.1f}%)  |  "
             f"GPU (1): {n_gpu} ({n_gpu/len(labels)*100:.1f}%)")

    df["ilp_label"] = labels
    print_summary(df)

    df.to_csv(out, index=False, float_format="%.8g")
    log.info(f"Guardado : {out}")


if __name__ == "__main__":
    main()