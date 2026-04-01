"""
    python ILP.py --input dataset.csv
    python ILP.py --input dataset.csv --output labeled.csv
    python ILP.py --input dataset.csv --time-limit 600
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
    print("ERROR: PuLP no está instalado.")
    print("       Instalar con: pip install pulp")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger()

ALPHA = 0.3 # peso energético

REQUIRED = [
    "kernel", 
    "N", 
    "sparsity",
    "cpu_exec_time_ms", 
    "gpu_exec_time_ms",
    "cpu_energy_j",     
    "gpu_energy_j",
    "gpu_mem_gb",       
    "gpu_mem_bw_gbps",
]

def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        log.error(f"Columnas faltantes en el CSV: {missing}")
        sys.exit(1)

    nan_counts = {c: int(df[c].isna().sum())
                  for c in REQUIRED if df[c].isna().any()}
    if nan_counts:
        log.warning(f"Filas con NaN: {nan_counts} → se descartan")

    df = df.dropna(subset=[
        "cpu_exec_time_ms", "gpu_exec_time_ms",
        "cpu_energy_j",     "gpu_energy_j",
    ]).copy()

    log.info(f"Filas válidas: {len(df)}")
    return df.reset_index(drop=True)

# Memoria requerida por tarea (restricción R1)
def memory_gb(row) -> float:
    N    = int(row["N"])
    k    = str(row["kernel"]).upper()
    prec = 8   # float64

    if k == "GEMM":
        return 3 * N * N * prec / 1e9
    elif k == "SPMV":
        sp  = float(row.get("sparsity", 0.01))
        nnz = max(int(N * N * sp), 1)
        return (nnz * (prec + 4) + (N + 1) * 4) / 1e9
    else:  # FFT
        return N * prec * 2 / 1e9

# Solver ILP exacto con PuLP y CPLEX
def solve_ilp(df: pd.DataFrame,
              time_limit: int = 300,
              verbose: bool   = False) -> tuple[np.ndarray, float]:
    n = len(df)

    T_cpu = df["cpu_exec_time_ms"].values.astype(float)
    T_gpu = df["gpu_exec_time_ms"].values.astype(float)
    E_cpu = df["cpu_energy_j"].values.astype(float)
    E_gpu = df["gpu_energy_j"].values.astype(float)

    # Factor de escala σ, normalización simétrica CPU + GPU
    # Convierte energía (J) a la misma escala dimensional que la latencia (ms)
    # sin sesgar hacia ninguno de los dos dispositivos
    sigma = (T_cpu.sum() + T_gpu.sum()) / (E_cpu.sum() + E_gpu.sum() + 1e-12)
    log.info(f"σ = {sigma:.6f}  (factor de escala energía→latencia)")

    M_GPU = float(df["gpu_mem_gb"].iloc[0])
    B_GPU = float(df["gpu_mem_bw_gbps"].iloc[0])

    # Memoria requerida por tarea (para R1)
    mem_req = np.array([memory_gb(row) for _, row in df.iterrows()])

    # BW requerido por tarea (para R2)
    has_bw  = "bw_gbps_gpu" in df.columns and df["bw_gbps_gpu"].notna().any()
    bw_req  = df["bw_gbps_gpu"].values.astype(float) if has_bw else np.zeros(n)

    # Construir el problema PuLP
    prob = pulp.LpProblem("CPU_GPU_Assignment", pulp.LpMinimize)

    # R6: Variables binarias
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Función objetivo
    ZL = pulp.lpSum(x[i] * T_gpu[i] + (1 - x[i]) * T_cpu[i] for i in range(n))
    ZE = pulp.lpSum((x[i] * E_gpu[i] + (1 - x[i]) * E_cpu[i]) * sigma
                    for i in range(n))
    prob += ZL + ALPHA * ZE, "Objetivo_Latencia_Energia"

    # R1: Capacidad de memoria GPU
    # Reserva el 10% para driver CUDA/ROCm y estructuras del sistema
    prob += (
        pulp.lpSum(x[i] * mem_req[i] for i in range(n)) <= M_GPU * 0.90,
        "R1_Memoria_GPU"
    )
    log.info(f"R1: límite memoria GPU = {M_GPU * 0.90:.2f} GB")

    # R2: Ancho de banda GPU
    # Evita saturación del bus HBM2e
    if has_bw and bw_req.sum() > 0:
        prob += (
            pulp.lpSum(x[i] * bw_req[i] for i in range(n)) <= B_GPU * 0.85,
            "R2_BW_GPU"
        )
        log.info(f"R2: límite BW GPU = {B_GPU * 0.85:.1f} GB/s  (activa)")
    else:
        log.info("R2: omitida (bw_gbps_gpu no disponible)")

    # R3: Fracción máxima de tareas en GPU
    # Evita saturación del bus PCIe y overhead de context switching
    max_gpu = int(n * 0.85)
    prob += (
        pulp.lpSum(x[i] for i in range(n)) <= max_gpu,
        "R3_Fraccion_Max_GPU"
    )
    log.info(f"R3: máx tareas GPU = {max_gpu} de {n} ({max_gpu/n*100:.0f}%)")

    # R4: SpMV con esparsidad extrema
    # Accesos altamente irregulares degradan la GPU por debajo de la CPU
    n_r4 = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if (str(row["kernel"]).upper() == "SPMV"
                and float(row.get("sparsity", 1.0)) < 0.005):
            prob += x[i] == 0, f"R4_SpMV_CPU_{i}"
            n_r4 += 1
    log.info(f"R4: {n_r4} tareas SpMV dispersas forzadas a CPU")

    # R5: GEMM grande
    # A esta escala la intensidad operativa O(N) garantiza speedup GPU ≥ 2x
    n_r5 = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if (str(row["kernel"]).upper() == "GEMM"
                and int(row["N"]) >= 2048):
            prob += x[i] == 1, f"R5_GEMM_GPU_{i}"
            n_r5 += 1
    log.info(f"R5: {n_r5} tareas GEMM N≥2048 forzadas a GPU")

    # Resolver
    try:
        solver = pulp.CPLEX_PY(
            timeLimit=time_limit,
            msg=int(verbose),
            gapRel=1e-6,
        )
    except Exception as e:
        log.error(f"CPLEX no disponible: {e}")
        log.error("Instalar CPLEX Python API: pip install cplex")
        sys.exit(1)
    log.info("Solver: CPLEX")

    t0     = time.time()
    status = prob.solve(solver)
    elapsed = time.time() - t0

    status_str = pulp.LpStatus[status]
    obj_val    = pulp.value(prob.objective)
    log.info(f"Estado : {status_str}")
    log.info(f"Tiempo : {elapsed:.3f}s")
    log.info(f"Obj    : {obj_val:.6f}")

    if status_str not in ("Optimal", "Feasible"):
        log.error(f"El solver no encontró solución ({status_str}). "
                  f"Verifica que el problema sea factible.")
        sys.exit(1)

    labels = np.array([int(round(pulp.value(x[i]))) for i in range(n)])
    return labels, obj_val

def print_summary(df: pd.DataFrame):
    print()
    print("=" * 65)
    print("  RESUMEN DE ASIGNACIÓN ILP")
    print("=" * 65)
    print(f"  Filas procesadas : {len(df)}")
    print(f"  α (peso energía) : {ALPHA}  (balance típico)")
    print("  Solver           : CPLEX (exacto)")
    print()

    g = df.groupby("kernel").agg(
        Total   = ("ilp_label", "count"),
        CPU     = ("ilp_label", lambda x: (x == 0).sum()),
        GPU     = ("ilp_label", lambda x: (x == 1).sum()),
        T_cpu   = ("cpu_exec_time_ms", "mean"),
        T_gpu   = ("gpu_exec_time_ms", "mean"),
        E_cpu   = ("cpu_energy_j",     "mean"),
        E_gpu   = ("gpu_energy_j",     "mean"),
    ).round(4)
    g["speedup"] = (g["T_cpu"] / g["T_gpu"]).round(2)
    g["GPU_%"]   = (g["GPU"] / g["Total"] * 100).round(1)

    print(g[["Total", "CPU", "GPU", "GPU_%",
             "T_cpu", "T_gpu", "speedup",
             "E_cpu", "E_gpu"]].to_string())

    optional = [c for c in ["gflops_cpu","gflops_gpu","bw_gbps_cpu","bw_gbps_gpu"]
                if c in df.columns]
    if optional:
        print()
        print("  Métricas adicionales (medias por kernel):")
        print(df.groupby("kernel")[optional].mean().round(3).to_string())

    print()
    print(f"  CPU (0) total : {(df['ilp_label']==0).sum()} "
          f"({(df['ilp_label']==0).mean()*100:.1f}%)")
    print(f"  GPU (1) total : {(df['ilp_label']==1).sum()} "
          f"({(df['ilp_label']==1).mean()*100:.1f}%)")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(
        description="Etiquetado ILP exacto (PuLP) sobre el dataset del clúster"
    )
    parser.add_argument("--input",      required=True,
                        help="CSV con las métricas del clúster")
    parser.add_argument("--output",     default=None,
                        help="CSV de salida (default: <input>_labeled.csv)")

    parser.add_argument("--time-limit", type=int, default=300,
                        help="Límite de tiempo del solver en segundos (default: 300)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Mostrar output interno del solver")
    args = parser.parse_args()

    out = args.output or args.input.replace(".csv", "_labeled.csv")

    log.info(f"Dataset  : {args.input}")
    log.info(f"α ILP    : {ALPHA}  (balance típico, fijo)")
    log.info("Solver   : CPLEX (exacto)")
    log.info(f"T. límite: {args.time_limit}s")

    # Carga y validación
    df_raw = pd.read_csv(args.input)
    log.info(f"Filas leídas : {len(df_raw)} | "
             f"Kernels: {df_raw['kernel'].value_counts().to_dict()}")

    df = validate(df_raw)
    if len(df) < len(df_raw):
        log.warning(f"Descartadas {len(df_raw) - len(df)} filas con NaN")

    # Resolver ILP
    log.info("Resolviendo ILP...")
    labels, obj = solve_ilp(df,
                            time_limit=args.time_limit,
                            verbose=args.verbose)

    df["ilp_label"] = labels
    print_summary(df)

    # Guardar 
    df.to_csv(out, index=False, float_format="%.8g")
    log.info(f"Dataset etiquetado guardado en: {out}")

if __name__ == "__main__":
    main()