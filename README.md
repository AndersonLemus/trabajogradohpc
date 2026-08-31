# Scheduling híbrido ILP + ML para sistemas heterogéneos CPU–GPU

Trabajo de grado — *Modelado predictivo de rendimiento en sistemas CPU–GPU
mediante un enfoque híbrido de programación lineal entera y aprendizaje
automático.*

**Autores:** Johan Sebastian Galvis Beltrán · Anderson Jahir Lemus Ramírez
**Institución:** Universidad Industrial de Santander (UIS)

---

## 1. Descripción

Este repositorio contiene el código, los datos y los modelos del proyecto. El
objetivo es decidir, para una tarea de HPC (GEMM, SpMV o FFT), si conviene
ejecutarla en **CPU** o en **GPU** minimizando de forma conjunta el **tiempo de
ejecución** y el **consumo energético**.

La propuesta es un flujo **ILP + ML** en dos tiempos:

1. **Offline** — un modelo de **Programación Lineal Entera (ILP)** resuelve la
   asignación óptima exacta sobre un dataset de instancias medidas en hardware
   real y produce la etiqueta de referencia (`ilp_label` ∈ {CPU=0, GPU=1}).
2. **Online** — clasificadores supervisados (**Random Forest** y **XGBoost**)
   aprenden esa política y se embeben en un *runtime* C++ que decide en
   microsegundos, sin resolver el ILP en tiempo de ejecución.

**Resultados principales:** los clasificadores reproducen la política del ILP
con **94,87 % de exactitud**; el modelo XGBoost embebido en el *runtime* C++
decide con una latencia media de **2,08 µs** (< 0,01 % del tiempo de las cargas
que el planificador envía a GPU).

---

## 2. Pipeline

```
 [1] Benchmarks            [2] Dataset             [3] ILP (etiquetado)
 benchmark_cpu_intel  ─┐                          ILP_thor.py
 benchmark_cpu_amd    ─┼─► raw_csvs/*.csv ──────► ILP_pacca.py   ─► results_*_labeled.csv
 benchmark_gpu_nvidia ─┤   (merge CPU + GPU)      ILP_exadell.py     (+ columna ilp_label)
 benchmark_amd.hip    ─┘

 [4] Entrenamiento ML                     [5] Export + runtime C++
 ML_models.py --input results_*_labeled   export_model.py  ─► ILP/generated/*.hpp
   ├─ rf_model.pkl / xgb_model.pkl         validate_predictions.cpp (sklearn == C++)
   ├─ X_test.npy / y_test.npy              runtime_main.cpp  ─► latencia de decisión
   └─ fig1..fig5 .png
```

---

## 3. Estructura del repositorio

| Ruta | Contenido |
|------|-----------|
| `benchmark_cpu_intel.cpp` | Benchmark CPU Intel (MKL): GEMM, SpMV, FFT + energía RAPL |
| `benchmark_cpu_amd.cpp` | Benchmark CPU AMD EPYC (OpenBLAS + FFTW + Eigen) |
| `benchmark_gpu_nvidia.cu` | Benchmark GPU NVIDIA (cuBLAS / cuSPARSE / cuFFT + NVML) |
| `benchmark_amd.hip.cpp` | Benchmark GPU AMD (rocBLAS / rocSPARSE / rocFFT) |
| `raw_csvs/` | Salidas crudas de los benchmarks por plataforma y nº de núcleos |
| `results_*_labeled.csv` | Dataset final etiquetado por el ILP (raíz = copia de referencia) |
| `ILP/ILP_thor.py`, `ILP/ILP_pacca.py`, `ILP/ILP_exadell.py` | Modelo ILP por plataforma (un ILP por cada valor de `CPU_Cores`) |
| `ILP/ML_models.py` | Entrenamiento RF + XGBoost con optimización de hiperparámetros (Optuna) |
| `ILP/export_model.py` | Exporta el modelo entrenado a un header C++ (`*.hpp`) |
| `ILP/include/` | `task_features.hpp` (vector de features), `tree_model.hpp` (inferencia de árboles) |
| `ILP/generated/` | Bosques embebidos (`rf_model.hpp`, `xgb_model.hpp`) + predicciones de muestra |
| `ILP/src/runtime_main.cpp` | Prototipo de *runtime*: consulta el modelo antes de despachar la tarea |
| `ILP/src/validate_predictions.cpp` | Verifica que la inferencia C++ == sklearn/xgboost |
| `ILP/fig1..fig5_*.png` | Figuras del capítulo de resultados |

> Los ejecutables compilados (`ILP/runtime_rf`, `ILP/runtime_xgb`,
> `ILP/validate_rf`, `ILP/validate_xgb`, `benchmark_*`) y los `.pkl` /`.npy`
> son artefactos reproducibles con los pasos de abajo; conviene moverlos a un
> `.gitignore`.

---

## 4. Requisitos

### Benchmarks (C++)
| Plataforma | Toolchain | Librerías |
|------------|-----------|-----------|
| CPU Intel | `g++ -O3 -march=native -std=c++17` | Intel oneAPI MKL |
| CPU AMD EPYC | `g++ -O3 -march=znver4 -fopenmp -std=c++17` | OpenBLAS, FFTW3, Eigen 3.4 |
| GPU NVIDIA | `nvcc -O3 -std=c++17` | CUDA Toolkit (cuBLAS, cuSPARSE, cuFFT), NVML |
| GPU AMD | `hipcc -O3 --offload-arch=gfx90a -std=c++17` | ROCm (rocBLAS, rocSPARSE, rocFFT) |

### Python (ILP + ML)
```bash
pip install numpy pandas scikit-learn xgboost optuna joblib matplotlib pulp
```
PuLP usa el solver CBC incluido; no requiere instalación adicional.

### Instalación de MKL (RHEL / Rocky)
```bash
sudo tee /etc/yum.repos.d/oneAPI.repo > /dev/null << 'EOF'
[oneAPI]
name=Intel oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
sudo dnf install intel-oneapi-mkl-devel
```

---

## 5. Reproducción paso a paso

### Fase 1 — Benchmarks

Los benchmarks aceptan `--output <archivo.csv>` y `--threads <n>` (CPU).
Cada ejecución recorre un barrido de tamaños de problema para los tres kernels.

**CPU Intel**
```bash
g++ -O3 -march=native -std=c++17 benchmark_cpu_intel.cpp \
    -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -fopenmp -lpthread -lm -ldl \
    -o benchmark_cpu_intel
./benchmark_cpu_intel --threads 16 --output raw_csvs/resultados_intel_16_cores.csv
```

**CPU AMD EPYC**
```bash
g++ -O3 -march=znver4 -fopenmp -std=c++17 benchmark_cpu_amd.cpp \
    -I<openblas>/include -I<eigen-3.4.0> -I<fftw>/include \
    -L<openblas>/lib -L<fftw>/lib \
    -lopenblas -lfftw3_omp -lfftw3 -lm -lpthread \
    -o benchmark_cpu_amd
./benchmark_cpu_amd --threads 64 --output raw_csvs/resultados_epyc_64_cores.csv
```

**GPU NVIDIA**
```bash
nvcc -O3 -arch=sm_52 -std=c++17 benchmark_gpu_nvidia.cu \
    -lnvidia-ml -lcublas -lcusparse -lcufft -lpthread -o benchmark_gpu_nvidia
./benchmark_gpu_nvidia --output raw_csvs/resultados_nvidia.csv
```

**GPU AMD**
```bash
hipcc -O3 --offload-arch=gfx90a -std=c++17 benchmark_amd.hip.cpp \
    -lrocblas -lrocsparse -lrocfft -lpthread -o benchmark_gpu_amd
./benchmark_gpu_amd --output raw_csvs/resultados_mi210.csv
```

Comandos SLURM de ejemplo para los nodos del clúster GUANE/UIS:
```bash
srun -p fat -w thor      --nodes=1 --ntasks=1 --cpus-per-task=8  --time=02:00:00 --pty bash -i
srun -p GPU -w paccaA100  --nodes=1 --ntasks=1 --cpus-per-task=8  --exclusive --time=48:00:00 --pty bash -i
srun -p amd -w exadell     --nodes=1 --ntasks=1 --cpus-per-task=8  --exclusive --time=48:00:00 --pty bash -i
# añade --gres=gpu:1 para reservar el acelerador
```

**Esquema de salida** (columnas relevantes tras combinar CPU + GPU):

`kernel, N, nnz, sparsity, op_intensity_cpu, mem_mb, cpu_time_s, cpu_energy_j,
gflops_cpu, bw_gbps_cpu, CPU_Cores, CPU_Freq, op_intensity_gpu, gpu_sm_count,
gpu_freq_ghz, gpu_mem_gb, gpu_mem_bw_gbps, gpu_time_s, gpu_energy_j, gflops_gpu,
bw_gbps_gpu, speedup`

### Fase 2 — Dataset

Se combinan las filas de CPU y de GPU del mismo kernel/tamaño en un único CSV
por plataforma (`ILP/results_thor.csv`, `ILP/results_pacca.csv`,
`ILP/results_exadell.csv`).

### Fase 3 — Etiquetado ILP

```bash
cd ILP
python ILP_thor.py --input results_thor.csv --output results_thor_labeled.csv
```

Un ILP independiente por cada configuración de `CPU_Cores`. Función objetivo:
`min  Σ [x·T_gpu + (1-x)·T_cpu]  +  α · Σ energía`, con variables binarias
`x_i` (1 = GPU).

| Parámetro | Bandera | Valor por defecto | Significado |
|-----------|---------|-------------------|-------------|
| `α` | `--alpha` | `0.3` | Peso de la energía en el objetivo |
| `γ` | — | `0.85` | Fracción máxima de tareas en GPU (R3) |
| Umbral R4 | `--r4-threshold` | `0.5` | `T_gpu / T_cpu` para forzar GEMM grande a GPU |
| Límite de tiempo | `--time-limit` | `300` s | Corte del solver por sub-problema |

Restricciones: **R1** memoria VRAM (≤ 90 % de la capacidad), **R2** ancho de
banda (≤ 85 % del pico), **R3** cota `γ` de tareas en GPU, **R4** incentivo para
GEMM grande claramente favorable a GPU.

### Fase 4 — Entrenamiento de los clasificadores

```bash
cd ILP
python ML_models.py --input results_thor_labeled.csv --output-dir . --trials 100
```

15 features (`kernel_encoded, N, nnz, sparsity, op_intensity_cpu/gpu,
cpu/gpu_time_s, cpu/gpu_energy_j, gflops_cpu/gpu, bw_gbps_cpu/gpu, speedup`),
target `ilp_label`. Split 80/20 estratificado, HPO con Optuna, validación
cruzada de 5 pliegues.

Genera: `rf_model.pkl`, `xgb_model.pkl`, `X_test.npy`, `y_test.npy` y
`fig1_model_evaluation.png` … `fig5_optuna_history.png`.

> **Nota de honestidad metodológica** (ver `ILP/include/task_features.hpp`):
> 5 de las 15 features (`gpu_time_s`, `gpu_energy_j`, `gflops_gpu`,
> `bw_gbps_gpu`, `speedup`) sólo se conocen *después* de ejecutar en GPU. El
> *runtime* opera en **modo replay** sobre features ya medidas para demostrar
> el mecanismo y medir su latencia; sustituirlas por estimaciones a priori es
> trabajo futuro.

### Fase 5 — Exportar a C++, validar y ejecutar el *runtime*

```bash
cd ILP
# 1. Exportar el modelo a un header + CSV de validación
python export_model.py --model xgb_model.pkl --kind xgb \
    --out generated/xgb_model.hpp \
    --X-test X_test.npy --sample-out generated/xgb_sample_predictions.csv

# 2. Validar: la inferencia C++ debe coincidir con Python
g++ -O3 -std=c++17 -DVALIDATE_XGB src/validate_predictions.cpp -o validate_xgb
./validate_xgb generated/xgb_sample_predictions.csv 1e-5

# 3. Runtime: mide la latencia de decisión y (si el CSV trae ilp_label)
#    compara contra la etiqueta real
g++ -O3 -std=c++17 -DRUNTIME_MODEL_XGB src/runtime_main.cpp -o runtime_xgb
./runtime_xgb results_thor_labeled.csv
```

Cambia `xgb` → `rf` y los macros `-DVALIDATE_RF` / `-DRUNTIME_MODEL_RF` para el
Random Forest. Sin argumento CSV, `runtime_*` usa 3 tareas sintéticas de
ejemplo.

---

## 6. Plataformas de hardware

| Alias | Nodo | CPU | GPU |
|-------|------|-----|-----|
| `thor` | HPE (nodo *fat*) | Intel Xeon | NVIDIA Tesla M40 (sm_52) |
| `pacca` | HPE XL290n G10+ | Intel Xeon Gold 5315Y (8 / 16 c) | NVIDIA A100 40 GB |
| `exadell` | Dell PowerEdge R7625 | AMD EPYC 9554 (hasta 64 c) | AMD Instinct MI210 64 GB |

Dataset final: **1 364 instancias**; el ILP asigna 1 206 a CPU (88,4 %) y
158 a GPU (11,6 %). Detalles y análisis completo en el documento de tesis.

---

## 7. Recursos complementarios

- **Repositorio:** <https://github.com/AndersonLemus/trabajogradohpc>
- **Notebook Colab:** <https://colab.research.google.com/drive/14bjIRm0TNDywzpddDNzft9y4cKC842_g?usp=sharing>
- **Documento de tesis:** `overleaf/` (fuente LaTeX y PDF compilado)

---

## 8. Licencia

Uso académico. Consultar con los autores antes de reutilizar el código o los
datos.
