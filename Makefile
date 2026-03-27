# Makefile — Benchmarks HPC clúster SC3 UIS
# ============================================================
# Uso:
#   make nvidia_a100      → compila para A100  (sm_80)
#   make nvidia_titanx    → compila para Titan X (sm_52)
#   make amd_mi210        → compila para MI210 (gfx90a)
#   make amd_mi210_sysfs  → MI210 con backend sysfs (sin amdsmi)
#   make all              → los tres targets NVIDIA + AMD sysfs
#   make clean

# ── Herramientas ─────────────────────────────────────────────
NVCC    = nvcc
HIPCC   = hipcc
CC      = gcc

# ── Flags comunes ────────────────────────────────────────────
CFLAGS_COMMON = -O3 -std=c++17 -Wall
PAPI_INC      = $(shell papi-config --prefix 2>/dev/null)/include
PAPI_LIB      = $(shell papi-config --prefix 2>/dev/null)/lib

# Si papi-config no está en PATH, ajusta manualmente:
# PAPI_INC = /usr/local/include
# PAPI_LIB = /usr/local/lib

PAPI_FLAGS  = -I$(PAPI_INC) -L$(PAPI_LIB) -lpapi
THREAD_FLAG = -lpthread

# ── NVIDIA ────────────────────────────────────────────────────
CUDA_HOME   ?= /usr/local/cuda
NVCC_FLAGS   = $(CFLAGS_COMMON) \
               -I$(CUDA_HOME)/include \
               -L$(CUDA_HOME)/lib64

NVIDIA_LIBS  = -lcublas -lcusparse -lcufft -lnvml

.PHONY: nvidia_a100 nvidia_titanx

nvidia_a100:
	$(NVCC) $(NVCC_FLAGS) -arch=sm_80 \
	    benchmark_nvidia.cu \
	    $(PAPI_FLAGS) $(THREAD_FLAG) $(NVIDIA_LIBS) \
	    -o benchmark_a100
	@echo "Compilado: benchmark_a100 (sm_80 — A100)"

nvidia_titanx:
	$(NVCC) $(NVCC_FLAGS) -arch=sm_52 \
	    benchmark_nvidia.cu \
	    $(PAPI_FLAGS) $(THREAD_FLAG) $(NVIDIA_LIBS) \
	    -o benchmark_titanx
	@echo "Compilado: benchmark_titanx (sm_52 — GTX Titan X)"

# ── AMD ROCm ─────────────────────────────────────────────────
ROCM_HOME   ?= /opt/rocm
HIPCC_FLAGS  = $(CFLAGS_COMMON) \
               --offload-arch=gfx90a \
               -I$(ROCM_HOME)/include

AMD_LIBS     = -L$(ROCM_HOME)/lib \
               -lrocblas -lrocsparse -lrocfft \
               -lamdsmi
AMD_LIBS_SYSFS = -L$(ROCM_HOME)/lib \
                 -lrocblas -lrocsparse -lrocfft

.PHONY: amd_mi210 amd_mi210_sysfs

amd_mi210:
	$(HIPCC) $(HIPCC_FLAGS) \
	    benchmark_amd.hip.cpp \
	    $(PAPI_FLAGS) $(THREAD_FLAG) $(AMD_LIBS) \
	    -o benchmark_mi210
	@echo "Compilado: benchmark_mi210 (gfx90a — MI210, backend amdsmi)"

amd_mi210_sysfs:
	$(HIPCC) $(HIPCC_FLAGS) -DUSE_SYSFS_POWER \
	    benchmark_amd.hip.cpp \
	    $(PAPI_FLAGS) $(THREAD_FLAG) $(AMD_LIBS_SYSFS) \
	    -o benchmark_mi210
	@echo "Compilado: benchmark_mi210 (gfx90a — MI210, backend sysfs)"

# ── Targets combinados ────────────────────────────────────────
.PHONY: all clean

all: nvidia_a100 nvidia_titanx amd_mi210_sysfs

clean:
	rm -f benchmark_a100 benchmark_titanx benchmark_mi210
	rm -f resultados_*.csv

# ── Ejecución de referencia ───────────────────────────────────
.PHONY: run_a100 run_titanx run_mi210

run_a100:
	./benchmark_a100 --output resultados_a100.csv

run_titanx:
	./benchmark_titanx --output resultados_titanx.csv

run_mi210:
	./benchmark_mi210 --output resultados_mi210.csv
