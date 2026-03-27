/*
 * benchmark_nvidia.cu
 * ============================================================
 * Benchmark HPC para plataformas NVIDIA del clúster SC3 — UIS
 *
 *   HPE ProLiant XL290n G10+  →  NVIDIA A100 40 GB   (sm_80)
 *   HPE ProLiant DL580 G7     →  NVIDIA GTX Titan X  (sm_52)
 *
 * Kernels:
 *   GEMM  — cuBLAS   (cublasDgemm, double precision)
 *   SpMV  — cuSPARSE (cusparseSpMV, CSR, double)
 *   FFT   — cuFFT    (cufftExecD2Z, 1D real→complex)
 *
 * Métricas por ejecución:
 *   gpu_time_ms    — CUDA Events
 *   gpu_energy_j   — NVML (hilo de muestreo a 10 ms)
 *   gflops         — FLOPs teóricos / tiempo
 *   bw_gbps        — bytes transferidos / tiempo
 *   cpu_*          — PAPI: FLOPS, L1/L2/L3 misses, ciclos, instrucciones
 *
 * Compilación:
 *   A100 (sm_80):
 *     nvcc -O3 -arch=sm_80 benchmark_nvidia.cu \
 *          -lcublas -lcusparse -lcufft -lnvml -lpapi -lpthread \
 *          -o benchmark_a100
 *
 *   Titan X (sm_52):
 *     nvcc -O3 -arch=sm_52 benchmark_nvidia.cu \
 *          -lcublas -lcusparse -lcufft -lnvml -lpapi -lpthread \
 *          -o benchmark_titanx
 *
 * Uso:
 *   ./benchmark_a100  --output resultados_a100.csv
 *   ./benchmark_titanx --output resultados_titanx.csv
 *
 * Salida CSV (una fila por (kernel, N, sparsity)):
 *   platform, kernel, N, nnz, precision,
 *   gpu_time_ms, gpu_energy_j, gflops, bw_gbps,
 *   cpu_flops, cpu_l1_miss, cpu_l2_miss, cpu_l3_miss,
 *   cpu_cycles, cpu_instructions
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cufft.h>
#include <nvml.h>
#include <papi.h>

/* ───────────────────────────── Macros de error ───────────────────────────── */

#define CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                              \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA %s:%d %s\n",                             \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CUBLAS_CHECK(x) do {                                            \
    cublasStatus_t _s = (x);                                           \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuBLAS %s:%d code=%d\n",                      \
                __FILE__, __LINE__, _s);                                \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CUSPARSE_CHECK(x) do {                                          \
    cusparseStatus_t _s = (x);                                         \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                \
        fprintf(stderr, "cuSPARSE %s:%d code=%d\n",                    \
                __FILE__, __LINE__, _s);                                \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CUFFT_CHECK(x) do {                                             \
    cufftResult _r = (x);                                              \
    if (_r != CUFFT_SUCCESS) {                                          \
        fprintf(stderr, "cuFFT %s:%d code=%d\n",                       \
                __FILE__, __LINE__, _r);                                \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define NVML_CHECK(x) do {                                              \
    nvmlReturn_t _r = (x);                                             \
    if (_r != NVML_SUCCESS) {                                           \
        fprintf(stderr, "NVML %s:%d %s\n",                             \
                __FILE__, __LINE__, nvmlErrorString(_r));               \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define PAPI_CHECK(x) do {                                              \
    int _r = (x);                                                       \
    if (_r != PAPI_OK)                                                  \
        fprintf(stderr, "PAPI %s:%d %s\n",                             \
                __FILE__, __LINE__, PAPI_strerror(_r));                 \
} while(0)

/* ──────────────────────────── Configuración ──────────────────────────────── */

#define WARMUP_RUNS      3
#define BENCH_RUNS       5
#define NVML_INTERVAL_MS 10   /* periodo de muestreo de potencia GPU */

/* Tamaños GEMM: matrices cuadradas N×N */
static const int GEMM_N[] = {
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2500, 3000, 4000, 5000, 6000, 7000, 8000
};
static const int N_GEMM = (int)(sizeof(GEMM_N)/sizeof(GEMM_N[0]));

/* Tamaños SpMV: matriz N×N */
static const int SPMV_N[] = {
    100, 200, 500, 1000, 2000, 4000, 8000,
    16000, 32000, 64000, 100000
};
static const int N_SPMV = (int)(sizeof(SPMV_N)/sizeof(SPMV_N[0]));

/* Sparsidades del diseño experimental: 10%, 1%, 0.1% */
static const double SPARSITIES[] = { 0.10, 0.01, 0.001 };
static const int N_SP = 3;

/* Tamaños FFT (potencias de 2 hasta 2^20 = 1M) */
static const int FFT_N[] = {
    256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144, 524288, 1048576
};
static const int N_FFT = (int)(sizeof(FFT_N)/sizeof(FFT_N[0]));

/* ─────────────────────────── Estructura de resultado ─────────────────────── */

typedef struct {
    char   kernel[16];
    int    N;
    long   nnz;
    char   prec[8];
    double gpu_time_ms;
    double gpu_energy_j;
    double gflops;
    double bw_gbps;
    long long cpu_fp_ops;
    long long cpu_l1_miss;
    long long cpu_l2_miss;
    long long cpu_l3_miss;
    long long cpu_cycles;
    long long cpu_instr;
} Result;

/* ────────────────────────── NVML: hilo de muestreo ──────────────────────── */

typedef struct {
    nvmlDevice_t dev;
    volatile int active;
    double       energy_j;
} NvmlCtx;

static void* nvml_thread(void* arg)
{
    NvmlCtx* ctx = (NvmlCtx*)arg;
    struct timespec ts = { 0, NVML_INTERVAL_MS * 1000000L };
    while (ctx->active) {
        unsigned int mw = 0;
        nvmlDeviceGetPowerUsage(ctx->dev, &mw);
        ctx->energy_j += (mw / 1000.0) * (NVML_INTERVAL_MS / 1000.0);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

/* ──────────────────────────────── PAPI ───────────────────────────────────── */

#define N_EV 6
static int papi_evset = PAPI_NULL;
static const int PAPI_EV[N_EV] = {
    PAPI_FP_OPS, PAPI_L1_DCM, PAPI_L2_DCM,
    PAPI_L3_TCM, PAPI_TOT_CYC, PAPI_TOT_INS
};

static void papi_setup(void)
{
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI init falló\n"); return;
    }
    PAPI_CHECK(PAPI_create_eventset(&papi_evset));
    for (int i = 0; i < N_EV; i++) {
        int r = PAPI_add_event(papi_evset, PAPI_EV[i]);
        if (r != PAPI_OK)
            fprintf(stderr, "PAPI: no se pudo agregar evento %d\n", i);
    }
}

/* ─────────────────────────── Escritura CSV ───────────────────────────────── */

static void csv_header(FILE* f)
{
    fprintf(f,
        "platform,kernel,N,nnz,precision,"
        "gpu_time_ms,gpu_energy_j,gflops,bw_gbps,"
        "cpu_fp_ops,cpu_l1_miss,cpu_l2_miss,cpu_l3_miss,"
        "cpu_cycles,cpu_instructions\n");
}

static void csv_row(FILE* f, const char* plat, const Result* r)
{
    fprintf(f,
        "%s,%s,%d,%ld,%s,"
        "%.6f,%.6f,%.4f,%.4f,"
        "%lld,%lld,%lld,%lld,%lld,%lld\n",
        plat, r->kernel, r->N, r->nnz, r->prec,
        r->gpu_time_ms, r->gpu_energy_j, r->gflops, r->bw_gbps,
        r->cpu_fp_ops, r->cpu_l1_miss, r->cpu_l2_miss,
        r->cpu_l3_miss, r->cpu_cycles, r->cpu_instr);
}

/* ════════════════════════════════════════════════════════════════════════════
 * GEMM — cuBLAS dgemm
 * ════════════════════════════════════════════════════════════════════════════
 * C = alpha*A*B + beta*C   (todas N×N, double)
 * FLOPs = 2*N^3
 * Bytes  = 3*N^2*8  (lectura A,B + escritura C)
 */
static Result run_gemm(int N, cublasHandle_t blas, nvmlDevice_t nvml)
{
    Result r; memset(&r, 0, sizeof(r));
    strcpy(r.kernel, "GEMM"); strcpy(r.prec, "double"); r.N = N;

    size_t sz = (size_t)N * N * sizeof(double);
    double *hA = (double*)malloc(sz);
    double *hB = (double*)malloc(sz);
    double *dA, *dB, *dC;

    srand(42);
    for (int i = 0; i < N*N; i++) {
        hA[i] = (double)rand()/RAND_MAX;
        hB[i] = (double)rand()/RAND_MAX;
    }

    CUDA_CHECK(cudaMalloc(&dA, sz));
    CUDA_CHECK(cudaMalloc(&dB, sz));
    CUDA_CHECK(cudaMalloc(&dC, sz));
    CUDA_CHECK(cudaMemcpy(dA, hA, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, sz));

    const double alpha = 1.0, beta = 0.0;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    /* warm-up */
    for (int w = 0; w < WARMUP_RUNS; w++)
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* medición */
    NvmlCtx ctx = { nvml, 1, 0.0 };
    pthread_t tid;
    pthread_create(&tid, NULL, nvml_thread, &ctx);

    long long pv[N_EV] = {0};
    double    tot_ms = 0.0;
    PAPI_CHECK(PAPI_reset(papi_evset));
    PAPI_CHECK(PAPI_start(papi_evset));

    for (int i = 0; i < BENCH_RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        tot_ms += ms;
    }

    PAPI_CHECK(PAPI_stop(papi_evset, pv));
    ctx.active = 0; pthread_join(tid, NULL);

    double avg = tot_ms / BENCH_RUNS;
    r.gpu_time_ms  = avg;
    r.gpu_energy_j = ctx.energy_j;
    r.gflops       = (2.0*N*N*N / 1e9) / (avg / 1e3);
    r.bw_gbps      = (3.0*N*N*sizeof(double) / 1e9) / (avg / 1e3);
    r.cpu_fp_ops   = pv[0]; r.cpu_l1_miss = pv[1];
    r.cpu_l2_miss  = pv[2]; r.cpu_l3_miss = pv[3];
    r.cpu_cycles   = pv[4]; r.cpu_instr   = pv[5];

    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB);
    return r;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SpMV — cuSPARSE CSR
 * ════════════════════════════════════════════════════════════════════════════
 * y = A * x   (A N×N sparse CSR, double)
 * FLOPs = 2 * nnz
 * Bytes  = nnz*8 (val) + nnz*4 (col) + (N+1)*4 (row) + N*8 (x) + N*8 (y)
 */
static Result run_spmv(int N, double sparsity,
                        cusparseHandle_t sp, nvmlDevice_t nvml)
{
    Result r; memset(&r, 0, sizeof(r));
    strcpy(r.kernel, "SpMV"); strcpy(r.prec, "double"); r.N = N;

    /* Construir CSR en host */
    long nnz = (long)((double)N * N * sparsity);
    if (nnz < N) nnz = N;

    int    *h_row = (int*)   calloc(N+1, sizeof(int));
    int    *h_col = (int*)   malloc(nnz  * sizeof(int));
    double *h_val = (double*)malloc(nnz  * sizeof(double));
    double *h_x   = (double*)malloc(N    * sizeof(double));

    srand(77);
    long per_row = nnz / N, rem = nnz % N, ptr = 0;
    h_row[0] = 0;
    for (int i = 0; i < N; i++) {
        long cnt = per_row + (i < rem ? 1 : 0);
        /* columnas equidistantes en [0, N) */
        for (long j = 0; j < cnt; j++) {
            h_col[ptr] = (int)(j * N / cnt) % N;
            h_val[ptr] = (double)rand()/RAND_MAX + 0.1;
            ptr++;
        }
        h_row[i+1] = (int)ptr;
    }
    nnz = ptr;
    r.nnz = nnz;

    for (int i = 0; i < N; i++)
        h_x[i] = (double)rand()/RAND_MAX;

    int    *d_row, *d_col;
    double *d_val, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row, (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col, nnz  *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, nnz  *sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,   N    *sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,   N    *sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row, h_row, (N+1)*sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, h_col, nnz  *sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, h_val, nnz  *sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,   h_x,   N    *sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, N*sizeof(double)));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, N, N, nnz,
        d_row, d_col, d_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_64F));

    const double alpha = 1.0, beta = 0.0;
    size_t buf_sz = 0; void* buf = NULL;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sp,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_sz));
    if (buf_sz) CUDA_CHECK(cudaMalloc(&buf, buf_sz));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    /* warm-up */
    for (int w = 0; w < WARMUP_RUNS; w++)
        CUSPARSE_CHECK(cusparseSpMV(sp,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* medición */
    NvmlCtx ctx = { nvml, 1, 0.0 };
    pthread_t tid;
    pthread_create(&tid, NULL, nvml_thread, &ctx);

    long long pv[N_EV] = {0};
    double    tot_ms = 0.0;
    PAPI_CHECK(PAPI_reset(papi_evset)); PAPI_CHECK(PAPI_start(papi_evset));

    for (int i = 0; i < BENCH_RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUSPARSE_CHECK(cusparseSpMV(sp,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        tot_ms += ms;
    }

    PAPI_CHECK(PAPI_stop(papi_evset, pv));
    ctx.active = 0; pthread_join(tid, NULL);

    double avg = tot_ms / BENCH_RUNS;
    double bytes = (double)nnz*(sizeof(double)+sizeof(int))
                 + (double)(N+1)*sizeof(int)
                 + 2.0*N*sizeof(double);
    r.gpu_time_ms  = avg;
    r.gpu_energy_j = ctx.energy_j;
    r.gflops       = (2.0*nnz / 1e9) / (avg / 1e3);
    r.bw_gbps      = (bytes / 1e9) / (avg / 1e3);
    r.cpu_fp_ops   = pv[0]; r.cpu_l1_miss = pv[1];
    r.cpu_l2_miss  = pv[2]; r.cpu_l3_miss = pv[3];
    r.cpu_cycles   = pv[4]; r.cpu_instr   = pv[5];

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    if (buf) CUDA_CHECK(cudaFree(buf));
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(d_row)); CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_val)); CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_row); free(h_col); free(h_val); free(h_x);
    return r;
}

/* ════════════════════════════════════════════════════════════════════════════
 * FFT — cuFFT D2Z (real double → complex double, 1D)
 * ════════════════════════════════════════════════════════════════════════════
 * FLOPs ≈ 5 * N * log2(N)
 * Bytes  = N*8 (entrada) + (N/2+1)*16 (salida)
 */
static Result run_fft(int N, nvmlDevice_t nvml)
{
    Result r; memset(&r, 0, sizeof(r));
    strcpy(r.kernel, "FFT"); strcpy(r.prec, "double"); r.N = N;

    double*             h_in  = (double*)malloc(N * sizeof(double));
    cufftDoubleComplex* h_out = (cufftDoubleComplex*)
                                 malloc((N/2+1)*sizeof(cufftDoubleComplex));
    for (int i = 0; i < N; i++)
        h_in[i] = sin(2.0*M_PI*i/N) + 0.3*cos(6.0*M_PI*i/N);

    double*             d_in;
    cufftDoubleComplex* d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N       *sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, (N/2+1) *sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N*sizeof(double), cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_D2Z, 1));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    /* warm-up */
    for (int w = 0; w < WARMUP_RUNS; w++)
        CUFFT_CHECK(cufftExecD2Z(plan, d_in, d_out));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* medición */
    NvmlCtx ctx = { nvml, 1, 0.0 };
    pthread_t tid;
    pthread_create(&tid, NULL, nvml_thread, &ctx);

    long long pv[N_EV] = {0};
    double    tot_ms = 0.0;
    PAPI_CHECK(PAPI_reset(papi_evset)); PAPI_CHECK(PAPI_start(papi_evset));

    for (int i = 0; i < BENCH_RUNS; i++) {
        /* reset para medir solo la FFT, no la transferencia */
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N*sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e0));
        CUFFT_CHECK(cufftExecD2Z(plan, d_in, d_out));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        tot_ms += ms;
    }

    PAPI_CHECK(PAPI_stop(papi_evset, pv));
    ctx.active = 0; pthread_join(tid, NULL);

    double avg   = tot_ms / BENCH_RUNS;
    double flops = 5.0 * N * log2((double)N);
    double bytes = (double)N*sizeof(double)
                 + (double)(N/2+1)*sizeof(cufftDoubleComplex);
    r.gpu_time_ms  = avg;
    r.gpu_energy_j = ctx.energy_j;
    r.gflops       = (flops / 1e9) / (avg / 1e3);
    r.bw_gbps      = (bytes / 1e9) / (avg / 1e3);
    r.cpu_fp_ops   = pv[0]; r.cpu_l1_miss = pv[1];
    r.cpu_l2_miss  = pv[2]; r.cpu_l3_miss = pv[3];
    r.cpu_cycles   = pv[4]; r.cpu_instr   = pv[5];

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out);
    return r;
}

/* ════════════════════════════════════════════════════════════════════════════
 * main
 * ════════════════════════════════════════════════════════════════════════════ */

int main(int argc, char* argv[])
{
    const char* out_path = "resultados_nvidia.csv";
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--output")) out_path = argv[i+1];

    /* Identificar GPU y plataforma */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU detectada  : %s  (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    printf("Memoria global : %.1f GB\n\n", prop.totalGlobalMem / 1e9);

    char platform[64];
    if      (strstr(prop.name, "A100"))  snprintf(platform, 64, "HPE_XL290n_A100");
    else if (strstr(prop.name, "TITAN") ||
             strstr(prop.name, "Titan")) snprintf(platform, 64, "HPE_DL580_TitanX");
    else                                 snprintf(platform, 64, "NVIDIA_%s", prop.name);

    /* NVML */
    NVML_CHECK(nvmlInit());
    nvmlDevice_t nvml;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &nvml));

    /* PAPI */
    papi_setup();

    /* Handles de bibliotecas */
    cublasHandle_t   blas;
    cusparseHandle_t sph;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUSPARSE_CHECK(cusparseCreate(&sph));

    /* CSV */
    FILE* fp = fopen(out_path, "w");
    if (!fp) { perror("fopen"); return 1; }
    csv_header(fp);
    printf("Resultados → %s\n\n", out_path);

    /* ════ GEMM ═════════════════════════════════════════════════════════════ */
    printf("=== GEMM (cuBLAS dgemm) ===\n");
    printf("%-6s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_gpu(ms)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_GEMM; i++) {
        int N = GEMM_N[i];
        if (3ULL*N*N*sizeof(double) > prop.totalGlobalMem * 0.78) {
            printf("N=%-6d SKIP (memoria insuficiente)\n", N); continue;
        }
        Result r = run_gemm(N, blas, nvml);
        printf("%-6d  %-12.4f  %-12.6f  %-10.2f  %-10.2f\n",
               N, r.gpu_time_ms, r.gpu_energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, platform, &r); fflush(fp);
    }

    /* ════ SpMV ═════════════════════════════════════════════════════════════ */
    printf("\n=== SpMV (cuSPARSE CSR) ===\n");
    printf("%-6s  %-8s  %-10s  %-12s  %-12s  %-10s\n",
           "N", "sparsity", "nnz", "T_gpu(ms)", "Energy(J)", "GFLOPS");

    for (int si = 0; si < N_SP; si++) {
        double sp = SPARSITIES[si];
        for (int i = 0; i < N_SPMV; i++) {
            int  N   = SPMV_N[i];
            long nnz = (long)((double)N*N*sp); if (nnz < N) nnz = N;
            size_t need = (size_t)nnz*(sizeof(double)+sizeof(int))
                        + (size_t)(N+1)*sizeof(int)
                        + (size_t)2*N*sizeof(double);
            if (need > prop.totalGlobalMem * 0.78) {
                printf("N=%-6d sp=%.3f SKIP\n", N, sp); continue;
            }
            Result r = run_spmv(N, sp, sph, nvml);
            printf("%-6d  %-8.3f  %-10ld  %-12.4f  %-12.6f  %-10.4f\n",
                   N, sp, r.nnz, r.gpu_time_ms, r.gpu_energy_j, r.gflops);
            csv_row(fp, platform, &r); fflush(fp);
        }
    }

    /* ════ FFT ══════════════════════════════════════════════════════════════ */
    printf("\n=== FFT (cuFFT D2Z) ===\n");
    printf("%-10s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_gpu(ms)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_FFT; i++) {
        int N = FFT_N[i];
        Result r = run_fft(N, nvml);
        printf("%-10d  %-12.4f  %-12.6f  %-10.4f  %-10.4f\n",
               N, r.gpu_time_ms, r.gpu_energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, platform, &r); fflush(fp);
    }

    /* Limpieza */
    fclose(fp);
    CUBLAS_CHECK(cublasDestroy(blas));
    CUSPARSE_CHECK(cusparseDestroy(sph));
    NVML_CHECK(nvmlShutdown());
    PAPI_CHECK(PAPI_cleanup_eventset(papi_evset));
    PAPI_CHECK(PAPI_destroy_eventset(&papi_evset));
    PAPI_shutdown();

    printf("\nCompletado. Resultados en: %s\n", out_path);
    return 0;
}
