/*
 *   GEMM  — rocBLAS   (rocblas_dgemm, double precision)
 *   SpMV  — rocSPARSE (rocsparse_dcsrmv, CSR, double)
 *   FFT   — rocFFT    (rocfft_execute, D2Z, 1D)
 *
 * Métricas por ejecución:
 *   gpu_time_ms    — hipEvent_t (análogo a CUDA Events)
 *   gpu_energy_j   — rocm-smi via /sys/class/drm o amdsmi (hilo 10 ms)
 *   gflops         — FLOPs teóricos / tiempo
 *   bw_gbps        — bytes transferidos / tiempo
 *   cpu_*          — PAPI: FLOPS, L1/L2/L3 misses, ciclos, instrucciones
 *
 * Compilación (ROCm ≥ 5.6):
 *   hipcc -O3 --offload-arch=gfx90a benchmark_amd.hip.cpp \
 *         -lrocblas -lrocsparse -lrocfft -lpapi -lpthread \
 *         -o benchmark_mi210
 *
 *   Si amdsmi no está disponible, usa la lectura sysfs:
 *   hipcc -O3 --offload-arch=gfx90a benchmark_amd.hip.cpp \
 *         -lrocblas -lrocsparse -lrocfft -lpapi -lpthread \
 *         -DUSE_SYSFS_POWER \
 *         -o benchmark_mi210
 *
 * Uso:
 *   ./benchmark_mi210 --output resultados_mi210.csv
 *
 * Salida CSV (misma estructura que benchmark_nvidia):
 *   platform, kernel, N, nnz, precision,
 *   gpu_time_ms, gpu_energy_j, gflops, bw_gbps,
 *   cpu_fp_ops, cpu_l1_miss, cpu_l2_miss, cpu_l3_miss,
 *   cpu_cycles, cpu_instructions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

/* HIP runtime */
#include <hip/hip_runtime.h>

/* rocBLAS */
#include <rocblas/rocblas.h>

/* rocSPARSE */
#include <rocsparse/rocsparse.h>

/* rocFFT */
#include <rocfft/rocfft.h>

/* PAPI */
#include <papi.h>

/* Medición de potencia AMD
 * Dos estrategias:
 *   1. amdsmi (ROCm ≥ 6.0) - más precisa
 *   2. sysfs hwmon - compatible con versiones anteriores
 * Se selecciona con -DUSE_SYSFS_POWER o sin ella (amdsmi).
 */

#ifndef USE_SYSFS_POWER
  /* amdsmi — disponible en ROCm ≥ 6.0 */
  #include <amd_smi/amdsmi.h>
  #define POWER_BACKEND "amdsmi"
#else
  #define POWER_BACKEND "sysfs"
#endif

/*  Macros error */

#define HIP_CHECK(x) do {                                               \
    hipError_t _e = (x);                                               \
    if (_e != hipSuccess) {                                             \
        fprintf(stderr, "HIP %s:%d %s\n",                              \
                __FILE__, __LINE__, hipGetErrorString(_e));             \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define ROCBLAS_CHECK(x) do {                                           \
    rocblas_status _s = (x);                                           \
    if (_s != rocblas_status_success) {                                 \
        fprintf(stderr, "rocBLAS %s:%d code=%d\n",                     \
                __FILE__, __LINE__, _s);                                \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define ROCSPARSE_CHECK(x) do {                                         \
    rocsparse_status _s = (x);                                         \
    if (_s != rocsparse_status_success) {                               \
        fprintf(stderr, "rocSPARSE %s:%d code=%d\n",                   \
                __FILE__, __LINE__, _s);                                \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define ROCFFT_CHECK(x) do {                                            \
    rocfft_status _s = (x);                                            \
    if (_s != rocfft_status_success) {                                  \
        fprintf(stderr, "rocFFT %s:%d code=%d\n",                      \
                __FILE__, __LINE__, _s);                                \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define PAPI_CHECK(x) do {                                              \
    int _r = (x);                                                       \
    if (_r != PAPI_OK)                                                  \
        fprintf(stderr, "PAPI %s:%d %s\n",                             \
                __FILE__, __LINE__, PAPI_strerror(_r));                 \
} while(0)

/* Configuración */

#define WARMUP_RUNS      3
#define BENCH_RUNS       5
#define POWER_INTERVAL_MS 10

static const int GEMM_N[] = {
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2500, 3000, 4000, 5000, 6000, 7000, 8000
};
static const int N_GEMM = (int)(sizeof(GEMM_N)/sizeof(GEMM_N[0]));

static const int SPMV_N[] = {
    100, 200, 500, 1000, 2000, 4000, 8000,
    16000, 32000, 64000, 100000
};
static const int N_SPMV = (int)(sizeof(SPMV_N)/sizeof(SPMV_N[0]));

static const double SPARSITIES[] = { 0.10, 0.01, 0.001 };
static const int    N_SP         = 3;

static const int FFT_N[] = {
    256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144, 524288, 1048576
};
static const int N_FFT = (int)(sizeof(FFT_N)/sizeof(FFT_N[0]));

/* Estructura de resultado */

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

/* Medición de potencia AMD */

typedef struct {
    volatile int active;
    double       energy_j;
    /* parámetros según backend */
#ifndef USE_SYSFS_POWER
    amdsmi_processor_handle_t handle;
#else
    char hwmon_path[256];   /* /sys/class/hwmon/hwmonX/power1_average */
#endif
} PowerCtx;

#ifdef USE_SYSFS_POWER
/* Localiza el archivo de potencia del primer dispositivo amdgpu en sysfs */
static int find_amdgpu_hwmon(char* path, size_t len)
{
    for (int idx = 0; idx < 20; idx++) {
        char name_path[256];
        snprintf(name_path, sizeof(name_path),
                 "/sys/class/hwmon/hwmon%d/name", idx);
        FILE* f = fopen(name_path, "r");
        if (!f) continue;
        char name[32] = {0};
        fgets(name, sizeof(name), f);
        fclose(f);
        if (strncmp(name, "amdgpu", 6) == 0) {
            snprintf(path, len,
                     "/sys/class/hwmon/hwmon%d/power1_average", idx);
            return 0;
        }
    }
    return -1;
}

static double read_power_w(const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f) return 0.0;
    long uw = 0;
    fscanf(f, "%ld", &uw);
    fclose(f);
    return uw / 1e6;   /* µW → W */
}
#endif  /* USE_SYSFS_POWER */

static void* power_thread(void* arg)
{
    PowerCtx* ctx = (PowerCtx*)arg;
    struct timespec ts = { 0, POWER_INTERVAL_MS * 1000000L };

    while (ctx->active) {
        double watts = 0.0;

#ifndef USE_SYSFS_POWER
        /* amdsmi: potencia en µW, dividir por 1e6 */
        uint64_t power_uw = 0;
        if (amdsmi_get_power_info(ctx->handle, &power_uw) ==
            AMDSMI_STATUS_SUCCESS)
            watts = (double)power_uw / 1e6;
#else
        watts = read_power_w(ctx->hwmon_path);
#endif
        ctx->energy_j += watts * (POWER_INTERVAL_MS / 1000.0);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void power_ctx_init(PowerCtx* ctx)
{
    ctx->active   = 1;
    ctx->energy_j = 0.0;

#ifndef USE_SYSFS_POWER
    /* Inicializar amdsmi y obtener handle del primer dispositivo */
    amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    uint32_t      dev_count = 0;
    amdsmi_processor_handle_t handles[8];
    amdsmi_get_processor_handles(AMDSMI_PROCESSOR_TYPE_AMD_GPU,
                                  &dev_count, handles);
    if (dev_count > 0) ctx->handle = handles[0];
#else
    if (find_amdgpu_hwmon(ctx->hwmon_path, sizeof(ctx->hwmon_path)) != 0) {
        fprintf(stderr, "Advertencia: no se encontró hwmon amdgpu. "
                        "gpu_energy_j será 0.\n");
        ctx->hwmon_path[0] = '\0';
    }
    printf("Backend de potencia: sysfs  (%s)\n", ctx->hwmon_path);
#endif
}

/* PAPI */

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

/* Escritura CSV */

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

/* Tiempo entre eventos HIP (análogo a cudaEventElapsedTime) */
static double hip_elapsed_ms(hipEvent_t start, hipEvent_t stop)
{
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return (double)ms;
}

/*
 * GEMM - rocBLAS dgemm
 * C = alpha*A*B + beta*C  (N×N, double)
 * FLOPs = 2*N^3
 * Bytes  = 3*N^2*8
 */
static Result run_gemm(int N, rocblas_handle blas, PowerCtx* pctx)
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

    HIP_CHECK(hipMalloc(&dA, sz));
    HIP_CHECK(hipMalloc(&dB, sz));
    HIP_CHECK(hipMalloc(&dC, sz));
    HIP_CHECK(hipMemcpy(dA, hA, sz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, sz));

    const double alpha = 1.0, beta = 0.0;

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    /* warm-up */
    for (int w = 0; w < WARMUP_RUNS; w++)
        ROCBLAS_CHECK(rocblas_dgemm(blas,
            rocblas_operation_none, rocblas_operation_none,
            N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
    HIP_CHECK(hipDeviceSynchronize());

    /* medición */
    pctx->active   = 1;
    pctx->energy_j = 0.0;
    pthread_t tid;
    pthread_create(&tid, NULL, power_thread, pctx);

    long long pv[N_EV] = {0};
    double    tot_ms = 0.0;
    PAPI_CHECK(PAPI_reset(papi_evset));
    PAPI_CHECK(PAPI_start(papi_evset));

    for (int i = 0; i < BENCH_RUNS; i++) {
        HIP_CHECK(hipEventRecord(e0, NULL));
        ROCBLAS_CHECK(rocblas_dgemm(blas,
            rocblas_operation_none, rocblas_operation_none,
            N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
        HIP_CHECK(hipEventRecord(e1, NULL));
        HIP_CHECK(hipEventSynchronize(e1));
        tot_ms += hip_elapsed_ms(e0, e1);
    }

    PAPI_CHECK(PAPI_stop(papi_evset, pv));
    pctx->active = 0; pthread_join(tid, NULL);

    double avg = tot_ms / BENCH_RUNS;
    r.gpu_time_ms  = avg;
    r.gpu_energy_j = pctx->energy_j;
    r.gflops       = (2.0*N*N*N / 1e9) / (avg / 1e3);
    r.bw_gbps      = (3.0*N*N*sizeof(double) / 1e9) / (avg / 1e3);
    r.cpu_fp_ops   = pv[0]; r.cpu_l1_miss = pv[1];
    r.cpu_l2_miss  = pv[2]; r.cpu_l3_miss = pv[3];
    r.cpu_cycles   = pv[4]; r.cpu_instr   = pv[5];

    HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    free(hA); free(hB);
    return r;
}

/*
 * SpMV — rocSPARSE CSR (dcsrmv)
 * y = alpha * A * x + beta * y  (CSR, double)
 * FLOPs = 2 * nnz
 */
static Result run_spmv(int N, double sparsity,
                        rocsparse_handle sp, PowerCtx* pctx)
{
    Result r; memset(&r, 0, sizeof(r));
    strcpy(r.kernel, "SpMV"); strcpy(r.prec, "double"); r.N = N;

    long nnz = (long)((double)N * N * sparsity);
    if (nnz < N) nnz = N;

    int    *h_row = (int*)   calloc(N+1, sizeof(int));
    int    *h_col = (int*)   malloc(nnz  * sizeof(int));
    double *h_val = (double*)malloc(nnz  * sizeof(double));
    double *h_x   = (double*)malloc(N    * sizeof(double));

    srand(77);
    long per = nnz/N, rem = nnz%N, ptr = 0;
    h_row[0] = 0;
    for (int i = 0; i < N; i++) {
        long cnt = per + (i < rem ? 1 : 0);
        for (long j = 0; j < cnt; j++) {
            h_col[ptr] = (int)(j * N / cnt) % N;
            h_val[ptr] = (double)rand()/RAND_MAX + 0.1;
            ptr++;
        }
        h_row[i+1] = (int)ptr;
    }
    nnz = ptr;
    r.nnz = nnz;
    for (int i = 0; i < N; i++) h_x[i] = (double)rand()/RAND_MAX;

    int    *d_row, *d_col;
    double *d_val, *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_row, (N+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col, nnz  *sizeof(int)));
    HIP_CHECK(hipMalloc(&d_val, nnz  *sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x,   N    *sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y,   N    *sizeof(double)));
    HIP_CHECK(hipMemcpy(d_row, h_row, (N+1)*sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, h_col, nnz  *sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_val, h_val, nnz  *sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x,   h_x,   N    *sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_y, 0, N*sizeof(double)));

    /* Descriptor de matriz CSR */
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    /* Análisis (buffer de la SpMV) */
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));
    ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(sp,
        rocsparse_operation_none,
        N, N, (rocsparse_int)nnz, descr,
        d_val, d_row, d_col, info));

    const double alpha = 1.0, beta = 0.0;
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));

    /* warm-up */
    for (int w = 0; w < WARMUP_RUNS; w++)
        ROCSPARSE_CHECK(rocsparse_dcsrmv(sp,
            rocsparse_operation_none,
            N, N, (rocsparse_int)nnz, &alpha, descr,
            d_val, d_row, d_col, info, d_x, &beta, d_y));
    HIP_CHECK(hipDeviceSynchronize());

    /* medición */
    pctx->active   = 1;
    pctx->energy_j = 0.0;
    pthread_t tid;
    pthread_create(&tid, NULL, power_thread, pctx);

    long long pv[N_EV] = {0};
    double    tot_ms = 0.0;
    PAPI_CHECK(PAPI_reset(papi_evset)); PAPI_CHECK(PAPI_start(papi_evset));

    for (int i = 0; i < BENCH_RUNS; i++) {
        HIP_CHECK(hipEventRecord(e0, NULL));
        ROCSPARSE_CHECK(rocsparse_dcsrmv(sp,
            rocsparse_operation_none,
            N, N, (rocsparse_int)nnz, &alpha, descr,
            d_val, d_row, d_col, info, d_x, &beta, d_y));
        HIP_CHECK(hipEventRecord(e1, NULL));
        HIP_CHECK(hipEventSynchronize(e1));
        tot_ms += hip_elapsed_ms(e0, e1);
    }

    PAPI_CHECK(PAPI_stop(papi_evset, pv));
    pctx->active = 0; pthread_join(tid, NULL);

    double avg   = tot_ms / BENCH_RUNS;
    double bytes = (double)nnz*(sizeof(double)+sizeof(int))
                 + (double)(N+1)*sizeof(int) + 2.0*N*sizeof(double);
    r.gpu_time_ms  = avg;
    r.gpu_energy_j = pctx->energy_j;
    r.gflops       = (2.0*nnz / 1e9) / (avg / 1e3);
    r.bw_gbps      = (bytes / 1e9) / (avg / 1e3);
    r.cpu_fp_ops   = pv[0]; r.cpu_l1_miss = pv[1];
    r.cpu_l2_miss  = pv[2]; r.cpu_l3_miss = pv[3];
    r.cpu_cycles   = pv[4]; r.cpu_instr   = pv[5];

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(d_row)); HIP_CHECK(hipFree(d_col));
    HIP_CHECK(hipFree(d_val)); HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    free(h_row); free(h_col); free(h_val); free(h_x);
    return r;
}

/*
 * FFT — rocFFT 1D real-to-complex (D2Z equivalente)
 * FLOPs ≈ 5 * N * log2(N)
 * Bytes  = N*8 + (N/2+1)*16
 *
 * rocFFT usa double2 (hipDoubleComplex) para salida compleja.
 * El plan se crea con rocfft_transform_type_real_forward.
 */
static Result run_fft(int N, PowerCtx* pctx)
{
    Result r; memset(&r, 0, sizeof(r));
    strcpy(r.kernel, "FFT"); strcpy(r.prec, "double"); r.N = N;

    double*           h_in  = (double*)malloc(N * sizeof(double));
    hipDoubleComplex* h_out = (hipDoubleComplex*)
                               malloc((N/2+1)*sizeof(hipDoubleComplex));
    for (int i = 0; i < N; i++)
        h_in[i] = sin(2.0*M_PI*i/N) + 0.3*cos(6.0*M_PI*i/N);

    double*           d_in;
    hipDoubleComplex* d_out;
    HIP_CHECK(hipMalloc(&d_in,  N       *sizeof(double)));
    HIP_CHECK(hipMalloc(&d_out, (N/2+1) *sizeof(hipDoubleComplex)));
    HIP_CHECK(hipMemcpy(d_in, h_in, N*sizeof(double), hipMemcpyHostToDevice));

    /* Crear plan rocFFT */
    rocfft_plan plan = NULL;
    size_t       fft_N = (size_t)N;
    ROCFFT_CHECK(rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,
        rocfft_transform_type_real_forward,
        rocfft_precision_double,
        1,            /* dimensiones */
        &fft_N,       /* tamaño */
        1,            /* número de transformadas */
        NULL));       /* sin descripción adicional */

    /* Buffer de trabajo */
    size_t work_sz = 0;
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &work_sz));
    void* work_buf = NULL;
    if (work_sz) HIP_CHECK(hipMalloc(&work_buf, work_sz));

    rocfft_execution_info exec_info = NULL;
    ROCFFT_CHECK(rocfft_execution_info_create(&exec_info));
    if (work_sz)
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(
            exec_info, work_buf, work_sz));

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));

    /* warm-up */
    void* in_ptr  = (void*)d_in;
    void* out_ptr = (void*)d_out;
    for (int w = 0; w < WARMUP_RUNS; w++)
        ROCFFT_CHECK(rocfft_execute(plan, &in_ptr, &out_ptr, exec_info));
    HIP_CHECK(hipDeviceSynchronize());

    /* medición */
    pctx->active   = 1;
    pctx->energy_j = 0.0;
    pthread_t tid;
    pthread_create(&tid, NULL, power_thread, pctx);

    long long pv[N_EV] = {0};
    double    tot_ms = 0.0;
    PAPI_CHECK(PAPI_reset(papi_evset)); PAPI_CHECK(PAPI_start(papi_evset));

    for (int i = 0; i < BENCH_RUNS; i++) {
        HIP_CHECK(hipMemcpy(d_in, h_in, N*sizeof(double),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipEventRecord(e0, NULL));
        ROCFFT_CHECK(rocfft_execute(plan, &in_ptr, &out_ptr, exec_info));
        HIP_CHECK(hipEventRecord(e1, NULL));
        HIP_CHECK(hipEventSynchronize(e1));
        tot_ms += hip_elapsed_ms(e0, e1);
    }

    PAPI_CHECK(PAPI_stop(papi_evset, pv));
    pctx->active = 0; pthread_join(tid, NULL);

    double avg   = tot_ms / BENCH_RUNS;
    double flops = 5.0 * N * log2((double)N);
    double bytes = (double)N*sizeof(double)
                 + (double)(N/2+1)*sizeof(hipDoubleComplex);
    r.gpu_time_ms  = avg;
    r.gpu_energy_j = pctx->energy_j;
    r.gflops       = (flops / 1e9) / (avg / 1e3);
    r.bw_gbps      = (bytes / 1e9) / (avg / 1e3);
    r.cpu_fp_ops   = pv[0]; r.cpu_l1_miss = pv[1];
    r.cpu_l2_miss  = pv[2]; r.cpu_l3_miss = pv[3];
    r.cpu_cycles   = pv[4]; r.cpu_instr   = pv[5];

    ROCFFT_CHECK(rocfft_execution_info_destroy(exec_info));
    ROCFFT_CHECK(rocfft_plan_destroy(plan));
    if (work_buf) HIP_CHECK(hipFree(work_buf));
    HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_out));
    free(h_in); free(h_out);
    return r;
}

int main(int argc, char* argv[])
{
    const char* out_path = "resultados_mi210.csv";
    for (int i = 1; i+1 < argc; i++)
        if (!strcmp(argv[i], "--output")) out_path = argv[i+1];

    /* Identificar GPU HIP */
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU detectada  : %s\n", prop.name);
    printf("Memoria global : %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Backend potencia: %s\n\n", POWER_BACKEND);

    /* Inicializar rocFFT (global) */
    ROCFFT_CHECK(rocfft_setup());

    /* Handles de bibliotecas */
    rocblas_handle   blas;
    rocsparse_handle sph;
    ROCBLAS_CHECK(rocblas_create_handle(&blas));
    ROCSPARSE_CHECK(rocsparse_create_handle(&sph));

    /* PAPI */
    papi_setup();

    /* Contexto de potencia */
    PowerCtx pctx;
    power_ctx_init(&pctx);

    /* CSV */
    FILE* fp = fopen(out_path, "w");
    if (!fp) { perror("fopen"); return 1; }
    csv_header(fp);
    printf("Resultados → %s\n\n", out_path);

    const char* platform = "Dell_R7625_MI210";

    /* GEMM */
    printf("=== GEMM (rocBLAS dgemm) ===\n");
    printf("%-6s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_gpu(ms)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_GEMM; i++) {
        int N = GEMM_N[i];
        if (3ULL*N*N*sizeof(double) > prop.totalGlobalMem * 0.78) {
            printf("N=%-6d SKIP\n", N); continue;
        }
        Result r = run_gemm(N, blas, &pctx);
        printf("%-6d  %-12.4f  %-12.6f  %-10.2f  %-10.2f\n",
               N, r.gpu_time_ms, r.gpu_energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, platform, &r); fflush(fp);
    }

    /* SpMV */
    printf("\n=== SpMV (rocSPARSE dcsrmv) ===\n");
    printf("%-6s  %-8s  %-10s  %-12s  %-12s  %-10s\n",
           "N", "sparsity", "nnz", "T_gpu(ms)", "Energy(J)", "GFLOPS");

    for (int si = 0; si < N_SP; si++) {
        double sp = SPARSITIES[si];
        for (int i = 0; i < N_SPMV; i++) {
            int  N   = SPMV_N[i];
            long nnz = (long)((double)N*N*sp); if (nnz < N) nnz = N;
            size_t need = (size_t)nnz*(sizeof(double)+sizeof(int))
                        + (size_t)(N+1)*sizeof(int) + 2ULL*N*sizeof(double);
            if (need > prop.totalGlobalMem * 0.78) {
                printf("N=%-6d sp=%.3f SKIP\n", N, sp); continue;
            }
            Result r = run_spmv(N, sp, sph, &pctx);
            printf("%-6d  %-8.3f  %-10ld  %-12.4f  %-12.6f  %-10.4f\n",
                   N, sp, r.nnz, r.gpu_time_ms, r.gpu_energy_j, r.gflops);
            csv_row(fp, platform, &r); fflush(fp);
        }
    }

    /* FFT */
    printf("\n=== FFT (rocFFT D2Z) ===\n");
    printf("%-10s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_gpu(ms)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_FFT; i++) {
        int N = FFT_N[i];
        Result r = run_fft(N, &pctx);
        printf("%-10d  %-12.4f  %-12.6f  %-10.4f  %-10.4f\n",
               N, r.gpu_time_ms, r.gpu_energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, platform, &r); fflush(fp);
    }

    /* Limpieza */
    fclose(fp);
    ROCBLAS_CHECK(rocblas_destroy_handle(blas));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(sph));
    ROCFFT_CHECK(rocfft_cleanup());
    PAPI_CHECK(PAPI_cleanup_eventset(papi_evset));
    PAPI_CHECK(PAPI_destroy_eventset(&papi_evset));
    PAPI_shutdown();
#ifndef USE_SYSFS_POWER
    amdsmi_shut_down();
#endif

    printf("\nCompletado. Resultados en: %s\n", out_path);
    return 0;
}
