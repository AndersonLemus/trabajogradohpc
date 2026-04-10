#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>
#include <rocfft/rocfft.h>

#ifdef USE_AMDSMI
  #include <amd_smi/amdsmi.h>
#endif

static constexpr int    GPU_SM_COUNT    = 104;
static constexpr double GPU_FREQ_GHZ   = 1.70;
static constexpr double GPU_MEM_GB     = 64.0;
static constexpr double GPU_MEM_BW_GBPS = 1638.0;

#define HIP_CHECK(x) do {                                                   \
    hipError_t _e = (x);                                                    \
    if (_e != hipSuccess) {                                                  \
        fprintf(stderr, "HIP %s:%d  %s\n",                                  \
                __FILE__, __LINE__, hipGetErrorString(_e));                  \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define ROCBLAS_CHECK(x) do {                                                \
    rocblas_status _s = (x);                                                 \
    if (_s != rocblas_status_success) {                                      \
        fprintf(stderr, "rocBLAS %s:%d  code=%d\n",                         \
                __FILE__, __LINE__, (int)_s);                                \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define ROCSPARSE_CHECK(x) do {                                              \
    rocsparse_status _s = (x);                                               \
    if (_s != rocsparse_status_success) {                                    \
        fprintf(stderr, "rocSPARSE %s:%d  code=%d\n",                       \
                __FILE__, __LINE__, (int)_s);                                \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define ROCFFT_CHECK(x) do {                                                 \
    rocfft_status _s = (x);                                                  \
    if (_s != rocfft_status_success) {                                       \
        fprintf(stderr, "rocFFT %s:%d  code=%d\n",                          \
                __FILE__, __LINE__, (int)_s);                                \
        exit(1);                                                             \
    }                                                                        \
} while(0)

static const int GEMM_N[] = {
    1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
    2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840,
    4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
    9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384
};
static const int N_GEMM = (int)(sizeof(GEMM_N) / sizeof(GEMM_N[0]));

static const int SPMV_N[] = {
    10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000,
    60000, 75000, 100000, 125000, 150000, 200000, 250000, 300000, 400000, 500000
};
static const int N_SPMV = (int)(sizeof(SPMV_N) / sizeof(SPMV_N[0]));

// Mismas densidades que benchmark_cpu_amd
static const double SPARSITIES[] = { 0.01, 0.001, 0.0001, 0.00001 };
static const int    N_SP         = 4;

static const int FFT_N[] = {
    16384, 24000, 32768, 48000, 65536, 100000,
    131072, 200000, 262144, 400000, 524288, 800000,
    1048576, 1600000, 2097152, 3200000, 4194304
};
static const int N_FFT = (int)(sizeof(FFT_N) / sizeof(FFT_N[0]));

static constexpr int WARMUP_RUNS      = 3;
static constexpr int BENCH_RUNS       = 5;
static constexpr int POWER_INTERVAL_MS = 10;

// ─────────────────────────────────────────────────────────────────────────────
// Resultado por medición
// ─────────────────────────────────────────────────────────────────────────────

struct Result {
    std::string kernel;
    int         N            = 0;
    long        nnz          = 0;
    double      sparsity     = 0.0;
    double      op_intensity = 0.0;
    double      gpu_time_s   = 0.0;
    double      gpu_energy_j = 0.0;
    double      gflops       = 0.0;
    double      bw_gbps      = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tiempo GPU — hipEvent_t
// ─────────────────────────────────────────────────────────────────────────────

static inline double hip_elapsed_ms(hipEvent_t t0, hipEvent_t t1)
{
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    return (double)ms;
}

// ─────────────────────────────────────────────────────────────────────────────
// Energía GPU — hilo de muestreo (sysfs hwmon o amdsmi)
// ─────────────────────────────────────────────────────────────────────────────

struct PowerCtx {
    volatile int active  = 0;
    double       energy_j = 0.0;
    char         hwmon_path[256] = {};
#ifdef USE_AMDSMI
    amdsmi_processor_handle_t handle = {};
#endif
};

// Encuentra el hwmon del primer dispositivo amdgpu
static bool find_amdgpu_hwmon(char* path, size_t len)
{
    for (int idx = 0; idx < 20; idx++) {
        char name_path[256];
        snprintf(name_path, sizeof(name_path),
                 "/sys/class/hwmon/hwmon%d/name", idx);
        FILE* f = fopen(name_path, "r");
        if (!f) continue;
        char name[32] = {};
        fgets(name, sizeof(name), f);
        fclose(f);
        if (strncmp(name, "amdgpu", 6) == 0) {
            snprintf(path, len, "/sys/class/hwmon/hwmon%d/power1_average", idx);
            return true;
        }
    }
    return false;
}

static double read_power_sysfs(const char* path)
{
    if (!path || path[0] == '\0') return 0.0;
    FILE* f = fopen(path, "r");
    if (!f) return 0.0;
    long uw = 0;
    fscanf(f, "%ld", &uw);
    fclose(f);
    return (double)uw / 1e6;   // µW → W
}

static void* power_thread(void* arg)
{
    PowerCtx* ctx = (PowerCtx*)arg;
    struct timespec ts = { 0, (long)POWER_INTERVAL_MS * 1000000L };

    while (ctx->active) {
        double watts = 0.0;

#ifdef USE_AMDSMI
        uint64_t power_uw = 0;
        if (amdsmi_get_power_info(ctx->handle, &power_uw) == AMDSMI_STATUS_SUCCESS)
            watts = (double)power_uw / 1e6;
#else
        watts = read_power_sysfs(ctx->hwmon_path);
#endif
        ctx->energy_j += watts * (POWER_INTERVAL_MS / 1000.0);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void power_init(PowerCtx* ctx)
{
#ifdef USE_AMDSMI
    amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    uint32_t count = 0;
    amdsmi_processor_handle_t handles[8];
    amdsmi_get_processor_handles(AMDSMI_PROCESSOR_TYPE_AMD_GPU, &count, handles);
    if (count > 0) {
        ctx->handle = handles[0];
        std::cout << "[POWER] amdsmi activo\n";
    } else {
        std::cerr << "[POWER] amdsmi: no se encontró GPU\n";
    }
#else
    if (find_amdgpu_hwmon(ctx->hwmon_path, sizeof(ctx->hwmon_path))) {
        std::cout << "[POWER] sysfs hwmon: " << ctx->hwmon_path << "\n";
    } else {
        std::cerr << "[POWER] hwmon amdgpu no encontrado — gpu_energy_j = 0\n";
        ctx->hwmon_path[0] = '\0';
    }
#endif
}

// Helpers para iniciar/detener el hilo de potencia
static pthread_t power_start(PowerCtx* ctx)
{
    ctx->active   = 1;
    ctx->energy_j = 0.0;
    pthread_t tid;
    pthread_create(&tid, NULL, power_thread, ctx);
    return tid;
}

static double power_stop(PowerCtx* ctx, pthread_t tid)
{
    ctx->active = 0;
    pthread_join(tid, NULL);
    return ctx->energy_j;
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV
// ─────────────────────────────────────────────────────────────────────────────

static void csv_header(std::ofstream& f)
{
    f << "kernel,N,nnz,sparsity,op_intensity,"
         "gpu_sm_count,gpu_freq_ghz,gpu_mem_gb,gpu_mem_bw_gbps,"
         "gpu_time_s,gpu_energy_j,gflops_GPU,BW_GPU\n";
}

static void csv_row(std::ofstream& f, const Result& r)
{
    f << r.kernel        << ","
      << r.N             << ","
      << r.nnz           << ","
      << r.sparsity      << ","
      << r.op_intensity  << ","
      << GPU_SM_COUNT    << ","
      << GPU_FREQ_GHZ    << ","
      << GPU_MEM_GB      << ","
      << GPU_MEM_BW_GBPS << ","
      << r.gpu_time_s    << ","
      << r.gpu_energy_j  << ","
      << r.gflops        << ","
      << r.bw_gbps       << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// GEMM — rocBLAS dgemm
// FLOPs = 2·N³  |  Bytes = 3·N²·8  |  I = N/12
// ─────────────────────────────────────────────────────────────────────────────

static Result run_gemm(int N, rocblas_handle blas, PowerCtx* pctx)
{
    Result r;
    r.kernel   = "GEMM";
    r.N        = N;
    r.sparsity = 0.0;

    const double flops = 2.0 * N * N * N;
    const double bytes = 3.0 * N * N * 8.0;
    r.op_intensity = flops / bytes;

    size_t sz = (size_t)N * N * sizeof(double);
    std::vector<double> hA(N*N), hB(N*N);
    srand(42);
    for (auto& v : hA) v = (double)rand() / RAND_MAX;
    for (auto& v : hB) v = (double)rand() / RAND_MAX;

    double *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, sz));
    HIP_CHECK(hipMalloc(&dB, sz));
    HIP_CHECK(hipMalloc(&dC, sz));
    HIP_CHECK(hipMemcpy(dA, hA.data(), sz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, sz));

    const double alpha = 1.0, beta = 0.0;

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    for (int w = 0; w < WARMUP_RUNS; w++)
        ROCBLAS_CHECK(rocblas_dgemm(blas,
            rocblas_operation_none, rocblas_operation_none,
            N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
    HIP_CHECK(hipDeviceSynchronize());

    pthread_t tid = power_start(pctx);
    double tot_ms = 0.0;

    for (int i = 0; i < BENCH_RUNS; i++) {
        HIP_CHECK(hipEventRecord(e0, NULL));
        ROCBLAS_CHECK(rocblas_dgemm(blas,
            rocblas_operation_none, rocblas_operation_none,
            N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
        HIP_CHECK(hipEventRecord(e1, NULL));
        HIP_CHECK(hipEventSynchronize(e1));
        tot_ms += hip_elapsed_ms(e0, e1);
    }

    r.gpu_energy_j = power_stop(pctx, tid);
    r.gpu_time_s   = (tot_ms / BENCH_RUNS) / 1e3;
    r.gflops       = (flops / 1e9) / r.gpu_time_s;
    r.bw_gbps      = (bytes / 1e9) / r.gpu_time_s;

    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// SpMV — rocSPARSE dcsrmv (CSR double)
// FLOPs = 2·nnz  |  Bytes = nnz·12 + (N+1)·4 + 2·N·8
// ─────────────────────────────────────────────────────────────────────────────

static Result run_spmv(int N, double sparsity,
                        rocsparse_handle sp, PowerCtx* pctx)
{
    Result r;
    r.kernel   = "SpMV";
    r.N        = N;
    r.sparsity = sparsity;

    // Cast a long antes de multiplicar para evitar overflow int32 (N >= 46341)
    long nnz = std::max((long)((long)N * N * sparsity), (long)N);

    std::vector<int>    h_row(N+1, 0), h_col(nnz);
    std::vector<double> h_val(nnz), h_x(N);

    srand(77);
    long per = nnz / N, rem = nnz % N, ptr = 0;
    h_row[0] = 0;
    for (int i = 0; i < N; i++) {
        long cnt = per + (i < rem ? 1 : 0);
        for (long j = 0; j < cnt; j++) {
            h_col[ptr] = (int)(j * N / cnt) % N;
            h_val[ptr] = (double)rand() / RAND_MAX + 0.1;
            ptr++;
        }
        h_row[i+1] = (int)ptr;
    }
    nnz = ptr;
    r.nnz = nnz;
    for (auto& v : h_x) v = (double)rand() / RAND_MAX;

    const double flops = 2.0 * nnz;
    const double bytes = (double)nnz * (8.0 + 4.0)
                       + (double)(N+1) * 4.0
                       + 2.0 * N * 8.0;
    r.op_intensity = flops / bytes;

    int    *d_row, *d_col;
    double *d_val, *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_row, (N+1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col, nnz   * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_val, nnz   * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x,   N     * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y,   N     * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_row, h_row.data(), (N+1)*sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, h_col.data(), nnz  *sizeof(int),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_val, h_val.data(), nnz  *sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x,   h_x.data(),  N    *sizeof(double),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_y, 0, N * sizeof(double)));

    rocsparse_mat_descr descr;
    rocsparse_mat_info  info;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));
    ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(sp,
        rocsparse_operation_none,
        N, N, (rocsparse_int)nnz, descr,
        d_val, d_row, d_col, info));

    const double alpha = 1.0, beta = 0.0;
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    for (int w = 0; w < WARMUP_RUNS; w++)
        ROCSPARSE_CHECK(rocsparse_dcsrmv(sp,
            rocsparse_operation_none,
            N, N, (rocsparse_int)nnz, &alpha, descr,
            d_val, d_row, d_col, info, d_x, &beta, d_y));
    HIP_CHECK(hipDeviceSynchronize());

    pthread_t tid = power_start(pctx);
    double tot_ms = 0.0;

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

    r.gpu_energy_j = power_stop(pctx, tid);
    r.gpu_time_s   = (tot_ms / BENCH_RUNS) / 1e3;
    r.gflops       = (flops / 1e9) / r.gpu_time_s;
    r.bw_gbps      = (bytes / 1e9) / r.gpu_time_s;

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(d_row)); HIP_CHECK(hipFree(d_col));
    HIP_CHECK(hipFree(d_val)); HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// FFT — rocFFT 1D real→complex double
// FLOPs ≈ 5·N·log₂(N)  |  Bytes = N·8 + (N/2+1)·16
// ─────────────────────────────────────────────────────────────────────────────

static Result run_fft(int N, PowerCtx* pctx)
{
    Result r;
    r.kernel   = "FFT";
    r.N        = N;
    r.sparsity = 0.0;

    const double flops = 5.0 * N * std::log2((double)N);
    const double bytes = (double)N * 8.0 + (double)(N/2+1) * 16.0;
    r.op_intensity = flops / bytes;

    std::vector<double> h_in(N);
    for (int i = 0; i < N; i++)
        h_in[i] = std::sin(2.0*M_PI*i/N) + 0.3*std::cos(6.0*M_PI*i/N);

    double*           d_in;
    double2* d_out;   /* double2 = par {re,im} de doubles; tipo correcto en ROCm */
    HIP_CHECK(hipMalloc(&d_in,  N       * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_out, (N/2+1) * sizeof(double2)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), N*sizeof(double), hipMemcpyHostToDevice));

    rocfft_plan plan = NULL;
    size_t fft_N     = (size_t)N;
    ROCFFT_CHECK(rocfft_plan_create(&plan,
        rocfft_placement_notinplace,
        rocfft_transform_type_real_forward,
        rocfft_precision_double,
        1, &fft_N, 1, NULL));

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
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    void* in_ptr  = (void*)d_in;
    void* out_ptr = (void*)d_out;

    for (int w = 0; w < WARMUP_RUNS; w++)
        ROCFFT_CHECK(rocfft_execute(plan, &in_ptr, &out_ptr, exec_info));
    HIP_CHECK(hipDeviceSynchronize());

    pthread_t tid = power_start(pctx);
    double tot_ms = 0.0;

    for (int i = 0; i < BENCH_RUNS; i++) {
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), N*sizeof(double),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipEventRecord(e0, NULL));
        ROCFFT_CHECK(rocfft_execute(plan, &in_ptr, &out_ptr, exec_info));
        HIP_CHECK(hipEventRecord(e1, NULL));
        HIP_CHECK(hipEventSynchronize(e1));
        tot_ms += hip_elapsed_ms(e0, e1);
    }

    r.gpu_energy_j = power_stop(pctx, tid);
    r.gpu_time_s   = (tot_ms / BENCH_RUNS) / 1e3;
    r.gflops       = (flops / 1e9) / r.gpu_time_s;
    r.bw_gbps      = (bytes / 1e9) / r.gpu_time_s;

    ROCFFT_CHECK(rocfft_execution_info_destroy(exec_info));
    ROCFFT_CHECK(rocfft_plan_destroy(plan));
    if (work_buf) HIP_CHECK(hipFree(work_buf));
    HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_out));
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    std::string out_path = "resultados_mi210.csv";
    for (int i = 1; i+1 < argc; i++)
        if (!strcmp(argv[i], "--output")) out_path = argv[i+1];

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "============================================================\n"
              << "  Benchmark GPU — AMD Instinct MI210  (Dell PowerEdge R7625)\n"
              << "============================================================\n"
              << "GPU detectada  : " << prop.name << "\n"
              << "Memoria global : " << prop.totalGlobalMem / 1e9 << " GB\n"
              << "Backend potencia: "
#ifdef USE_AMDSMI
              << "amdsmi\n"
#else
              << "sysfs hwmon\n"
#endif
              << "Resultados     : " << out_path << "\n\n";

    ROCFFT_CHECK(rocfft_setup());

    rocblas_handle   blas;
    rocsparse_handle sph;
    ROCBLAS_CHECK(rocblas_create_handle(&blas));
    ROCSPARSE_CHECK(rocsparse_create_handle(&sph));

    PowerCtx pctx;
    power_init(&pctx);

    std::ofstream fp(out_path);
    if (!fp) { std::cerr << "No se pudo abrir: " << out_path << "\n"; return 1; }
    csv_header(fp);

    // ── GEMM ─────────────────────────────────────────────────────────────────
    std::cout << "=== GEMM (rocBLAS dgemm) ===\n";
    printf("%-6s  %-12s  %-12s  %-10s  %-10s  %-10s\n",
           "N", "T_gpu(s)", "Energy(J)", "GFLOPS_GPU", "BW_GPU(GB/s)", "Op.Int.");

    for (int i = 0; i < N_GEMM; i++) {
        int N = GEMM_N[i];
        if (3ULL * N * N * sizeof(double) > (size_t)(prop.totalGlobalMem * 0.78)) {
            printf("N=%-6d  SKIP (VRAM)\n", N); continue;
        }
        auto r = run_gemm(N, blas, &pctx);
        printf("%-6d  %-12.4f  %-12.6f  %-10.2f  %-10.4f  %-10.4f\n",
               N, r.gpu_time_s, r.gpu_energy_j, r.gflops, r.bw_gbps, r.op_intensity);
        csv_row(fp, r); fp.flush();
    }

    // ── SpMV ─────────────────────────────────────────────────────────────────
    std::cout << "\n=== SpMV (rocSPARSE dcsrmv) ===\n";
    printf("%-6s  %-8s  %-12s  %-12s  %-12s  %-10s\n",
           "N", "sparsity", "nnz", "T_gpu(s)", "Energy(J)", "GFLOPS_GPU");

    for (int si = 0; si < N_SP; si++) {
        double sp = SPARSITIES[si];
        for (int i = 0; i < N_SPMV; i++) {
            int  N   = SPMV_N[i];
            long nnz = std::max((long)((long)N * N * sp), (long)N);

            /* rocsparse_dcsrmv recibe nnz como rocsparse_int (int32).
             * Límite conservador: 2^30 = 1,073,741,824.
             * Por encima de ese valor rocSPARSE devuelve code=4 (invalid_size)
             * en gfx90a con ROCm 7.2, aunque el nnz quepa en int32. */
            static constexpr long NNZ_ROCSPARSE_MAX = (1L << 30);
            if (nnz > NNZ_ROCSPARSE_MAX) {
                printf("N=%-6d sp=%.5f  SKIP (nnz=%ldM > rocsparse int32 limit)\n",
                       N, sp, nnz / 1000000L); continue;
            }

            size_t need = (size_t)nnz * (sizeof(double) + sizeof(int))
                        + (size_t)(N+1) * sizeof(int)
                        + (size_t)2 * N * sizeof(double);
            if (need > (size_t)(prop.totalGlobalMem * 0.78)) {
                printf("N=%-6d sp=%.5f  SKIP (VRAM)\n", N, sp); continue;
            }
            auto r = run_spmv(N, sp, sph, &pctx);
            printf("%-6d  %-8.5f  %-12ld  %-12.4f  %-12.6f  %-10.4f\n",
                   N, sp, r.nnz, r.gpu_time_s, r.gpu_energy_j, r.gflops);
            csv_row(fp, r); fp.flush();
        }
    }

    // ── FFT ───────────────────────────────────────────────────────────────────
    printf("\n=== FFT (rocFFT D2Z) ===\n");
    printf("%-10s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_gpu(s)", "Energy(J)", "GFLOPS_GPU", "BW_GPU(GB/s)");

    for (int i = 0; i < N_FFT; i++) {
        auto r = run_fft(FFT_N[i], &pctx);
        printf("%-10d  %-12.4f  %-12.6f  %-10.4f  %-10.4f\n",
               r.N, r.gpu_time_s, r.gpu_energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, r); fp.flush();
    }

    // ── Limpieza ──────────────────────────────────────────────────────────────
    ROCBLAS_CHECK(rocblas_destroy_handle(blas));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(sph));
    ROCFFT_CHECK(rocfft_cleanup());
#ifdef USE_AMDSMI
    amdsmi_shut_down();
#endif

    std::cout << "\nCompletado. Resultados en: " << out_path << "\n";
    return 0;
}
