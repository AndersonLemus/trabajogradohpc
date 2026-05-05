#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cufft.h>
#include <nvml.h>

#define CUDA_CHECK(x) do {                                                  \
    cudaError_t _e = (x);                                                   \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA %s:%d  %s\n",                                 \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define CUBLAS_CHECK(x) do {                                                 \
    cublasStatus_t _s = (x);                                                 \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "cuBLAS %s:%d  code=%d\n",                          \
                __FILE__, __LINE__, (int)_s);                                \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define CUSPARSE_CHECK(x) do {                                               \
    cusparseStatus_t _s = (x);                                               \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                     \
        fprintf(stderr, "cuSPARSE %s:%d  code=%d\n",                        \
                __FILE__, __LINE__, (int)_s);                                \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define CUFFT_CHECK(x) do {                                                  \
    cufftResult _r = (x);                                                    \
    if (_r != CUFFT_SUCCESS) {                                               \
        fprintf(stderr, "cuFFT %s:%d  code=%d\n",                           \
                __FILE__, __LINE__, (int)_r);                                \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define NVML_CHECK(x) do {                                                   \
    nvmlReturn_t _r = (x);                                                   \
    if (_r != NVML_SUCCESS) {                                                \
        fprintf(stderr, "NVML %s:%d  %s\n",                                 \
                __FILE__, __LINE__, nvmlErrorString(_r));                    \
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
    1000, 2000, 5000, 10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000,
    60000, 75000, 100000, 125000, 150000, 200000, 250000, 300000, 400000, 500000,
    1000000, 2000000, 5000000, 10000000
};
static const int N_SPMV = (int)(sizeof(SPMV_N) / sizeof(SPMV_N[0]));

static const double SPARSITIES[] = { 0.01, 0.001, 0.0001, 0.00001 };
static const int    N_SP         = 4;

static const int FFT_N[] = {
    16384, 24000, 32768, 48000, 65536, 100000,
    131072, 200000, 262144, 400000, 524288, 800000,
    1048576, 1600000, 2097152, 3000000, 3200000, 4194304,
    6000000, 6400000, 8388608, 12000000, 12800000, 16777216,
    25000000, 33554432, 50000000, 67108864
};
static const int N_FFT = (int)(sizeof(FFT_N) / sizeof(FFT_N[0]));

static constexpr int WARMUP_RUNS      = 3;
static constexpr int BENCH_RUNS       = 5;
static constexpr int NVML_INTERVAL_MS = 10;

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

struct NvmlCtx {
    nvmlDevice_t dev;
    volatile int active   = 0;
    double       energy_j = 0.0;
};

static void* nvml_thread(void* arg)
{
    NvmlCtx* ctx = (NvmlCtx*)arg;
    struct timespec ts = { 0, (long)NVML_INTERVAL_MS * 1000000L };
    while (ctx->active) {
        unsigned int mw = 0;
        nvmlDeviceGetPowerUsage(ctx->dev, &mw);
        ctx->energy_j += (mw / 1000.0) * (NVML_INTERVAL_MS / 1000.0);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static pthread_t nvml_start(NvmlCtx* ctx, nvmlDevice_t dev)
{
    ctx->dev      = dev;
    ctx->active   = 1;
    ctx->energy_j = 0.0;
    pthread_t tid;
    pthread_create(&tid, NULL, nvml_thread, ctx);
    return tid;
}

static double nvml_stop(NvmlCtx* ctx, pthread_t tid)
{
    ctx->active = 0;
    pthread_join(tid, NULL);
    return ctx->energy_j;
}

double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) return (v[n/2 - 1] + v[n/2]) / 2.0;
    return v[n/2];
}

static int    g_sm_count    = 0;
static double g_freq_ghz    = 0.0;
static double g_mem_gb      = 0.0;
static double g_mem_bw_gbps = 0.0;

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
      << g_sm_count      << ","
      << g_freq_ghz      << ","
      << g_mem_gb        << ","
      << g_mem_bw_gbps   << "," 
      << r.gpu_time_s    << ","
      << r.gpu_energy_j  << ","
      << r.gflops        << ","
      << r.bw_gbps       << "\n";
}

static Result run_gemm(int N, cublasHandle_t blas, nvmlDevice_t nvml)
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
    CUDA_CHECK(cudaMalloc(&dA, sz));
    CUDA_CHECK(cudaMalloc(&dB, sz));
    CUDA_CHECK(cudaMalloc(&dC, sz));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, sz));

    const double alpha = 1.0, beta = 0.0;

    std::vector<double> times(BENCH_RUNS);
    std::vector<double> energies(BENCH_RUNS);

    for (int w = 0; w < WARMUP_RUNS; w++){
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ta = std::chrono::steady_clock::now();
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
    CUDA_CHECK(cudaDeviceSynchronize()); // Crucial en GPU
    auto tb = std::chrono::steady_clock::now();
    
    double t1rep = std::chrono::duration<double>(tb - ta).count();
    int inner_reps = (int)(0.2 / t1rep);
    if (inner_reps < 5)    inner_reps = 5;
    if (inner_reps > 5000) inner_reps = 5000;

    for (int i = 0; i < BENCH_RUNS; i++) {
        
        NvmlCtx ctx;
        pthread_t tid = nvml_start(&ctx, nvml);

        auto t_start = std::chrono::steady_clock::now();

        for (int ir = 0; ir < inner_reps; ir++) {
            CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto t_end = std::chrono::steady_clock::now();

        energies[i] = nvml_stop(&ctx, tid) / inner_reps;
        times[i] = std::chrono::duration<double>(t_end - t_start).count() / inner_reps;
    }

    r.gpu_time_s   = median(times);
    r.gpu_energy_j = median(energies);
    r.gflops       = (flops / 1e9) / r.gpu_time_s;
    r.bw_gbps      = (bytes / 1e9) / r.gpu_time_s;

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return r;
}   

static Result run_spmv(int N, double sparsity, cusparseHandle_t sp, nvmlDevice_t nvml)
{
    Result r; r.N = N; r.sparsity = sparsity; r.kernel = "SpMV";

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
    CUDA_CHECK(cudaMalloc(&d_row, (N+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col, nnz   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, nnz   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,   N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,   N     * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row, h_row.data(), (N+1)*sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, h_col.data(), nnz  *sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), nnz  *sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,   h_x.data(),   N    *sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(double)));

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

    for (int w = 0; w < WARMUP_RUNS; w++)
        CUSPARSE_CHECK(cusparseSpMV(sp,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ta = std::chrono::steady_clock::now();
    CUSPARSE_CHECK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
    CUDA_CHECK(cudaDeviceSynchronize()); // Crucial en GPU
    auto tb = std::chrono::steady_clock::now();
    
    double t1rep = std::chrono::duration<double>(tb - ta).count();
    int inner_reps = (int)(0.2 / t1rep);
    if (inner_reps < 10)   inner_reps = 10;
    if (inner_reps > 2000) inner_reps = 2000;

    std::vector<double> times(BENCH_RUNS);
    std::vector<double> energies(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; i++) {
        
        NvmlCtx ctx;
        pthread_t tid = nvml_start(&ctx, nvml);
        auto t_start = std::chrono::steady_clock::now();

        for (int ir = 0; ir < inner_reps; ir++) {
            CUSPARSE_CHECK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto t_end = std::chrono::steady_clock::now();
        energies[i] = nvml_stop(&ctx, tid) / inner_reps;
        times[i] = std::chrono::duration<double>(t_end - t_start).count() / inner_reps;
    }

    r.gpu_time_s   = median(times);
    r.gpu_energy_j = median(energies);
    r.gflops       = (flops / 1e9) / r.gpu_time_s;
    r.bw_gbps      = (bytes / 1e9) / r.gpu_time_s;


    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    if (buf) CUDA_CHECK(cudaFree(buf));
    CUDA_CHECK(cudaFree(d_row)); CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_val)); CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return r;
}

static Result run_fft(int N, nvmlDevice_t nvml)
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

    double*             d_in;
    cufftDoubleComplex* d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, (N/2+1) * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N*sizeof(double), cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_D2Z, 1));

    for (int w = 0; w < WARMUP_RUNS; w++)
        CUFFT_CHECK(cufftExecD2Z(plan, d_in, d_out));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ta = std::chrono::steady_clock::now();
    CUFFT_CHECK(cufftExecD2Z(plan, d_in, d_out));
    CUDA_CHECK(cudaDeviceSynchronize()); // Crucial en GPU
    auto tb = std::chrono::steady_clock::now();
    
    double t1rep = std::chrono::duration<double>(tb - ta).count();
    int inner_reps = (int)(0.2 / t1rep);
    if (inner_reps < 5)    inner_reps = 5;
    if (inner_reps > 5000) inner_reps = 5000;

    std::vector<double> times(BENCH_RUNS);
    std::vector<double> energies(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; i++) {
        
        NvmlCtx ctx;
        pthread_t tid = nvml_start(&ctx, nvml);

        auto t_start = std::chrono::steady_clock::now();

        // Ráfaga asíncrona a la GPU
        for (int ir = 0; ir < inner_reps; ir++) {
            CUFFT_CHECK(cufftExecD2Z(plan, d_in, d_out));
        }

        // Bloquear CPU hasta que la GPU termine
        CUDA_CHECK(cudaDeviceSynchronize()); 
        auto t_end = std::chrono::steady_clock::now();
        
        energies[i] = nvml_stop(&ctx, tid) / inner_reps;
        times[i] = std::chrono::duration<double>(t_end - t_start).count() / inner_reps;
    }

    r.gpu_time_s   = median(times);
    r.gpu_energy_j = median(energies);
    r.gflops       = (flops / 1e9) / r.gpu_time_s;
    r.bw_gbps      = (bytes / 1e9) / r.gpu_time_s;

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    return r;
}

int main(int argc, char* argv[])
{
    const char* out_path = "resultados_nvidia.csv";
    for (int i = 1; i+1 < argc; i++)
        if (!strcmp(argv[i], "--output")) out_path = argv[i+1];

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    g_sm_count    = prop.multiProcessorCount;
    g_freq_ghz    = prop.clockRate / 1e6;
    g_mem_gb      = prop.totalGlobalMem / 1e9;
    g_mem_bw_gbps = 2.0 * (prop.memoryClockRate * 1e3)
                       * (prop.memoryBusWidth / 8.0) / 1e9;

    char platform[64];
    if      (strstr(prop.name, "A100"))  snprintf(platform, 64, "HPE_XL290n_A100");
    else if (strstr(prop.name, "TITAN") ||
             strstr(prop.name, "Titan")) snprintf(platform, 64, "HPE_DL580_TitanX");
    else                                 snprintf(platform, 64, "NVIDIA_%s", prop.name);

    std::cout << "============================================================\n"
              << "  Benchmark GPU — NVIDIA  (" << platform << ")\n"
              << "============================================================\n"
              << "GPU           : " << prop.name << "  (sm_"
                                   << prop.major << prop.minor << ")\n"
              << "SMs           : " << g_sm_count    << "\n"
              << "Frecuencia    : " << g_freq_ghz    << " GHz\n"
              << "Memoria       : " << g_mem_gb      << " GB\n"
              << "BW pico       : " << g_mem_bw_gbps << " GB/s\n"
              << "Resultados    : " << out_path << "\n\n";

    NVML_CHECK(nvmlInit());
    nvmlDevice_t nvml;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &nvml));

    cublasHandle_t   blas;
    cusparseHandle_t sph;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUSPARSE_CHECK(cusparseCreate(&sph));

    std::ofstream fp(out_path);
    if (!fp) { std::cerr << "No se pudo abrir: " << out_path << "\n"; return 1; }
    csv_header(fp);

    std::cout << "=== GEMM (cuBLAS dgemm) ===\n";
    printf("%-6s  %-12s  %-12s  %-12s  %-12s  %-10s\n",
           "N", "T_gpu(s)", "Energy(J)", "GFLOPS_GPU", "BW_GPU(GB/s)", "Op.Int.");

    for (int i = 0; i < N_GEMM; i++) {
        int N = GEMM_N[i];
        if (3ULL * N * N * sizeof(double) > (size_t)(prop.totalGlobalMem * 0.78)) {
            printf("N=%-6d  SKIP (VRAM)\n", N); continue;
        }
        auto r = run_gemm(N, blas, nvml);
        printf("%-6d  %-12.6f  %-12.6f  %-12.2f  %-12.4f  %-10.4f\n",
               N, r.gpu_time_s, r.gpu_energy_j, r.gflops, r.bw_gbps, r.op_intensity);
        csv_row(fp, r); fp.flush();
    }

    std::cout << "\n=== SpMV (cuSPARSE CSR) ===\n";
    printf("%-6s  %-8s  %-12s  %-12s  %-12s  %-12s\n",
           "N", "sparsity", "nnz", "T_gpu(s)", "Energy(J)", "GFLOPS_GPU");

    for (int si = 0; si < N_SP; si++) {
        double sp = SPARSITIES[si];
        for (int i = 0; i < N_SPMV; i++) {
            int  N   = SPMV_N[i];
            long nnz = std::max((long)((long)N * N * sp), (long)N);
            size_t need = (size_t)nnz * (sizeof(double) + sizeof(int))
                        + (size_t)(N+1) * sizeof(int)
                        + (size_t)2 * N * sizeof(double);
            if (need > (size_t)(prop.totalGlobalMem * 0.78)) {
                printf("N=%-6d sp=%.5f  SKIP (VRAM)\n", N, sp); continue;
            }
            auto r = run_spmv(N, sp, sph, nvml);
            printf("%-6d  %-8.5f  %-12ld  %-12.6f  %-12.6f  %-12.4f\n",
                   N, sp, r.nnz, r.gpu_time_s, r.gpu_energy_j, r.gflops);
            csv_row(fp, r); fp.flush();
        }
    }

    printf("\n=== FFT (cuFFT D2Z) ===\n");
    printf("%-10s  %-12s  %-12s  %-12s  %-12s\n",
           "N", "T_gpu(s)", "Energy(J)", "GFLOPS_GPU", "BW_GPU(GB/s)");

    for (int i = 0; i < N_FFT; i++) {
        auto r = run_fft(FFT_N[i], nvml);
        printf("%-10d  %-12.6f  %-12.6f  %-12.4f  %-12.4f\n",
               r.N, r.gpu_time_s, r.gpu_energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, r); fp.flush();
    }

    CUBLAS_CHECK(cublasDestroy(blas));
    CUSPARSE_CHECK(cusparseDestroy(sph));
    NVML_CHECK(nvmlShutdown());

    std::cout << "\nCompletado. Resultados en: " << out_path << "\n";
    return 0;
}
