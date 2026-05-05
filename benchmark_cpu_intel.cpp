#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

#include <mkl.h>
#include <mkl_spblas.h>
#include <mkl_dfti.h>

int g_cpu_cores = 8;
static const int    BENCH_RUNS   = 5;
static const double CPU_FREQ_GHZ = 2.27;
static const double CPU_CACHE_MB = 12.0;

static const int GEMM_N[] = {
    1024,  1152,  1280,  1408,  1536,  1664,  1792,  1920,
    // 2048,  2304,  2560,  2816,  3072,  3328,  3584,  3840,
    // 4096,  4608,  5120,  5632,  6144,  6656,  7168,  7680,  8192,
    // 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384
};
static const int N_GEMM = (int)(sizeof(GEMM_N) / sizeof(GEMM_N[0]));

static const int SPMV_N[] = {
    1000, 2000, 5000, 10000, 12500, 15000,
    // 60000, 75000, 100000, 125000, 150000, 200000, 250000, 300000, 400000, 500000,
    // 1000000, 2000000, 5000000, 10000000
};
static const int N_SPMV = (int)(sizeof(SPMV_N) / sizeof(SPMV_N[0]));

static const double SPARSITIES[] = { 0.01, 0.001, 0.0001, 0.00001 };
static const int    N_SP         = 4;

static const int FFT_N[] = {
    16384,    24000,    32768,    48000,    65536,   100000,
   131072,   200000,   262144,   400000,   524288,   800000,
//   1048576,  1600000,  2097152,  3000000,  3200000,  4194304,
//   6000000,  6400000,  8388608, 12000000, 12800000, 16777216,
//  25000000, 33554432, 50000000, 67108864
};
static const int N_FFT = (int)(sizeof(FFT_N) / sizeof(FFT_N[0]));

static const long NNZ_CPU_MAX = (1L << 30);


using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline double elapsed_s(TimePoint t0, TimePoint t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}

static int cmp_double(const void* a, const void* b)
{
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static double median(double* v, int n)
{
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    int m = n / 2;
    return (n % 2 == 0) ? (v[m-1] + v[m]) / 2.0 : v[m];
}

// =========================================================================
// MÓDULO RAPL MULTI-SOCKET
// =========================================================================
#define RAPL_MAX 4

struct RaplState {
    long long uj[RAPL_MAX];
};

static int       rapl_fds[RAPL_MAX];       
static long long rapl_max_uj[RAPL_MAX];          
static int       rapl_n = 0;

static void rapl_init(void)
{
    // Escaneamos dinámicamente hasta 4 posibles sockets en la tarjeta madre
    for (int i = 0; i < RAPL_MAX; i++) {
        char path_energy[256];
        char path_max[256];
        snprintf(path_energy, sizeof(path_energy), "/sys/class/powercap/intel-rapl:%d/energy_uj", i);
        snprintf(path_max, sizeof(path_max), "/sys/class/powercap/intel-rapl:%d/max_energy_range_uj", i);

        int fd = open(path_energy, O_RDONLY);
        if (fd >= 0) {
            rapl_fds[rapl_n] = fd; // Mantener el archivo abierto (cero I/O latencia luego)
            
            FILE* fmax = fopen(path_max, "r");
            if (fmax) {
                fscanf(fmax, "%lld", &rapl_max_uj[rapl_n]);
                fclose(fmax);
            } else {
                rapl_max_uj[rapl_n] = 262143328850ULL; // Fallback típico de Intel
            }
            
            printf("[RAPL] Detectado Socket %d (Max: %.3f J)\n", i, rapl_max_uj[rapl_n] / 1e6);
            rapl_n++;
        }
    }

    if (rapl_n == 0) {
        fprintf(stderr,
            "[RAPL] No disponible — Energia_CPU_J = 0.\n"
            "       sudo chmod a+r /sys/class/powercap/intel-rapl*/energy_uj\n");
    }
}

static RaplState rapl_read(void)
{
    RaplState state = {0};
    char buf[64];
    for (int i = 0; i < rapl_n; i++) {
        ssize_t nb = pread(rapl_fds[i], buf, sizeof(buf) - 1, 0);
        if (nb > 0) {
            buf[nb] = '\0';
            state.uj[i] = atoll(buf);
        }
    }
    return state;
}

static double rapl_calc_joules(RaplState s0, RaplState s1)
{
    double total_joules = 0.0;
    for (int i = 0; i < rapl_n; i++) {
        long long delta;
        if (s1.uj[i] >= s0.uj[i]) {
            delta = s1.uj[i] - s0.uj[i];
        } else if (rapl_max_uj[i] > 0) {
            delta = rapl_max_uj[i] - s0.uj[i] + s1.uj[i];
        } else {
            delta = 0;
        }
        total_joules += (double)delta / 1e6;
    }
    return total_joules;
}

static void rapl_close(void)
{
    for (int i = 0; i < rapl_n; i++)
        close(rapl_fds[i]);
}

static void csv_header(FILE* f)
{
    fprintf(f,
        "Kernel,N,nnz,Sparsity,OI,Memoria_MB,"
        "Tiempo_CPU_s,Energia_CPU_J,GFLOPS_CPU,BW_Medido_GBps,"
        "CPU_Cores,CPU_Freq\n");
}

static void csv_row(FILE* f,
                    const char* kernel, int N, long nnz, double sparsity,
                    double oi, double mem_mb,
                    double time_s, double energy_j,
                    double gflops, double bw_gbps)
{
    fprintf(f, "%s,%d,%ld,%.6g,%.6g,%.4f,%.9g,%.6g,%.4f,%.4f,%d,%.2f\n",
            kernel, N, nnz, sparsity, oi, mem_mb,
            time_s, energy_j, gflops, bw_gbps,
            g_cpu_cores, CPU_FREQ_GHZ);
}

static void run_gemm(int N, FILE* fp)
{
    size_t sz = (size_t)N * N * sizeof(double);
    double* A = (double*)mkl_malloc(sz, 64);
    double* B = (double*)mkl_malloc(sz, 64);
    double* C = (double*)mkl_calloc((size_t)N * N, sizeof(double), 64);
    if (!A || !B || !C) { fprintf(stderr, "mkl_malloc GEMM N=%d\n", N); exit(1); }

    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    const double flops  = 2.0 * N * N * N;
    const double bytes  = 3.0 * N * N * 8.0;
    const double oi     = flops / bytes;
    const double mem_mb = bytes / 1e6;
    const double alpha  = 1.0, beta = 0.0;

    const int WARMUP = 3;
    for (int w = 0; w < WARMUP; w++)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, alpha, A, N, B, N, beta, C, N);

    TimePoint ta = Clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, alpha, A, N, B, N, beta, C, N);
    TimePoint tb = Clock::now();
    
    double t1rep = elapsed_s(ta, tb);
    int inner_reps = (int)(0.2 / t1rep);
    if (inner_reps < 5)    inner_reps = 5;    
    if (inner_reps > 2000) inner_reps = 2000;

    std::vector<double> times(BENCH_RUNS);
    std::vector<double> energies(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; i++) {
        
        RaplState state0 = rapl_read();
        TimePoint t0  = Clock::now();
        
        for(int ir = 0; ir < inner_reps; ir++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, alpha, A, N, B, N, beta, C, N);
        }
        
        TimePoint t1  = Clock::now();
        RaplState state1 = rapl_read();
        
        times[i]    = elapsed_s(t0, t1) / inner_reps;
        energies[i] = rapl_calc_joules(state0, state1) / inner_reps;
    }

    double t_med  = median(times.data(), BENCH_RUNS);
    double e_med  = median(energies.data(), BENCH_RUNS);
    double gflops = (flops / 1e9) / t_med;
    double bw     = (bytes / 1e9) / t_med;

    printf("%-6d  %-12.6f  %-12.6f  %-10.2f  %-10.4f  %-10.4f\n",
           N, t_med, e_med, gflops, bw, oi);
    csv_row(fp, "GEMM", N, 0, 0.0, oi, mem_mb, t_med, e_med, gflops, bw);
    fflush(fp);

    mkl_free(A); mkl_free(B); mkl_free(C);
}


static void run_spmv(int N, double sparsity, FILE* fp)
{
    long nnz = (long)((long)N * N * sparsity);
    if (nnz < N) nnz = N;

    if (nnz > NNZ_CPU_MAX) {
        printf("  N=%-8d sp=%.6f SKIP (nnz=%ldM > int32 limit)\n", N, sparsity, nnz / 1000000L);
        return;
    }

    size_t need = (size_t)nnz * 12 + (size_t)(N+1) * 4 + (size_t)2 * N * 8;
    if (need > (size_t)(192e9 * 0.78)) {
        printf("  N=%-8d sp=%.6f SKIP (RAM)\n", N, sparsity);
        return;
    }

    int*    h_row = (int*)   malloc((size_t)(N+1) * sizeof(int));
    int*    h_col = (int*)   malloc((size_t)nnz   * sizeof(int));
    double* h_val = (double*)malloc((size_t)nnz   * sizeof(double));
    double* x     = (double*)malloc((size_t)N     * sizeof(double));
    double* y     = (double*)calloc((size_t)N,       sizeof(double));
    if (!h_row || !h_col || !h_val || !x || !y) {
        fprintf(stderr, "malloc SpMV N=%d\n", N); exit(1);
    }

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
    for (int i = 0; i < N; i++) x[i] = (double)rand() / RAND_MAX;

    const double flops  = 2.0 * nnz;
    const double bytes  = (double)nnz * (8.0 + 4.0) + (double)(N+1) * 4.0 + 2.0 * N * 8.0;
    const double oi     = flops / bytes;
    const double mem_mb = bytes / 1e6;

    sparse_matrix_t A_mkl;
    mkl_sparse_d_create_csr(&A_mkl, SPARSE_INDEX_BASE_ZERO, N, N, h_row, h_row + 1, h_col, h_val);
    mkl_sparse_optimize(A_mkl);
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT };

    const int WARMUP = 3;
    for (int w = 0; w < WARMUP; w++) {
        memset(y, 0, (size_t)N * sizeof(double));
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, descr, x, 0.0, y);
    }

    memset(y, 0, (size_t)N * sizeof(double));
    TimePoint ta = Clock::now();
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, descr, x, 0.0, y);
    TimePoint tb = Clock::now();
    
    double t1rep = elapsed_s(ta, tb);
    int inner_reps = (int)(0.2 / t1rep);
    if (inner_reps < 10)   inner_reps = 10;
    if (inner_reps > 2000) inner_reps = 2000;

    std::vector<double> times(BENCH_RUNS), energies(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; i++) {
        memset(y, 0, (size_t)N * sizeof(double));

        RaplState state0 = rapl_read();
        TimePoint t0  = Clock::now();

        for (int ir = 0; ir < inner_reps; ir++)
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, descr, x, 0.0, y);

        TimePoint t1  = Clock::now();
        RaplState state1 = rapl_read();

        times[i]    = elapsed_s(t0, t1) / inner_reps;
        energies[i] = rapl_calc_joules(state0, state1) / inner_reps;
    }

    double t_med  = median(times.data(),    BENCH_RUNS);
    double e_med  = median(energies.data(), BENCH_RUNS);
    double gflops = (flops / 1e9) / t_med;
    double bw     = (bytes / 1e9) / t_med;

    printf("  %-8d %-10.6f %-12ld %-12.6f %-12.6f %-10.4f\n",
           N, sparsity, nnz, t_med, e_med, gflops);
    csv_row(fp, "SpMV", N, nnz, sparsity, oi, mem_mb, t_med, e_med, gflops, bw);
    fflush(fp);

    mkl_sparse_destroy(A_mkl);
    free(h_row); free(h_col); free(h_val); free(x); free(y);
}


static void run_fft(int N, FILE* fp)
{
    double* in  = (double*)mkl_malloc((size_t)N * sizeof(double), 64);
    double* out = (double*)mkl_malloc((size_t)(N/2+1) * 2 * sizeof(double), 64);
    if (!in || !out) { fprintf(stderr, "mkl_malloc FFT N=%d\n", N); exit(1); }

    for (int i = 0; i < N; i++)
        in[i] = sin(2.0 * M_PI * i / N) + 0.3 * cos(6.0 * M_PI * i / N);
    double* in_orig = (double*)malloc((size_t)N * sizeof(double));
    memcpy(in_orig, in, (size_t)N * sizeof(double));

    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N);
    DftiSetValue(desc, DFTI_PLACEMENT,              DFTI_NOT_INPLACE);
    DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS,   1);
    DftiCommitDescriptor(desc);

    const double flops  = 5.0 * (double)N * log2((double)N);
    const double bytes  = (double)N * 8.0 + (double)(N/2+1) * 16.0;
    const double oi     = flops / bytes;
    const double mem_mb = bytes / 1e6;

    const int WARMUP = 3;
    for (int w = 0; w < WARMUP; w++) {
        memcpy(in, in_orig, (size_t)N * sizeof(double));
        DftiComputeForward(desc, in, out);
    }

    memcpy(in, in_orig, (size_t)N * sizeof(double));
    TimePoint ta = Clock::now();
    DftiComputeForward(desc, in, out);
    TimePoint tb = Clock::now();
    
    double t1rep = elapsed_s(ta, tb);
    int inner_reps = (int)(0.2 / t1rep);
    if (inner_reps < 5)    inner_reps = 5;
    if (inner_reps > 5000) inner_reps = 5000;

    std::vector<double> times(BENCH_RUNS), energies(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; i++) {
        memcpy(in, in_orig, (size_t)N * sizeof(double));

        RaplState state0 = rapl_read();
        TimePoint t0  = Clock::now();

        for (int ir = 0; ir < inner_reps; ir++)
            DftiComputeForward(desc, in, out);

        TimePoint t1  = Clock::now();
        RaplState state1 = rapl_read();

        times[i]    = elapsed_s(t0, t1) / inner_reps;
        energies[i] = rapl_calc_joules(state0, state1) / inner_reps;
    }

    double t_med  = median(times.data(),    BENCH_RUNS);
    double e_med  = median(energies.data(), BENCH_RUNS);
    double gflops = (flops / 1e9) / t_med;
    double bw     = (bytes / 1e9) / t_med;

    printf("%-12d %-12.6f %-12.6f %-10.4f %-10.4f\n",
           N, t_med, e_med, gflops, bw);
    csv_row(fp, "FFT", N, 0, 0.0, oi, mem_mb, t_med, e_med, gflops, bw);
    fflush(fp);

    DftiFreeDescriptor(&desc);
    mkl_free(in); mkl_free(out); free(in_orig);
}

int main(int argc, char* argv[])
{
    const char* out_path = "resultados_intel_8_felix.csv";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output")  && i+1 < argc) out_path    = argv[++i];
        if (!strcmp(argv[i], "--threads") && i+1 < argc) g_cpu_cores = atoi(argv[++i]);
    }

    mkl_set_num_threads(g_cpu_cores);

    printf("============================================================\n"
           " Benchmark CPU — Intel Xeon Gold 5315Y (HPE XL290n G10+)\n"
           "============================================================\n");
    printf("Hilos MKL  : %d\n",        g_cpu_cores);
    printf("CPU freq   : %.2f GHz\n",  CPU_FREQ_GHZ);
    printf("L3 cache   : %.0f MB\n",   CPU_CACHE_MB);
    printf("Resultados : %s\n\n",      out_path);

    rapl_init();

    FILE* fp = fopen(out_path, "w");
    if (!fp) { fprintf(stderr, "fopen: %s\n", out_path); return 1; }
    csv_header(fp);

    printf("=== GEMM (MKL cblas_dgemm) ===\n");
    printf("%-6s %-12s %-12s %-10s %-10s %-10s\n",
           "N", "T_cpu(s)", "Energy(J)", "GFLOPS", "BW(GB/s)", "Op.Int.");

    for (int i = 0; i < N_GEMM; i++) {
        int N = GEMM_N[i];
        if (3ULL * N * N * sizeof(double) > 150ULL << 30) {
            printf("N=%-6d SKIP (RAM)\n", N); continue;
        }
        run_gemm(N, fp);
    }

    printf("\n=== SpMV (MKL mkl_sparse_d_mv, CSR) ===\n");
    printf("  %-8s %-10s %-12s %-12s %-12s %-10s\n",
           "N", "sparsity", "nnz", "T_cpu(s)", "Energy(J)", "GFLOPS");

    for (int si = 0; si < N_SP; si++) {
        double sp = SPARSITIES[si];
        for (int i = 0; i < N_SPMV; i++)
            run_spmv(SPMV_N[i], sp, fp);
    }

    printf("\n=== FFT (MKL DFTI real→compleja, double) ===\n");
    printf("%-12s %-12s %-12s %-10s %-10s\n",
           "N", "T_cpu(s)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_FFT; i++)
        run_fft(FFT_N[i], fp);

    fclose(fp);
    rapl_close();   

    printf("\nCompletado. Resultados en: %s\n", out_path);
    return 0;
}
