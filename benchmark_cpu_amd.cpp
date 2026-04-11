#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <unistd.h>

#include <cblas.h>
extern "C" {
#include <cs.h>
}
#include <fftw3.h>

#ifdef max_size
#  undef max_size
#endif
#ifdef min_size
#  undef min_size
#endif

static const int    CPU_CORES    = 64;
static const double CPU_FREQ_GHZ = 3.10;
static const double CPU_CACHE_MB = 384.0;

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

struct Result {
    char   kernel[16];
    int    N;
    long   nnz;
    double sparsity;
    double op_intensity;
    double time_ms;
    double energy_j;
    double gflops;
    double bw_gbps;
    double mem_mb;
};

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline double elapsed_ms(TimePoint t0, TimePoint t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int cmp_double(const void* a, const void* b)
{
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double median(double* v, int n)
{
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    int m = n / 2;
    return (n % 2 == 0) ? (v[m-1] + v[m]) / 2.0 : v[m];
}

#define RAPL_MAX 16
static char rapl_paths[RAPL_MAX][256];
static int  rapl_n = 0;

static void rapl_init(void)
{
    const char* base = "/sys/class/powercap/amd_energy";
    DIR* d = opendir(base);
    if (d) {
        struct dirent* e;
        while ((e = readdir(d)) && rapl_n < RAPL_MAX) {
            if (e->d_name[0] == '.') continue;
            snprintf(rapl_paths[rapl_n], 256,
                     "%s/%s/energy_input", base, e->d_name);
            FILE* f = fopen(rapl_paths[rapl_n], "r");
            if (f) { fclose(f); rapl_n++; }
        }
        closedir(d);
    }
    if (rapl_n > 0) {
        printf("[RAPL] sysfs amd_energy: %d dominio(s)\n", rapl_n);
        return;
    }
    const char* ip[] = {
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl:0/energy_uj"
    };
    for (int i = 0; i < 2; i++) {
        FILE* f = fopen(ip[i], "r");
        if (f) {
            fclose(f);
            strncpy(rapl_paths[0], ip[i], 255);
            rapl_n = 1;
            printf("[RAPL] sysfs intel-rapl: %s\n", rapl_paths[0]);
            return;
        }
    }
    fprintf(stderr,
        "[RAPL] No disponible — cpu_energy_j = 0.\n"
        "       sudo chmod a+r "
        "/sys/class/powercap/amd_energy/*/energy_input\n"
        "       sudo modprobe amd_energy\n");
}

static long long rapl_read_uj(void)
{
    long long tot = 0;
    for (int i = 0; i < rapl_n; i++) {
        FILE* f = fopen(rapl_paths[i], "r");
        if (!f) continue;
        long long v = 0;
        fscanf(f, "%lld", &v);
        fclose(f);
        tot += v;
    }
    return tot;
}

static void csv_header(FILE* f)
{
    fprintf(f,
        "Kernel,N,Sparsity,OI,Memoria_MB,"
        "Tiempo_CPU_s,Energia_CPU_J,GFLOPS_CPU,BW_Medido_GBps,"
        "CPU_Cores,CPU_Freq\n");
}

static void csv_row(FILE* f, const Result* r)
{
    fprintf(f,
        "%s,%d,%.6g,%.6g,%.6g,"
        "%.9g,%.6g,%.4f,%.4f,"
        "%d,%.2f\n",
        r->kernel, r->N, r->sparsity, r->op_intensity, r->mem_mb,
        r->time_ms / 1e3, r->energy_j, r->gflops, r->bw_gbps,
        CPU_CORES, CPU_FREQ_GHZ);
}

/* ─────────────────────────────────────────────────────────────────────────
 * GEMM — cblas_dgemm
 * Bug corregido: el documento tenía 4 problemas en run_gemm:
 *   1. Llave '{' extra que dejaba run_gemm sin cerrar → error de compilación
 *   2. 'auto t1 = Clock::now()' capturada pero times[i] usaba Clock::now()
 *      de nuevo → t1 no se usaba nunca (el tiempo medido incluía rapl_read)
 *   3. 'energies' malloc'd pero free() faltaba al final
 *   4. Medición RAPL dentro del loop (overhead de fopen en cada iteración)
 *      → movida fuera: delta total / runs da la energía media por ejecución
 * ───────────────────────────────────────────────────────────────────────── */
static Result run_gemm(int N, int nthreads)
{
    Result r;
    memset(&r, 0, sizeof(r));
    strncpy(r.kernel, "GEMM", 15);
    r.N = N;

    openblas_set_num_threads(nthreads);

    size_t sz = (size_t)N * N * sizeof(double);
    double* A = (double*)malloc(sz);
    double* B = (double*)malloc(sz);
    double* C = (double*)calloc((size_t)N * N, sizeof(double));
    if (!A || !B || !C) { fprintf(stderr, "malloc GEMM N=%d\n", N); exit(1); }

    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    const double flops = 2.0 * N * N * N;
    const double bytes = 3.0 * N * N * 8.0;
    r.op_intensity = flops / bytes;
    r.mem_mb       = bytes / 1e6;

    int runs = (N < 2000) ? 20 : (N < 4000) ? 10 : 5;
    double* times    = (double*)malloc((size_t)runs * sizeof(double));
    double* energies = (double*)malloc((size_t)runs * sizeof(double));
    const double alpha = 1.0, beta = 0.0;

    /* RAPL por ejecucion individual — evita desbordamiento del contador
     * (~262 s en intel-rapl 32 bits) cuando N es grande y runs*T > 262 s */
    for (int i = 0; i < runs; i++) {
        long long uj0 = rapl_read_uj();
        TimePoint t0  = Clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, alpha, A, N, B, N, beta, C, N);
        TimePoint t1  = Clock::now();
        long long uj1 = rapl_read_uj();
        times[i]    = elapsed_ms(t0, t1);
        energies[i] = (uj1 >= uj0) ? (double)(uj1 - uj0) / 1e6 : 0.0;
    }

    r.time_ms  = median(times, runs);
    r.energy_j = median(energies, runs);
    r.gflops   = (flops / 1e9) / (r.time_ms / 1e3);
    r.bw_gbps  = (bytes / 1e9) / (r.time_ms / 1e3);

    free(A); free(B); free(C); free(times); free(energies);
    return r;
}

static Result run_spmv(int N, double sparsity)
{
    Result r;
    memset(&r, 0, sizeof(r));
    strncpy(r.kernel, "SpMV", 15);
    r.N        = N;
    r.sparsity = sparsity;

    long nnz = (long)((long)N * N * sparsity);
    if (nnz < N) nnz = N;

    cs* T = cs_spalloc(N, N, nnz, 1, 1);
    if (!T) { fprintf(stderr, "cs_spalloc\n"); exit(1); }

    srand(77);
    long per = nnz / N, rem = nnz % N, ptr = 0;
    for (int col = 0; col < N && ptr < nnz; col++) {
        long cnt = per + (col < rem ? 1 : 0);
        for (long j = 0; j < cnt && ptr < nnz; j++) {
            cs_entry(T, (int)(j * N / cnt) % N, col,
                     (double)rand() / RAND_MAX + 0.1);
            ptr++;
        }
    }
    cs* A = cs_compress(T);
    cs_spfree(T);
    if (!A) { fprintf(stderr, "cs_compress\n"); exit(1); }
    r.nnz = (A->nz < 0) ? A->p[N] : A->nz;

    double* x = (double*)malloc((size_t)N * sizeof(double));
    double* y = (double*)calloc((size_t)N,  sizeof(double));
    if (!x || !y) { fprintf(stderr, "malloc SpMV\n"); exit(1); }
    for (int i = 0; i < N; i++) x[i] = (double)rand() / RAND_MAX;

    const double flops = 2.0 * r.nnz;
    const double bytes = r.nnz * (8.0 + 4.0) + (N + 1) * 4.0 + 2.0 * N * 8.0;
    r.op_intensity = flops / bytes;
    r.mem_mb       = bytes / 1e6;

    long ne = r.nnz > 0 ? r.nnz : (long)N;
    int runs = (ne <     10000) ? 5000 :
               (ne <    100000) ? 500  :
               (ne <   1000000) ? 50   :
               (ne <  10000000) ? 10   : 5;

    double* times = (double*)malloc((size_t)runs * sizeof(double));

    long long uj0 = rapl_read_uj();
    for (int i = 0; i < runs; i++) {
        memset(y, 0, (size_t)N * sizeof(double));
        TimePoint t0 = Clock::now();
        cs_gaxpy(A, x, y);
        TimePoint t1 = Clock::now();
        times[i] = elapsed_ms(t0, t1);
    }
    long long uj1 = rapl_read_uj();

    r.time_ms  = median(times, runs);
    r.energy_j = (uj1 >= uj0) ? (double)(uj1 - uj0) / 1e6 / runs : 0.0;
    r.gflops   = (flops / 1e9) / (r.time_ms / 1e3);
    r.bw_gbps  = (bytes / 1e9) / (r.time_ms / 1e3);

    cs_spfree(A);
    free(x); free(y); free(times);
    return r;
}

static Result run_fft(int N, int nthreads)
{
    Result r;
    memset(&r, 0, sizeof(r));
    strncpy(r.kernel, "FFT", 15);
    r.N = N;

    fftw_init_threads();
    fftw_plan_with_nthreads(nthreads);

    double*       in  = (double*)      fftw_malloc((size_t)N * sizeof(double));
    fftw_complex* out = (fftw_complex*)fftw_malloc((size_t)(N/2+1) * sizeof(fftw_complex));
    if (!in || !out) { fprintf(stderr, "fftw_malloc N=%d\n", N); exit(1); }

    for (int i = 0; i < N; i++)
        in[i] = sin(2.0 * M_PI * i / N) + 0.3 * cos(6.0 * M_PI * i / N);

    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_MEASURE);
    if (!plan) { fprintf(stderr, "fftw_plan N=%d\n", N); exit(1); }

    const double flops = 5.0 * N * log2((double)N);
    const double bytes = N * 8.0 + (N/2 + 1) * 16.0;
    r.op_intensity = flops / bytes;
    r.mem_mb       = bytes / 1e6;

    int runs = (N <=    1024) ? 5000 :
               (N <=    8192) ? 1000 :
               (N <=   65536) ? 200  :
               (N <=  524288) ? 20   : 5;

    double* times = (double*)malloc((size_t)runs * sizeof(double));

    long long uj0 = rapl_read_uj();
    for (int i = 0; i < runs; i++) {
        for (int j = 0; j < N; j++)
            in[j] = sin(2.0 * M_PI * j / N) + 0.3 * cos(6.0 * M_PI * j / N);
        TimePoint t0 = Clock::now();
        fftw_execute(plan);
        TimePoint t1 = Clock::now();
        times[i] = elapsed_ms(t0, t1);
    }
    long long uj1 = rapl_read_uj();

    r.time_ms  = median(times, runs);
    r.energy_j = (uj1 >= uj0) ? (double)(uj1 - uj0) / 1e6 / runs : 0.0;
    r.gflops   = (flops / 1e9) / (r.time_ms / 1e3);
    r.bw_gbps  = (bytes / 1e9) / (r.time_ms / 1e3);

    fftw_destroy_plan(plan);
    fftw_free(in); fftw_free(out);
    fftw_cleanup_threads();
    free(times);
    return r;
}

int main(int argc, char* argv[])
{
    const char* out_path = "resultados_epyc.csv";
    int nthreads = 64;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output")  && i+1 < argc) out_path = argv[++i];
        if (!strcmp(argv[i], "--threads") && i+1 < argc) nthreads = atoi(argv[++i]);
    }

    printf("============================================================\n"
           "  Benchmark CPU — AMD EPYC 9554  (Dell PowerEdge R7625)\n"
           "============================================================\n");
    printf("Hilos OpenBLAS/FFTW : %d\n", nthreads);
    printf("CPU cores / freq    : %d cores / %.2f GHz\n", CPU_CORES, CPU_FREQ_GHZ);
    printf("L3 cache            : %.0f MB\n", CPU_CACHE_MB);
    printf("Resultados          : %s\n\n", out_path);

    rapl_init();

    FILE* fp = fopen(out_path, "w");
    if (!fp) { fprintf(stderr, "fopen: %s\n", out_path); return 1; }
    csv_header(fp);

    printf("=== GEMM (OpenBLAS cblas_dgemm) ===\n");
    printf("%-6s  %-12s  %-12s  %-10s  %-10s  %-10s\n",
           "N", "T_cpu(s)", "Energy(J)", "GFLOPS", "BW(GB/s)", "Op.Int.");

    for (int i = 0; i < N_GEMM; i++) {
        int N = GEMM_N[i];
        if (3ULL * N * N * sizeof(double) > 350ULL << 30) {
            printf("N=%-6d  SKIP (RAM)\n", N); continue;
        }
        Result r = run_gemm(N, nthreads);
        printf("%-6d  %-12.6f  %-12.6f  %-10.2f  %-10.4f  %-10.4f\n",
               N, r.time_ms / 1e3, r.energy_j, r.gflops, r.bw_gbps, r.op_intensity);
        csv_row(fp, &r); fflush(fp);
    }

    printf("\n=== SpMV (CSparse cs_gaxpy, CSC) ===\n");
    printf("%-6s  %-8s  %-12s  %-12s  %-12s  %-10s\n",
           "N", "sparsity", "nnz", "T_cpu(s)", "Energy(J)", "GFLOPS");

    for (int si = 0; si < N_SP; si++) {
        double sp = SPARSITIES[si];
        for (int i = 0; i < N_SPMV; i++) {
            int  N   = SPMV_N[i];
            long nnz = (long)((long)N * N * sp);
            if (nnz < N) nnz = N;
            size_t need = (size_t)nnz * 12 + (size_t)(N+1) * 4 + (size_t)2 * N * 8;
            if (need > 350ULL << 30) {
                printf("N=%-6d sp=%.5f  SKIP (RAM)\n", N, sp); continue;
            }
            Result r = run_spmv(N, sp);
            printf("%-6d  %-8.5f  %-12ld  %-12.6f  %-12.6f  %-10.4f\n",
                   N, sp, r.nnz, r.time_ms / 1e3, r.energy_j, r.gflops);
            csv_row(fp, &r); fflush(fp);
        }
    }

    printf("\n=== FFT (FFTW3 r2c double, %d hilos) ===\n", nthreads);
    printf("%-10s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_cpu(s)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_FFT; i++) {
        Result r = run_fft(FFT_N[i], nthreads);
        printf("%-10d  %-12.6f  %-12.6f  %-10.4f  %-10.4f\n",
               r.N, r.time_ms / 1e3, r.energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, &r); fflush(fp);
    }

    fclose(fp);
    printf("\nCompletado. Resultados en: %s\n", out_path);
    return 0;
}
