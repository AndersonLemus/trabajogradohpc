#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cblas.h>
#include <fftw3.h>
#include <Eigen/Sparse>
#include <Eigen/Core>

using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Tripleta = Eigen::Triplet<double>;

static constexpr int BENCH_RUNS = 5;

int g_cpu_cores = 16; 
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
    10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000,
    60000, 75000, 100000, 125000, 150000, 200000, 250000,
    300000, 400000, 500000
};
static const int N_SPMV = (int)(sizeof(SPMV_N) / sizeof(SPMV_N[0]));

static const double SPARSITIES[] = { 0.01, 0.001, 0.0001, 0.00001 };
static const int    N_SP         = 4;

static const int FFT_N[] = {
    16384, 24000, 32768, 48000, 65536, 100000,
    131072, 200000, 262144, 400000, 524288, 800000,
    1048576, 1600000, 2097152, 3200000, 4194304
};
static const int N_FFT = (int)(sizeof(FFT_N) / sizeof(FFT_N[0]));

struct Result {
    char   kernel[16];
    int    N;
    long   nnz;
    double sparsity;
    double op_intensity;
    double time_s;       
    double energy_j;
    double gflops;
    double bw_gbps;
    double mem_mb;
};

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline double elapsed_s(TimePoint t0, TimePoint t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

static double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? (v[m-1] + v[m]) / 2.0 : v[m];
}


#define RAPL_MAX 16

struct RaplState {
    long long uj[RAPL_MAX];
};

static int       rapl_fds[RAPL_MAX];
static long long rapl_max_uj[RAPL_MAX];
static int       rapl_n = 0;

static void rapl_init(void) {
    const char* amd_base = "/sys/class/powercap/amd_energy";
    DIR* d = opendir(amd_base);
    if (d) {
        struct dirent* e;
        while ((e = readdir(d)) && rapl_n < RAPL_MAX) {
            if (e->d_name[0] == '.') continue;
            char path[256];
            snprintf(path, sizeof(path), "%s/%s/energy_input", amd_base, e->d_name);
            int fd = open(path, O_RDONLY);
            if (fd >= 0) {
                rapl_fds[rapl_n] = fd;
                rapl_max_uj[rapl_n] = 0; 
                rapl_n++;
            }
        }
        closedir(d);
    }

    if (rapl_n > 0) {
        printf("[RAPL] amd_energy: %d dominio(s) abiertos (AMD EPYC).\n", rapl_n);
        return;
    }

    const char* intel_paths[] = {
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl:0/energy_uj"
    };
    for (int i = 0; i < 2; i++) {
        int fd = open(intel_paths[i], O_RDONLY);
        if (fd >= 0) {
            rapl_fds[rapl_n] = fd;
            rapl_max_uj[rapl_n] = 262143328850ULL;
            printf("[RAPL] intel-rapl: %s abierto.\n", intel_paths[i]);
            rapl_n++;
            return;
        }
    }

    fprintf(stderr,
        "[RAPL] No disponible — energy_j = 0 en todos los kernels.\n");
}

static RaplState rapl_read(void) {
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

static double rapl_calc_joules(RaplState s0, RaplState s1) {
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

static void rapl_close(void) {
    for (int i = 0; i < rapl_n; i++) close(rapl_fds[i]);
}

static void csv_header(FILE* f) {
    fprintf(f,
        "Kernel,N,Sparsity,OI,Memoria_MB,"
        "Tiempo_CPU_s,Energia_CPU_J,GFLOPS_CPU,BW_Medido_GBps,"
        "CPU_Cores,CPU_Freq\n");
}

static void csv_row(FILE* f, const Result* r) {
    fprintf(f,
        "%s,%d,%.6g,%.6g,%.6g,"
        "%.9g,%.6g,%.4f,%.4f,"
        "%d,%.2f\n",
        r->kernel, r->N, r->sparsity, r->op_intensity, r->mem_mb,
        r->time_s, r->energy_j, r->gflops, r->bw_gbps,
        g_cpu_cores, CPU_FREQ_GHZ); 
}

static Result run_gemm(int N) {
    Result r;
    memset(&r, 0, sizeof(r));
    strncpy(r.kernel, "GEMM", 15);
    r.N = N;

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
    const double alpha = 1.0, beta = 0.0;

    for (int i = 0; i < 3; i++)
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

    std::vector<double> times(BENCH_RUNS), energies(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; i++) {
        RaplState state0 = rapl_read();
        TimePoint t0  = Clock::now();

        for (int ir = 0; ir < inner_reps; ir++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, alpha, A, N, B, N, beta, C, N);
        }

        TimePoint t1  = Clock::now();
        RaplState state1 = rapl_read();

        times[i]    = elapsed_s(t0, t1) / inner_reps;
        energies[i] = rapl_calc_joules(state0, state1) / inner_reps;
    }

    r.time_s   = median(times);
    r.energy_j = median(energies);
    r.gflops   = (flops / 1e9) / r.time_s;
    r.bw_gbps  = (bytes / 1e9) / r.time_s;

    free(A); free(B); free(C);
    return r;
}

static Result run_spmv(int N, double sparsity) {
    Result r;
    memset(&r, 0, sizeof(r));
    r.N = N;
    r.sparsity = sparsity;
    strncpy(r.kernel, "SpMV_Eigen",15);

    long nnz = std::max((long)(N * N * sparsity), (long)N);
    r.nnz = nnz;

    std::vector<Tripleta> listaTripletas;
    listaTripletas.reserve(nnz);

    srand(77);
    long per = nnz / N;
    long rem = nnz % N;

    for (int i = 0; i < N; i++) {
        long cnt = per + (i < rem ? 1 : 0);
        for (long j = 0; j < cnt; j++) {
            int col = rand() % N; 
            double val = (double)rand() / RAND_MAX + 0.1;
            listaTripletas.emplace_back(i, col, val);
        }
    }

    SpMat A(N, N);
    A.setFromTriplets(listaTripletas.begin(), listaTripletas.end());
    A.makeCompressed();

    Eigen::VectorXd x = Eigen::VectorXd::Random(N);
    Eigen::VectorXd y(N);

    const double flops = 2.0 * nnz;
    const double bytes = nnz * (8.0 + 4.0) + (N + 1) * 4.0 + 2.0 * N * 8.0;
    r.op_intensity = flops / bytes;
    r.mem_mb = bytes / 1e6;

    for (int i = 0; i < 3; i++) y = A * x; 

    TimePoint ta = Clock::now();
    y = A * x;
    TimePoint tb = Clock::now();
    
    double t1rep = elapsed_s(ta, tb);
    int inner_reps = (int)(0.2 / t1rep);
    if (inner_reps < 10)   inner_reps = 10;
    if (inner_reps > 2000) inner_reps = 2000;

    std::vector<double> times(BENCH_RUNS), energies(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; i++) {
        RaplState state0 = rapl_read();
        TimePoint t0  = Clock::now();

        for (int ir = 0; ir < inner_reps; ir++) {
            y = A * x;
        }

        TimePoint t1  = Clock::now();
        RaplState state1 = rapl_read();

        times[i]    = elapsed_s(t0, t1) / inner_reps;
        energies[i] = rapl_calc_joules(state0, state1) / inner_reps;
    }

    r.time_s   = median(times);
    r.energy_j = median(energies);
    r.gflops   = (flops / 1e9) / r.time_s;
    r.bw_gbps  = (bytes / 1e9) / r.time_s;
    
    double res = y.sum(); 
    if (res == -999999) std::cout << " "; 
    return r;
}

static Result run_fft(int N) {
    Result r;
    memset(&r, 0, sizeof(r));
    strncpy(r.kernel, "FFT", 15);
    r.N = N;

    double*       in  = (double*)      fftw_malloc((size_t)N * sizeof(double));
    fftw_complex* out = (fftw_complex*)fftw_malloc((size_t)(N/2+1) * sizeof(fftw_complex));
    if (!in || !out) { fprintf(stderr, "fftw_malloc N=%d\n", N); exit(1); }

    for (int i = 0; i < N; i++)
        in[i] = sin(2.0 * M_PI * i / N) + 0.3 * cos(6.0 * M_PI * i / N);

    std::vector<double> in_orig(in, in + N);

    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_MEASURE);
    if (!plan) { fprintf(stderr, "fftw_plan N=%d\n", N); exit(1); }

    const double flops = 5.0 * (double)N * std::log2((double)N);
    const double bytes = N * 8.0 + (N/2 + 1) * 16.0;
    r.op_intensity = flops / bytes;
    r.mem_mb       = bytes / 1e6;

    for (int i = 0; i < 3; i++) {
        memcpy(in, in_orig.data(), N * sizeof(double));
        fftw_execute(plan);
    }

    memcpy(in, in_orig.data(), N * sizeof(double));
    TimePoint ta = Clock::now();
    fftw_execute(plan);
    TimePoint tb = Clock::now();
    
    double t1rep = elapsed_s(ta, tb);
    int inner_reps = (int)(0.2 / t1rep);
    if (inner_reps < 5)    inner_reps = 5;
    if (inner_reps > 5000) inner_reps = 5000;

    std::vector<double> times(BENCH_RUNS), energies(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; i++) {
        memcpy(in, in_orig.data(), N * sizeof(double));

        RaplState state0 = rapl_read();
        TimePoint t0  = Clock::now();

        for (int ir = 0; ir < inner_reps; ir++) {
            fftw_execute(plan);
        }

        TimePoint t1  = Clock::now();
        RaplState state1 = rapl_read();

        times[i]    = elapsed_s(t0, t1) / inner_reps;
        energies[i] = rapl_calc_joules(state0, state1) / inner_reps;
    }

    r.time_s   = median(times);
    r.energy_j = median(energies);
    r.gflops   = (flops / 1e9) / r.time_s;
    r.bw_gbps  = (bytes / 1e9) / r.time_s;

    fftw_destroy_plan(plan);
    fftw_free(in); fftw_free(out);
    fftw_cleanup_threads();
    return r;
}

int main(int argc, char* argv[]) {
    const char* out_path = "resultados_epyc_16.csv";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output")  && i+1 < argc) out_path  = argv[++i];
        if (!strcmp(argv[i], "--threads") && i+1 < argc) g_cpu_cores  = atoi(argv[++i]);
    }

    openblas_set_num_threads(g_cpu_cores);
    omp_set_num_threads(g_cpu_cores);
    Eigen::setNbThreads(g_cpu_cores); 

    fftw_init_threads();
    fftw_plan_with_nthreads(g_cpu_cores);

    printf("============================================================\n"
           "  Benchmark CPU — AMD EPYC 9554  (Dell PowerEdge R7625)\n"
           "============================================================\n");
    printf("Hilos OpenBLAS/FFTW : %d\n", g_cpu_cores); // Mostrar los hilos reales en pantalla
    printf("Frecuencia          : %.2f GHz\n", CPU_FREQ_GHZ);
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
        Result r = run_gemm(N);
        printf("%-6d  %-12.6f  %-12.6f  %-10.2f  %-10.4f  %-10.4f\n",
               N, r.time_s, r.energy_j, r.gflops, r.bw_gbps, r.op_intensity);
        csv_row(fp, &r); fflush(fp);
    }

    printf("\n=== SpMV (Eigen SpMat, CSC) ===\n");
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
                   N, sp, r.nnz, r.time_s, r.energy_j, r.gflops);
            csv_row(fp, &r); fflush(fp);
        }
    }

    printf("\n=== FFT (FFTW3 r2c double) ===\n");
    printf("%-10s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_cpu(s)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_FFT; i++) {
        Result r = run_fft(FFT_N[i]); 
        printf("%-10d  %-12.6f  %-12.6f  %-10.4f  %-10.4f\n",
               r.N, r.time_s, r.energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, &r); fflush(fp);
    }

    fclose(fp);
    rapl_close();  
    printf("\nCompletado. Resultados en: %s\n", out_path);
    return 0;
}
