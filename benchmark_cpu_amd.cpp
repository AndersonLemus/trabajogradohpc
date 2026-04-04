/*
 *   Dell PowerEdge R7625
 *   AMD EPYC 9554 — 64 cores — 3.10 GHz — L3: 384 MB
 *   384 GB DDR5 — BW pico: 460.8 GB/s
 *
 * Kernels:
 *   GEMM  — OpenBLAS  (cblas_dgemm, double)
 *   SpMV  — SuiteSparse CSparse (cs_gaxpy, CSC double)
 *   FFT   — FFTW3     (fftw_plan_dft_r2c_1d, double)
 *
 * Uso:
 *   ./benchmark_epyc --output resultados_epyc.csv
 *   ./benchmark_epyc --output resultados_epyc.csv --threads 64
 * ============================================================
 */

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

#include <cblas.h>
extern "C" {
#include <cs.h>
}
#include <fftw3.h>
#include <papi.h>   /* solo para RAPL component (backend 3) */

#include <dirent.h>
#include <unistd.h>

// Hardware del nodo
static constexpr int    CPU_CORES     = 64;
static constexpr double CPU_FREQ_GHZ  = 3.20;
static constexpr double CPU_CACHE_MB  = 384.0;   // L3 total EPYC 9554

static const int GEMM_N[] = {
    1000, 1500, 2000, 2500, 3000, 3500,
    4000, 5000, 6000, 7000, 8000
};
static const int N_GEMM = sizeof(GEMM_N) / sizeof(GEMM_N[0]);

static const int SPMV_N[] = {
    100, 200, 500, 1000, 2000, 4000, 8000,
    16000, 32000, 64000, 100000
};
static const int N_SPMV = sizeof(SPMV_N) / sizeof(SPMV_N[0]);

static const double SPARSITIES[] = { 0.001, 0.0001, 0.00001 };
static const int    N_SP         = 3;

static const int FFT_N[] = {
    256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144, 524288, 1048576
};
static const int N_FFT = sizeof(FFT_N) / sizeof(FFT_N[0]);

// Resultado por medición
struct Result {
    std::string kernel;
    int         N          = 0;
    long        nnz        = 0;
    double      sparsity   = 0.0;
    double      op_intensity = 0.0;   // FLOP / byte
    double      time_ms    = 0.0;
    double      energy_j   = 0.0;
    double      gflops     = 0.0;
    double      bw_gbps    = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tiempo — std::chrono::steady_clock
// ─────────────────────────────────────────────────────────────────────────────

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline double elapsed_ms(TimePoint t0, TimePoint t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double median(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? (v[m-1] + v[m]) / 2.0 : v[m];
}

// RAPL
// El driver amd_energy actualiza el contador cada ~1ms.
// Se mide el delta sobre TODAS las repeticiones y se divide entre runs para obtener la energía media por ejecución.
static std::vector<std::string> rapl_paths;
static int  rapl_papi_evset = PAPI_NULL;
static bool rapl_use_papi   = false;

static void rapl_init()
{
    // Backend 1: sysfs amd_energy (AMD EPYC, kernel >= 5.11)
    const char* amd_base = "/sys/class/powercap/amd_energy";
    DIR* d = opendir(amd_base);
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d))) {
            if (ent->d_name[0] == '.') continue;
            std::string path = std::string(amd_base) + "/"
                             + ent->d_name + "/energy_input";
            FILE* f = fopen(path.c_str(), "r");
            if (f) { rapl_paths.push_back(path); fclose(f); }
        }
        closedir(d);
    }
    if (!rapl_paths.empty()) {
        std::cout << "[RAPL] sysfs amd_energy: "
                  << rapl_paths.size() << " dominio(s)\n";
        return;
    }

    // Backend 2: sysfs intel-rapl (fallback)
    for (auto& p : {
        std::string("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"),
        std::string("/sys/class/powercap/intel-rapl:0/energy_uj")
    }) {
        FILE* f = fopen(p.c_str(), "r");
        if (f) { rapl_paths.push_back(p); fclose(f); break; }
    }
    if (!rapl_paths.empty()) {
        std::cout << "[RAPL] sysfs intel-rapl: " << rapl_paths[0] << "\n";
        return;
    }

    // Backend 3: PAPI RAPL component
    if (PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT) {
        int rapl_code = 0;
        if (PAPI_event_name_to_code(
                (char*)"rapl:::PACKAGE_ENERGY:PACKAGE0", &rapl_code) == PAPI_OK &&
            PAPI_create_eventset(&rapl_papi_evset) == PAPI_OK &&
            PAPI_add_event(rapl_papi_evset, rapl_code) == PAPI_OK &&
            PAPI_start(rapl_papi_evset) == PAPI_OK) {
            rapl_use_papi = true;
            std::cout << "[RAPL] PAPI component activo\n";
            return;
        }
    }

    std::cerr << "[RAPL] No disponible — cpu_energy_j = 0.\n"
              << "       Para habilitar:\n"
              << "         sudo chmod a+r "
                 "/sys/class/powercap/amd_energy/*/energy_input\n"
              << "         sudo modprobe amd_energy\n";
}

static long long rapl_read_uj()
{
    if (rapl_use_papi) {
        long long val = 0LL;
        PAPI_read(rapl_papi_evset, &val);
        return val / 1000LL;
    }
    long long total = 0;
    for (auto& path : rapl_paths) {
        FILE* f = fopen(path.c_str(), "r");
        if (!f) continue;
        long long v = 0;
        fscanf(f, "%lld", &v);
        fclose(f);
        total += v;
    }
    return total;
}

// CSV
static void csv_header(std::ofstream& f)
{
    f << "kernel,N,nnz,sparsity,op_intensity,"
         "cpu_cores,cpu_freq_ghz,cpu_cache_mb,"
         "cpu_time_ms,cpu_energy_j,gflops,bw_gbps\n";
}

static void csv_row(std::ofstream& f, const Result& r)
{
    f << r.kernel      << ","
      << r.N           << ","
      << r.nnz         << ","
      << r.sparsity    << ","
      << r.op_intensity << ","
      << CPU_CORES     << ","
      << CPU_FREQ_GHZ  << ","
      << CPU_CACHE_MB  << ","
      << r.time_ms     << ","
      << r.energy_j    << ","
      << r.gflops      << ","
      << r.bw_gbps     << "\n";
}

// GEMM - OpenBLAS cblas_dgemm
static Result run_gemm(int N, int nthreads)
{
    Result r;
    r.kernel   = "GEMM";
    r.N        = N;
    r.sparsity = 0.0;   // GEMM es denso
    openblas_set_num_threads(nthreads);

    std::vector<double> A(N * N), B(N * N), C(N * N, 0.0);
    srand(42);
    for (auto& v : A) v = (double)rand() / RAND_MAX;
    for (auto& v : B) v = (double)rand() / RAND_MAX;

    const double alpha = 1.0, beta = 0.0;
    const double flops = 2.0 * N * N * N;
    const double bytes = 3.0 * N * N * 8.0;
    r.op_intensity = flops / bytes;   // ≈ N/12

    int runs = (N < 2000) ? 20 : (N < 4000) ? 10 : 5;
    std::vector<double> times(runs);

    long long uj_start = rapl_read_uj();

    for (int i = 0; i < runs; i++) {
        auto t0 = Clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
        times[i] = elapsed_ms(t0, Clock::now());
    }

    long long uj_end = rapl_read_uj();

    r.time_ms  = median(times);
    r.energy_j = (double)(uj_end - uj_start) / 1e6 / runs;
    r.gflops   = (flops / 1e9) / (r.time_ms / 1e3);
    r.bw_gbps  = (bytes / 1e9) / (r.time_ms / 1e3);
    return r;
}

// SpMV - SuiteSparse CSparse
static Result run_spmv(int N, double sparsity)
{
    Result r;
    r.kernel   = "SpMV";
    r.N        = N;
    r.sparsity = sparsity;

    long nnz = std::max((long)((long)N * N * sparsity), (long)N);

    cs* T = cs_spalloc(N, N, nnz, 1, 1);
    if (!T) { std::cerr << "cs_spalloc falló\n"; exit(1); }

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
    if (!A) { std::cerr << "cs_compress falló\n"; exit(1); }
    r.nnz = (A->nz < 0) ? A->p[N] : A->nz;

    std::vector<double> x(N), y(N, 0.0);
    for (auto& v : x) v = (double)rand() / RAND_MAX;

    const double flops = 2.0 * r.nnz;
    const double bytes = r.nnz * (8.0 + 4.0) + (N + 1) * 4.0 + 2.0 * N * 8.0;
    r.op_intensity = flops / bytes;

    // runs dinámico según nnz
    long nnz_est = r.nnz > 0 ? r.nnz : (long)N;
    int runs = (nnz_est <     10000) ? 5000 :
               (nnz_est <    100000) ? 500  :
               (nnz_est <   1000000) ? 50   :
               (nnz_est <  10000000) ? 10   : 5;

    std::vector<double> times(runs);

    long long uj_start = rapl_read_uj();

    for (int i = 0; i < runs; i++) {
        std::fill(y.begin(), y.end(), 0.0);
        auto t0 = Clock::now();
        cs_gaxpy(A, x.data(), y.data());
        times[i] = elapsed_ms(t0, Clock::now());
    }

    long long uj_end = rapl_read_uj();

    r.time_ms  = median(times);
    r.energy_j = (double)(uj_end - uj_start) / 1e6 / runs;
    r.gflops   = (flops / 1e9) / (r.time_ms / 1e3);
    r.bw_gbps  = (bytes / 1e9) / (r.time_ms / 1e3);

    cs_spfree(A);
    return r;
}

// FFT — FFTW3 fftw_plan_dft_r2c_1d (real→complex double)
static Result run_fft(int N, int nthreads)
{
    Result r;
    r.kernel   = "FFT";
    r.N        = N;
    r.sparsity = 0.0;

    fftw_init_threads();
    fftw_plan_with_nthreads(nthreads);

    double*       in  = (double*)      fftw_malloc(N * sizeof(double));
    fftw_complex* out = (fftw_complex*)fftw_malloc((N/2+1) * sizeof(fftw_complex));
    if (!in || !out) { std::cerr << "fftw_malloc falló\n"; exit(1); }

    for (int i = 0; i < N; i++)
        in[i] = std::sin(2.0 * M_PI * i / N) + 0.3 * std::cos(6.0 * M_PI * i / N);

    // FFTW_MEASURE: FFTW prueba estrategias y elige la más rápida para este HW.
    // El plan se crea una sola vez y se reutiliza en todas las repeticiones.
    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_MEASURE);
    if (!plan) { std::cerr << "fftw_plan falló\n"; exit(1); }

    const double flops = 5.0 * N * std::log2((double)N);
    const double bytes = N * 8.0 + (N/2 + 1) * 16.0;
    r.op_intensity = flops / bytes;

    int runs = (N <=    1024) ? 5000 :
               (N <=    8192) ? 1000 :
               (N <=   65536) ? 200  :
               (N <=  524288) ? 20   : 5;

    std::vector<double> times(runs);

    long long uj_start = rapl_read_uj();

    for (int i = 0; i < runs; i++) {
        // Reinicializar entrada: fftw_execute puede modificar el buffer in
        for (int j = 0; j < N; j++)
            in[j] = std::sin(2.0 * M_PI * j / N) + 0.3 * std::cos(6.0 * M_PI * j / N);
        auto t0 = Clock::now();
        fftw_execute(plan);
        times[i] = elapsed_ms(t0, Clock::now());
    }

    long long uj_end = rapl_read_uj();

    r.time_ms  = median(times);
    r.energy_j = (double)(uj_end - uj_start) / 1e6 / runs;
    r.gflops   = (flops / 1e9) / (r.time_ms / 1e3);
    r.bw_gbps  = (bytes / 1e9) / (r.time_ms / 1e3);

    fftw_destroy_plan(plan);
    fftw_free(in); fftw_free(out);
    fftw_cleanup_threads();
    return r;
}

int main(int argc, char* argv[])
{
    std::string out_path = "resultados_epyc.csv";
    int nthreads = 64;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output")  && i+1 < argc) out_path  = argv[++i];
        if (!strcmp(argv[i], "--threads") && i+1 < argc) nthreads  = atoi(argv[++i]);
    }

    std::cout << "============================================================\n"
              << "  Benchmark CPU — AMD EPYC 9554  (Dell PowerEdge R7625)\n"
              << "============================================================\n"
              << "Hilos OpenBLAS/FFTW : " << nthreads    << "\n"
              << "CPU cores / freq    : " << CPU_CORES   << " cores / "
                                          << CPU_FREQ_GHZ << " GHz\n"
              << "L3 caché            : " << CPU_CACHE_MB << " MB\n"
              << "Resultados          : " << out_path    << "\n\n";

    rapl_init();

    std::ofstream fp(out_path);
    if (!fp) { std::cerr << "No se pudo abrir: " << out_path << "\n"; return 1; }
    csv_header(fp);

    // GEMM
    std::cout << "=== GEMM (OpenBLAS cblas_dgemm) ===\n";
    printf("%-6s  %-12s  %-12s  %-10s  %-10s  %-10s\n",
           "N", "T_cpu(ms)", "Energy(J)", "GFLOPS", "BW(GB/s)", "Op.Int.");

    for (int i = 0; i < N_GEMM; i++) {
        int N = GEMM_N[i];
        if (3ULL * N * N * sizeof(double) > 350ULL << 30) {
            printf("N=%-6d  SKIP (RAM)\n", N); continue;
        }
        auto r = run_gemm(N, nthreads);
        printf("%-6d  %-12.4f  %-12.6f  %-10.2f  %-10.4f  %-10.4f\n",
               N, r.time_ms, r.energy_j, r.gflops, r.bw_gbps, r.op_intensity);
        csv_row(fp, r); fp.flush();
    }

    // SpMV
    std::cout << "\n=== SpMV (CSparse cs_gaxpy, CSC) ===\n";
    printf("%-6s  %-8s  %-12s  %-12s  %-12s  %-10s\n",
           "N", "sparsity", "nnz", "T_cpu(ms)", "Energy(J)", "GFLOPS");

    for (int si = 0; si < N_SP; si++) {
        double sp = SPARSITIES[si];
        for (int i = 0; i < N_SPMV; i++) {
            int  N   = SPMV_N[i];
            long nnz = std::max((long)((long)N * N * sp), (long)N);
            size_t need = (size_t)nnz * 12 + (size_t)(N + 1) * 4 + (size_t)2 * N * 8;
            if (need > 350ULL << 30) {
                printf("N=%-6d sp=%.5f  SKIP (RAM)\n", N, sp); continue;
            }
            auto r = run_spmv(N, sp);
            printf("%-6d  %-8.5f  %-12ld  %-12.4f  %-12.6f  %-10.4f\n",
                   N, sp, r.nnz, r.time_ms, r.energy_j, r.gflops);
            csv_row(fp, r); fp.flush();
        }
    }

    // FFT
    printf("\n=== FFT (FFTW3 r2c double, %d hilos) ===\n", nthreads);
    printf("%-10s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_cpu(ms)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_FFT; i++) {
        auto r = run_fft(FFT_N[i], nthreads);
        printf("%-10d  %-12.4f  %-12.6f  %-10.4f  %-10.4f\n",
               r.N, r.time_ms, r.energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, r); fp.flush();
    }

    // Limpieza
    if (rapl_use_papi && rapl_papi_evset != PAPI_NULL) {
        long long dummy;
        PAPI_stop(rapl_papi_evset, &dummy);
        PAPI_cleanup_eventset(rapl_papi_evset);
        PAPI_destroy_eventset(&rapl_papi_evset);
        PAPI_shutdown();
    }

    std::cout << "\nCompletado. Resultados en: " << out_path << "\n";
    return 0;
}