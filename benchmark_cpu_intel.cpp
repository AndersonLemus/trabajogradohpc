**
 * Compilación (Intel MKL):
 *   icpx -O3 -march=native -std=c++17 hpc_dataset.cpp \
 *        -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
 *        -lmkl_rt -lmkl_dfti \
 *        -lpthread -lm -o hpc_dataset

 *   ./hpc_dataset --kernel gemm --n 500 [--header]
 *   ./hpc_dataset --kernel spmv --n 500 --sparsity 0.01 [--header]
 *   ./hpc_dataset --kernel fft  --n 512 [--header]
 *
 */
 
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <mkl.h>
#include <mkl_spblas.h>
#include <mkl_dfti.h>
 
// Parámetros de medición
static const int N_WARMUP = 3;
static const int N_REPS   = 7;
 
// Características estáticas del hardware
static const std::string PLATFORM     = "Felix";
static const int         CPU_CORES    = 8;
static const double      CPU_FREQ_GHZ = 2.27;
// static const double   CACHE_L3_MiB  = 96;   

static std::string rapl_path = "";
 
static void rapl_init() {
    for (auto& p : {
        std::string("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"),
        std::string("/sys/class/powercap/intel-rapl:0/energy_uj")
    }) {
        FILE* f = fopen(p.c_str(), "r");
        if (f) { rapl_path = p; fclose(f); return; }
    }
    fprintf(stderr, "[INFO] RAPL no disponible o sin permisos. "
                    "Energia_CPU_J = 0.0 en el CSV.\n");
}
 
static long long rapl_read_uj() {
    if (rapl_path.empty()) return 0LL;
    FILE* f = fopen(rapl_path.c_str(), "r");
    if (!f) return 0LL;
    long long v = 0;
    fscanf(f, "%lld", &v);
    fclose(f);
    return v;
}
 
// Utilidades
static double median_d(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? (v[m-1] + v[m]) / 2.0 : v[m];
}
 

struct Medicion {
    double tiempo_s;      
    double energia_j;     
    double gflops;        
    double bw_gbps;       
};
 

template<typename KernelFn>

static Medicion medir(KernelFn kernel_fn,
                       double flops_totales,
                       double bytes_totales)
{
    // Warm-up: estabiliza CPU y caché
    for (int i = 0; i < N_WARMUP; ++i) kernel_fn();
 
    std::vector<double> times(N_REPS);
    std::vector<double> energies(N_REPS);
 
    for (int i = 0; i < N_REPS; ++i) {
        long long e_start = rapl_read_uj();
        auto t0 = std::chrono::steady_clock::now();

        kernel_fn();

        auto t1 = std::chrono::steady_clock::now();
        long long e_end = rapl_read_uj();

long long e_end = rapl_read_uj();
        times[i] = std::chrono::duration<double>(t1 - t0).count();

        if (e_end >= e_start) {
            energies[i] = (double)(e_end - e_start) / 1e6; // Convertir micro-Joules a Joules
        } else {
            energies[i] = 0.0; 
        }
    }
 
    double t_med  = median_d(times);
    double e_med  = median_d(energies);

    double gflops = (flops_totales / 1e9) / t_med;
    double bw     = (bytes_totales / 1e9) / t_med;
 
    return { t_med, e_med, gflops, bw };
}
 
// GEMM
static void run_gemm(int n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(0.1, 1.0);
    std::vector<double> A(n*n), B(n*n), C(n*n, 0.0);
    for (auto& v : A) v = dis(gen);
    for (auto& v : B) v = dis(gen);
 
    double alpha = 1.0, beta = 0.0;
 
   

    double flops = 2.0 * n * n * n;
    double bytes = 3.0 * n * n * sizeof(double);
    double oi = (double)n / 12.0;
 
    auto kernel = [&]() {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, alpha, A.data(), n, B.data(), n, beta, C.data(), n);
    };
 
    Medicion m = medir(kernel, flops, bytes);
 
    printf("GEMM,%d,0.0,%.4f,%.2f,%.6f,%.6f,%.2f,0.0,0.0,%.2f,%d,%.2f\n",
n,
           oi,
           bytes / (1024.0 * 1024.0),   // Memoria_MB
           m.tiempo_s,
           m.energia_j,
           m.gflops,
           m.bw_gbps,
           CPU_CORES,
           CPU_FREQ_GHZ
           );
}
 

// SpMV — formato CSR con densidad configurable
static void run_spmv(int n, double sparsity) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> val_dist(0.1, 2.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
 
    // Construir matriz CSR
    std::vector<double> values;
    std::vector<int>    col_idx;
    std::vector<int>    row_ptr;
    row_ptr.push_back(0);
 
    for (int i = 0; i < n; ++i) {
        bool diag = false;
        for (int j = 0; j < n; ++j) {
            if (coin(gen) < sparsity || (!diag && j == i)) {
                col_idx.push_back(j);
                values.push_back(val_dist(gen));
                if (j == i) diag = true;
            }
        }
        row_ptr.push_back((int)col_idx.size());
    }
 
    int nnz = (int)values.size();
 
    sparse_matrix_t A_mkl;
    mkl_sparse_d_create_csr(&A_mkl, SPARSE_INDEX_BASE_ZERO,
                             n, n,
                             row_ptr.data(), row_ptr.data() + 1,
                             col_idx.data(), values.data());
        mkl_sparse_optimize(A_mkl);
 
    std::vector<double> x(n, 1.0), y(n, 0.0);
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL,
        SPARSE_FILL_MODE_FULL,
                                   SPARSE_DIAG_NON_UNIT };
 

    double flops = 2.0 * nnz;
    double bytes = (double)nnz * (sizeof(double) + sizeof(int))
                 + 2.0 * n * sizeof(double);
 
    double oi = flops/bytes;
 
    auto kernel = [&]() {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                        1.0, A_mkl, descr,
                        x.data(), 0.0, y.data());
    };
 
    Medicion m = medir(kernel, flops, bytes);
 
    printf("SpMV,%d,%.4f,%.4f,%.2f,%.6f,%.6f,%.2f,0.0,0.0,%.2f,%d,%.2f\n",
           n,
           sparsity,
           oi,
           bytes / (1024.0 * 1024.0),
           m.tiempo_s,
           m.energia_j,
           m.gflops,
           m.bw_gbps,
           CPU_CORES,
           CPU_FREQ_GHZ
           );
 
    mkl_sparse_destroy(A_mkl);
}
 
// FFT — transformada 1D compleja in-place con MKL DFTI
static void run_fft(int n) {

    std::vector<double> x(2 * n);
    for (int i = 0; i < n; ++i) {
        x[2*i]   = std::cos(2.0 * M_PI * i / n);
        x[2*i+1] = std::sin(2.0 * M_PI * i / n);
    }
    std::vector<double> x_orig = x;
 
    // Crear descriptor DFTI (una sola vez, fuera de la medición)
    DFTI_DESCRIPTOR_HANDLE desc;
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)n);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0);
    DftiCommitDescriptor(desc);
 
    double flops = 5.0 * n * std::log2((double)n);
    double bytes = 4.0 * n * sizeof(double);
 
    // OI = (5/16)*log2(N) (ec. 17 del plan)
    double oi = (5.0 / 16.0) * std::log2((double)n);
 
    // La FFT in-place destruye x — se restaura antes de cada repetición
    auto kernel = [&]() {
        x = x_orig;
        DftiComputeForward(desc, x.data());
    };
 
    Medicion m = medir(kernel, flops, bytes);
 
    printf("FFT,%d,0.0,%.4f,%.2f,%.6f,%.6f,%.2f,0.0,0.0,%.2f,%d,%.2f\n",
           n,
           oi,
           bytes / (1024.0 * 1024.0),
           m.tiempo_s,
           m.energia_j,
           m.gflops,
           m.bw_gbps,
           CPU_CORES,
           CPU_FREQ_GHZ
           );
 
    DftiFreeDescriptor(&desc);
}
 

int main(int argc, char** argv) {
 
    std::string kernel_name = "";
    int    n        = 0;
    double sparsity = 0.01;
    bool   header   = false;
 
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kernel"   && i+1 < argc) kernel_name = argv[++i];
        else if (arg == "--n"   && i+1 < argc) n           = std::atoi(argv[++i]);
        else if (arg == "--sparsity" && i+1 < argc) sparsity = std::atof(argv[++i]);
        else if (arg == "--header")             header      = true;
}
 
    if (kernel_name.empty() || n <= 0) {
        fprintf(stderr,
            "Uso: %s --kernel <gemm|spmv|fft> --n <N> "
            "[--sparsity <val>] [--header]\n", argv[0]);
        return EXIT_FAILURE;
    }
 
    rapl_init();
 
    if (header) {
        printf("Kernel,N,Sparsity,OI_FLOPs_Byte,Memoria_MB,"
               "Tiempo_CPU_s,Energia_CPU_J,GFLOPS_CPU,"
               "Tiempo_GPU_s,Energia_GPU_J,"
               "BW_CPU_GBps,CPU_CORES,CPU_FREQ_GHZ\n");
    }
 
    if      (kernel_name == "gemm") run_gemm(n);
    else if (kernel_name == "spmv") run_spmv(n, sparsity);
    else if (kernel_name == "fft")  run_fft(n);
    else {
        fprintf(stderr, "Kernel desconocido: '%s'. "
                        "Opciones: gemm, spmv, fft\n", kernel_name.c_str());
        return EXIT_FAILURE;
    }
 
    return EXIT_SUCCESS;
}
