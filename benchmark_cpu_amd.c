/*
 * benchmark_cpu_amd.c
 * ============================================================
 * Benchmark HPC — CPU AMD del clúster SC3 — UIS
 *
 *   Dell PowerEdge R7625
 *   AMD EPYC 9554 — 64 cores — 3.10 GHz
 *   384 GB DDR5 — BW: 460.8 GB/s
 *
 * Kernels:
 *   GEMM  — OpenBLAS  (cblas_dgemm, double precision)
 *   SpMV  — SuiteSparse CSparse (cs_gaxpy, CSC double)
 *   FFT   — FFTW3     (fftw_plan_r2c_1d, double)
 *
 * Métricas por ejecución:
 *   cpu_time_ms    — clock_gettime(CLOCK_MONOTONIC)
 *   cpu_energy_j   — RAPL via /sys/class/powercap/intel-rapl
 *                    (también disponible en AMD desde kernel 5.11+
 *                     como /sys/class/powercap/amd_energy)
 *   gflops         — FLOPs teóricos / tiempo
 *   bw_gbps        — bytes transferidos / tiempo
 *   cpu_fp_ops     — PAPI_FP_OPS
 *   cpu_l1_miss    — PAPI_L1_DCM
 *   cpu_l2_miss    — PAPI_L2_DCM
 *   cpu_l3_miss    — PAPI_L3_TCM
 *   cpu_cycles     — PAPI_TOT_CYC
 *   cpu_instr      — PAPI_TOT_INS
 *
 * Compilación (en el nodo AMD):
 *   gcc -O3 -march=znver4 -fopenmp benchmark_cpu_amd.c \
 *       -lopenblas -lcsparse -lfftw3 -lpapi -lm -lpthread \
 *       -o benchmark_epyc
 *
 *   # Si SuiteSparse está en ruta no estándar:
 *   gcc -O3 -march=znver4 -fopenmp benchmark_cpu_amd.c \
 *       -I/usr/include/suitesparse \
 *       -lopenblas -lcsparse -lfftw3 -lpapi -lm -lpthread \
 *       -o benchmark_epyc
 *
 * Uso:
 *   ./benchmark_epyc --output resultados_epyc.csv
 *   ./benchmark_epyc --output resultados_epyc.csv --threads 64
 *
 * Salida CSV (misma estructura que benchmark_nvidia/amd GPU):
 *   platform, kernel, N, nnz, precision,
 *   cpu_time_ms, cpu_energy_j, gflops, bw_gbps,
 *   cpu_fp_ops, cpu_l1_miss, cpu_l2_miss, cpu_l3_miss,
 *   cpu_cycles, cpu_instructions
 * ============================================================
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>

/* OpenBLAS — GEMM */
#include <cblas.h>

/* SuiteSparse CSparse — SpMV */
#include <cs.h>

/* FFTW3 — FFT */
#include <fftw3.h>

/* PAPI — contadores hardware */
#include <papi.h>

/* ───────────────────────────── Macros de error ───────────────────────────── */

#define PAPI_CHECK(x) do {                                              \
    int _r = (x);                                                       \
    if (_r != PAPI_OK)                                                  \
        fprintf(stderr, "PAPI %s:%d  %s\n",                            \
                __FILE__, __LINE__, PAPI_strerror(_r));                 \
} while(0)

#define DIE(msg) do {                                                   \
    fprintf(stderr, "ERROR %s:%d  %s\n", __FILE__, __LINE__, msg);     \
    exit(EXIT_FAILURE);                                                 \
} while(0)

/* ──────────────────────────── Configuración ─────────────────────────────── */

#define WARMUP_RUNS   3
#define BENCH_RUNS    5
#define RAPL_POLL_MS  10    /* periodo de muestreo RAPL en ms */

/* Tamaños GEMM (matrices cuadradas N×N) */
static const int GEMM_N[] = {
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2500, 3000, 4000, 5000, 6000, 7000, 8000
};
static const int N_GEMM = (int)(sizeof(GEMM_N)/sizeof(GEMM_N[0]));

/* Tamaños SpMV (matriz N×N dispersa) */
static const int SPMV_N[] = {
    100, 200, 500, 1000, 2000, 4000, 8000,
    16000, 32000, 64000, 100000
};
static const int N_SPMV = (int)(sizeof(SPMV_N)/sizeof(SPMV_N[0]));

/* Sparsidades: 10%, 1%, 0.1% */
static const double SPARSITIES[] = { 0.10, 0.01, 0.001 };
static const int    N_SP         = 3;

/* Tamaños FFT (potencias de 2 hasta 2^20 = 1 M) */
static const int FFT_N[] = {
    256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144, 524288, 1048576
};
static const int N_FFT = (int)(sizeof(FFT_N)/sizeof(FFT_N[0]));

/* ─────────────────────────── Estructura de resultado ────────────────────── */

typedef struct {
    char   kernel[16];
    int    N;
    long   nnz;
    char   prec[8];
    double cpu_time_ms;
    double cpu_energy_j;
    double gflops;
    double bw_gbps;
    long long fp_ops;
    long long l1_miss;
    long long l2_miss;
    long long l3_miss;
    long long cycles;
    long long instr;
} Result;

/* ────────────────────────── Tiempo de pared ─────────────────────────────── */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ──────────────────────── RAPL: energía CPU via sysfs ───────────────────────
 *
 * AMD EPYC (kernel ≥ 5.11) expone el consumo del paquete CPU en:
 *   /sys/class/powercap/amd_energy/amd_energy_pkg<X>/energy_input  (µJ)
 *
 * Como respaldo (si amd_energy no está disponible), se intenta:
 *   /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj          (µJ)
 *
 * Ambas rutas devuelven microjulios acumulados (contador que se reinicia
 * al llegar a max_energy_range_uj).
 *
 * Para habilitar lectura sin root en Fedora/RHEL:
 *   sudo chmod a+r /sys/class/powercap/amd_energy/*/

#define RAPL_MAX_PATHS 4

typedef unsigned long long energy_input;

typedef struct {
    char   path[256];    /* ruta al archivo energy_input / energy_uj */
    int    valid;
} RaplDomain;

static RaplDomain rapl_domains[RAPL_MAX_PATHS];
static int        n_rapl = 0;

/*
 * rapl_find_paths() — detecta automáticamente las rutas RAPL disponibles.
 * Busca primero amd_energy (AMD), luego intel-rapl (fallback).
 */
static void rapl_find_paths(void)
{
    n_rapl = 0;

    /* ── Intento 1: amd_energy (AMD EPYC kernel ≥ 5.11) ─────────────────── */
    const char* amd_base = "/sys/class/powercap/amd_energy";
    DIR* d = opendir(amd_base);
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) && n_rapl < RAPL_MAX_PATHS) {
            if (ent->d_name[0] == '.') continue;
            snprintf(rapl_domains[n_rapl].path,
                     sizeof(rapl_domains[n_rapl].path),
                     "%s/%s/energy_input", amd_base, ent->d_name);
            FILE* f = fopen(rapl_domains[n_rapl].path, "r");
            if (f) {
                rapl_domains[n_rapl].valid = 1;
                n_rapl++;
                fclose(f);
            }
        }
        closedir(d);
    }

    /* ── Intento 2: intel-rapl (disponible también en algunos AMD) ──────── */
    if (n_rapl == 0) {
        const char* intel_paths[] = {
            "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
            "/sys/class/powercap/intel-rapl:0/energy_uj",
        };
        for (int i = 0; i < 2 && n_rapl < RAPL_MAX_PATHS; i++) {
            FILE* f = fopen(intel_paths[i], "r");
            if (f) {
                strncpy(rapl_domains[n_rapl].path, intel_paths[i], 255);
                rapl_domains[n_rapl].valid = 1;
                n_rapl++;
                fclose(f);
            }
        }
    }

    if (n_rapl == 0) {
        fprintf(stderr,
            "[RAPL] Advertencia: no se encontró interfaz de energía.\n"
            "       cpu_energy_j = 0 en el CSV.\n"
            "       Para habilitarla: sudo chmod a+r "
            "/sys/class/powercap/amd_energy/*/energy_input\n");
    } else {
        printf("[RAPL] %d dominio(s) encontrado(s):\n", n_rapl);
        for (int i = 0; i < n_rapl; i++)
            printf("       %s\n", rapl_domains[i].path);
    }
}

/* Lee la energía acumulada total (µJ) de todos los dominios RAPL activos */
static long long rapl_read_uj(void)
{
    long long total = 0;
    for (int i = 0; i < n_rapl; i++) {
        if (!rapl_domains[i].valid) continue;
        FILE* f = fopen(rapl_domains[i].path, "r");
        if (!f) continue;
        long long v = 0;
        fscanf(f, "%lld", &v);
        fclose(f);
        total += v;
    }
    return total;
}

/* ─────────────────────────────── PAPI ───────────────────────────────────── */

#define N_EV 6
static int papi_evset = PAPI_NULL;
static const int PAPI_EV[N_EV] = {
    PAPI_FP_OPS,   /* operaciones FP totales  */
    PAPI_L1_DCM,   /* L1 data cache misses    */
    PAPI_L2_DCM,   /* L2 data cache misses    */
    PAPI_L3_TCM,   /* L3 total cache misses   */
    PAPI_TOT_CYC,  /* ciclos totales          */
    PAPI_TOT_INS,  /* instrucciones totales   */
};

static void papi_setup(void)
{
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "[PAPI] init falló — contadores CPU deshabilitados\n");
        return;
    }
    PAPI_CHECK(PAPI_create_eventset(&papi_evset));
    for (int i = 0; i < N_EV; i++) {
        int r = PAPI_add_event(papi_evset, PAPI_EV[i]);
        if (r != PAPI_OK)
            fprintf(stderr, "[PAPI] No se pudo agregar evento %d: %s\n",
                    PAPI_EV[i], PAPI_strerror(r));
    }
}

static inline void papi_start(void)
{
    if (papi_evset == PAPI_NULL) return;
    PAPI_CHECK(PAPI_reset(papi_evset));
    PAPI_CHECK(PAPI_start(papi_evset));
}

static inline void papi_stop(long long* vals)
{
    if (papi_evset == PAPI_NULL) { memset(vals, 0, N_EV*sizeof(long long)); return; }
    PAPI_CHECK(PAPI_stop(papi_evset, vals));
}

/* ─────────────────────────── Escritura CSV ──────────────────────────────── */

static void csv_header(FILE* f)
{
    fprintf(f,
        "platform,kernel,N,nnz,precision,"
        "cpu_time_ms,cpu_energy_j,gflops,bw_gbps,"
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
        r->cpu_time_ms, r->cpu_energy_j,
        r->gflops, r->bw_gbps,
        r->fp_ops, r->l1_miss, r->l2_miss,
        r->l3_miss, r->cycles, r->instr);
}

/* ════════════════════════════════════════════════════════════════════════════
 * GEMM — OpenBLAS cblas_dgemm
 * ════════════════════════════════════════════════════════════════════════════
 * C = alpha*A*B + beta*C  (N×N, double, Row-Major)
 * FLOPs = 2 * N^3
 * Bytes  = 3 * N^2 * 8  (lectura A, B + escritura C)
 */
static Result run_gemm(int N, int nthreads, const char* plat)
{
    Result r; memset(&r, 0, sizeof(r));
    strcpy(r.kernel, "GEMM"); strcpy(r.prec, "double"); r.N = N;

    /* OpenBLAS: configurar número de hilos antes de cada llamada */
    openblas_set_num_threads(nthreads);

    size_t sz = (size_t)N * N * sizeof(double);
    double* A  = (double*)malloc(sz);
    double* B  = (double*)malloc(sz);
    double* C  = (double*)calloc(N*N, sizeof(double));
    if (!A || !B || !C) DIE("malloc GEMM");

    srand(42);
    for (int i = 0; i < N*N; i++) {
        A[i] = (double)rand()/RAND_MAX;
        B[i] = (double)rand()/RAND_MAX;
    }

    const double alpha = 1.0, beta = 0.0;

    /* warm-up */
    for (int w = 0; w < WARMUP_RUNS; w++)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, alpha, A, N, B, N, beta, C, N);

    /* medición */
    long long pv[N_EV] = {0};
    double    tot_ms   = 0.0;
    double    e_start, e_stop;

    papi_start();
    long long uj_start = rapl_read_uj();
    double    t_start  = now_ms();

    for (int i = 0; i < BENCH_RUNS; i++) {
        double t0 = now_ms();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, alpha, A, N, B, N, beta, C, N);
        tot_ms += now_ms() - t0;
    }

    long long uj_end = rapl_read_uj();
    papi_stop(pv);

    double avg   = tot_ms / BENCH_RUNS;
    double flops = 2.0 * N * N * N;
    double bytes = 3.0 * N * N * sizeof(double);

    r.cpu_time_ms  = avg;
    r.cpu_energy_j = (uj_end - uj_start) / 1e6;   /* µJ → J */
    r.gflops       = (flops / 1e9) / (avg / 1e3);
    r.bw_gbps      = (bytes / 1e9) / (avg / 1e3);
    r.fp_ops       = pv[0]; r.l1_miss = pv[1];
    r.l2_miss      = pv[2]; r.l3_miss = pv[3];
    r.cycles       = pv[4]; r.instr   = pv[5];

    free(A); free(B); free(C);
    return r;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SpMV — SuiteSparse CSparse (cs_gaxpy)
 * ════════════════════════════════════════════════════════════════════════════
 * y += A * x   (A N×N en formato CSC, double)
 * FLOPs = 2 * nnz
 * Bytes  = nnz*8 (val) + nnz*4 (row_idx) + (N+1)*4 (col_ptr)
 *        + N*8 (x) + N*8 (y)
 *
 * Nota: cs_gaxpy opera sobre matrices CSC (Compressed Sparse Column).
 * La construcción se hace con cs_spalloc + cs_entry + cs_compress.
 */
static Result run_spmv(int N, double sparsity, const char* plat)
{
    Result r; memset(&r, 0, sizeof(r));
    strcpy(r.kernel, "SpMV"); strcpy(r.prec, "double"); r.N = N;

    long nnz = (long)((double)N * N * sparsity);
    if (nnz < N) nnz = N;

    /* Construir triplete y comprimir a CSC */
    cs* T = cs_spalloc(N, N, nnz, 1, 1);   /* triplete */
    if (!T) DIE("cs_spalloc");

    srand(77);
    long per = nnz / N, rem = nnz % N, ptr = 0;
    for (int col = 0; col < N && ptr < nnz; col++) {
        long cnt = per + (col < rem ? 1 : 0);
        for (long j = 0; j < cnt && ptr < nnz; j++) {
            int row = (int)(j * N / cnt) % N;
            double val = (double)rand()/RAND_MAX + 0.1;
            cs_entry(T, row, col, val);
            ptr++;
        }
    }

    cs* A = cs_compress(T);   /* CSC comprimida */
    cs_spfree(T);
    if (!A) DIE("cs_compress");

    r.nnz = A->nz < 0 ? A->p[N] : A->nz;   /* nz real en CSC */

    double* x = (double*)malloc(N * sizeof(double));
    double* y = (double*)calloc(N, sizeof(double));
    if (!x || !y) DIE("malloc SpMV");
    for (int i = 0; i < N; i++) x[i] = (double)rand()/RAND_MAX;

    /* warm-up */
    for (int w = 0; w < WARMUP_RUNS; w++) {
        memset(y, 0, N * sizeof(double));
        cs_gaxpy(A, x, y);
    }

    /* medición */
    long long pv[N_EV] = {0};
    double    tot_ms   = 0.0;

    papi_start();
    long long uj_start = rapl_read_uj();

    for (int i = 0; i < BENCH_RUNS; i++) {
        memset(y, 0, N * sizeof(double));
        double t0 = now_ms();
        cs_gaxpy(A, x, y);
        tot_ms += now_ms() - t0;
    }

    long long uj_end = rapl_read_uj();
    papi_stop(pv);

    double avg   = tot_ms / BENCH_RUNS;
    long   nz    = r.nnz;
    double bytes = (double)nz * (sizeof(double) + sizeof(int))
                 + (double)(N+1) * sizeof(int)
                 + 2.0 * N * sizeof(double);

    r.cpu_time_ms  = avg;
    r.cpu_energy_j = (uj_end - uj_start) / 1e6;
    r.gflops       = (2.0 * nz / 1e9) / (avg / 1e3);
    r.bw_gbps      = (bytes / 1e9) / (avg / 1e3);
    r.fp_ops       = pv[0]; r.l1_miss = pv[1];
    r.l2_miss      = pv[2]; r.l3_miss = pv[3];
    r.cycles       = pv[4]; r.instr   = pv[5];

    cs_spfree(A);
    free(x); free(y);
    return r;
}

/* ════════════════════════════════════════════════════════════════════════════
 * FFT — FFTW3 (fftw_plan_dft_r2c_1d, double, real→complex)
 * ════════════════════════════════════════════════════════════════════════════
 * FLOPs ≈ 5 * N * log2(N)
 * Bytes  = N*8 (entrada) + (N/2+1)*16 (salida compleja)
 *
 * Se usa FFTW_MEASURE en el plan para que FFTW optimice el algoritmo
 * al hardware específico del EPYC (Zen 4, AVX-512).
 * El plan se crea una sola vez y se reutiliza en todos los BENCH_RUNS.
 */
static Result run_fft(int N, int nthreads, const char* plat)
{
    Result r; memset(&r, 0, sizeof(r));
    strcpy(r.kernel, "FFT"); strcpy(r.prec, "double"); r.N = N;

    /* FFTW con hilos — debe llamarse antes de crear el plan */
    fftw_init_threads();
    fftw_plan_with_nthreads(nthreads);

    double*          in  = (double*)         fftw_malloc(N * sizeof(double));
    fftw_complex*    out = (fftw_complex*)    fftw_malloc((N/2+1) * sizeof(fftw_complex));
    if (!in || !out) DIE("fftw_malloc");

    for (int i = 0; i < N; i++)
        in[i] = sin(2.0*M_PI*i/N) + 0.3*cos(6.0*M_PI*i/N);

    /*
     * FFTW_MEASURE: mide varias estrategias y elige la más rápida.
     * Puede tardar varios segundos en la primera llamada — es normal.
     * Para producción: guardar el plan con fftw_export_wisdom_to_filename.
     */
    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_MEASURE);
    if (!plan) DIE("fftw_plan_dft_r2c_1d");

    /* warm-up con el plan ya optimizado */
    for (int w = 0; w < WARMUP_RUNS; w++) {
        for (int i = 0; i < N; i++)
            in[i] = sin(2.0*M_PI*i/N);
        fftw_execute(plan);
    }

    /* medición */
    long long pv[N_EV] = {0};
    double    tot_ms   = 0.0;

    papi_start();
    long long uj_start = rapl_read_uj();

    for (int i = 0; i < BENCH_RUNS; i++) {
        /* reset de entrada para medición limpia */
        for (int j = 0; j < N; j++)
            in[j] = sin(2.0*M_PI*j/N) + 0.3*cos(6.0*M_PI*j/N);
        double t0 = now_ms();
        fftw_execute(plan);
        tot_ms += now_ms() - t0;
    }

    long long uj_end = rapl_read_uj();
    papi_stop(pv);

    double avg   = tot_ms / BENCH_RUNS;
    double flops = 5.0 * N * log2((double)N);
    double bytes = (double)N * sizeof(double)
                 + (double)(N/2+1) * sizeof(fftw_complex);

    r.cpu_time_ms  = avg;
    r.cpu_energy_j = (uj_end - uj_start) / 1e6;
    r.gflops       = (flops / 1e9) / (avg / 1e3);
    r.bw_gbps      = (bytes / 1e9) / (avg / 1e3);
    r.fp_ops       = pv[0]; r.l1_miss = pv[1];
    r.l2_miss      = pv[2]; r.l3_miss = pv[3];
    r.cycles       = pv[4]; r.instr   = pv[5];

    fftw_destroy_plan(plan);
    fftw_free(in); fftw_free(out);
    fftw_cleanup_threads();
    return r;
}

/* ════════════════════════════════════════════════════════════════════════════
 * main
 * ════════════════════════════════════════════════════════════════════════════ */

int main(int argc, char* argv[])
{
    const char* out_path = "resultados_epyc.csv";
    int nthreads = 64;   /* AMD EPYC 9554: 64 cores físicos */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output") && i+1 < argc)
            out_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i+1 < argc)
            nthreads = atoi(argv[++i]);
    }

    printf("============================================================\n");
    printf("  Benchmark CPU — AMD EPYC 9554  (Dell PowerEdge R7625)\n");
    printf("============================================================\n");
    printf("Hilos OpenBLAS/FFTW : %d\n", nthreads);
    printf("Bibliotecas          : OpenBLAS + CSparse + FFTW3\n");
    printf("Métricas CPU         : PAPI + RAPL\n");
    printf("Resultados           : %s\n\n", out_path);

    const char* platform = "Dell_R7625_EPYC9554";

    /* RAPL */
    rapl_find_paths();

    /* PAPI */
    papi_setup();

    /* CSV */
    FILE* fp = fopen(out_path, "w");
    if (!fp) { perror("fopen"); return 1; }
    csv_header(fp);

    /* ════ GEMM ═════════════════════════════════════════════════════════════ */
    printf("=== GEMM (OpenBLAS cblas_dgemm) ===\n");
    printf("%-6s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_cpu(ms)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_GEMM; i++) {
        int N = GEMM_N[i];
        /* Verificar que caben 3 matrices en RAM (384 GB disponibles) */
        if (3ULL * N * N * sizeof(double) > 350ULL * 1024*1024*1024) {
            printf("N=%-6d  SKIP (RAM insuficiente)\n", N); continue;
        }
        Result r = run_gemm(N, nthreads, platform);
        printf("%-6d  %-12.4f  %-12.6f  %-10.2f  %-10.2f\n",
               N, r.cpu_time_ms, r.cpu_energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, platform, &r); fflush(fp);
    }

    /* ════ SpMV ═════════════════════════════════════════════════════════════ */
    printf("\n=== SpMV (CSparse cs_gaxpy, CSC) ===\n");
    printf("%-6s  %-8s  %-10s  %-12s  %-12s  %-10s\n",
           "N", "sparsity", "nnz", "T_cpu(ms)", "Energy(J)", "GFLOPS");

    for (int si = 0; si < N_SP; si++) {
        double sp = SPARSITIES[si];
        for (int i = 0; i < N_SPMV; i++) {
            int  N   = SPMV_N[i];
            long nnz = (long)((double)N * N * sp);
            if (nnz < N) nnz = N;
            /* Verificar RAM para la matriz CSC */
            size_t need = (size_t)nnz * (sizeof(double) + sizeof(int))
                        + (size_t)(N+1) * sizeof(int)
                        + (size_t)2 * N * sizeof(double);
            if (need > 350ULL * 1024*1024*1024) {
                printf("N=%-6d sp=%.3f  SKIP\n", N, sp); continue;
            }
            Result r = run_spmv(N, sp, platform);
            printf("%-6d  %-8.3f  %-10ld  %-12.4f  %-12.6f  %-10.4f\n",
                   N, sp, r.nnz, r.cpu_time_ms, r.cpu_energy_j, r.gflops);
            csv_row(fp, platform, &r); fflush(fp);
        }
    }

    /* ════ FFT ══════════════════════════════════════════════════════════════ */
    printf("\n=== FFT (FFTW3 r2c double, %d hilos) ===\n", nthreads);
    printf("%-10s  %-12s  %-12s  %-10s  %-10s\n",
           "N", "T_cpu(ms)", "Energy(J)", "GFLOPS", "BW(GB/s)");

    for (int i = 0; i < N_FFT; i++) {
        int N = FFT_N[i];
        Result r = run_fft(N, nthreads, platform);
        printf("%-10d  %-12.4f  %-12.6f  %-10.4f  %-10.4f\n",
               N, r.cpu_time_ms, r.cpu_energy_j, r.gflops, r.bw_gbps);
        csv_row(fp, platform, &r); fflush(fp);
    }

    /* Limpieza */
    fclose(fp);
    if (papi_evset != PAPI_NULL) {
        PAPI_CHECK(PAPI_cleanup_eventset(papi_evset));
        PAPI_CHECK(PAPI_destroy_eventset(&papi_evset));
        PAPI_shutdown();
    }

    printf("\nCompletado. Resultados en: %s\n", out_path);
    return 0;
}
