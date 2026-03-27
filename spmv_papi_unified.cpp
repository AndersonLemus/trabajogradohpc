/**
 * spmv_papi_final.cpp
 *
 * Benchmark SpMV (MKL sparse) con PAPI.
 * Misma estructura que gemm_papi_final.cpp:
 *   Fase 1 — tiempo limpio (mediana de N_REPS ejecuciones sin PAPI)
 *   Fase 2 — contadores PAPI (mediana de N_COUNTER_REPS por lote)
 *
 * Operación medida: y = A * x
 *   A : matriz dispersa N×N en formato CSR con densidad DENSITY
 *   x : vector denso de entrada (N doubles)
 *   y : vector denso de salida  (N doubles)
 *
 * Compilación:
 *   icpx -O2 -o spmv_papi spmv_papi_final.cpp \
 *        -lpapi -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
 *
 * Uso:
 *   ./spmv_papi <N> [--density <0.0-1.0>] [--verbose]
 *
 *   N         : dimensión de la matriz cuadrada (filas = columnas = N)
 *   --density : fracción de entradas no cero (por defecto 0.01 = 1%)
 *   --verbose : muestra detalle de lotes y mediciones
 *
 * Salida CSV → stdout:
 *   N,nnz,tipo,lote_o_fase,evento,valor,rel_dev_%
 *
 * Notas sobre SpMV vs GEMM:
 *   - El kernel es mucho más corto que GEMM; para N pequeños el tiempo
 *     de overhead de PAPI puede ser comparable al kernel. Usa N >= 10000.
 *   - El patrón de acceso a memoria es irregular (depende de la estructura
 *     de la matriz dispersa), por lo que los contadores de caché variarán
 *     más que en GEMM. Un rel_dev alto en L1_TCM/TLB_DM es esperado.
 *   - La matriz se genera con entradas aleatorias pero con la misma semilla
 *     en todas las ejecuciones para reproducibilidad.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <papi.h>
#include <mkl.h>
#include <mkl_spblas.h>

// ─────────────────────────────────────────────────────────────────────────────
// Parámetros de ejecución
// ─────────────────────────────────────────────────────────────────────────────
static const int    N_REPS         = 7;
static const int    N_WARMUP       = 2;
static const int    N_COUNTER_REPS = 5;
static const double DEFAULT_DENSITY = 0.01;   // 1% de entradas no cero

// ─────────────────────────────────────────────────────────────────────────────
// Catálogo de eventos PAPI
// ─────────────────────────────────────────────────────────────────────────────
struct EventDef { const char* name; int code; };

static const EventDef ALL_EVENTS[] = {
    { "PAPI_TOT_CYC", PAPI_TOT_CYC }, { "PAPI_TOT_INS", PAPI_TOT_INS },
    { "PAPI_RES_STL", PAPI_RES_STL }, { "PAPI_REF_CYC", PAPI_REF_CYC },
    { "PAPI_LD_INS",  PAPI_LD_INS  }, { "PAPI_SR_INS",  PAPI_SR_INS  },
    { "PAPI_LST_INS", PAPI_LST_INS },
    { "PAPI_L1_TCM",  PAPI_L1_TCM  }, { "PAPI_L1_DCM",  PAPI_L1_DCM  },
    { "PAPI_L1_ICM",  PAPI_L1_ICM  }, { "PAPI_L1_DCH",  PAPI_L1_DCH  },
    { "PAPI_L1_DCA",  PAPI_L1_DCA  }, { "PAPI_L1_DCR",  PAPI_L1_DCR  },
    { "PAPI_L1_DCW",  PAPI_L1_DCW  }, { "PAPI_L1_LDM",  PAPI_L1_LDM  },
    { "PAPI_L1_STM",  PAPI_L1_STM  },
    { "PAPI_L2_TCM",  PAPI_L2_TCM  }, { "PAPI_L2_DCM",  PAPI_L2_DCM  },
    { "PAPI_L2_ICM",  PAPI_L2_ICM  }, { "PAPI_L2_DCH",  PAPI_L2_DCH  },
    { "PAPI_L2_DCA",  PAPI_L2_DCA  }, { "PAPI_L2_DCR",  PAPI_L2_DCR  },
    { "PAPI_L2_DCW",  PAPI_L2_DCW  }, { "PAPI_L2_LDM",  PAPI_L2_LDM  },
    { "PAPI_L2_STM",  PAPI_L2_STM  },
    { "PAPI_L3_TCM",  PAPI_L3_TCM  }, { "PAPI_L3_LDM",  PAPI_L3_LDM  },
    { "PAPI_L3_DCA",  PAPI_L3_DCA  }, { "PAPI_L3_DCR",  PAPI_L3_DCR  },
    { "PAPI_L3_DCW",  PAPI_L3_DCW  },
    { "PAPI_TLB_DM",  PAPI_TLB_DM  }, { "PAPI_TLB_IM",  PAPI_TLB_IM  },
    { "PAPI_TLB_TL",  PAPI_TLB_TL  },
    { "PAPI_BR_INS",  PAPI_BR_INS  }, { "PAPI_BR_CN",   PAPI_BR_CN   },
    { "PAPI_BR_UCN",  PAPI_BR_UCN  }, { "PAPI_BR_TKN",  PAPI_BR_TKN  },
    { "PAPI_BR_NTK",  PAPI_BR_NTK  }, { "PAPI_BR_MSP",  PAPI_BR_MSP  },
    { "PAPI_BR_PRC",  PAPI_BR_PRC  },
    { "PAPI_FP_OPS",  PAPI_FP_OPS  }, { "PAPI_SP_OPS",  PAPI_SP_OPS  },
    { "PAPI_DP_OPS",  PAPI_DP_OPS  }, { "PAPI_FP_INS",  PAPI_FP_INS  },
    { "PAPI_VEC_SP",  PAPI_VEC_SP  }, { "PAPI_VEC_DP",  PAPI_VEC_DP  },
};
static const int N_EVENTS = sizeof(ALL_EVENTS) / sizeof(ALL_EVENTS[0]);

// ─────────────────────────────────────────────────────────────────────────────
// Estructura CSR + handle MKL
// ─────────────────────────────────────────────────────────────────────────────
struct CSRMatrix {
    int                  nrows, ncols, nnz;
    std::vector<double>  values;    // valores no cero  [nnz]
    std::vector<int>     col_idx;   // índice de columna [nnz]
    std::vector<int>     row_ptr;   // puntero de fila  [nrows+1]
    sparse_matrix_t      mkl_handle;
};

// ─────────────────────────────────────────────────────────────────────────────
// Utilidades
// ─────────────────────────────────────────────────────────────────────────────
static void fatal(int retval, const char* ctx) {
    fprintf(stderr, "PAPI error en '%s': %s (código %d)\n",
            ctx, PAPI_strerror(retval), retval);
    exit(EXIT_FAILURE);
}

static void fatal_mkl(sparse_status_t st, const char* ctx) {
    fprintf(stderr, "MKL sparse error en '%s': código %d\n", ctx, (int)st);
    exit(EXIT_FAILURE);
}

static double median_d(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? (v[m-1] + v[m]) / 2.0 : v[m];
}

static long long median_ll(std::vector<long long>& v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? (v[m-1] + v[m]) / 2 : v[m];
}

// ─────────────────────────────────────────────────────────────────────────────
// Generación de la matriz dispersa en formato CSR
//
// Se usa una distribución uniforme con semilla fija para reproducibilidad.
// Cada fila tiene en promedio density*ncols entradas no cero, distribuidas
// aleatoriamente entre las columnas.
// ─────────────────────────────────────────────────────────────────────────────
static CSRMatrix build_csr(int n, double density) {
    CSRMatrix mat;
    mat.nrows = mat.ncols = n;

    std::mt19937 rng(42);                          // semilla fija → reproducible
    std::uniform_real_distribution<double> val_dist(0.1, 2.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    mat.row_ptr.reserve(n + 1);
    mat.row_ptr.push_back(0);

    for (int i = 0; i < n; ++i) {
        // Garantizamos al menos 1 entrada por fila (diagonal) para que
        // la matriz no sea trivialmente singular y SpMV sea no trivial.
        bool diag_added = false;
        for (int j = 0; j < n; ++j) {
            if (coin(rng) < density || (!diag_added && j == i)) {
                mat.col_idx.push_back(j);
                mat.values.push_back(val_dist(rng));
                if (j == i) diag_added = true;
            }
        }
        mat.row_ptr.push_back((int)mat.col_idx.size());
    }

    mat.nnz = (int)mat.values.size();

    // Crear handle MKL sparse (inspección + optimización interna)
    sparse_status_t st = mkl_sparse_d_create_csr(
        &mat.mkl_handle,
        SPARSE_INDEX_BASE_ZERO,
        mat.nrows, mat.ncols,
        mat.row_ptr.data(),
        mat.row_ptr.data() + 1,
        mat.col_idx.data(),
        mat.values.data()
    );
    if (st != SPARSE_STATUS_SUCCESS) fatal_mkl(st, "mkl_sparse_d_create_csr");

    // Hint de uso: operación mv repetida, optimiza layout interno
    st = mkl_sparse_set_mv_hint(mat.mkl_handle,
                                SPARSE_OPERATION_NON_TRANSPOSE,
                                {SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL,
                                 SPARSE_DIAG_NON_UNIT},
                                N_REPS + N_COUNTER_REPS * 10);
    if (st != SPARSE_STATUS_SUCCESS) fatal_mkl(st, "mkl_sparse_set_mv_hint");

    st = mkl_sparse_optimize(mat.mkl_handle);
    if (st != SPARSE_STATUS_SUCCESS) fatal_mkl(st, "mkl_sparse_optimize");

    return mat;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel SpMV: y = alpha * A * x + beta * y
// ─────────────────────────────────────────────────────────────────────────────
static void run_spmv(const CSRMatrix& mat,
                     const std::vector<double>& x,
                     std::vector<double>& y)
{
    static const struct matrix_descr descr = {
        SPARSE_MATRIX_TYPE_GENERAL,
        SPARSE_FILL_MODE_FULL,
        SPARSE_DIAG_NON_UNIT
    };
    double alpha = 1.0, beta = 0.0;
    sparse_status_t st = mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha, mat.mkl_handle, descr,
        x.data(), beta, y.data()
    );
    if (st != SPARSE_STATUS_SUCCESS) fatal_mkl(st, "mkl_sparse_d_mv");
}

// ─────────────────────────────────────────────────────────────────────────────
// Warm-up de caché: toca x e y para que estén en L3/RAM antes de medir
// ─────────────────────────────────────────────────────────────────────────────
static void cache_warmup(const CSRMatrix& mat,
                          const std::vector<double>& x,
                          std::vector<double>& y)
{
    run_spmv(mat, x, y);
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-partición greedy de eventos en lotes compatibles
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<std::vector<EventDef>> build_batches(bool verbose) {
    std::vector<EventDef> available;
    for (int i = 0; i < N_EVENTS; ++i)
        if (PAPI_query_event(ALL_EVENTS[i].code) == PAPI_OK)
            available.push_back(ALL_EVENTS[i]);

    if (verbose)
        fprintf(stderr, "%zu eventos disponibles.\n\n", available.size());

    std::vector<int>                   test_sets;
    std::vector<std::vector<EventDef>> batches;

    for (auto& ev : available) {
        bool placed = false;
        for (size_t b = 0; b < batches.size(); ++b) {
            if (PAPI_add_event(test_sets[b], ev.code) == PAPI_OK) {
                batches[b].push_back(ev);
                placed = true;
                break;
            }
        }
        if (!placed) {
            int ns = PAPI_NULL;
            if (PAPI_create_eventset(&ns) != PAPI_OK) continue;
            if (PAPI_add_event(ns, ev.code) == PAPI_OK) {
                test_sets.push_back(ns);
                batches.push_back({ ev });
            } else {
                PAPI_destroy_eventset(&ns);
                if (verbose)
                    fprintf(stderr, "[SKIP] %s no pudo añadirse solo.\n", ev.name);
            }
        }
    }

    for (int& es : test_sets) {
        PAPI_cleanup_eventset(es);
        PAPI_destroy_eventset(&es);
    }

    if (verbose) {
        fprintf(stderr, "Auto-partición: %zu lote(s).\n\n", batches.size());
        for (size_t b = 0; b < batches.size(); ++b) {
            fprintf(stderr, "  Lote %zu (%zu eventos):", b+1, batches[b].size());
            for (auto& e : batches[b]) fprintf(stderr, " %s", e.name);
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }
    return batches;
}

// ─────────────────────────────────────────────────────────────────────────────
// FASE 1 — Tiempo limpio (sin PAPI)
// ─────────────────────────────────────────────────────────────────────────────
static double measure_time(const CSRMatrix& mat,
                            const std::vector<double>& x,
                            std::vector<double>& y,
                            bool verbose)
{
    if (verbose)
        fprintf(stderr, "Fase TIEMPO: %d warm-ups + %d reps...\n", N_WARMUP, N_REPS);

    for (int i = 0; i < N_WARMUP; ++i)
        run_spmv(mat, x, y);

    std::vector<double> times(N_REPS);
    for (int i = 0; i < N_REPS; ++i) {
        long long t0 = PAPI_get_real_usec();
        run_spmv(mat, x, y);
        long long t1 = PAPI_get_real_usec();
        times[i] = (t1 - t0) / 1e6;
        if (verbose) fprintf(stderr, "  rep %d: %.6f s\n", i+1, times[i]);
    }

    double med = median_d(times);
    if (verbose) fprintf(stderr, "  Mediana: %.6f s\n\n", med);
    return med;
}

// ─────────────────────────────────────────────────────────────────────────────
// FASE 2 — Contadores PAPI (mediana de N_COUNTER_REPS por lote)
// ─────────────────────────────────────────────────────────────────────────────
struct BatchResult {
    std::vector<EventDef>  events;
    std::vector<long long> medians;
    std::vector<double>    rel_devs;
};

static BatchResult measure_batch(const std::vector<EventDef>& events,
                                  const CSRMatrix& mat,
                                  const std::vector<double>& x,
                                  std::vector<double>& y,
                                  bool verbose)
{
    const int ne = (int)events.size();
    BatchResult res;
    res.events = events;
    res.medians.resize(ne, 0LL);
    res.rel_devs.resize(ne, 0.0);

    std::vector<std::vector<long long>> runs(N_COUNTER_REPS,
                                             std::vector<long long>(ne, 0LL));

    int EventSet = PAPI_NULL;
    int ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK) fatal(ret, "create_eventset");
    for (auto& e : events) {
        ret = PAPI_add_event(EventSet, e.code);
        if (ret != PAPI_OK) fatal(ret, e.name);
    }

    for (int r = 0; r < N_COUNTER_REPS; ++r) {
        cache_warmup(mat, x, y);
        std::fill(y.begin(), y.end(), 0.0);

        ret = PAPI_start(EventSet);
        if (ret != PAPI_OK) fatal(ret, "PAPI_start");

        run_spmv(mat, x, y);

        ret = PAPI_stop(EventSet, runs[r].data());
        if (ret != PAPI_OK) fatal(ret, "PAPI_stop");

        ret = PAPI_reset(EventSet);
        if (ret != PAPI_OK) fatal(ret, "PAPI_reset");
    }

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);

    for (int e = 0; e < ne; ++e) {
        std::vector<long long> vals(N_COUNTER_REPS);
        for (int r = 0; r < N_COUNTER_REPS; ++r) vals[r] = runs[r][e];

        long long med = median_ll(vals);
        long long mn  = *std::min_element(vals.begin(), vals.end());
        long long mx  = *std::max_element(vals.begin(), vals.end());

        res.medians[e]  = med;
        res.rel_devs[e] = (med > 0) ? (double)(mx - mn) / (double)med * 100.0 : 0.0;

        if (verbose)
            fprintf(stderr, "    %s: mediana=%lld  dev=%.1f%%\n",
                    events[e].name, med, res.rel_devs[e]);
    }
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <N> [--density <val>] [--verbose]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int    n       = std::atoi(argv[1]);
    double density = DEFAULT_DENSITY;
    bool   verbose = false;

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--density" && i+1 < argc)
            density = std::atof(argv[++i]);
        if (std::string(argv[i]) == "--verbose")
            verbose = true;
    }

    if (n <= 0 || density <= 0.0 || density > 1.0) {
        fprintf(stderr, "Error: N debe ser positivo y density en (0, 1].\n");
        return EXIT_FAILURE;
    }

    // ── PAPI init ─────────────────────────────────────────────────────────────
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) fatal(retval, "PAPI_library_init");

    // ── Construcción de la matriz y vectores ──────────────────────────────────
    if (verbose)
        fprintf(stderr, "Construyendo matriz CSR %d×%d  density=%.4f...\n",
                n, n, density);

    CSRMatrix mat = build_csr(n, density);

    if (verbose)
        fprintf(stderr, "NNZ = %d  (%.4f%%)\n\n",
                mat.nnz, 100.0 * mat.nnz / ((double)n * n));

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> vdist(0.5, 1.5);
    std::vector<double> x(n), y(n, 0.0);
    for (auto& v : x) v = vdist(rng);

    // ── Auto-partición de lotes ───────────────────────────────────────────────
    auto batches = build_batches(verbose);
    if (batches.empty()) {
        fprintf(stderr, "Error: no hay eventos medibles.\n");
        return EXIT_FAILURE;
    }

    // ── FASE 1: Tiempo limpio ─────────────────────────────────────────────────
    double t_median = measure_time(mat, x, y, verbose);
    // Cabecera implícita: N,nnz,lote,evento,valor,rel_dev_%
    printf("%d,%d,tiempo,MEDIANA,%.6f,0.00\n", n, mat.nnz, t_median);

    // ── FASE 2: Contadores por lote ───────────────────────────────────────────
    for (size_t b = 0; b < batches.size(); ++b) {
        if (verbose)
            fprintf(stderr, "Fase CONTADORES lote %zu/%zu...\n",
                    b+1, batches.size());

        auto res = measure_batch(batches[b], mat, x, y, verbose);

        for (size_t e = 0; e < res.events.size(); ++e)
            printf("%d,%d,%zu,%s,%lld,%.2f\n",
                   n, mat.nnz, b+1,
                   res.events[e].name,
                   res.medians[e],
                   res.rel_devs[e]);
    }

    // ── Limpieza MKL ──────────────────────────────────────────────────────────
    mkl_sparse_destroy(mat.mkl_handle);
    PAPI_shutdown();
    return EXIT_SUCCESS;
}