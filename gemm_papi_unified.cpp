/**
 * gemm_papi_final.cpp
 *
 * Benchmark GEMM (MKL) con PAPI y separación correcta de tiempo vs contadores.
 *
 * Estrategia:
 *   1. Fase de TIEMPO:   ejecuta GEMM N_REPS veces sin PAPI. Toma la mediana.
 *   2. Fase de CONTADORES: un GEMM por lote, con warm-up de caché previo.
 *
 *

 * Salida CSV → stdout:
 *   Línea de tiempo:    N,tiempo,MEDIANA,<mediana_seg>
 *   Línea de contador:  N,contador,<lote>,<evento>,<valor>
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <papi.h>
#include <mkl.h>
#include <chrono>


static const int N_REPS          = 7;  // repeticiones para la mediana de TIEMPO
static const int N_WARMUP        = 2;  // warm-ups antes de la fase de tiempo
static const int N_COUNTER_REPS  = 5;  // repeticiones por lote de CONTADORES
// Con 5 reps se detecta variación sin multiplicar demasiado el tiempo total.

//Catálogo de eventos
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

//Manejo de error.
static void fatal(int retval, const char* ctx) {
    fprintf(stderr, "PAPI error en '%s': %s (código %d)\n",
            ctx, PAPI_strerror(retval), retval);
    exit(EXIT_FAILURE);
}

// Cálculo de la mediana.
static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? (v[m-1] + v[m]) / 2.0 : v[m];
}

// Warm-up de caché: antes de medir contadores. Para N grandes esto garantiza que todos los
// lotes ven el mismo estado inicial (datos fuera de L1/L2, en L3 o RAM).
static void cache_warmup(int n,
                          const std::vector<double>& A,
                          const std::vector<double>& B,
                          std::vector<double>& C)
{
    double alpha = 1.0, beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, alpha, A.data(), n, B.data(), n, beta, C.data(), n);
}

// Auto-partición greedy de eventos en lotes compatibles
static std::vector<std::vector<EventDef>> build_batches(bool verbose) {

    std::vector<EventDef> available;
    for (int i = 0; i < N_EVENTS; ++i)
        if (PAPI_query_event(ALL_EVENTS[i].code) == PAPI_OK)
            available.push_back(ALL_EVENTS[i]);

    int rapl_cpu_code = 0;
    if (PAPI_event_name_to_code((char*)"rapl:::PACKAGE_ENERGY:PACKAGE0", &rapl_cpu_code) == PAPI_OK) {
        available.push_back({"ENERGIA_CPU_nJ", rapl_cpu_code});
        if (verbose) fprintf(stderr, "Sensor RAPL CPU detectado y añadido.\n");
    }

    int rapl_ram_code = 0;
    if (PAPI_event_name_to_code((char*)"rapl:::DRAM_ENERGY:PACKAGE0", &rapl_ram_code) == PAPI_OK) {
        available.push_back({"ENERGIA_RAM_nJ", rapl_ram_code});
        if (verbose) fprintf(stderr, "Sensor RAPL RAM detectado y añadido.\n");
    }

    if (verbose)
        fprintf(stderr, "%zu eventos disponibles en este hardware.\n\n",
                available.size());

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
        fprintf(stderr, "Auto-partición: %zu lote(s) necesarios.\n\n", batches.size());
        for (size_t b = 0; b < batches.size(); ++b) {
            fprintf(stderr, "  Lote %zu (%zu eventos):", b+1, batches[b].size());
            for (auto& e : batches[b]) fprintf(stderr, " %s", e.name);
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }

    return batches;
}


// C = alpha * (A x B) + Beta * C
// Param 0, la forma en la que la columna se guarda Row or columna.
//CblasNoTrans No transpongas la matriz A.
//CblasNoTrans No transpongas la matriz B.
//Dimensiones de las matrices (M, N, K) (Recordar que (M ° K) *(K ° N) = (M * N)
// //A.data(Es el puntero directo donde empieza la matriz A)
//n le dice a MKL cuántas casilla de memoria debe saltar para encontrar el inicio.
// //B.data(): Puntero inicio de B.
//n saltos de memoria para la matriz B.
//C.data() Puntero donde MKL va a guardar la respuesta.
//n saltos de memoria para la matriz C.
// FASE 1 — Calculo del tiempo limpio sin papi
// Ejecuta N_WARMUP veces para estabilizar frecuencia y caché,
// luego N_REPS mediciones reales. Reporta la mediana.
static inline double measure_time(int n,
                            const std::vector<double>& A,
                            const std::vector<double>& B,
                            std::vector<double>& C,
                            bool verbose)
{
    double alpha = 1.0, beta = 0.0;

    if (verbose) fprintf(stderr, "Fase TIEMPO: %d warm-ups + %d repeticiones...\n",
                         N_WARMUP, N_REPS);

    // Warm-up: activa turbo boost y estabiliza el estado del procesador
    for (int i = 0; i < N_WARMUP; ++i)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, alpha, A.data(), n, B.data(), n, beta, C.data(), n);

    // Mediciones
    std::vector<double> times(N_REPS);
    for (int i = 0; i < N_REPS; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, alpha, A.data(), n, B.data(), n, beta, C.data(), n);
        auto t1 = std::chrono::steady_clock::now(); 
        std::chrono::duration<double> elapsed = t1 - t0;
        times[i] = elapsed.count();
    }

    double med = median(times);
    if (verbose) fprintf(stderr, "  Mediana: %.6f s\n\n", med);
    return med;
}

// ─────────────────────────────────────────────────────────────────────────────
// FASE 2 — Contadores (N_COUNTER_REPS GEMMs por lote, mediana por evento)
//
// ¿Por qué repetir si los contadores son casi deterministas?
//   - Eventos de caché (L1_TCM, L2_DCM, TLB_DM) varían según el estado inicial
//     del caché, que nunca es idéntico entre ejecuciones.
//   - RES_STL y REF_CYC son sensibles a interferencia del SO.
//   - Con la mediana capturamos el valor "real" y con rel_dev sabemos qué tan
//     confiable es cada contador: rel_dev < 1% → determinista, > 5% → variable.
// ─────────────────────────────────────────────────────────────────────────────
struct BatchResult {
    std::vector<EventDef>  events;
    std::vector<long long> medians;   // mediana de N_COUNTER_REPS lecturas
    std::vector<double>    rel_devs;  // (max - min) / mediana * 100  [%]
};

static long long median_ll(std::vector<long long>& v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? (v[m-1] + v[m]) / 2 : v[m];
}

static BatchResult measure_batch(const std::vector<EventDef>& events,
                                  int n,
                                  const std::vector<double>& A,
                                  const std::vector<double>& B,
                                  std::vector<double>& C,
                                  bool verbose)
{
    const int ne = (int)events.size();
    BatchResult res;
    res.events = events;
    res.medians.resize(ne, 0LL);
    res.rel_devs.resize(ne, 0.0);

    // Matriz de lecturas: runs[rep][evento]
    std::vector<std::vector<long long>> runs(N_COUNTER_REPS,
                                             std::vector<long long>(ne, 0LL));

    // Construir EventSet una sola vez para todas las reps
    int EventSet = PAPI_NULL;
    int ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK) fatal(ret, "create_eventset fase contadores");
    for (auto& e : events) {
        ret = PAPI_add_event(EventSet, e.code);
        if (ret != PAPI_OK) fatal(ret, e.name);
    }

    double alpha = 1.0, beta = 0.0;

    for (int r = 0; r < N_COUNTER_REPS; ++r) {
        // Warm-up antes de CADA repetición: mismo estado de caché inicial
        cache_warmup(n, A, B, C);
        std::fill(C.begin(), C.end(), 0.0);

        ret = PAPI_start(EventSet);
        if (ret != PAPI_OK) fatal(ret, "PAPI_start");

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, alpha, A.data(), n, B.data(), n, beta, C.data(), n);

        ret = PAPI_stop(EventSet, runs[r].data());
        if (ret != PAPI_OK) fatal(ret, "PAPI_stop");

        // Resetear contadores para la siguiente repetición
        ret = PAPI_reset(EventSet);
        if (ret != PAPI_OK) fatal(ret, "PAPI_reset");
    }

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);

    // Calcular mediana y desviación relativa por evento
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
        fprintf(stderr, "Uso: %s <N> [--verbose]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = std::atoi(argv[1]);
    if (n <= 0) { fprintf(stderr, "N debe ser positivo.\n"); return EXIT_FAILURE; }

    bool verbose = false;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--verbose") verbose = true;

    // ── PAPI init ─────────────────────────────────────────────────────────────
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) fatal(retval, "PAPI_library_init");

    // ── Matrices ──────────────────────────────────────────────────────────────
    std::vector<double> A(n * n, 1.0);
    std::vector<double> B(n * n, 2.0);
    std::vector<double> C(n * n, 0.0);

    // ── Auto-partición de lotes ───────────────────────────────────────────────
    auto batches = build_batches(verbose);
    if (batches.empty()) {
        fprintf(stderr, "Error: no hay eventos medibles.\n");
        return EXIT_FAILURE;
    }

    // ── FASE 1: Tiempo limpio ─────────────────────────────────────────────────
    double t_median = measure_time(n, A, B, C, verbose);

    // Formato CSV: N, lote_o_"tiempo", evento, valor, rel_dev_%
    printf("%d,tiempo,MEDIANA,%.6f,0.00\n", n, t_median);

    // ── FASE 2: Contadores por lote ──────────────────────────────────────────
    // Primero se recogen todos los resultados-
    std::vector<BatchResult> all_results;
    for (size_t b = 0; b < batches.size(); ++b) {
        if (verbose)
            fprintf(stderr, "Fase CONTADORES lote %zu/%zu (%d reps)...\n",
                    b+1, batches.size(), N_COUNTER_REPS);
        all_results.push_back(measure_batch(batches[b], n, A, B, C, verbose));
        }

    //Salida CSV - Una línea por evento, tiempo como columna separada.
    //Formato: N, evento, valor, rel_dev%, tiempo_seg
        for (auto& res: all_results){
            for(size_t e = 0; e < res.events.size(); ++e)
                printf("%d,%s,%lld,%.2f,%.9f\n",
                   n,
                   res.events[e].name,
                   res.medians[e],
                   res.rel_devs[e],
                   t_median);
    }

    PAPI_shutdown();
    return EXIT_SUCCESS;
}