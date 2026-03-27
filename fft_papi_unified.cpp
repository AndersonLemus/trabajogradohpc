/**
 * fft_papi_final.cpp
 *
 * Benchmark FFT (MKL DFTI) con PAPI.
 * Misma estructura que gemm_papi_final.cpp:
 *   Fase 1 — tiempo limpio (mediana de N_REPS ejecuciones sin PAPI)
 *   Fase 2 — contadores PAPI (mediana de N_COUNTER_REPS por lote)
 *
 * Operación medida: FFT 1D in-place de doble precisión compleja
 *   x : vector de N números complejos (2*N doubles intercalados: re, im, re, im...)
 *   Se mide solo el DFTI_FORWARD (DFT directa).
 *   El descriptor DFTI se crea una sola vez fuera de la medición.
 *
 * ¿Por qué 1D compleja?
 *   Es el caso base de FFT en HPC. La FFT real-to-complex (DFTI_REAL) es
 *   una variante; si la necesitas puedes cambiar DFTI_COMPLEX → DFTI_REAL
 *   y ajustar el tamaño del buffer (N/2 + 1 valores complejos de salida).
 *
 * Compilación:
 *   icpx -O2 -o fft_papi fft_papi_final.cpp \
 *        -lpapi -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
 *
 * Uso:
 *   ./fft_papi <N> [--verbose]
 *
 *   N : número de puntos de la FFT.
 *       MKL DFTI es más eficiente cuando N es potencia de 2 o tiene
 *       factores primos pequeños (2, 3, 5, 7). Para benchmarking usa
 *       potencias de 2: 1024, 4096, 16384, 65536, 262144, 1048576.
 *
 * Salida CSV → stdout:
 *   N,tipo,lote_o_fase,evento,valor,rel_dev_%
 *
 * Notas sobre FFT vs GEMM:
 *   - La FFT es más intensiva en TLB y caché que GEMM para N grandes,
 *     porque accede a la memoria en patrones bit-reversal (no secuencial).
 *   - FP_OPS es aproximadamente N * log2(N) * 5 operaciones de doble precisión.
 *   - Para N < 1024 el kernel es tan corto que el overhead de PAPI distorsiona
 *     los contadores. Usa N >= 4096 para mediciones fiables.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <papi.h>
#include <mkl.h>
#include <mkl_dfti.h>

// ─────────────────────────────────────────────────────────────────────────────
// Parámetros de ejecución
// ─────────────────────────────────────────────────────────────────────────────
static const int N_REPS         = 7;
static const int N_WARMUP       = 2;
static const int N_COUNTER_REPS = 5;

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
// Utilidades
// ─────────────────────────────────────────────────────────────────────────────
static void fatal_papi(int retval, const char* ctx) {
    fprintf(stderr, "PAPI error en '%s': %s (código %d)\n",
            ctx, PAPI_strerror(retval), retval);
    exit(EXIT_FAILURE);
}

static void fatal_dfti(MKL_LONG st, const char* ctx) {
    fprintf(stderr, "MKL DFTI error en '%s': %s\n",
            ctx, DftiErrorMessage(st));
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
// Creación del descriptor DFTI
//
// Se crea UNA sola vez y se reutiliza en todas las mediciones.
// Esto es correcto metodológicamente: en aplicaciones reales el descriptor
// se crea en la inicialización y se reutiliza en el loop de cómputo.
// Medir la creación del descriptor contaminaría el benchmark con overhead
// de inicialización que no es representativo del rendimiento sostenido.
// ─────────────────────────────────────────────────────────────────────────────
static DFTI_DESCRIPTOR_HANDLE create_dfti(int n) {
    DFTI_DESCRIPTOR_HANDLE desc;
    MKL_LONG st;

    st = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)n);
    if (st != DFTI_NO_ERROR) fatal_dfti(st, "DftiCreateDescriptor");

    // In-place: la FFT sobreescribe el buffer de entrada con la salida.
    // Es más eficiente en memoria (no necesita buffer extra).
    st = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
    if (st != DFTI_NO_ERROR) fatal_dfti(st, "DftiSetValue PLACEMENT");

    // Sin factor de escala en la transformada directa (comportamiento estándar).
    st = DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0);
    if (st != DFTI_NO_ERROR) fatal_dfti(st, "DftiSetValue FORWARD_SCALE");

    st = DftiCommitDescriptor(desc);
    if (st != DFTI_NO_ERROR) fatal_dfti(st, "DftiCommitDescriptor");

    return desc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel FFT: transforma in-place el buffer x
//
// Importante: la FFT in-place DESTRUYE el contenido de x.
// Por eso el warm-up y cada repetición de contadores deben restaurar x
// antes de llamar a este kernel — de lo contrario estaríamos haciendo
// FFT de la FFT, que no es la operación que queremos medir.
// ─────────────────────────────────────────────────────────────────────────────
static void run_fft(DFTI_DESCRIPTOR_HANDLE desc,
                    std::vector<double>& x_re_im)
{
    MKL_LONG st = DftiComputeForward(desc, x_re_im.data());
    if (st != DFTI_NO_ERROR) fatal_dfti(st, "DftiComputeForward");
}

// ─────────────────────────────────────────────────────────────────────────────
// Warm-up de caché: lleva el buffer a L3/RAM en estado conocido
// Restaura el buffer original después del warm-up.
// ─────────────────────────────────────────────────────────────────────────────
static void cache_warmup(DFTI_DESCRIPTOR_HANDLE desc,
                          std::vector<double>& x_re_im,
                          const std::vector<double>& x_original)
{
    run_fft(desc, x_re_im);
    x_re_im = x_original;   // restaurar para que la siguiente medición sea válida
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-partición greedy
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
static double measure_time(DFTI_DESCRIPTOR_HANDLE desc,
                            std::vector<double>& x_re_im,
                            const std::vector<double>& x_original,
                            bool verbose)
{
    if (verbose)
        fprintf(stderr, "Fase TIEMPO: %d warm-ups + %d reps...\n", N_WARMUP, N_REPS);

    // Warm-up: estabiliza CPU y caché; restaura x después de cada uno
    for (int i = 0; i < N_WARMUP; ++i) {
        run_fft(desc, x_re_im);
        x_re_im = x_original;
    }

    std::vector<double> times(N_REPS);
    for (int i = 0; i < N_REPS; ++i) {
        // Restaurar x antes de cada medición (la FFT in-place lo destruye)
        x_re_im = x_original;
        long long t0 = PAPI_get_real_usec();
        run_fft(desc, x_re_im);
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
                                  DFTI_DESCRIPTOR_HANDLE desc,
                                  std::vector<double>& x_re_im,
                                  const std::vector<double>& x_original,
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
    if (ret != PAPI_OK) fatal_papi(ret, "create_eventset");
    for (auto& e : events) {
        ret = PAPI_add_event(EventSet, e.code);
        if (ret != PAPI_OK) fatal_papi(ret, e.name);
    }

    for (int r = 0; r < N_COUNTER_REPS; ++r) {
        // Warm-up de caché y restauración del buffer
        cache_warmup(desc, x_re_im, x_original);
        // Restaurar una vez más para la medición real
        x_re_im = x_original;

        ret = PAPI_start(EventSet);
        if (ret != PAPI_OK) fatal_papi(ret, "PAPI_start");

        run_fft(desc, x_re_im);

        ret = PAPI_stop(EventSet, runs[r].data());
        if (ret != PAPI_OK) fatal_papi(ret, "PAPI_stop");

        ret = PAPI_reset(EventSet);
        if (ret != PAPI_OK) fatal_papi(ret, "PAPI_reset");
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
        fprintf(stderr, "Uso: %s <N> [--verbose]\n", argv[0]);
        fprintf(stderr, "  N : puntos de FFT (preferiblemente potencia de 2)\n");
        return EXIT_FAILURE;
    }

    int  n       = std::atoi(argv[1]);
    bool verbose = false;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--verbose") verbose = true;

    if (n <= 0) {
        fprintf(stderr, "Error: N debe ser positivo.\n");
        return EXIT_FAILURE;
    }

    // ── PAPI init ─────────────────────────────────────────────────────────────
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) fatal_papi(retval, "PAPI_library_init");

    // ── Descriptor DFTI (se crea una vez, fuera de la medición) ──────────────
    if (verbose)
        fprintf(stderr, "Creando descriptor DFTI para N=%d...\n", n);
    DFTI_DESCRIPTOR_HANDLE desc = create_dfti(n);

    // ── Buffer de entrada: N números complejos = 2*N doubles (re, im, re, im…) -
    // Inicializar con valores no triviales para que la FFT haga trabajo real.
    std::vector<double> x_original(2 * n);
    for (int i = 0; i < n; ++i) {
        x_original[2*i]   = std::cos(2.0 * M_PI * i / n);   // parte real
        x_original[2*i+1] = std::sin(2.0 * M_PI * i / n);   // parte imaginaria
    }
    // Esta señal (exponencial compleja) tiene energía concentrada en un único
    // bin de frecuencia, lo que es representativo de casos reales.

    std::vector<double> x_re_im = x_original;   // copia de trabajo

    // ── Auto-partición de lotes ───────────────────────────────────────────────
    auto batches = build_batches(verbose);
    if (batches.empty()) {
        fprintf(stderr, "Error: no hay eventos medibles.\n");
        return EXIT_FAILURE;
    }

    // ── FASE 1: Tiempo limpio ─────────────────────────────────────────────────
    double t_median = measure_time(desc, x_re_im, x_original, verbose);
    // Formato CSV: N,lote_o_fase,evento,valor,rel_dev_%
    printf("%d,tiempo,MEDIANA,%.6f,0.00\n", n, t_median);

    // ── FASE 2: Contadores por lote ───────────────────────────────────────────
    for (size_t b = 0; b < batches.size(); ++b) {
        if (verbose)
            fprintf(stderr, "Fase CONTADORES lote %zu/%zu...\n",
                    b+1, batches.size());

        auto res = measure_batch(batches[b], desc, x_re_im, x_original, verbose);

        for (size_t e = 0; e < res.events.size(); ++e)
            printf("%d,%zu,%s,%lld,%.2f\n",
                   n, b+1,
                   res.events[e].name,
                   res.medians[e],
                   res.rel_devs[e]);
    }

    // ── Limpieza ──────────────────────────────────────────────────────────────
    DftiFreeDescriptor(&desc);
    PAPI_shutdown();
    return EXIT_SUCCESS;
}