// runtime_main.cpp
//
// Prototipo de runtime que consulta al modelo ML embebido ANTES de
// despachar una tarea a CPU o GPU (Objetivo específico 5 de la
// metodología). Este archivo:
//
//   1. Carga el bosque embebido (RF o XGBoost, según el macro de
//      compilación RUNTIME_MODEL_RF / RUNTIME_MODEL_XGB).
//   2. Recibe un lote de tareas (aquí: generadas sintéticamente o
//      leídas de un CSV — ver leerTareasDesdeCSV) con sus features
//      ya medidas ["modo replay", ver la ADVERTENCIA en
//      task_features.hpp].
//   3. Por cada tarea, mide el tiempo de inferencia (predictProba)
//      y decide el dispositivo.
//   4. Reporta latencia de decisión (media, p50, p99) — la métrica
//      que se debe comparar contra el tiempo de resolución del ILP
//      exacto para justificar el enfoque híbrido ILP+ML.
//   5. Deja un punto de enganche (TODO) donde iría el despacho real
//      a tus binarios GEMM/SpMV/FFT en CPU/GPU.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <optional>

#include "../include/task_features.hpp"
#include "../include/tree_model.hpp"

#if defined(RUNTIME_MODEL_RF)
    #include "../generated/rf_model.hpp"
    namespace gen = sched::generated::rf;
    static const char* MODEL_NAME = "Random Forest";
#elif defined(RUNTIME_MODEL_XGB)
    #include "../generated/xgb_model.hpp"
    namespace gen = sched::generated::xgb;
    static const char* MODEL_NAME = "XGBoost";
#else
    #error "Define RUNTIME_MODEL_RF o RUNTIME_MODEL_XGB al compilar"
#endif

using sched::TaskFeatures;
using sched::Device;
using sched::KernelType;

// ---------------------------------------------------------------------
// Carga de tareas desde CSV (modo "replay"): cada fila debe traer las
// 15 features en el mismo orden que FEATURE_COLS (ver task_features.hpp).
// Si el CSV trae una columna adicional numérica después de las 15
// (por ejemplo 'ilp_label', 0 o 1), se captura como etiqueta real para
// poder comparar la decisión del modelo contra ella. Cualquier otra
// columna extra (texto como 'kernel', u otras numéricas después de la
// primera) se ignora.
// ---------------------------------------------------------------------
struct LoadedTask {
    TaskFeatures features;
    int true_label = -1; // -1 = sin etiqueta disponible
};

std::vector<LoadedTask> leerTareasDesdeCSV(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("No se pudo abrir " + path);

    std::string header;
    std::getline(in, header);

    std::vector<LoadedTask> tasks;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<double> v;
        std::string cell;
        // Las primeras N_FEATURES columnas: features numéricas.
        for (std::size_t i = 0; i < sched::N_FEATURES && std::getline(ss, cell, ','); ++i) {
            v.push_back(std::stod(cell));
        }
        if (v.size() < sched::N_FEATURES) continue;

        LoadedTask lt;
        TaskFeatures& tf = lt.features;
        tf.kernel_encoded   = static_cast<KernelType>(static_cast<int>(v[0]));
        tf.N = v[1]; tf.nnz = v[2]; tf.sparsity = v[3];
        tf.op_intensity_cpu = v[4]; tf.op_intensity_gpu = v[5];
        tf.cpu_time_s = v[6]; tf.gpu_time_s = v[7];
        tf.cpu_energy_j = v[8]; tf.gpu_energy_j = v[9];
        tf.gflops_cpu = v[10]; tf.gflops_gpu = v[11];
        tf.bw_gbps_cpu = v[12]; tf.bw_gbps_gpu = v[13];
        tf.speedup = v[14];

        // Columna extra opcional: intentar leer 'ilp_label' como el
        // siguiente token numérico (0 o 1). Si no hay más tokens, o el
        // siguiente token no es numérico (p. ej. es 'GEMM'), se ignora.
        if (std::getline(ss, cell, ',')) {
            try {
                lt.true_label = std::stoi(cell);
            } catch (const std::exception&) {
                lt.true_label = -1;
            }
        }
        tasks.push_back(lt);
    }
    return tasks;
}

// Genera tareas sintéticas de ejemplo cuando no se pasa un CSV, solo
// para poder correr el binario "out of the box" y ver el mecanismo.
std::vector<LoadedTask> tareasDeEjemplo() {
    std::vector<LoadedTask> tasks;

    TaskFeatures gemm_grande;
    gemm_grande.kernel_encoded = KernelType::GEMM;
    gemm_grande.N = 16384;
    gemm_grande.cpu_time_s = 39.2; gemm_grande.gpu_time_s = 0.30;
    gemm_grande.cpu_energy_j = 5570; gemm_grande.gpu_energy_j = 100;
    gemm_grande.gflops_cpu = 180; gemm_grande.gflops_gpu = 9000;
    gemm_grande.bw_gbps_cpu = 40; gemm_grande.bw_gbps_gpu = 900;
    gemm_grande.op_intensity_cpu = 50; gemm_grande.op_intensity_gpu = 50;
    gemm_grande.speedup = gemm_grande.cpu_time_s / gemm_grande.gpu_time_s;
    tasks.push_back({gemm_grande, -1});

    TaskFeatures spmv_disperso;
    spmv_disperso.kernel_encoded = KernelType::SpMV;
    spmv_disperso.N = 1e6; spmv_disperso.sparsity = 1e-5;
    spmv_disperso.nnz = spmv_disperso.N * spmv_disperso.sparsity;
    spmv_disperso.cpu_time_s = 0.03; spmv_disperso.gpu_time_s = 0.08;
    spmv_disperso.cpu_energy_j = 1.0; spmv_disperso.gpu_energy_j = 3.0;
    spmv_disperso.speedup = spmv_disperso.cpu_time_s / spmv_disperso.gpu_time_s;
    tasks.push_back({spmv_disperso, -1});

    TaskFeatures fft_pequena;
    fft_pequena.kernel_encoded = KernelType::FFT;
    fft_pequena.N = 1024;
    fft_pequena.cpu_time_s = 1e-4; fft_pequena.gpu_time_s = 2e-2;
    fft_pequena.speedup = fft_pequena.cpu_time_s / fft_pequena.gpu_time_s;
    tasks.push_back({fft_pequena, -1});

    return tasks;
}

// ---------------------------------------------------------------------
// Punto de despacho real (TODO: reemplazar por llamadas a tus binarios).
// ---------------------------------------------------------------------
void despachar(const TaskFeatures& tf, Device decision) {
    // TODO: aquí va la integración real, por ejemplo:
    //
    //   switch (tf.kernel_encoded) {
    //     case KernelType::GEMM:
    //       decision == Device::GPU ? run_gemm_gpu(tf.N) : run_gemm_cpu(tf.N);
    //       break;
    //     case KernelType::SpMV:
    //       decision == Device::GPU ? run_spmv_gpu(tf.N, tf.sparsity)
    //                                : run_spmv_cpu(tf.N, tf.sparsity);
    //       break;
    //     case KernelType::FFT:
    //       decision == Device::GPU ? run_fft_gpu(tf.N) : run_fft_cpu(tf.N);
    //       break;
    //   }
    //
    // Por ahora solo se simula (no-op) para que el prototipo compile y
    // corra sin tus binarios de kernels.
    (void)tf; (void)decision;
}

int main(int argc, char** argv) {
    std::cout << "=== Runtime híbrido ILP+ML — modelo embebido: "
              << MODEL_NAME << " ===\n\n";

    sched::Forest forest = gen::buildEmbeddedForest();

    std::vector<LoadedTask> tasks;
    if (argc >= 2) {
        std::cout << "Cargando tareas desde CSV: " << argv[1] << "\n\n";
        tasks = leerTareasDesdeCSV(argv[1]);
    } else {
        std::cout << "Sin CSV de entrada: usando 3 tareas de ejemplo "
                     "(pásale un CSV como argv[1] para modo replay real)\n\n";
        tasks = tareasDeEjemplo();
    }

    if (tasks.empty()) {
        std::cerr << "No se cargaron tareas.\n";
        return 1;
    }

    std::vector<double> latencias_us;
    latencias_us.reserve(tasks.size());

    int n_cpu = 0, n_gpu = 0;

    // Métricas contra etiqueta real (solo si el CSV traía 'ilp_label').
    bool hay_etiquetas = false;
    int tp = 0, tn = 0, fp = 0, fn = 0; // clase positiva = GPU (1)
    std::vector<std::size_t> filas_falladas;

    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const auto& tf = tasks[i].features;

        auto t0 = std::chrono::high_resolution_clock::now();
        double proba = forest.predictProba(tf);
        Device decision = (proba >= 0.5) ? Device::GPU : Device::CPU;
        auto t1 = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        latencias_us.push_back(us);

        decision == Device::GPU ? ++n_gpu : ++n_cpu;
        despachar(tf, decision);

        int real = tasks[i].true_label;
        if (real == 0 || real == 1) {
            hay_etiquetas = true;
            int pred = (decision == Device::GPU) ? 1 : 0;
            if (pred == 1 && real == 1) ++tp;
            else if (pred == 0 && real == 0) ++tn;
            else if (pred == 1 && real == 0) ++fp;
            else if (pred == 0 && real == 1) ++fn;

            if (pred != real) filas_falladas.push_back(i);
        }
    }

    // Reporte de latencia de decisión (para comparar contra el tiempo
    // de resolución del ILP exacto en Python/solver).
    std::sort(latencias_us.begin(), latencias_us.end());
    double media = std::accumulate(latencias_us.begin(), latencias_us.end(), 0.0)
                   / latencias_us.size();
    double p50 = latencias_us[latencias_us.size() / 2];
    double p99 = latencias_us[static_cast<std::size_t>(latencias_us.size() * 0.99)];

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Tareas procesadas   : " << tasks.size() << "\n";
    std::cout << "Decisiones CPU/GPU  : " << n_cpu << " / " << n_gpu << "\n\n";
    std::cout << "Latencia de decisión (predictProba), en microsegundos:\n";
    std::cout << "  media : " << media << " us\n";
    std::cout << "  p50   : " << p50   << " us\n";
    std::cout << "  p99   : " << p99   << " us\n";
    std::cout << "  min   : " << latencias_us.front() << " us\n";
    std::cout << "  max   : " << latencias_us.back()  << " us\n";

    if (hay_etiquetas) {
        int total_etiquetadas = tp + tn + fp + fn;
        double accuracy = static_cast<double>(tp + tn) / total_etiquetadas;
        double precision_gpu = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall_gpu    = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;

        std::cout << "\n--- Comparación contra ilp_label (etiqueta real) ---\n";
        std::cout << "Tareas con etiqueta : " << total_etiquetadas << " / " << tasks.size() << "\n";
        std::cout << "Accuracy            : " << std::setprecision(4) << accuracy << "\n";
        std::cout << "Precision (GPU)     : " << precision_gpu << "\n";
        std::cout << "Recall (GPU)        : " << recall_gpu << "\n\n";
        std::cout << "Matriz de confusión:\n";
        std::cout << "                  predicho CPU   predicho GPU\n";
        std::cout << "  real CPU        " << std::setw(10) << tn
                   << "     " << std::setw(10) << fp << "\n";
        std::cout << "  real GPU        " << std::setw(10) << fn
                   << "     " << std::setw(10) << tp << "\n";

        if (!filas_falladas.empty()) {
            std::cout << "\nFilas mal clasificadas (fila = orden en el CSV, 0-indexado), "
                       << "hasta 20:\n";
            std::size_t shown = 0;
            for (auto idx : filas_falladas) {
                if (shown++ >= 20) {
                    std::cout << "  ... y " << (filas_falladas.size() - 20) << " más.\n";
                    break;
                }
                const auto& tf = tasks[idx].features;
                double proba = forest.predictProba(tf);
                std::cout << "  fila " << idx
                          << "  kernel=" << static_cast<int>(tf.kernel_encoded)
                          << "  N=" << tf.N
                          << "  real=" << (tasks[idx].true_label == 1 ? "GPU" : "CPU")
                          << "  predicho=" << (proba >= 0.5 ? "GPU" : "CPU")
                          << "  P(GPU)=" << std::setprecision(4) << proba << "\n";
            }
        }
    } else {
        std::cout << "\n(El CSV no trae columna 'ilp_label' como 16ª columna; "
                     "no se calcula accuracy contra etiqueta real.)\n";
    }

    return 0;
}
