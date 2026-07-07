// task_features.hpp
//
// Estructura del vector de características que consume el modelo embebido.
// El ORDEN de los campos debe coincidir EXACTAMENTE con FEATURE_COLS en
// ML_models.py, porque los árboles exportados referencian las features
// por índice posicional (f0, f1, ..., f14), no por nombre.
//
//   FEATURE_COLS = [
//       'kernel_encoded', 'N', 'nnz', 'sparsity',
//       'op_intensity_cpu', 'op_intensity_gpu',
//       'cpu_time_s', 'gpu_time_s',
//       'cpu_energy_j', 'gpu_energy_j',
//       'gflops_cpu', 'gflops_gpu',
//       'bw_gbps_cpu', 'bw_gbps_gpu',
//       'speedup',
//   ]
//
// ADVERTENCIA (léase antes de usar en producción):
// Cinco de estas quince features (gpu_time_s, gpu_energy_j, gflops_gpu,
// bw_gbps_gpu, speedup) se MIDEN después de ejecutar la tarea en GPU.
// En un runtime real que debe decidir ANTES de ejecutar, esos valores no
// existen todavía. Este prototipo opera en modo "replay": consume vectores
// de características ya medidos (p. ej. tomados del dataset del ILP) para
// demostrar el mecanismo de consulta al modelo y medir su latencia de
// decisión frente al ILP exacto. Sustituir esas 5 features por estimaciones
// a priori (o reentrenar el modelo solo con features disponibles antes de
// ejecutar) es trabajo futuro — ver Sección 4.4.5 / Conclusiones.

#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace sched {

constexpr std::size_t N_FEATURES = 15;

// Debe coincidir con el LabelEncoder de sklearn ajustado sobre
// df['kernel'] en build_features(). Verifica el mapping impreso por
// ML_models.py ("Kernel enc. : {...}") antes de asumir este orden:
// por defecto LabelEncoder ordena alfabéticamente las categorías.
enum class KernelType : int {
    FFT  = 0,
    GEMM = 1,
    SpMV = 2,
};

struct TaskFeatures {
    KernelType kernel_encoded;
    double N               = 0.0;
    double nnz             = 0.0;
    double sparsity        = 0.0;
    double op_intensity_cpu = 0.0;
    double op_intensity_gpu = 0.0;
    double cpu_time_s      = 0.0;
    double gpu_time_s      = 0.0;   // ver ADVERTENCIA arriba
    double cpu_energy_j    = 0.0;
    double gpu_energy_j    = 0.0;   // ver ADVERTENCIA arriba
    double gflops_cpu      = 0.0;
    double gflops_gpu      = 0.0;   // ver ADVERTENCIA arriba
    double bw_gbps_cpu     = 0.0;
    double bw_gbps_gpu     = 0.0;   // ver ADVERTENCIA arriba
    double speedup         = 0.0;   // ver ADVERTENCIA arriba

    // Serializa en el mismo orden posicional que FEATURE_COLS.
    std::array<double, N_FEATURES> toVector() const {
        return {
            static_cast<double>(kernel_encoded),
            N, nnz, sparsity,
            op_intensity_cpu, op_intensity_gpu,
            cpu_time_s, gpu_time_s,
            cpu_energy_j, gpu_energy_j,
            gflops_cpu, gflops_gpu,
            bw_gbps_cpu, bw_gbps_gpu,
            speedup,
        };
    }
};

enum class Device : int { CPU = 0, GPU = 1 };

inline std::string toString(Device d) {
    return d == Device::GPU ? "GPU" : "CPU";
}

} // namespace sched
