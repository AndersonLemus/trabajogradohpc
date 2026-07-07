// validate_predictions.cpp
//
// Verifica que la reimplementación en C++ del bosque exportado produce
// las MISMAS probabilidades que sklearn/xgboost en Python, sobre una
// muestra real del conjunto de prueba (generada por export_model.py).
//
// Esto es un paso OBLIGATORIO antes de confiar en el runtime: una
// reimplementación manual de inferencia de árboles es fácil de tener
// sutilmente mal (dirección de "missing", orden de features, manejo
// del base_score de XGBoost, etc.), y este programa es la red de
// seguridad que lo detecta.
//
// Uso:
//   ./validate_predictions <ruta_al_csv_de_validacion> [tolerancia]
//
// El CSV tiene columnas: f0,f1,...,f14,python_proba  (sin encabezado
// de "kernel_encoded" etc., solo los 15 valores numéricos en orden).

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

#include "../include/task_features.hpp"
#include "../include/tree_model.hpp"

// Selecciona aquí qué modelo exportado se está validando.
// Comenta/descomenta según corresponda, o compila dos binarios distintos
// (ver CMakeLists.txt: validate_rf y validate_xgb).
#if defined(VALIDATE_RF)
    #include "../generated/rf_model.hpp"
    namespace gen = sched::generated::rf;
#elif defined(VALIDATE_XGB)
    #include "../generated/xgb_model.hpp"
    namespace gen = sched::generated::xgb;
#else
    #error "Define VALIDATE_RF o VALIDATE_XGB al compilar (-DVALIDATE_RF)"
#endif

struct Row {
    std::array<double, sched::N_FEATURES> x;
    double python_proba;
};

static std::vector<Row> loadCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("No se pudo abrir " + path);

    std::string header;
    std::getline(in, header); // descartar encabezado

    std::vector<Row> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        Row r{};
        for (std::size_t i = 0; i < sched::N_FEATURES; ++i) {
            std::getline(ss, cell, ',');
            r.x[i] = std::stod(cell);
        }
        std::getline(ss, cell, ','); // python_proba
        r.python_proba = std::stod(cell);
        rows.push_back(r);
    }
    return rows;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <csv_validacion> [tolerancia=1e-6]\n";
        return 2;
    }
    std::string csv_path = argv[1];
    double tol = (argc >= 3) ? std::stod(argv[2]) : 1e-6;

    auto rows = loadCsv(csv_path);
    sched::Forest forest = gen::buildEmbeddedForest();

    double max_diff = 0.0, sum_diff = 0.0;
    std::size_t n_fail = 0;
    std::vector<std::tuple<std::size_t, double, double, double>> worst; // idx, diff, python, cpp

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        sched::TaskFeatures tf; // no la usamos directamente: evaluamos el
        // vector crudo para no depender del mapping semántico aquí.
        // Construimos la predicción llamando a la lógica interna del
        // Forest sobre el arreglo ya ordenado (toVector-compatible).
        double cpp_proba;
        {
            // Reutilizamos Forest::predictProba pasando un TaskFeatures
            // "sintético": como toVector() reproduce el mismo orden,
            // copiamos los 15 valores en los campos correspondientes.
            tf.kernel_encoded   = static_cast<sched::KernelType>(static_cast<int>(r.x[0]));
            tf.N                = r.x[1];
            tf.nnz              = r.x[2];
            tf.sparsity         = r.x[3];
            tf.op_intensity_cpu = r.x[4];
            tf.op_intensity_gpu = r.x[5];
            tf.cpu_time_s       = r.x[6];
            tf.gpu_time_s       = r.x[7];
            tf.cpu_energy_j     = r.x[8];
            tf.gpu_energy_j     = r.x[9];
            tf.gflops_cpu       = r.x[10];
            tf.gflops_gpu       = r.x[11];
            tf.bw_gbps_cpu      = r.x[12];
            tf.bw_gbps_gpu      = r.x[13];
            tf.speedup          = r.x[14];
            cpp_proba = forest.predictProba(tf);
        }

        double diff = std::fabs(cpp_proba - r.python_proba);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (diff > tol) {
            ++n_fail;
            worst.emplace_back(i, diff, r.python_proba, cpp_proba);
        }
    }

    double mean_diff = rows.empty() ? 0.0 : sum_diff / rows.size();

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Filas evaluadas   : " << rows.size() << "\n";
    std::cout << "Diferencia máxima : " << max_diff  << "\n";
    std::cout << "Diferencia media  : " << mean_diff << "\n";
    std::cout << "Tolerancia        : " << tol << "\n";
    std::cout << "Filas fuera de tolerancia: " << n_fail << " / " << rows.size() << "\n";

    if (n_fail > 0) {
        std::sort(worst.begin(), worst.end(),
                  [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });

        std::cout << "\nFilas mas problematicas (hasta 10):\n";
        std::size_t shown = 0;
        for (const auto& [idx, diff, py, cpp] : worst) {
            if (shown++ >= 10) break;
            std::cout << "  fila " << idx << "  diff=" << diff
                      << "  python=" << py << "  cpp=" << cpp << "\n";
            const auto& x = rows[idx].x;
            bool has_nan = false, has_inf = false;
            for (double v : x) {
                if (std::isnan(v)) has_nan = true;
                if (std::isinf(v)) has_inf = true;
            }
            std::cout << "    contiene NaN: " << (has_nan ? "SI" : "no")
                      << "   contiene Inf: " << (has_inf ? "SI" : "no") << "\n";
            std::cout << "    features: [";
            for (std::size_t k = 0; k < x.size(); ++k) {
                std::cout << x[k] << (k + 1 < x.size() ? ", " : "");
            }
            std::cout << "]\n\n";
        }

        std::cout << "[FALLA] La reimplementación en C++ NO coincide con Python "
                     "dentro de la tolerancia. No usar este modelo en el runtime "
                     "hasta resolver la discrepancia.\n"
                     "Ejecuta tambien: python3 cross_check_xgb.py --model xgb_model.pkl "
                     "--X-test X_test.npy   para aislar si el bug esta en el algoritmo "
                     "(Python puro) o en la traduccion a C++.\n";
        return 1;
    }
    std::cout << "\n[OK] La reimplementación en C++ coincide con Python.\n";
    return 0;
}
