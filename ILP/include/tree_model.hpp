// tree_model.hpp
//
// Motor de inferencia genérico para bosques de árboles (Random Forest o
// XGBoost) exportados como datos estáticos por export_model.py. No depende
// de sklearn, xgboost, ONNX Runnite ni ninguna librería externa: solo
// aritmética de punto flotante y comparaciones, por lo que compila y corre
// en cualquier nodo del clúster con un compilador C++17.
//
// Cómo se representa un árbol:
//   - Un árbol es un arreglo plano de Node.
//   - Cada nodo interno tiene: índice de feature, umbral, e índices de
//     los hijos izquierdo/derecho/"missing" (el que se sigue si el valor
//     de la feature es NaN).
//   - Cada hoja tiene is_leaf=true y leaf_value con el valor de salida:
//       * Random Forest -> probabilidad de clase GPU en esa hoja
//         (ya normalizada, igual que sklearn tree_.value).
//       * XGBoost        -> margen (log-odds) aportado por ese árbol,
//         que se SUMA a través de todos los árboles antes de aplicar
//         sigmoid (igual que la agregación interna de XGBoost).

#pragma once
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
#include "task_features.hpp"

namespace sched {

struct Node {
    int    feature;     // índice en [0, N_FEATURES) ; -1 si es hoja
    double threshold;
    int    left;
    int    right;
    int    missing;     // hijo a seguir si el valor es NaN
    bool   is_leaf;
    double leaf_value;  // solo válido si is_leaf == true
};

struct Tree {
    std::vector<Node> nodes; // nodes[0] es la raíz

    double eval(const std::array<double, N_FEATURES>& x) const {
        int idx = 0;
        // Límite de profundidad defensivo por si el árbol exportado
        // tuviera un ciclo accidental (no debería ocurrir nunca).
        for (int guard = 0; guard < 10000; ++guard) {
            const Node& n = nodes[idx];
            if (n.is_leaf) return n.leaf_value;

            double v = x[n.feature];
            if (std::isnan(v)) {
                idx = n.missing;
            } else if (v < n.threshold) {
                idx = n.left;
            } else {
                idx = n.right;
            }
        }
        throw std::runtime_error("Tree::eval: profundidad excedida (¿ciclo en el árbol exportado?)");
    }
};

enum class ModelKind { RANDOM_FOREST, XGBOOST };

struct Forest {
    ModelKind kind;
    double base_margin = 0.0;   // solo se usa si kind == XGBOOST
    std::vector<Tree> trees;

    // Probabilidad de que la tarea sea GPU-óptima, en [0, 1].
    double predictProba(const TaskFeatures& tf) const {
        auto x = tf.toVector();

        if (kind == ModelKind::RANDOM_FOREST) {
            double sum = 0.0;
            for (const auto& t : trees) sum += t.eval(x);
            return sum / static_cast<double>(trees.size());
        }

        // XGBOOST: suma de márgenes de todos los árboles + sigmoid.
        double margin = base_margin;
        for (const auto& t : trees) margin += t.eval(x);
        return 1.0 / (1.0 + std::exp(-margin));
    }

    Device predict(const TaskFeatures& tf, double threshold = 0.5) const {
        return predictProba(tf) >= threshold ? Device::GPU : Device::CPU;
    }
};

} // namespace sched
