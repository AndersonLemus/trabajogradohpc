import argparse
import json
import joblib
import numpy as np

FEATURE_COLS = [
    'kernel_encoded', 'N', 'nnz', 'sparsity',
    'op_intensity_cpu', 'op_intensity_gpu',
    'cpu_time_s', 'gpu_time_s',
    'cpu_energy_j', 'gpu_energy_j',
    'gflops_cpu', 'gflops_gpu',
    'bw_gbps_cpu', 'bw_gbps_gpu',
    'speedup',
]
N_FEATURES = len(FEATURE_COLS)


def extract_rf_trees(rf_model):
    trees = []
    for est in rf_model.estimators_:
        t = est.tree_
        nodes = []
        for i in range(t.node_count):
            if t.children_left[i] == -1: 
                counts = t.value[i][0]
                total = counts.sum()
                p_gpu = float(counts[1] / total) if total > 0 else 0.0
                nodes.append(dict(feature=-1, threshold=0.0,
                                   left=-1, right=-1, missing=-1,
                                   is_leaf=True, leaf_value=p_gpu))
            else:
                nodes.append(dict(
                    feature=int(t.feature[i]),
                    threshold=float(t.threshold[i]),
                    left=int(t.children_left[i]),
                    right=int(t.children_right[i]),
                    missing=int(t.children_left[i]),
                    is_leaf=False, leaf_value=0.0,
                ))
        trees.append(nodes)
    return trees, 0.0

def extract_xgb_trees(xgb_model):
    booster = xgb_model.get_booster()
    dumps = booster.get_dump(dump_format='json')

    try:
        cfg = json.loads(booster.save_config())
        raw = cfg['learner']['learner_model_param']['base_score']
        base_score = float(str(raw).strip('[]'))
    except Exception as e:
        print(f"  [AVISO] No se pudo parsear base_score ({e}); usando 0.5")
        base_score = 0.5
    eps = 1e-12
    base_score = min(max(base_score, eps), 1 - eps)
    base_margin = float(np.log(base_score / (1.0 - base_score)))

    trees = []
    for dump in dumps:
        raw = json.loads(dump)
        flat = []
        id_map = {}

        def visit(node):
            new_id = len(flat)
            flat.append(None) 
            id_map[node['nodeid']] = new_id

            if 'leaf' in node:
                flat[new_id] = dict(feature=-1, threshold=0.0,
                                     left=-1, right=-1, missing=-1,
                                     is_leaf=True,
                                     leaf_value=float(node['leaf']))
                return new_id

            fname = node['split']
            fidx = int(fname[1:]) if fname.startswith('f') else int(fname)

            yes_id = visit_child(node, node['yes'])
            no_id = visit_child(node, node['no'])
            missing_raw = node.get('missing', node['yes'])

            flat[new_id] = dict(
                feature=fidx,
                threshold=float(node['split_condition']),
                left=yes_id, right=no_id,
                missing=yes_id if missing_raw == node['yes'] else no_id,
                is_leaf=False, leaf_value=0.0,
            )
            return new_id

        def visit_child(parent, child_nodeid):
            child = next(c for c in parent['children'] if c['nodeid'] == child_nodeid)
            return visit(child)

        visit(raw)
        trees.append(flat)

    return trees, base_margin

def emit_header(trees, base_margin, kind, out_path):
    kind_enum = "RANDOM_FOREST" if kind == "rf" else "XGBOOST"
    namespace = kind

    lines = []
    lines.append("// AUTOGENERADO por export_model.py — no editar a mano.")
    lines.append(f"// Modelo: {kind_enum} | {len(trees)} árboles | "
                  f"{N_FEATURES} features")
    lines.append("#pragma once")
    lines.append('#include "../include/tree_model.hpp"')
    lines.append("")
    lines.append(f"namespace sched {{ namespace generated {{ namespace {namespace} {{")
    lines.append("")

    for ti, nodes in enumerate(trees):
        lines.append(f"static const std::vector<Node> tree_{ti}_nodes = {{")
        for n in nodes:
            leaf = "true" if n["is_leaf"] else "false"
            lines.append(
                f"    {{ {n['feature']}, {n['threshold']!r}, "
                f"{n['left']}, {n['right']}, {n['missing']}, "
                f"{leaf}, {n['leaf_value']!r} }},"
            )
        lines.append("};")
        lines.append("")

    lines.append("inline sched::Forest buildEmbeddedForest() {")
    lines.append("    sched::Forest f;")
    lines.append(f"    f.kind = sched::ModelKind::{kind_enum};")
    lines.append(f"    f.base_margin = {base_margin!r};")
    lines.append(f"    f.trees.reserve({len(trees)});")
    for ti in range(len(trees)):
        lines.append(f"    f.trees.push_back(sched::Tree{{ tree_{ti}_nodes }});")
    lines.append("    return f;")
    lines.append("}")
    lines.append("")
    lines.append(f"}}}} }} // namespace sched::generated::{namespace}")
    lines.append("")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  [OK] Header escrito en {out_path}  "
          f"({len(trees)} árboles, "
          f"{sum(len(t) for t in trees)} nodos totales)")

def emit_validation_csv(model, X_test, out_path, n_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test[idx]
    proba = model.predict_proba(X_sample)[:, 1]

    with open(out_path, "w") as fh:
        fh.write(",".join(FEATURE_COLS) + ",python_proba\n")
        for row, p in zip(X_sample, proba):
            fh.write(",".join(repr(float(v)) for v in row) + f",{float(p)!r}\n")
    print(f"  [OK] CSV de validación escrito en {out_path} ({n} filas)")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Ruta al .pkl del modelo (joblib.dump)")
    ap.add_argument("--kind", required=True, choices=["rf", "xgb"])
    ap.add_argument("--out", required=True, help="Ruta de salida del header .hpp")
    ap.add_argument("--X-test", default=None, help="Ruta a X_test.npy (opcional, para validación)")
    ap.add_argument("--y-test", default=None, help="Ruta a y_test.npy (opcional, no usado aún)")
    ap.add_argument("--sample-out", default=None, help="Ruta de salida del CSV de validación")
    args = ap.parse_args()

    print(f"Cargando modelo ({args.kind}) desde {args.model} ...")
    model = joblib.load(args.model)

    if args.kind == "rf":
        trees, base_margin = extract_rf_trees(model)
    else:
        trees, base_margin = extract_xgb_trees(model)

    emit_header(trees, base_margin, args.kind, args.out)

    if args.X_test and args.sample_out:
        X_test = np.load(args.X_test)
        emit_validation_csv(model, X_test, args.sample_out)
    elif args.X_test or args.sample_out:
        print("  [AVISO] Se necesitan --X-test Y --sample-out juntos para "
              "generar el CSV de validación; se omite este paso.")


if __name__ == "__main__":
    main()