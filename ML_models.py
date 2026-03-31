"""
  python ML_models.py --input dataset.csv
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 10

PALETTE = {
    'rf':   '#2E86AB',
    'xgb':  '#E84855',
    'cpu':  '#F18F01',
    'gpu':  '#5C6BC0',
    'bg':   '#F8F9FA',
    'grid': '#DEE2E6',
    'text': '#212529',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FEATURE_COLS = [
    # Características del problema
    'kernel_encoded',       # GEMM=0, FFT=1, SpMV=2  (LabelEncoder)
    'N',                    # tamaño del problema
    'sparsity',             # solo relevante para SpMV; 0.01 para el resto
    # Hardware CPU
    'cpu_cores',
    'cpu_freq_ghz',
    'cpu_mem_bw_gbps',
    # Hardware GPU
    'gpu_sm_count',
    'gpu_freq_ghz',
    'gpu_mem_bw_gbps',
    'gpu_mem_gb',
    # Métricas de rendimiento y energía medidas en el clúster
    'gflops_cpu',
    'gflops_gpu',
    'bw_gbps_cpu',
    'bw_gbps_gpu',
    'cpu_exec_time_ms',
    'gpu_exec_time_ms',
    'cpu_energy_j',
    'gpu_energy_j',
    # Feature derivada — codifica ventaja relativa de la GPU
    'speedup_ratio',        # T_cpu / T_gpu  (calculado en build_features)
]
TARGET_COL = 'ilp_label'

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Verificar columnas mínimas requeridas
    required = ['kernel', 'N', 'sparsity',
                'cpu_exec_time_ms', 'gpu_exec_time_ms',
                'cpu_energy_j', 'gpu_energy_j', TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")

    # Avisar sobre opcionales que faltan (no rompen la ejecución)
    optional = ['gflops_cpu', 'gflops_gpu', 'bw_gbps_cpu', 'bw_gbps_gpu',
                'cpu_cores', 'cpu_freq_ghz', 'cpu_mem_bw_gbps',
                'gpu_sm_count', 'gpu_freq_ghz', 'gpu_mem_bw_gbps', 'gpu_mem_gb']
    for col in optional:
        if col not in df.columns:
            print(f"  [AVISO] Columna '{col}' no encontrada → se rellena con 0")
            df[col] = 0.0

    return df.reset_index(drop=True)

def build_features(df: pd.DataFrame):
    df = df.copy()

    # Codificación del tipo de kernel
    le = LabelEncoder()
    df['kernel_encoded'] = le.fit_transform(df['kernel'])

    # Alias para las figuras de análisis por kernel
    df['kernel_type'] = df['kernel']

    # Feature derivada: ventaja de latencia GPU
    df['speedup_ratio'] = (
        df['cpu_exec_time_ms'] / (df['gpu_exec_time_ms'] + 1e-9)
    )

    # Rellenar NaN en columnas opcionales de rendimiento
    for col in ['gflops_cpu', 'gflops_gpu', 'bw_gbps_cpu', 'bw_gbps_gpu']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df, le

def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test

def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

def build_xgb(y_train: np.ndarray) -> XGBClassifier:
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        scale_pos_weight=n0 / (n1 + 1e-9),
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

def evaluate(model, X_test, y_test, X_train, y_train, name: str):
    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, pred)
    f1    = f1_score(y_test, pred, average='macro')
    auc   = roc_auc_score(y_test, proba)

    n_minority = min(int((y_train == 0).sum()), int((y_train == 1).sum()))
    n_splits   = min(5, max(2, n_minority))
    if n_splits < 5:
        print(f'  [AVISO] Clase minoritaria muy pequeña ({n_minority} '
              f'muestras en train) -> CV reducido a {n_splits} pliegues')
    cv    = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc', n_jobs=-1,
    )
    sep = '─' * (50 - len(name))
    print(f"── {name} {sep}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Macro  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    if hasattr(model, 'oob_score_'):
        print(f"  OOB Score : {model.oob_score_:.4f}")
    print(f"  CV-AUC    : {cv.mean():.4f} ± {cv.std():.4f}")
    print()
    print(classification_report(y_test, pred, target_names=['CPU (0)', 'GPU (1)']))
    return pred, proba, acc, f1, auc, cv

def fig1_evaluation(y_test, rf_pred, rf_proba, xgb_pred, xgb_proba,
                    rf_auc, xgb_auc, rf_acc, rf_f1, rf_cv,
                    xgb_acc, xgb_f1, xgb_cv, out: str):

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('CPU-GPU Task Assignment — Model Evaluation\n'
                 'Random Forest vs XGBoost',
                 fontsize=14, fontweight='bold', color=PALETTE['text'], y=0.98)

    # Matrices de confusión
    for ax, pred, cmap, title in [
        (axes[0,0], rf_pred,  'Blues', 'Confusion Matrix — Random Forest'),
        (axes[0,1], xgb_pred, 'Reds',  'Confusion Matrix — XGBoost'),
    ]:
        ax.set_facecolor(PALETTE['bg'])
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, pred),
            display_labels=['CPU (0)', 'GPU (1)']
        ).plot(ax=ax, colorbar=False, cmap=cmap)
        ax.set_title(title, fontweight='bold', pad=10)
        for t in ax.texts: t.set_fontsize(13)

    # Curvas ROC
    ax = axes[1, 0]; ax.set_facecolor(PALETTE['bg'])
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_proba)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
    ax.plot(fpr_rf,  tpr_rf,  color=PALETTE['rf'],  lw=2,
            label=f'Random Forest (AUC={rf_auc:.4f})')
    ax.plot(fpr_xgb, tpr_xgb, color=PALETTE['xgb'], lw=2,
            label=f'XGBoost       (AUC={xgb_auc:.4f})')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.fill_between(fpr_rf,  tpr_rf,  alpha=0.08, color=PALETTE['rf'])
    ax.fill_between(fpr_xgb, tpr_xgb, alpha=0.08, color=PALETTE['xgb'])
    ax.set_title('ROC Curves', fontweight='bold', pad=10)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.legend(fontsize=8); ax.grid(True, color=PALETTE['grid'], alpha=0.7)

    # Precision-Recall
    ax = axes[1, 1]; ax.set_facecolor(PALETTE['bg'])
    pr_rf,  rc_rf,  _ = precision_recall_curve(y_test, rf_proba)
    pr_xgb, rc_xgb, _ = precision_recall_curve(y_test, xgb_proba)
    baseline = y_test.mean()
    ax.plot(rc_rf,  pr_rf,  color=PALETTE['rf'],  lw=2, label='Random Forest')
    ax.plot(rc_xgb, pr_xgb, color=PALETTE['xgb'], lw=2, label='XGBoost')
    ax.axhline(baseline, color='gray', lw=1.2, ls='--',
               label=f'Baseline ({baseline:.2f})')
    ax.fill_between(rc_rf,  pr_rf,  alpha=0.07, color=PALETTE['rf'])
    ax.fill_between(rc_xgb, pr_xgb, alpha=0.07, color=PALETTE['xgb'])
    ax.set_title('Precision-Recall Curves', fontweight='bold', pad=10)
    ax.set(xlabel='Recall', ylabel='Precision')
    ax.legend(fontsize=8); ax.grid(True, color=PALETTE['grid'], alpha=0.7)

    # Comparativa de métricas
    ax = axes[2, 0]; ax.set_facecolor(PALETTE['bg'])
    mnames   = ['Accuracy', 'F1 Macro', 'ROC-AUC', 'CV-AUC (mean)']
    rf_vals  = [rf_acc,  rf_f1,  rf_auc,  rf_cv.mean()]
    xgb_vals = [xgb_acc, xgb_f1, xgb_auc, xgb_cv.mean()]
    xp = np.arange(len(mnames)); w = 0.35
    b1 = ax.bar(xp-w/2, rf_vals,  w, label='Random Forest',
                color=PALETTE['rf'],  alpha=0.85, edgecolor='white')
    b2 = ax.bar(xp+w/2, xgb_vals, w, label='XGBoost',
                color=PALETTE['xgb'], alpha=0.85, edgecolor='white')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(xp); ax.set_xticklabels(mnames, fontsize=8.5)
    yfloor = max(0.5, min(rf_vals + xgb_vals) - 0.05)
    ax.set_ylim(yfloor, 1.02)
    ax.set_title('Performance Metrics Comparison', fontweight='bold', pad=10)
    ax.set_ylabel('Score'); ax.legend(fontsize=8)
    ax.grid(True, axis='y', color=PALETTE['grid'], alpha=0.7)

    # Distribución de probabilidades predichas
    ax = axes[2, 1]; ax.set_facecolor(PALETTE['bg'])
    bins = np.linspace(0, 1, 35)
    ax.hist(rf_proba[y_test==0],  bins=bins, alpha=0.55, color=PALETTE['cpu'],
            label='RF — True CPU',  density=True)
    ax.hist(rf_proba[y_test==1],  bins=bins, alpha=0.55, color=PALETTE['gpu'],
            label='RF — True GPU',  density=True)
    ax.hist(xgb_proba[y_test==0], bins=bins, alpha=0.35, color=PALETTE['cpu'],
            label='XGB — True CPU', density=True, histtype='step', lw=2, ls='--')
    ax.hist(xgb_proba[y_test==1], bins=bins, alpha=0.35, color=PALETTE['gpu'],
            label='XGB — True GPU', density=True, histtype='step', lw=2, ls='--')
    ax.axvline(0.5, color='black', lw=1.2, ls=':', label='Threshold=0.5')
    ax.set_title('Predicted Probability Distributions', fontweight='bold', pad=10)
    ax.set(xlabel='P(GPU)', ylabel='Density')
    ax.legend(fontsize=7.5); ax.grid(True, color=PALETTE['grid'], alpha=0.7)

    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    print(f"  [OK] {out}")


def fig2_importance(rf_model, xgb_model, X_test, y_test, out: str):

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('Feature Importance Analysis\n'
                 'Random Forest (MDI) vs XGBoost (Gain)',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])

    # RF - Mean Decrease in Impurity
    rf_imp = pd.Series(
        rf_model.feature_importances_, index=FEATURE_COLS
    ).sort_values()
    ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
    colors = [PALETTE['rf'] if v > rf_imp.mean() else '#ADB5BD' for v in rf_imp]
    bars = ax.barh(rf_imp.index, rf_imp.values,
                   color=colors, edgecolor='white', height=0.7)
    ax.axvline(rf_imp.mean(), color='black', ls='--', lw=1, alpha=0.6, label='Mean')
    ax.set_title('Random Forest — MDI Importance', fontweight='bold', pad=10)
    ax.set_xlabel('MDI Importance'); ax.legend(fontsize=8)
    ax.grid(True, axis='x', color=PALETTE['grid'], alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        if w > 0.01:
            ax.text(w+0.001, bar.get_y()+bar.get_height()/2,
                    f'{w:.3f}', va='center', fontsize=7)

    # XGBoost - Importance by Gain (nativa)
    booster  = xgb_model.get_booster()
    gain_raw = booster.get_score(importance_type='gain')
    xgb_imp  = pd.Series(
        {FEATURE_COLS[int(k[1:])]: v for k, v in gain_raw.items()},
    ).reindex(FEATURE_COLS).fillna(0).sort_values()

    ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
    colors2 = [PALETTE['xgb'] if v > xgb_imp.mean() else '#ADB5BD' for v in xgb_imp]
    bars2 = ax.barh(xgb_imp.index, xgb_imp.values,
                    color=colors2, edgecolor='white', height=0.7)
    ax.axvline(xgb_imp.mean(), color='black', ls='--', lw=1, alpha=0.6, label='Mean')
    ax.set_title('XGBoost — Importance by Gain', fontweight='bold', pad=10)
    ax.set_xlabel('Gain (mejora total en función objetivo)'); ax.legend(fontsize=8)
    ax.grid(True, axis='x', color=PALETTE['grid'], alpha=0.6)
    for bar in bars2:
        w = bar.get_width()
        if w > xgb_imp.max() * 0.03:
            ax.text(w + xgb_imp.max()*0.01, bar.get_y()+bar.get_height()/2,
                    f'{w:.1f}', va='center', fontsize=7)

    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    print(f"  [OK] {out}")


def fig3_learning(rf_model, xgb_model, X, y, X_train, y_train,
                  df, idx_test, rf_pred, xgb_pred, y_test, out: str):

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('Learning Dynamics & Per-Kernel Analysis',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])

    # Curvas de aprendizaje
    for ax, model, color, title in [
        (axes[0,0], rf_model,  PALETTE['rf'],  'Learning Curve — Random Forest'),
        (axes[0,1], xgb_model, PALETTE['xgb'], 'Learning Curve — XGBoost'),
    ]:
        ax.set_facecolor(PALETTE['bg'])
        n_min = min(int((y_train == 0).sum()), int((y_train == 1).sum()))
        lc_cv = min(4, max(2, n_min))
        # Tamaño mínimo: garantiza al menos lc_cv muestras de la clase rara
        min_frac = max(0.15, (lc_cv * 2) / len(y_train))
        tsz, tsc, vsc = learning_curve(
            model, X_train, y_train, cv=lc_cv, scoring='roc_auc',
            train_sizes=np.linspace(min_frac, 1.0, 8), n_jobs=-1,
            error_score='raise',
        )
        ax.fill_between(tsz, tsc.mean(1)-tsc.std(1), tsc.mean(1)+tsc.std(1),
                        alpha=0.15, color=color)
        ax.fill_between(tsz, vsc.mean(1)-vsc.std(1), vsc.mean(1)+vsc.std(1),
                        alpha=0.15, color=PALETTE['gpu'])
        ax.plot(tsz, tsc.mean(1), 'o-', color=color,          lw=2, ms=5, label='Train AUC')
        ax.plot(tsz, vsc.mean(1), 's-', color=PALETTE['gpu'], lw=2, ms=5, label='Val AUC')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set(xlabel='Training samples', ylabel='ROC-AUC')
        ax.legend(fontsize=8); ax.grid(True, color=PALETTE['grid'], alpha=0.7)

    # 10-fold CV boxplot
    ax = axes[1, 0]; ax.set_facecolor(PALETTE['bg'])
    skf    = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_rf  = cross_val_score(rf_model,  X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_xgb = cross_val_score(xgb_model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    bplot  = ax.boxplot([cv_rf, cv_xgb], labels=['Random Forest', 'XGBoost'],
                        patch_artist=True, widths=0.4,
                        medianprops=dict(color='white', lw=2))
    bplot['boxes'][0].set_facecolor(PALETTE['rf']  + 'CC')
    bplot['boxes'][1].set_facecolor(PALETTE['xgb'] + 'CC')
    for i, (scores, color) in enumerate(
        [(cv_rf, PALETTE['rf']), (cv_xgb, PALETTE['xgb'])], 1
    ):
        ax.scatter(np.random.normal(i, 0.06, len(scores)), scores,
                   alpha=0.6, s=20, color=color, zorder=5)
    ax.set_title('10-Fold CV Distribution', fontweight='bold', pad=10)
    ax.set_ylabel('ROC-AUC (10-fold CV)')
    ax.grid(True, axis='y', color=PALETTE['grid'], alpha=0.7)
    ymin = max(0.5, min(cv_rf.min(), cv_xgb.min()) - 0.02)
    ax.set_ylim(ymin, 1.01)

    # Accuracy por tipo de kernel
    ax = axes[1, 1]; ax.set_facecolor(PALETTE['bg'])
    kernels     = df['kernel_type'].unique().tolist()
    test_kernel = df.loc[idx_test, 'kernel_type'].values
    rf_pk, xgb_pk = {}, {}
    for k in kernels:
        mask      = test_kernel == k
        rf_pk[k]  = accuracy_score(y_test[mask], rf_pred[mask])  if mask.sum() else 0
        xgb_pk[k] = accuracy_score(y_test[mask], xgb_pred[mask]) if mask.sum() else 0

    xk = np.arange(len(kernels)); w = 0.32
    b1 = ax.bar(xk-w/2, [rf_pk[k]  for k in kernels], w,
                label='Random Forest', color=PALETTE['rf'],  alpha=0.85, edgecolor='white')
    b2 = ax.bar(xk+w/2, [xgb_pk[k] for k in kernels], w,
                label='XGBoost',       color=PALETTE['xgb'], alpha=0.85, edgecolor='white')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xk); ax.set_xticklabels(kernels)
    yfloor = max(0.5, min(list(rf_pk.values()) + list(xgb_pk.values())) - 0.05)
    ax.set_ylim(yfloor, 1.02)
    ax.set_title('Accuracy per Kernel Type', fontweight='bold', pad=10)
    ax.set_ylabel('Accuracy'); ax.legend(fontsize=8)
    ax.grid(True, axis='y', color=PALETTE['grid'], alpha=0.7)
    for i, k in enumerate(kernels):
        ax.text(i, yfloor + 0.005, f'n={(test_kernel==k).sum()}',
                ha='center', fontsize=7.5, color='gray')

    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    print(f"  [OK] {out}")


def fig4_confidence(df, idx_test, y_test, rf_pred, rf_proba, xgb_proba, out: str):
    """
    Confianza del predictor vs speedup_ratio, coloreado por tipo de kernel.
    speedup_ratio = T_cpu / T_gpu — calculado en build_features().
    """
    kernels     = df['kernel_type'].unique().tolist()
    cmap_k      = {k: c for k, c in zip(
                    kernels, [PALETTE['rf'], PALETTE['xgb'], PALETTE['gpu']])}
    speedup     = df.loc[idx_test, 'speedup_ratio'].values
    test_kernel = df.loc[idx_test, 'kernel_type'].values

    # Acotar el eje X para evitar outliers extremos
    sp_max = np.percentile(speedup, 95)
    xlim   = [-0.5, max(sp_max, 2.0)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('Prediction Confidence vs Speedup Ratio',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])

    # RF: correcto vs error
    ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
    correct = (rf_pred == y_test)
    ax.scatter(speedup[correct],  rf_proba[correct],
               c=PALETTE['rf'],  alpha=0.40, s=12, label='Correcto')
    ax.scatter(speedup[~correct], rf_proba[~correct],
               c=PALETTE['cpu'], alpha=0.80, s=30, marker='x', lw=1.5, label='Error')
    ax.axhline(0.5, color='black', lw=1.2, ls='--', alpha=0.7, label='Umbral 0.5')
    ax.axvline(1.0, color='gray',  lw=1.0, ls=':',  alpha=0.7, label='Speedup = 1')
    ax.set_xlim(xlim)
    ax.set_title('RF: Confianza vs Speedup ratio', fontweight='bold', pad=10)
    ax.set(xlabel='Speedup ratio  (T_cpu / T_gpu)', ylabel='RF — P(GPU)')
    ax.legend(fontsize=8); ax.grid(True, color=PALETTE['grid'], alpha=0.6)

    # XGBoost: coloreado por kernel
    ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
    for k in kernels:
        mask = test_kernel == k
        if mask.sum() == 0: continue
        ax.scatter(speedup[mask], xgb_proba[mask],
                   c=cmap_k.get(k, '#888888'), alpha=0.55, s=18, label=k)
    ax.axhline(0.5, color='black', lw=1.2, ls='--', alpha=0.7, label='Umbral 0.5')
    ax.axvline(1.0, color='gray',  lw=1.0, ls=':',  alpha=0.7, label='Speedup = 1')
    ax.set_xlim(xlim)
    ax.set_title('XGBoost: Confianza vs Speedup (por kernel)',
                 fontweight='bold', pad=10)
    ax.set(xlabel='Speedup ratio  (T_cpu / T_gpu)', ylabel='XGBoost — P(GPU)')
    ax.legend(fontsize=8); ax.grid(True, color=PALETTE['grid'], alpha=0.6)

    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    print(f"  [OK] {out}")

def main():
    parser = argparse.ArgumentParser(
        description='CPU-GPU ML Predictor — Random Forest + XGBoost'
    )
    parser.add_argument('--input',      required=True,
                        help='CSV del clúster etiquetado por el ILP')
    parser.add_argument('--output-dir', default='.',
                        help='Directorio para las figuras (default: .)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out = args.output_dir

    print("=" * 65)
    print("  CPU-GPU TASK ASSIGNMENT — ML PIPELINE")
    print("=" * 65)
    print(f"  Dataset  : {args.input}")

    df = load_and_validate(args.input)
    print(f"  Muestras : {len(df)}")
    print(f"  CPU (0)  : {(df[TARGET_COL]==0).sum()} "
          f"({(df[TARGET_COL]==0).mean()*100:.1f}%)")
    print(f"  GPU (1)  : {(df[TARGET_COL]==1).sum()} "
          f"({(df[TARGET_COL]==1).mean()*100:.1f}%)")
    print(f"  Kernels  : {df['kernel'].value_counts().to_dict()}")
    print()

    df, le = build_features(df)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  Features     : {len(FEATURE_COLS)}")
    print(f"  Target       : '{TARGET_COL}'")
    print(f"  Kernel enc.  : {mapping}")
    print(f"  speedup_ratio: min={df['speedup_ratio'].min():.3f}  "
          f"max={df['speedup_ratio'].max():.3f}  "
          f"mean={df['speedup_ratio'].mean():.3f}")
    print()

    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print()

    rf_model = build_rf()
    rf_model.fit(X_train, y_train)

    xgb_model = build_xgb(y_train)
    xgb_model.fit(X_train, y_train)

    rf_pred,  rf_proba,  rf_acc,  rf_f1,  rf_auc,  rf_cv = evaluate(
        rf_model, X_test, y_test, X_train, y_train, "RANDOM FOREST"
    )
    xgb_pred, xgb_proba, xgb_acc, xgb_f1, xgb_auc, xgb_cv = evaluate(
        xgb_model, X_test, y_test, X_train, y_train, "XGBOOST"
    )

    X_full = df[FEATURE_COLS].values.astype(float)
    y_full = df[TARGET_COL].values.astype(int)

    fig1_evaluation(
        y_test, rf_pred, rf_proba, xgb_pred, xgb_proba,
        rf_auc, xgb_auc, rf_acc, rf_f1, rf_cv,
        xgb_acc, xgb_f1, xgb_cv,
        out=f'{out}/fig1_model_evaluation.png',
    )
    fig2_importance(
        rf_model, xgb_model, X_test, y_test,
        out=f'{out}/fig2_feature_importance.png',
    )
    fig3_learning(
        rf_model, xgb_model, X_full, y_full, X_train, y_train,
        df, idx_test, rf_pred, xgb_pred, y_test,
        out=f'{out}/fig3_learning_dynamics.png',
    )
    fig4_confidence(
        df, idx_test, y_test, rf_pred, rf_proba, xgb_proba,
        out=f'{out}/fig4_confidence_analysis.png',
    )

    print()
    print("=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)
    print(f"  {'Modelo':<26} {'Accuracy':>10} {'F1 Macro':>10} {'ROC-AUC':>10}")
    print(f"  {'─'*56}")
    print(f"  {'Random Forest':<26} {rf_acc:>10.4f} {rf_f1:>10.4f} {rf_auc:>10.4f}")
    print(f"  {'XGBoost':<26} {xgb_acc:>10.4f} {xgb_f1:>10.4f} {xgb_auc:>10.4f}")
    print()
    print(f"  Figuras guardadas en: {out}/")
    print("=" * 65)

if __name__ == '__main__':
    main()