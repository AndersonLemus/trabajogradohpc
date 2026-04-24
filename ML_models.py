"""
  python cpu_gpu_ml_models.py --input dataset.csv
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5        # pliegues para Optuna (rápido) y evaluación final
OPTUNA_TRIALS = 100       # nº de trials por modelo (sobreescribible con --trials)
OPTUNA_CV     = 5        # pliegues dentro de Optuna

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
    'kernel_encoded',   # GEMM=0, FFT=1, SpMV=2  (LabelEncoder)
    'N',                # tamaño del problema
    'nnz',              # entradas no cero (0 para GEMM/FFT)
    'sparsity',         # densidad de la matriz (0 para GEMM/FFT)
    'op_intensity_cpu', # FLOPs/byte medido en CPU
    'op_intensity_gpu', # FLOPs/byte medido en GPU
    'cpu_time_s',       # tiempo de ejecución CPU (segundos)
    'gpu_time_s',       # tiempo de ejecución GPU (segundos)
    'cpu_energy_j',     # energía CPU (julios)
    'gpu_energy_j',     # energía GPU (julios)
    'gflops_cpu',       # rendimiento efectivo CPU
    'gflops_gpu',       # rendimiento efectivo GPU
    'bw_gbps_cpu',      # ancho de banda medido CPU
    'bw_gbps_gpu',      # ancho de banda medido GPU
    'speedup',          # T_cpu / T_gpu  (feature derivada)
]
TARGET_COL = 'ilp_label'

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        'kernel', 
        'N', 
        'nnz', 
        'sparsity',
        'cpu_time_s', 
        'gpu_time_s',
        'cpu_energy_j', 
        'gpu_energy_j',
        'gflops_cpu', 
        'gflops_gpu',
        'bw_gbps_cpu', 
        'bw_gbps_gpu',
        TARGET_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")
    for col in ['op_intensity_cpu', 'op_intensity_gpu']:
        if col not in df.columns:
            if 'op_intensity' in df.columns:
                print(f"  [AVISO] '{col}' no encontrada → usando 'op_intensity'")
                df[col] = df['op_intensity']
            else:
                print(f"  [AVISO] '{col}' no encontrada → se rellena con 0")
                df[col] = 0.0
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df['kernel_encoded'] = le.fit_transform(df['kernel'])
    df['kernel_type']    = df['kernel']
    df['speedup'] = df['cpu_time_s'] / (df['gpu_time_s'] + 1e-9)
    # nnz = 0 para GEMM y FFT si no se rellenó antes
    if 'nnz' not in df.columns:
        df['nnz'] = 0
    df['nnz'] = df['nnz'].fillna(0).astype(float)
    # Garantizar que columnas de rendimiento no tengan NaN
    for col in ['gflops_cpu', 'gflops_gpu', 'bw_gbps_cpu', 'bw_gbps_gpu',
                'op_intensity_cpu', 'op_intensity_gpu']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df, le


def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test

def _cv_auc(model, X_tr, y_tr, n_splits: int) -> float:
    n_minority = min(int((y_tr == 0).sum()), int((y_tr == 1).sum()))
    folds = min(n_splits, max(2, n_minority))
    skf   = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_tr, y_tr,
                             cv=skf, scoring='roc_auc', n_jobs=-1)
    return float(scores.mean())

def optimize_rf(X_train: np.ndarray, y_train: np.ndarray,
                n_trials: int = OPTUNA_TRIALS) -> dict:
    """
    Busca los mejores hiperparámetros para RandomForest con Optuna TPE.
    Espacio de búsqueda:
      n_estimators       [100, 800]
      max_depth          [3, 30] o None
      min_samples_split  [2, 20]
      min_samples_leaf   [1, 10]
      max_features       ['sqrt', 'log2', 0.3, 0.5, 0.7]
      max_samples        [0.5, 1.0]  (bootstrap fraction)
    """
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())

    def objective(trial: optuna.Trial) -> float:
        max_depth_flag = trial.suggest_categorical('max_depth_none', [True, False])
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 800, step=50),
            'max_depth':         None if max_depth_flag else
                                 trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':      trial.suggest_categorical(
                                     'max_features',
                                     ['sqrt', 'log2', 0.3, 0.5, 0.7]),
            'max_samples':       trial.suggest_float('max_samples', 0.5, 1.0),
            'class_weight':      'balanced',
            'n_jobs':            -1,
            'random_state':      RANDOM_STATE,
        }
        model = RandomForestClassifier(**params)
        return _cv_auc(model, X_train, y_train, OPTUNA_CV)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params.copy()
    # Reconstruir max_depth desde los dos params separados
    if best.pop('max_depth_none'):
        best['max_depth'] = None
    best['class_weight'] = 'balanced'
    best['oob_score']    = True
    best['n_jobs']       = -1
    best['random_state'] = RANDOM_STATE

    print(f"  [RF]  Mejor AUC en búsqueda : {study.best_value:.4f}")
    print(f"  [RF]  Mejores hiperparámetros:")
    for k, v in best.items():
        if k not in ('n_jobs', 'random_state', 'oob_score'):
            print(f"          {k:<25} = {v}")

    return best, study


def optimize_xgb(X_train: np.ndarray, y_train: np.ndarray,
                 n_trials: int = OPTUNA_TRIALS) -> dict:
    """
    Busca los mejores hiperparámetros para XGBoost con Optuna TPE.
    Espacio de búsqueda:
      n_estimators      [100, 800]
      max_depth         [3, 12]
      learning_rate     [0.005, 0.3]  (log-uniforme)
      subsample         [0.5, 1.0]
      colsample_bytree  [0.4, 1.0]
      colsample_bylevel [0.4, 1.0]
      reg_lambda        [1e-3, 10.0]  (log-uniforme)
      reg_alpha         [1e-4, 5.0]   (log-uniforme)
      min_child_weight  [1, 20]
      gamma             [0.0, 5.0]
      grow_policy       ['depthwise', 'lossguide']
    """
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    spw = n0 / (n1 + 1e-9)   # scale_pos_weight fijo según desbalance

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators':       trial.suggest_int('n_estimators', 100, 800, step=50),
            'max_depth':          trial.suggest_int('max_depth', 3, 12),
            'learning_rate':      trial.suggest_float('learning_rate',
                                                       0.005, 0.3, log=True),
            'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel':  trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'reg_lambda':         trial.suggest_float('reg_lambda',
                                                       1e-3, 10.0, log=True),
            'reg_alpha':          trial.suggest_float('reg_alpha',
                                                       1e-4, 5.0, log=True),
            'min_child_weight':   trial.suggest_int('min_child_weight', 1, 20),
            'gamma':              trial.suggest_float('gamma', 0.0, 5.0),
            'grow_policy':        trial.suggest_categorical(
                                      'grow_policy',
                                      ['depthwise', 'lossguide']),
            'scale_pos_weight':   spw,
            'eval_metric':        'logloss',
            'random_state':       RANDOM_STATE,
            'n_jobs':             -1,
            'verbosity':          0,
        }
        model = XGBClassifier(**params)
        return _cv_auc(model, X_train, y_train, OPTUNA_CV)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params.copy()
    best['scale_pos_weight'] = spw
    best['eval_metric']      = 'logloss'
    best['random_state']     = RANDOM_STATE
    best['n_jobs']           = -1
    best['verbosity']        = 0

    print(f"  [XGB] Mejor AUC en búsqueda : {study.best_value:.4f}")
    print(f"  [XGB] Mejores hiperparámetros:")
    for k, v in best.items():
        if k not in ('n_jobs', 'random_state', 'verbosity',
                     'eval_metric', 'scale_pos_weight'):
            print(f"          {k:<25} = {v}")

    return best, study

def build_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def build_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(**params)

def evaluate(model, X_test, y_test, X_train, y_train, name: str):
    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, pred)
    f1    = f1_score(y_test, pred, average='macro')
    auc   = roc_auc_score(y_test, proba)

    n_minority = min(int((y_train == 0).sum()), int((y_train == 1).sum()))
    n_splits   = min(CV_FOLDS, max(2, n_minority))
    cv = cross_val_score(
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

# Figuras
def fig1_evaluation(y_test, rf_pred, rf_proba, xgb_pred, xgb_proba,
                    rf_auc, xgb_auc, rf_acc, rf_f1, rf_cv,
                    xgb_acc, xgb_f1, xgb_cv, out: str):
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('CPU-GPU Task Assignment — Model Evaluation\n'
                 'Random Forest vs XGBoost  (Optuna HPO)',
                 fontsize=14, fontweight='bold', color=PALETTE['text'], y=0.98)

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
                 'Random Forest (MDI) vs XGBoost (Gain)  [Optuna HPO]',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])

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
    ax.set_xlabel('Gain'); ax.legend(fontsize=8)
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
    fig.suptitle('Learning Dynamics & Per-Kernel Analysis  [Optuna HPO]',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])

    for ax, model, color, title in [
        (axes[0,0], rf_model,  PALETTE['rf'],  'Learning Curve — Random Forest'),
        (axes[0,1], xgb_model, PALETTE['xgb'], 'Learning Curve — XGBoost'),
    ]:
        ax.set_facecolor(PALETTE['bg'])
        n_min   = min(int((y_train == 0).sum()), int((y_train == 1).sum()))
        lc_cv   = min(4, max(2, n_min))
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
    ax.set_title(f'{CV_FOLDS}-Fold CV Distribution', fontweight='bold', pad=10)
    ax.set_ylabel('ROC-AUC')
    ax.grid(True, axis='y', color=PALETTE['grid'], alpha=0.7)
    ymin = max(0.5, min(cv_rf.min(), cv_xgb.min()) - 0.02)
    ax.set_ylim(ymin, 1.01)

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
    kernels     = df['kernel_type'].unique().tolist()
    cmap_k      = {k: c for k, c in zip(
                    kernels, [PALETTE['rf'], PALETTE['xgb'], PALETTE['gpu']])}
    speedup     = df.loc[idx_test, 'speedup'].values
    test_kernel = df.loc[idx_test, 'kernel_type'].values
    sp_max      = np.percentile(speedup, 95)
    xlim        = [-0.5, max(sp_max, 2.0)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('Prediction Confidence vs Speedup Ratio  [Optuna HPO]',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])

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

# Figura Optuna
def fig5_optuna(rf_study, xgb_study, out: str):
    """
    Dos subplots:
      - Izquierda : historia de trials (valor objetivo por trial) para RF y XGBoost
      - Derecha   : importancia de hiperparámetros estimada por Optuna (fANOVA)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('Optuna Hyperparameter Optimization History',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])

    # Historia de optimización
    ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
    for study, color, label in [
        (rf_study,  PALETTE['rf'],  'Random Forest'),
        (xgb_study, PALETTE['xgb'], 'XGBoost'),
    ]:
        vals = [t.value for t in study.trials if t.value is not None]
        # Mejor acumulado hasta el trial i
        best_so_far = np.maximum.accumulate(vals)
        ax.plot(range(1, len(vals)+1), vals,
                color=color, alpha=0.30, lw=1)
        ax.plot(range(1, len(best_so_far)+1), best_so_far,
                color=color, lw=2, label=f'{label} (best={best_so_far[-1]:.4f})')
    ax.set_title('Optimization History (ROC-AUC en CV)', fontweight='bold', pad=10)
    ax.set(xlabel='Trial', ylabel='ROC-AUC (CV)')
    ax.legend(fontsize=9); ax.grid(True, color=PALETTE['grid'], alpha=0.7)

    # Importancia de hiperparámetros (fANOVA de Optuna)
    ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
    try:
        from optuna.importance import get_param_importances
        rf_imp  = get_param_importances(rf_study)
        xgb_imp = get_param_importances(xgb_study)

        # Normalizar y ordenar
        rf_s  = pd.Series(rf_imp).sort_values(ascending=True)
        xgb_s = pd.Series(xgb_imp).sort_values(ascending=True)

        # Dibujar barras horizontales solapadas
        all_params = sorted(set(rf_s.index) | set(xgb_s.index))
        yp = np.arange(len(all_params))
        w  = 0.4
        ax.barh(yp - w/2,
                [rf_s.get(p, 0)  for p in all_params], w,
                color=PALETTE['rf'],  alpha=0.85, label='Random Forest',
                edgecolor='white')
        ax.barh(yp + w/2,
                [xgb_s.get(p, 0) for p in all_params], w,
                color=PALETTE['xgb'], alpha=0.85, label='XGBoost',
                edgecolor='white')
        ax.set_yticks(yp); ax.set_yticklabels(all_params, fontsize=8)
        ax.set_title('Hyperparameter Importance (fANOVA)',
                     fontweight='bold', pad=10)
        ax.set_xlabel('Importancia relativa')
        ax.legend(fontsize=8)
        ax.grid(True, axis='x', color=PALETTE['grid'], alpha=0.6)
    except Exception as e:
        ax.text(0.5, 0.5, f'fANOVA no disponible\n({e})',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        ax.set_title('Hyperparameter Importance', fontweight='bold', pad=10)

    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    print(f"  [OK] {out}")

def main():
    parser = argparse.ArgumentParser(
        description='CPU-GPU ML Predictor — RF + XGBoost con Optuna HPO'
    )
    parser.add_argument('--input',      required=True,
                        help='CSV del clúster etiquetado por el ILP')
    parser.add_argument('--output-dir', default='.',
                        help='Directorio para las figuras (default: .)')
    parser.add_argument('--trials',     type=int, default=OPTUNA_TRIALS,
                        help=f'Nº de trials Optuna por modelo (default: {OPTUNA_TRIALS})')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out = args.output_dir

    print("=" * 65)
    print("  CPU-GPU TASK ASSIGNMENT — ML PIPELINE  [Optuna HPO]")
    print("=" * 65)
    print(f"  Dataset  : {args.input}")
    print(f"  Trials   : {args.trials} por modelo")

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
    print(f"  speedup:       min={df['speedup'].min():.3f}  "
          f"max={df['speedup'].max():.3f}  "
          f"mean={df['speedup'].mean():.3f}")
    print()

    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print()

    print("  Optimizando Random Forest con Optuna...")
    rf_params,  rf_study  = optimize_rf( X_train, y_train, n_trials=args.trials)
    print()
    print("  Optimizando XGBoost con Optuna...")
    xgb_params, xgb_study = optimize_xgb(X_train, y_train, n_trials=args.trials)
    print()

    print("  Entrenando modelos finales con mejores hiperparámetros...")
    rf_model  = build_rf(rf_params)
    xgb_model = build_xgb(xgb_params)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    print()

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
    fig5_optuna(
        rf_study, xgb_study,
        out=f'{out}/fig5_optuna_history.png',
    )

    print()
    print("=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)
    print(f"  {'Modelo':<26} {'Accuracy':>10} {'F1 Macro':>10} {'ROC-AUC':>10}")
    print(f"  {'─'*56}")
    print(f"  {'Random Forest (Optuna)':<26} {rf_acc:>10.4f} {rf_f1:>10.4f} {rf_auc:>10.4f}")
    print(f"  {'XGBoost (Optuna)':<26} {xgb_acc:>10.4f} {xgb_f1:>10.4f} {xgb_auc:>10.4f}")
    print()
    print(f"  Figuras guardadas en: {out}/")
    print("=" * 65)


if __name__ == '__main__':
    main()
