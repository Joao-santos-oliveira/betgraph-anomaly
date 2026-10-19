"""Resume ROC entre seeds do mesmo modelo e condicao, sem misturar folds."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gerar(directory, condition='context_only', model='gnn'):
    metrics = pd.read_csv(directory / 'auc_conferida.csv')
    metrics = metrics[(metrics.condition == condition) & (metrics.model == model)].sort_values('seed')
    if len(metrics) < 2:
        raise ValueError('Sao necessarias pelo menos duas seeds')
    grid = np.linspace(0, 1, 1001)
    curves = []
    fig, ax = plt.subplots(figsize=(9, 7.5))
    originals = []
    for row in metrics.itertuples():
        roc = pd.read_csv(directory / f'roc_{condition}_{model}_{row.seed}.csv')
        # interp usa o ultimo TPR em FPRs repetidas, preservando o topo do salto vertical.
        interpolated = np.interp(grid, roc.fpr, roc.tpr)
        curves.append(interpolated)
        originals.append(roc)
        ax.plot(roc.fpr, roc.tpr, color='gray', alpha=.4, lw=1)
    mean = np.mean(curves, axis=0)
    sd = np.std(curves, axis=0, ddof=1)
    auc_mean, auc_sd = metrics.auc.mean(), metrics.auc.std(ddof=1)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.8, label='Referência aleatória (AUC = 0,500)')
    ax.fill_between(grid, np.maximum(0, mean-sd), np.minimum(1, mean+sd),
                    color='blue', alpha=.16, label='± 1 desvio-padrão entre seeds')
    ax.plot(grid, mean, color='blue', lw=3, label=f'ROC média (AUCs = {auc_mean:.4f} ± {auc_sd:.4f})')
    for idx, color, label in [(int(np.argmax(metrics.auc)), 'forestgreen', 'Maior AUC'),
                              (int(np.argmin(metrics.auc)), 'red', 'Menor AUC')]:
        row = metrics.iloc[idx]
        roc = originals[idx]
        ax.plot(roc.fpr, roc.tpr, color=color, lw=1.6,
                label=f'{label}: seed {row.seed} (AUC = {row.auc:.4f})')
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='Taxa de falsos positivos (FPR)',
           ylabel='Taxa de verdadeiros positivos (TPR)',
           title=f'ROC de validação — {model.upper()} / {condition}\n{len(metrics)} seeds · mesma divisão e mesmos pares')
    ax.grid(alpha=.25)
    ax.legend(loc='lower right', fontsize=10)
    fig.text(.5, .015, 'Faixa descritiva, não intervalo de confiança · tarefa de conexões, não de fraude',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, 1])
    stem = directory / f'roc_resumo_{condition}_{model}'
    fig.savefig(stem.with_suffix('.png'), dpi=180)
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    pd.DataFrame({'fpr': grid, 'tpr_media': mean, 'tpr_desvio_amostral': sd}).to_csv(stem.with_suffix('.csv'), index=False)
    print(f'{stem}: AUC {auc_mean:.6f} +/- {auc_sd:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    args = parser.parse_args()
    for condition in ['context_only', 'full_day']:
        for model in ['gnn', 'mlp']:
            gerar(args.input, condition, model)
