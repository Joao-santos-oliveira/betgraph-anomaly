"""Figura descritiva do piloto; usa somente a copia versionada de metricas.csv."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    folder = Path(__file__).resolve().parent
    frame = pd.read_csv(folder / 'metricas.csv')
    frame = frame.loc[frame.universe == 'completo']
    means = frame.groupby(['scenario', 'model'])[['recovered_100', 'ap']].mean()
    methods = ['gnn', 'mlp', 'features']
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, title in zip(axes, ['recovered_100', 'ap'],
                                 ['Alvos recuperados no top 100', 'Average Precision (AP)']):
        for scenario, shift, color, label in [('original', -.18, '#8795a1', 'Antes'),
                                               ('injetado', .18, '#246ba3', 'Depois')]:
            values = [means.loc[(scenario, method), metric] for method in methods]
            bars = ax.bar(x + shift, values, width=.34, color=color, label=label)
            ax.bar_label(bars, labels=[f'{value:.2f}' if metric == 'recovered_100' else f'{value:.4f}' for value in values], padding=4, fontsize=10)
        ax.set_xticks(x, ['GNN + IF', 'MLP + IF', 'Features + IF'])
        ax.set_title(title)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 4.2)
    axes[0].set_ylabel('Quantidade entre os 50 alvos')
    axes[0].axhline(1, color='#8a6437', linestyle=':', label='Expectativa aleatória: 1')
    axes[1].set_ylim(0, .033)
    axes[1].axhline(.01, color='#8a6437', linestyle=':', label='Prevalência: 1%')
    for ax in axes: ax.legend(fontsize=9)
    fig.suptitle('Piloto sintético v2 — recuperação de 50 alvos em 5.000 carteiras', fontsize=14)
    fig.text(.5, .025, 'GNN/MLP: médias das seeds 42, 73 e 2026 · features: IF seed 42 · rótulos artificiais', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .07, 1, .92])
    for extension in ['png', 'pdf']:
        fig.savefig(folder / f'recuperacao_piloto_v2.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
