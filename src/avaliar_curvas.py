"""Curvas ROC de validacao e perdas por epoca; nunca avalia o teste reservado."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from modelo_gnn import MODELOS
from preparar_gnn import FORWARD
from ranking_gnn import sha256


def gerar(experiment, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    protocol = json.loads((experiment / 'protocolo.json').read_text())
    seeds = protocol['model_seeds']
    output.mkdir(parents=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    learning, loss_axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    reference = None
    rows = []
    for col, condition in enumerate(protocol['conditions']):
        division = experiment / condition / 'divisao.pt'
        graph = torch.load(division, map_location='cpu', weights_only=False)['context']
        ax = axes[col]
        for row, (kind, color) in enumerate([('gnn', '#1464b4'), ('mlp', '#d5691c')]):
            for i, seed in enumerate(seeds):
                run = experiment / condition / f'{kind}_seed_{seed}'
                saved = torch.load(run / 'melhor_modelo.pt', map_location='cpu', weights_only=False)
                assert saved['sha256_divisao'] == sha256(division)
                evaluation = torch.load(run / 'validacao.pt', weights_only=True)
                if reference is None:
                    reference = evaluation
                assert all(torch.equal(evaluation[k], reference[k]) for k in ['pairs', 'labels'])
                model = MODELOS[kind](**saved['model_config'])
                model.load_state_dict(saved['model_state'])
                model.eval()
                with torch.no_grad():
                    logits = model.decode(model(graph.x_dict, graph.edge_index_dict), evaluation['pairs']).numpy()
                labels = evaluation['labels'].numpy()
                fpr, tpr, thresholds = roc_curve(labels, logits)
                auc = roc_auc_score(labels, logits)
                recorded = json.loads((run / 'treino_manifest.json').read_text())['melhor_validacao']['auc']
                assert np.isclose(auc, recorded, atol=1e-10)
                pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold_logit': thresholds}).to_csv(
                    output / f'roc_{condition}_{kind}_{seed}.csv', index=False)
                ax.plot(fpr, tpr, color=color, linestyle=['-', '--', ':'][i % 3], lw=1.8,
                        label=f'{kind.upper()} · {seed} · AUC {auc:.4f}')
                rows.append({'condition': condition, 'model': kind, 'seed': seed, 'auc': auc})
                history = pd.read_csv(run / 'historico.csv')
                loss_ax = loss_axes[row, col]
                line_color = ['#1464b4', '#a04398', '#16856a'][i % 3]
                loss_ax.plot(history.epoch, history.train_loss, color=line_color, alpha=.6, linestyle='--')
                loss_ax.plot(history.epoch, history.val_loss, color=line_color, label=str(seed))
                loss_ax.set_title(f'{kind.upper()} · {condition}')
                loss_ax.set_ylabel('BCE (menor é melhor)')
                loss_ax.set_xlabel('Época')
                loss_ax.grid(alpha=.15)
                loss_ax.legend(title='Seed', fontsize=9)
        edges = graph[FORWARD].edge_index
        du = torch.bincount(edges[0], minlength=graph['user'].num_nodes)
        de = torch.bincount(edges[1], minlength=graph['event'].num_nodes)
        pairs = reference['pairs']
        scores = (du[pairs[0]] * de[pairs[1]]).numpy()
        fpr, tpr, thresholds = roc_curve(labels, scores)
        degree_auc = roc_auc_score(labels, scores)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}).to_csv(output / f'roc_{condition}_grau.csv', index=False)
        ax.plot(fpr, tpr, color='#69737b', lw=1.7, label=f'Graus · AUC {degree_auc:.4f}')
        ax.plot([0, 1], [0, 1], color='#a8adb3', linestyle='--', label='Referência aleatória · 0.5')
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Taxa de falsos positivos (FPR)',
               ylabel='Taxa de verdadeiros positivos (TPR)',
               title='Features do dia inteiro' if condition == 'full_day' else 'Features de contexto')
        ax.grid(alpha=.15)
        ax.legend(loc='lower right', fontsize=8.5)
    fig.suptitle('ROC de validação — reconstrução de conexões', fontsize=17, fontweight='bold')
    fig.text(.5, .015, 'Mesmos pares entre modelos e seeds · checkpoints selecionados por validação · não avalia fraude', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .04, 1, .94])
    learning.suptitle('Aprendizado por época — linha contínua: validação; tracejada: treino', fontsize=14)
    learning.tight_layout(rect=[0, 0, 1, .95])
    for figure, name in [(fig, 'roc_validacao'), (learning, 'curvas_aprendizado')]:
        figure.savefig(output / f'{name}.png', dpi=180)
        figure.savefig(output / f'{name}.pdf')
        plt.close(figure)
    pd.DataFrame(rows).to_csv(output / 'auc_conferida.csv', index=False)
    (output / 'manifest.json').write_text(json.dumps({'source': str(experiment.resolve()),
        'sha256_protocolo': sha256(experiment / 'protocolo.json'), 'seeds': seeds,
        'teste_avaliado': False, 'metrica': 'ROC de ligacao com negativos uniformes; nao fraude',
        'matplotlib': matplotlib.__version__}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gerar(args.input, args.output)
