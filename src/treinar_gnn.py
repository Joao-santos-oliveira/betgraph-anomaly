"""Treina a GNN em CPU; seleciona por validacao e preserva o teste."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import torch
import torch_geometric
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from modelo_gnn import GNNBipartida
from preparar_features import ROOT
from preparar_gnn import FORWARD, REVERSE


def codigos(pairs, n_events):
    """Codifica cada par em um inteiro para verificar intersecoes."""
    return set((pairs[0] * n_events + pairs[1]).tolist())


def amostrar_negativos(n_users, n_events, count, forbidden, rng):
    """Amostra pares uniformemente sem repeticao e sem construir matriz densa."""
    total = n_users * n_events
    if count < 1 or count > total - len(forbidden):
        raise ValueError('Quantidade de negativos inviavel')
    chosen, seen = [], set()
    for _ in range(1000):
        candidates = rng.integers(0, total, size=max(64, 2 * (count - len(chosen))))
        for value in candidates:
            value = int(value)
            if value not in forbidden and value not in seen:
                chosen.append(value)
                seen.add(value)
                if len(chosen) == count:
                    values = np.array(chosen, dtype=np.int64)
                    return torch.from_numpy(np.stack([values // n_events, values % n_events]))
    raise ValueError('Amostragem lenta demais: grafo possivelmente muito denso')


def validar_divisao(split):
    """Rejeita particoes inconsistentes ou conexoes expostas no contexto."""
    graph = split['context']
    graph.validate(raise_on_error=True)
    nu, ne = graph['user'].num_nodes, graph['event'].num_nodes
    if not torch.equal(graph[REVERSE].edge_index, graph[FORWARD].edge_index.flip(0)):
        raise ValueError('Arestas reversas inconsistentes')
    parts = [graph[FORWARD].edge_index] + [split[k] for k in ['train_positive', 'val_positive', 'test_positive']]
    groups = []
    for pairs in parts + [split['all_positive']]:
        if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.shape[1] == 0:
            raise ValueError('Particao vazia ou formato de pares invalido')
        if (pairs < 0).any() or (pairs[0] >= nu).any() or (pairs[1] >= ne).any():
            raise ValueError('Endpoint fora dos limites')
        group = codigos(pairs, ne)
        if len(group) != pairs.shape[1]:
            raise ValueError('Pares repetidos na particao')
        groups.append(group)
    observed = groups.pop()
    if set.union(*groups) != observed:
        raise ValueError('Particoes nao cobrem all_positive')
    for i, group in enumerate(groups):
        if any(group & previous for previous in groups[:i]):
            raise ValueError('Vazamento de conexoes entre particoes')
    for kind in ['user', 'event']:
        if not torch.isfinite(graph[kind].x).all():
            raise ValueError('Features nao finitas')
    return observed


def exemplos(positive, negative):
    """Une pares e alvos binarios, sempre positivos antes dos negativos."""
    pairs = torch.cat([positive, negative], dim=1)
    labels = torch.cat([torch.ones(positive.shape[1]), torch.zeros(negative.shape[1])])
    return pairs, labels


def metricas(logits, labels):
    """Metricas de conexao, nunca de deteccao de fraude."""
    if not torch.isfinite(logits).all():
        raise ValueError('Logits nao finitos')
    y, scores = labels.numpy(), logits.numpy()
    two_classes = len(np.unique(y)) == 2
    return {'n': len(y), 'positivos': int(y.sum()),
            'loss': float(F.binary_cross_entropy_with_logits(logits, labels)),
            'ap': float(average_precision_score(y, scores)) if two_classes else None,
            'auc': float(roc_auc_score(y, scores)) if two_classes else None}


def avaliar_validacao(model, graph, pairs, labels):
    """Avalia sem dropout e separa pares com endpoints sem vizinhos."""
    model.eval()
    with torch.no_grad():
        embeddings = model(graph.x_dict, graph.edge_index_dict)
        logits = model.decode(embeddings, pairs)
    result = metricas(logits, labels)
    context = graph[FORWARD].edge_index
    user_degree = torch.bincount(context[0], minlength=graph['user'].num_nodes)
    event_degree = torch.bincount(context[1], minlength=graph['event'].num_nodes)
    has_context = (user_degree[pairs[0]] > 0) & (event_degree[pairs[1]] > 0)
    for key, mask in [('ambos_com_vizinhos', has_context), ('algum_sem_vizinhos', ~has_context)]:
        result[key] = metricas(logits[mask], labels[mask]) if mask.any() else {'n': 0}
    return result


def treinar(source, output, seed=42, epochs=100, patience=15, lr=.001,
            hidden=64, embedding=32, dropout=.1, threads=2, min_delta=.0001):
    """Treina com negativos renovados, checkpoint e early stopping por loss."""
    if min(epochs, patience, threads) < 1 or lr <= 0 or min_delta < 0:
        raise ValueError('Parametros de treino invalidos')
    # Cada execucao ocupa pasta nova para nao sobrescrever experimentos anteriores.
    if output.exists():
        raise FileExistsError(f'Escolha outra pasta de saida: {output}')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    split = torch.load(source, map_location='cpu', weights_only=False)
    observed = validar_divisao(split)
    graph = split['context']
    nu, ne = graph['user'].num_nodes, graph['event'].num_nodes
    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 1)
    val_negative = amostrar_negativos(nu, ne, split['val_positive'].shape[1], observed, rng_val)
    test_negative = amostrar_negativos(nu, ne, split['test_positive'].shape[1],
                                       observed | codigos(val_negative, ne), np.random.default_rng(seed + 2))
    train_forbidden = observed | codigos(val_negative, ne) | codigos(test_negative, ne)
    val_pairs, val_labels = exemplos(split['val_positive'], val_negative)
    config = {'user_features': graph['user'].x.shape[1], 'event_features': graph['event'].x.shape[1],
              'hidden': hidden, 'embedding': embedding, 'dropout': dropout}
    model = GNNBipartida(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    initial = avaliar_validacao(model, graph, val_pairs, val_labels)
    output.mkdir(parents=True)
    torch.save({'pairs': val_pairs, 'labels': val_labels}, output / 'validacao.pt')
    test_pairs, test_labels = exemplos(split['test_positive'], test_negative)
    torch.save({'pairs': test_pairs, 'labels': test_labels}, output / 'teste_reservado.pt')
    best_loss, best_epoch, stale = initial['loss'], 0, 0
    best_metrics = initial
    checkpoint = output / 'melhor_modelo.pt'

    def salvar(epoch, metrics):
        torch.save({'model_state': model.state_dict(), 'model_config': config,
                    'epoch': epoch, 'validation': metrics, 'seed': seed}, checkpoint)

    salvar(0, initial)
    history = []
    started = perf_counter()
    print(f'Antes do treino: val_loss={initial["loss"]:.6f} AP={initial["ap"]:.4f}', flush=True)
    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()
        negative = amostrar_negativos(nu, ne, split['train_positive'].shape[1], train_forbidden, rng_train)
        pairs, labels = exemplos(split['train_positive'], negative)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model(graph.x_dict, graph.edge_index_dict)
        logits = model.decode(z, pairs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError('Loss nao finita')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., error_if_nonfinite=True)
        optimizer.step()
        validation = avaliar_validacao(model, graph, val_pairs, val_labels)
        duration = perf_counter() - epoch_start
        history.append({'epoch': epoch, 'train_loss': float(loss.detach()),
                        'val_loss': validation['loss'], 'val_ap': validation['ap'],
                        'val_auc': validation['auc'], 'gradient_norm': float(norm), 'seconds': duration})
        if validation['loss'] < best_loss - min_delta:
            best_loss, best_epoch, stale = validation['loss'], epoch, 0
            best_metrics = validation
            salvar(epoch, validation)
        else:
            stale += 1
        pd.DataFrame(history).to_csv(output / 'historico.csv', index=False)
        if epoch == 1 or epoch % 10 == 0 or stale >= patience:
            print(f'Epoca {epoch}: train={loss.item():.5f} val={validation["loss"]:.5f} AP={validation["ap"]:.4f} tempo={duration:.3f}s', flush=True)
        if stale >= patience:
            break
    elapsed = perf_counter() - started
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model_state'])
    restored = avaliar_validacao(model, graph, val_pairs, val_labels)
    if not np.isclose(restored['loss'], best_metrics['loss'], rtol=1e-7, atol=1e-8):
        raise RuntimeError('Checkpoint nao reproduziu a validacao')
    report = {
        'seed': seed, 'source': str(source.resolve()),
        'sha256_divisao': hashlib.sha256(source.read_bytes()).hexdigest(),
        'config': config, 'lr': lr, 'epochs_limit': epochs, 'patience': patience,
        'min_delta': min_delta, 'gradient_clip': 5., 'threads': threads, 'device': 'cpu',
        'optimizer': 'Adam', 'negative_ratio': 1, 'sampling': 'uniforme; treino renovado; validacao e teste fixos e excluidos do treino',
        'epochs_executadas': len(history), 'melhor_epoca': best_epoch,
        'parada': 'patience' if stale >= patience else 'limite_epocas',
        'validacao_inicial': initial, 'melhor_validacao': restored,
        'tempo_loop_segundos': elapsed,
        'mediana_epoca_segundos': float(np.median([h['seconds'] for h in history])),
        'teste_avaliado': False,
        'limitacoes': ['features agregadas do recorte inteiro', 'negativos uniformes podem ser faceis',
                       'metricas de conexao, nao de fraude', 'sem teste de generalizacao temporal'],
        'versoes': {'python': platform.python_version(), 'torch': torch.__version__,
                    'pyg': torch_geometric.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
    }
    (output / 'treino_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Melhor epoca: {best_epoch}; val_loss={restored["loss"]:.6f}; AP={restored["ap"]:.4f}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/dev/gnn_seed_42/divisao.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--embedding', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or args.input.parent / f'treino_{stamp}_seed_{args.seed}'
    treinar(args.input, output, args.seed, args.epochs, args.patience, args.lr,
            args.hidden, args.embedding, args.dropout, args.threads)


if __name__ == '__main__':
    main()
