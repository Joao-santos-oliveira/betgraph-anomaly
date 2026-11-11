"""Coparticipacao repetida com ajuste aproximado pelos graus bipartidos."""
import numpy as np
from scipy import sparse
from scipy.stats import poisson


def scores_relacionais(pairs, n_users, n_events, top_partners=3, block_size=256):
    """Sem rotulos: retorna scores e parceiros que explicam a maior surpresa.

    Cauda Poisson e uma aproximacao descritiva, nao um teste calibrado.
    Calcula blocos de carteiras, sem criar uma matriz densa N x N.
    """
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.dtype.kind not in 'iu':
        raise ValueError('Pares devem ser inteiros [2, E]')
    if min(n_users, n_events, top_partners, block_size) < 1:
        raise ValueError('Dimensoes invalidas')
    if pairs.size and (pairs.min() < 0 or pairs[0].max() >= n_users or pairs[1].max() >= n_events):
        raise ValueError('Endpoint invalido')
    matrix = sparse.csr_matrix((np.ones(pairs.shape[1], dtype=np.int64), pairs), shape=(n_users, n_events))
    if matrix.nnz != pairs.shape[1]:
        raise ValueError('Pares duplicados')
    du = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    de = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    edges = pairs.shape[1]
    factor = float(np.sum(de * (de - 1)) / (edges * (edges - 1))) if edges > 1 else 0.
    score = np.zeros(n_users)
    raw = np.zeros(n_users)
    partner = np.full(n_users, -1, dtype=int)
    shared = np.zeros(n_users, dtype=int)
    expected = np.zeros(n_users)
    for start in range(0, n_users, block_size):
        overlap = (matrix[start:start+block_size] @ matrix.T).tocsr()
        for local in range(overlap.shape[0]):
            u = start + local
            lo, hi = overlap.indptr[local:local+2]
            vs, cs = overlap.indices[lo:hi], overlap.data[lo:hi]
            mask = (vs != u) & (cs >= 2)
            vs, cs = vs[mask], cs[mask]
            if not len(vs):
                continue
            lam = du[u] * du[vs] * factor
            # Cauda calculada em log; clipping numerico documentado em 300 decadas.
            surprise = np.minimum(-poisson.logsf(cs - 1, lam) / np.log(10.), 300.)
            if not np.isfinite(surprise).all():
                raise ValueError('Surpresa nao finita')
            order = np.lexsort((vs, -surprise))[:top_partners]
            score[u] = surprise[order].sum() / top_partners
            raw[u] = np.sort(cs)[-top_partners:].sum() / top_partners
            best = order[0]
            partner[u], shared[u], expected[u] = vs[best], cs[best], lam[best]
    return {'surpresa_coparticipacao': score, 'contagem_coparticipacao': raw,
            'melhor_parceiro_indice': partner, 'mercados_compartilhados': shared,
            'sobreposicao_esperada_aprox': expected, 'grau_contexto': du}
