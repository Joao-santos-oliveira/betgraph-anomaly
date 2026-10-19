"""Encoders GraphSAGE e MLP sem mensagens, com decoder de conexoes."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from preparar_gnn import FORWARD, REVERSE


class GNNBipartida(nn.Module):
    """Duas camadas: features -> hidden -> embeddings para cada tipo de no."""

    def __init__(self, user_features, event_features, hidden=64, embedding=32, dropout=.1):
        super().__init__()
        if min(user_features, event_features, hidden, embedding) < 1 or not 0 <= dropout < 1:
            raise ValueError('Dimensoes ou dropout invalidos')
        self.dropout = dropout
        self.layer1 = HeteroConv({
            FORWARD: SAGEConv((user_features, event_features), hidden),
            REVERSE: SAGEConv((event_features, user_features), hidden),
        }, aggr='sum')
        self.layer2 = HeteroConv({
            FORWARD: SAGEConv((hidden, hidden), embedding),
            REVERSE: SAGEConv((hidden, hidden), embedding),
        }, aggr='sum')
        # Decoder simples: aprende a compatibilidade entre os dois embeddings.
        self.decoder = nn.Sequential(nn.Linear(2 * embedding, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x_dict, edge_index_dict):
        """Combina features proprias e vizinhas; retorna um embedding por no."""
        hidden = self.layer1(x_dict, edge_index_dict)
        hidden = {key: F.dropout(F.relu(value), p=self.dropout, training=self.training)
                  for key, value in hidden.items()}
        return self.layer2(hidden, edge_index_dict)

    def decode(self, embeddings, pairs):
        """Retorna logits, sem sigmoid, apropriados para BCEWithLogitsLoss."""
        user = embeddings['user'][pairs[0]]
        event = embeddings['event'][pairs[1]]
        return self.decoder(torch.cat([user, event], dim=-1)).squeeze(-1)


class MLPSemMensagens(nn.Module):
    """Controle: mesmos tipos de no e decoder, sem consultar vizinhos."""

    def __init__(self, user_features, event_features, hidden=64, embedding=32, dropout=.1):
        super().__init__()
        if min(user_features, event_features, hidden, embedding) < 1 or not 0 <= dropout < 1:
            raise ValueError('Dimensoes ou dropout invalidos')
        self.encoders = nn.ModuleDict({kind: nn.Sequential(
            nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, embedding))
            for kind, size in [('user', user_features), ('event', event_features)]})
        self.decoder = nn.Sequential(nn.Linear(2 * embedding, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x_dict, edge_index_dict):
        return {kind: encoder(x_dict[kind]) for kind, encoder in self.encoders.items()}

    def decode(self, embeddings, pairs):
        return self.decoder(torch.cat([embeddings['user'][pairs[0]],
                                       embeddings['event'][pairs[1]]], dim=-1)).squeeze(-1)


MODELOS = {'gnn': GNNBipartida, 'mlp': MLPSemMensagens}
