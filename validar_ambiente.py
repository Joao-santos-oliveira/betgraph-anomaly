"""Valida operacoes CPU e propagacao de gradientes no PyG."""
import torch
import torch_geometric
from torch_geometric.nn import SAGEConv

torch.manual_seed(42)
assert (torch.ones(2, 2) @ torch.ones(2, 2)).tolist() == [[2., 2.], [2., 2.]]
x = torch.randn(3, 4)
y = SAGEConv(4, 2)(x, torch.tensor([[0, 1, 2], [1, 2, 0]]))
y.sum().backward()
assert y.shape == (3, 2) and torch.isfinite(y).all()
print('CPU e PyG forward/backward OK:', torch.__version__, torch_geometric.__version__)
