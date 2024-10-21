from typing import List
import numpy as np
import torch


class ValueNetwork(torch.nn.Module):
    def __init__(
        self, num_inputs: int, num_hiddens: List[int], num_outputs: int
    ) -> None:
        super(ValueNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x)
        for layer in self.layers:
            x = layer(x)
        return x


class FHValueNetwork(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_hiddens: List[int],
        num_outputs: int,
        H: int,
    ) -> None:
        super(FHValueNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h

        time_layers = []
        for h in range(H):
            action_layer = torch.nn.Linear(num_inputs, num_outputs)
            action_layer.weight.data.mul_(0.1)
            action_layer.bias.data.mul_(0.0)
            time_layers.append(action_layer)
        self.time_layers = torch.nn.ParameterList(time_layers)

    def forward(self, x: np.ndarray, h: int) -> torch.Tensor:
        x = torch.from_numpy(x)
        for layer in self.layers:
            x = layer(x)
        return self.time_layers[h](x)


class PARAFAC(torch.nn.Module):
    def __init__(
        self, dims: np.ndarray, k: int, scale: float = 1.0, nA: int = 1
    ) -> None:
        super().__init__()

        self.nA = nA
        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale * torch.randn(dim, k, dtype=torch.double, requires_grad=True)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices: np.ndarray) -> torch.Tensor:
        prod = torch.ones(self.k, dtype=torch.double)
        for i in range(len(indices)):
            idx = indices[i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if len(indices) < len(self.factors):
            res = []
            for cols in zip(
                *[self.factors[-(a + 1)].t() for a in reversed(range(self.nA))]
            ):
                kr = cols[0]
                for j in range(1, self.nA):
                    kr = torch.kron(kr, cols[j])
                res.append(kr)
            factors_action = torch.stack(res, dim=1)
            return torch.matmul(prod, factors_action.T)
        return torch.sum(prod, dim=-1)
