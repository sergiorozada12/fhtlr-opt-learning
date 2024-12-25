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

class LinearModel(torch.nn.Module):
    def __init__(self, ds: int, da: int) -> None:
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(ds, da, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.001)

    def forward(self, state: np.array) -> torch.Tensor:
        state = torch.from_numpy(state)
        return self.linear(state)
    
class FHLinearModel(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        H: int,
    ) -> None:
        super(FHLinearModel, self).__init__()

    
        time_layers = []
        for h in range(H):
            action_layer = torch.nn.Linear(num_inputs, num_outputs, bias=False)
            torch.nn.init.normal_(action_layer.weight, mean=0.0, std=0.001)
            time_layers.append(action_layer)
        self.time_layers = torch.nn.ParameterList(time_layers)

    def forward(self, x: np.ndarray, h: int) -> torch.Tensor:
        x = torch.from_numpy(x)
        return self.time_layers[h](x)
    


class RBFmodel(torch.nn.Module):
    def __init__(self, num_inputs, num_rbf_features, num_outputs):
        super(RBFmodel, self).__init__()
        self.num_rbf_features = num_rbf_features
        self.centers = torch.randn(num_rbf_features, num_inputs).double()
        self.linear = torch.nn.Linear(num_rbf_features, num_outputs)

    def radial_basis(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        dist = torch.cdist(x, self.centers)
        return torch.exp(-dist.pow(2))

    def forward(self, x):
        rbf_feats = self.radial_basis(x)
        return self.linear(rbf_feats)

class FHRBFmodel(torch.nn.Module):
    def __init__(self, num_inputs, num_rbf_features, num_outputs, H):
        super(FHRBFmodel, self).__init__()
        center_h = []
        linear_h = []
        for h in range(H):
            self.num_rbf_features = num_rbf_features
            center_h.append(torch.randn(num_rbf_features, num_inputs).double())
            linear_h.append(torch.nn.Linear(num_rbf_features, num_outputs))
        self.centers_h = torch.nn.ParameterList(center_h)
        self.linear_h = torch.nn.ParameterList(linear_h)
    def radial_basis(self, x,h):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        dist = torch.cdist(x, self.centers_h[h])
        return torch.exp(-dist.pow(2)).squeeze()

    def forward(self, x, h):
        x = torch.from_numpy(x)
        rbf_feats = self.radial_basis(x,h)
        return self.linear_h[h](rbf_feats)
