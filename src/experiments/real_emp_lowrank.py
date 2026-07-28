"""Restartable DFHQN and out-of-core PARAFAC experiment for real cases."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from src.agents.dqn import DFHqn
from src.trainer import run_experiment
import src.experiments.battery as battery_config
import src.experiments.wireless as wireless_config

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path("results")
DEFAULT_RANKS = (10_000,)
ENVIRONMENTS = {
    "battery": (battery_config, 50_000, [32]),
    "wireless": (wireless_config, 200_000, [32]),
}


def _paths(name):
    prefix = RESULTS_DIR / f"real_emp_lowrank_{name}"
    return {"returns": Path(f"{prefix}_returns.npy"),
            "model": Path(f"{prefix}_dfhqn.pt"),
            "tensor": Path(f"{prefix}_q_tensor.npy")}


def _make_agent(config, hidden_layers):
    alpha = getattr(config, "ALPHA_DFHQN", config.ALPHA_DQN)
    return DFHqn(config.DISCRETIZER, alpha, config.H, config.BUFFER_SIZE,
                 hidden_layers=hidden_layers)


def train_dfhqn(name, config, episodes, hidden_layers, force=False):
    paths = _paths(name)
    agent = _make_agent(config, hidden_layers)
    if not force and paths["returns"].exists() and paths["model"].exists():
        agent.Q.load_state_dict(torch.load(paths["model"], map_location="cpu"))
        print(f"[{name}] training artefacts exist; skipping step 1")
        return agent
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    decay = getattr(config, "EPS_DECAY", (0.9999) ** (30_000 / episodes))
    returns = run_experiment(n=0, E=episodes, H=config.H, eps=1.0,
                             eps_decay=decay, env=config.generate_env(), agent=agent)
    np.save(paths["returns"], np.asarray(returns, dtype=np.float64)[None, :])
    tmp = paths["model"].with_suffix(".tmp")
    torch.save(agent.Q.state_dict(), tmp)
    os.replace(tmp, paths["model"])
    print(f"\n[{name}] saved step 1 artefacts")
    return agent


def tensor_shape(config):
    d = config.DISCRETIZER
    return tuple([config.H] + d.bucket_states.astype(int).tolist()
                 + d.bucket_actions.astype(int).tolist())


def build_q_tensor(name, config, agent, batch_size=256, force=False):
    path = _paths(name)["tensor"]
    shape = tensor_shape(config)
    if not force and path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape == shape:
            print(f"[{name}] tensor exists; skipping step 2")
            return existing
        raise ValueError(f"{path} has shape {existing.shape}, expected {shape}")
    tmp = path.with_suffix(".tmp.npy")
    q_tensor = open_memmap(tmp, mode="w+", dtype=np.float32, shape=shape)
    d = config.DISCRETIZER
    state_dims = tuple(d.bucket_states.astype(int))
    n_states = int(np.prod(state_dims))
    n_actions = int(np.prod(d.bucket_actions))
    spacing = (d.max_points_states - d.min_points_states) / (d.bucket_states - 1)
    agent.Q.eval()
    print(f"[{name}] building tensor {shape} ({q_tensor.nbytes / 2**30:.2f} GiB)")
    with torch.inference_mode():
        for h in range(config.H):
            out = q_tensor[h].reshape(n_states, n_actions)
            for start in range(0, n_states, batch_size):
                stop = min(start + batch_size, n_states)
                idx = np.column_stack(np.unravel_index(np.arange(start, stop), state_dims))
                states = d.min_points_states + idx * spacing
                out[start:stop] = agent.Q(states.astype(np.float64), h).cpu().numpy()
            q_tensor.flush()
            print(f"[{name}] tensor timestep {h + 1}/{config.H}")
    del q_tensor
    os.replace(tmp, path)
    return np.load(path, mmap_mode="r")


def _flat_stats(array, chunk_size):
    flat = array.reshape(-1)
    total = total_sq = 0.0
    for start in range(0, flat.size, chunk_size):
        x = np.asarray(flat[start:start + chunk_size], dtype=np.float64)
        total += x.sum()
        total_sq += np.dot(x, x)
    mean = total / flat.size
    return mean, max(total_sq - flat.size * mean * mean, 0.0)


def _fit_sampled_parafac(array, rank, mean, steps, sample_size, lr):
    rng = np.random.default_rng(rank)
    torch.manual_seed(rank)
    shape = tuple(array.shape)
    factors = [torch.nn.Parameter(0.1 * torch.randn(n, rank)) for n in shape]
    optimizer = torch.optim.Adam(factors, lr=lr)
    flat = array.reshape(-1)
    for _ in range(steps):
        flat_idx = rng.integers(0, flat.size, size=sample_size)
        multi = np.unravel_index(flat_idx, shape)
        target = torch.from_numpy(np.asarray(flat[flat_idx], dtype=np.float64)) - mean
        product = torch.ones((sample_size, rank), dtype=torch.float64)
        for factor, idx in zip(factors, multi):
            product *= factor[torch.from_numpy(idx.astype(np.int64))]
        loss = torch.mean((product.sum(1) - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return [factor.detach().cpu() for factor in factors]


def _exact_nfe(array, factors, mean, norm_sq, chunk_size):
    flat, shape = array.reshape(-1), tuple(array.shape)
    error_sq, rank = 0.0, factors[0].shape[1]
    for start in range(0, flat.size, chunk_size):
        stop = min(start + chunk_size, flat.size)
        multi = np.unravel_index(np.arange(start, stop), shape)
        prediction = torch.ones((stop - start, rank), dtype=torch.float64)
        for factor, idx in zip(factors, multi):
            prediction *= factor[torch.from_numpy(idx.astype(np.int64))]
        target = torch.from_numpy(np.asarray(flat[start:stop], dtype=np.float64)) - mean
        error_sq += torch.sum((target - prediction.sum(1)) ** 2).item()
    return 0.0 if norm_sq == 0 else float(np.sqrt(error_sq / norm_sq))


def evaluate_parafac(name, q_tensor, ranks=DEFAULT_RANKS, force=False,
                     steps=1_000, sample_size=8_192, chunk_size=250_000, lr=0.03):
    output = RESULTS_DIR / "real_emp_lowrank_parafac_errors.json"
    data = json.loads(output.read_text()) if output.exists() else {}
    ranks = [int(k) for k in ranks]
    previous = data.get(name, {})
    saved_ranks = [int(rank) for rank in previous.get("ranks", [])]
    saved_errors = previous.get("errors", [])
    if len(saved_ranks) != len(saved_errors):
        raise ValueError(
            f"{output}: '{name}' has {len(saved_ranks)} ranks but "
            f"{len(saved_errors)} errors"
        )
    results_by_rank = dict(zip(saved_ranks, saved_errors))
    ranks_to_evaluate = ranks if force else [
        rank for rank in ranks if rank not in results_by_rank
    ]
    if not ranks_to_evaluate:
        print(f"[{name}] PARAFAC results exist; skipping step 3")
        return previous
    mean, norm_sq = _flat_stats(q_tensor, chunk_size)
    for rank in ranks_to_evaluate:
        factors = _fit_sampled_parafac(q_tensor, rank, mean, steps, sample_size, lr)
        error = _exact_nfe(q_tensor, factors, mean, norm_sq, chunk_size)
        if rank not in results_by_rank:
            saved_ranks.append(rank)
        results_by_rank[rank] = error
        print(f"[{name}] rank {rank}: NFE = {error:.8f}")
    data[name] = {"ranks": saved_ranks,
                  "errors": [results_by_rank[rank] for rank in saved_ranks],
                  "shape": list(q_tensor.shape),
                  "max_rank": int(np.prod(q_tensor.shape) // max(q_tensor.shape)),
                  "metric": "mean-centered normalized Frobenius error",
                  "solver": "uniform-entry stochastic PARAFAC; exact streamed NFE"}
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, output)
    return data[name]


def run_real_emp_lowrank(ranks=DEFAULT_RANKS, force=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (config, episodes, hidden_layers) in ENVIRONMENTS.items():
        agent = train_dfhqn(name, config, episodes, hidden_layers, force=force)
        tensor = build_q_tensor(name, config, agent, force=force)
        evaluate_parafac(name, tensor, ranks=ranks, force=force)


if __name__ == "__main__":
    run_real_emp_lowrank()
