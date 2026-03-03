import os
import json
import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import parafac

from src.environments import PendulumTerminalReward, CartPoleTerminalReward
from src.agents.dqn import DFHqn
from src.trainer import run_experiment
import src.experiments.pendulum as pendulum_config
import src.experiments.cartpole as cartpole_config

tl.set_backend('pytorch')

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def normalize_frobenius_error(tensor, reconstructed_tensor):
    # NFE = ||T - T_rec||_F / ||T||_F
    error_tensor = tensor - reconstructed_tensor
    norm_error = torch.linalg.norm(error_tensor)
    norm_original = torch.linalg.norm(tensor)
    if norm_original == 0:
        return 0.0
    return (norm_error / norm_original).item()

def evaluate_parafac_ranks(Q_tensor, max_rank=40, step=2):
    tensor_tl = tl.tensor(Q_tensor, dtype=torch.float64)
    # Mean centering as in gridworld plot_tensor_rank
    tensor_tl -= tensor_tl.mean()
    
    ranks = list(range(1, max_rank + 1, step))
    errors = []
    
    for rank in ranks:
        # Avoid rank > prod(dims) in small tensors
        try:
            factors = parafac(tensor_tl, rank=rank, init='random', n_iter_max=100, tol=1e-8, svd='numpy_svd')
            reconstructed_tl = tl.cp_to_tensor(factors)
            nfe = normalize_frobenius_error(tensor_tl, reconstructed_tl)
            errors.append(nfe)
            print(f"Rank {rank}: NFE = {nfe}")
        except Exception as e:
            print(f"Failed to decompose at rank {rank}: {e}")
            errors.append(None)
            
    return ranks, errors

def build_q_tensor(agent, discretizer, H):
    # dimensions: H x |S_1| x |S_2| ... x |A|
    bucket_states = discretizer.bucket_states.astype(int).tolist()
    bucket_actions = discretizer.bucket_actions.astype(int).tolist()
    dims = [H] + bucket_states + bucket_actions
    Q_tensor = np.zeros(dims)
    
    # Iterate over all indices to build the tensor.
    print(f"Building Q tensor of shape {dims}...")
    for idx_tuple in np.ndindex(*dims):
        h = idx_tuple[0]
        s_idx_tuple = idx_tuple[1:-len(bucket_actions)]
        a_idx_tuple = idx_tuple[-len(bucket_actions):]
        
        # We need continuous state vector
        s_continuous = discretizer.get_state_from_index(s_idx_tuple)
        
        # Flattened a_idx (since Q evaluates all actions at once and returns a flat vector of values)
        a_idx_flat = np.ravel_multi_index(a_idx_tuple, discretizer.bucket_actions)
        
        # Forward pass DFHQN
        # Note: DFHQN Q network returns value for all actions at state s, time h
        with torch.no_grad():
            q_values = agent.Q(s_continuous, h)
            Q_tensor[idx_tuple] = q_values[a_idx_flat].item()
            
    return Q_tensor

def train_and_eval_env(env_name, config_module, eps=20_000):
    print(f"--- Starting {env_name} PARAFAC tracking ---")
    env = config_module.generate_env()
    discretizer = config_module.DISCRETIZER
    H = config_module.H
    
    agent = DFHqn(
        discretizer, 
        config_module.ALPHA_DQN, 
        H, 
        config_module.BUFFER_SIZE, 
        hidden_layers=[64, 64]
    )
    
    # Train DFHQN on a larger number of episodes to ensure a good solution
    # eps=20_000 for good convergence. 
    # Use decay matching the eps.
    eps_decay = (0.9999) ** (30_000 / eps) if eps > 0 else 0.9999
    
    # Run the experiment
    print(f"Training DFHQN on {env_name} for {eps} episodes...")
    _ = run_experiment(
        n=0, 
        E=eps, 
        H=H, 
        eps=1.0, 
        eps_decay=eps_decay, 
        env=env, 
        agent=agent
    )
    
    # Extract tensor
    Q_tensor = build_q_tensor(agent, discretizer, H)
    
    # Evaluate ranks
    print("Evaluating PARAFAC ranks...")
    ranks, errors = evaluate_parafac_ranks(Q_tensor, max_rank=100, step=2)
    
    return {"ranks": ranks, "errors": errors}

def run_gym_simulations():
    os.makedirs('results', exist_ok=True)
    results_file = 'results/gym_parafac_errors.json'
    
    results = {}
    
    # 1. Pendulum
    res_pendulum = train_and_eval_env("Pendulum", pendulum_config, eps=20_000)
    results["pendulum"] = res_pendulum
    
    # 2. CartPole
    res_cartpole = train_and_eval_env("CartPole", cartpole_config, eps=20_000)
    results["cartpole"] = res_cartpole
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {results_file}")

if __name__ == "__main__":
    run_gym_simulations()
