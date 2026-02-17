import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import random
from multiprocessing import Pool
from typing import List, Optional, Union

import numpy as np
import torch
from gymnasium import Env

from src.environments import SourceChannelCodingEnv
from src.utils import Discretizer
from src.agents.dqn import DFHqn, Dqn
from src.agents.fhtlr import FHMaxTlr, FHTlr
from src.agents.ql import QLearning, FHQLearning
from src.agents.bf import FHLinear
from src.agents.rbf import FHRBF
from src.trainer import run_experiment


torch.set_num_threads(1)


# Constants
GAMMA = 0.8
H = 4
C = 4
MAX_BITS = 100.0

N_EXPS = 100
EPISODES = 40_000
BUFFER_SIZE = 1_000
BUFFER_SIZE_TLR = 5
ALPHA_DQN = 0.01
ALPHA_FHRBF = 0.1
ALPHA_LINEAR = 0.1

ALPHA_FHTLR = 0.11
ALPHA_FHTLR_MAX = 0.2

ALPHA_FHTLR_MAX_ER = 0.2
ALPHA_FHTLR_ER = 0.1

ALPHA_QL = 10
K = 15
SCALE = 0.5
W_DECAY = 0.0
EPS_DECAY = 0.99999


DISCRETIZER = Discretizer(
    min_points_states=[0, 0, 0, 0, 0],
    max_points_states=[MAX_BITS, 4, 4, 4, 4],
    bucket_states=[20, 10, 10, 10, 10],
    min_points_actions=[0, 0, 0, 0, 0],
    max_points_actions=[1, 1, 1, 1, 1],
    bucket_actions=[10, 10, 10, 10, 10],
)


def generate_env() -> Env:
    """Creates instances of SourceChannelCodingEnv."""
    env = SourceChannelCodingEnv(
        T=H,
        K=C,
        data_initial=MAX_BITS,
        rho=0.9,
        sigma=0.2,
        beta_success=2.0
    )
    return env


def get_agent(agent_name: str):
    """Factory function to create agents based on name."""
    if agent_name == 'dqn':
        return Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE)
    elif agent_name == 'dfhqn':
        return DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE)
    elif agent_name == "fhtlr_max":
        return FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_MAX, H, K, SCALE, w_decay=W_DECAY, buffer_size=1)
    elif agent_name == "fhtlr_true":
        return FHTlr(DISCRETIZER, ALPHA_FHTLR, H, K, SCALE, w_decay=W_DECAY, buffer_size=1)
    elif agent_name == "fhtlr_max_er":
        return FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_MAX_ER, H, K, SCALE, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    elif agent_name == "fhtlr_true_er":
        return FHTlr(DISCRETIZER, ALPHA_FHTLR_ER, H, K, SCALE, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    elif agent_name == "fhql":
        return FHQLearning(DISCRETIZER, ALPHA_QL, H, 0.1, 1)
    elif agent_name == "fhbf":
        return FHLinear(DISCRETIZER, ALPHA_LINEAR, H, BUFFER_SIZE)
    elif agent_name == "fhrbf":
        return FHRBF(DISCRETIZER, ALPHA_FHRBF, H, BUFFER_SIZE)
    elif agent_name == "ql":
        return QLearning(DISCRETIZER, ALPHA_LINEAR, 0.99)
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")


def run_experiment_with_agent(agent_name: str, n_exp: int) -> List[float]:
    """Runs a single experiment for a specific agent."""
    random.seed(n_exp)
    np.random.seed(n_exp)
    torch.manual_seed(n_exp)
    
    agent = get_agent(agent_name)
    
    Gs = run_experiment(
        n=n_exp,
        E=EPISODES,
        H=H,
        eps=1.0,
        eps_decay=EPS_DECAY,
        env=generate_env(),
        agent=agent
    )
    return Gs


def run_parallel(names: List[str], agents: List[str], n_exps: int, delta_exp: int = 0):
    """Runs experiments in parallel."""
    with Pool() as pool:
        # Generate arguments for starmap
        args = []
        for i in range(len(agents)):
            for n_exp in range(n_exps):
                args.append((agents[i], n_exp + delta_exp))
        
        results = pool.starmap(run_experiment_with_agent, args)

    # Process and save results
    idx = 0
    for name in names:
        file_path = f"results/channelcoding_{name}.npy"
        
        # Slice results for the current agent
        current_agent_results = results[idx : idx + n_exps]
        
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            combined_data = np.concatenate((existing_data, current_agent_results), axis=0)
        else:
            combined_data = current_agent_results
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        np.save(file_path, combined_data)
        idx += n_exps


def run_channel_coding_simulations():
    """Runs the main simulations."""
    # Uncomment lines to run specific agents
    # run_parallel(['dqn'], ['dqn'], N_EXPS)
    # run_parallel(['dfhqn'], ['dfhqn'], N_EXPS)
    run_parallel(['fhtlr_max'], ['fhtlr_max'], N_EXPS)
    run_parallel(['fhtlr_true'], ['fhtlr_true'], N_EXPS)
    run_parallel(['fhtlr_max_er'], ['fhtlr_max_er'], N_EXPS)
    run_parallel(['fhtlr_true_er'], ['fhtlr_true_er'], N_EXPS)
    # run_parallel(['fhrbf'], ['fhrbf'], N_EXPS)


if __name__ == "__main__":
    run_channel_coding_simulations()
