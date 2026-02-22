import os
from multiprocessing import Pool
import random
import numpy as np
import torch

from src.environments import CartPoleTerminalReward
from src.utils import Discretizer
from src.agents.dqn import Dqn, DFHqn
from src.agents.fhtlr import FHTlr, FHMaxTlr
from src.agents.ql import FHQLearning
from src.agents.rbf import FHRBF
from src.trainer import run_experiment

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

# Parameters from cartpole_playground.ipynb
H = 10
MAX_FORCE = 10.0
LOW_THDOT = 0.0
HIGH_THDOT = 1.0
EPISODES = 4000
BUFFER_SIZE = 100
ALPHA_DQN = 0.02
GAMMA = 1.0

# Tensor method hyperparameters
K = 20
SCALE_max = 0.02
SCALE_true = 0.02
SCALE_true_er = 0.02
W_DECAY = 0
BUFFER_SIZE_TLR = 100

ALPHA_FHTLR_max = 0.02
ALPHA_FHTLR_true = 0.02
ALPHA_FHTLR_max_er = 0.02
ALPHA_FHTLR_true_er = 0.02
ALPHA_QL = 0.1
ALPHA_FHRBF = 0.1

EPS_DECAY = 0.9999

DISCRETIZER = Discretizer(
    min_points_states=[-4.8, -0.5, -0.418, -0.9],
    max_points_states=[4.8, 0.5, 0.418, 0.9],
    bucket_states=[10, 10, 10, 10], 
    min_points_actions=[-MAX_FORCE],
    max_points_actions=[MAX_FORCE],
    bucket_actions=[11], 
)

# Ensure discretizer uses float64
DISCRETIZER.min_points_states = DISCRETIZER.min_points_states.astype(np.float64)
DISCRETIZER.max_points_states = DISCRETIZER.max_points_states.astype(np.float64)
DISCRETIZER.min_points_actions = DISCRETIZER.min_points_actions.astype(np.float64)
DISCRETIZER.max_points_actions = DISCRETIZER.max_points_actions.astype(np.float64)


def generate_env():
    # Parameters from notebook code execution
    return CartPoleTerminalReward(T=H, max_force=MAX_FORCE, low_thdot=LOW_THDOT, high_thdot=HIGH_THDOT)

def run_experiment_with_agent(agent_name, n_exp):
    random.seed(n_exp)
    np.random.seed(n_exp)
    torch.manual_seed(n_exp)

    agent = None
    if agent_name == 'dqn':
        agent = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE, hidden_layers=[64, 64])
    elif agent_name == 'dfhqn':
        agent = DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE, hidden_layers=[64, 64])
    elif agent_name == 'fhtlr_max':
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_max, H, K, SCALE_max, w_decay=W_DECAY, buffer_size=1)
    elif agent_name == 'fhtlr_true':
        agent = FHTlr(DISCRETIZER, ALPHA_FHTLR_true, H, K, SCALE_true, w_decay=W_DECAY, buffer_size=1)
    elif agent_name == 'fhtlr_max_er':
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_max_er, H, K, SCALE_max, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    elif agent_name == 'fhtlr_true_er':
        agent = FHTlr(DISCRETIZER, ALPHA_FHTLR_true_er, H, K, SCALE_true_er, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    elif agent_name == 'fhql':
        agent = FHQLearning(DISCRETIZER, ALPHA_QL, H, 0.1, 1)
    elif agent_name == 'fhrbf':
        agent = FHRBF(DISCRETIZER, ALPHA_FHRBF, H, BUFFER_SIZE)
    
    if agent is None:
        raise ValueError(f"Unknown agent: {agent_name}")

    Gs = run_experiment(n=n_exp, E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=generate_env(), agent=agent)
    return Gs

def run_paralell(names, agents, n_exps, delta_exp=0):
    with Pool() as pool:
        tasks = []
        for i in range(len(agents)):
            for n in range(n_exps):
                tasks.append((agents[i], n + delta_exp))
        
        results_flat = pool.starmap(run_experiment_with_agent, tasks)
    
    results_per_agent = {} 
    
    current_idx = 0
    for i in range(len(agents)):
        agent_results = results_flat[current_idx : current_idx + n_exps]
        current_idx += n_exps
        
        combined_data = np.array(agent_results)
        
        name = names[i]
        file_path = f"results/cartpole_{name}.npy"
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            combined_data = np.concatenate((existing_data, combined_data), axis=0)
        
        np.save(file_path, combined_data)


def run_cartpole_simulations(n_exps=100):
    #agents_to_run = ['dqn', 'dfhqn', 'fhtlr_max', 'fhtlr_true', 'fhtlr_max_er', 'fhtlr_true_er', 'fhrbf', 'fhql']
    agents_to_run = ['fhtlr_true_er']
    for agent in agents_to_run:
        print(f"Running experiments for {agent}...")
        run_paralell([agent], [agent], n_exps)

if __name__ == "__main__":
    run_cartpole_simulations(n_exps=100) 
