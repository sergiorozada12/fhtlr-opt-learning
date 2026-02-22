import os
from multiprocessing import Pool
import random
import numpy as np
import torch

from src.environments import PendulumTerminalReward
from src.utils import Discretizer
from src.agents.dqn import Dqn, DFHqn
from src.agents.fhtlr import FHTlr, FHMaxTlr
from src.agents.ql import FHQLearning
from src.agents.rbf import FHRBF
from src.trainer import run_experiment

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

# Parameters from pendulum_terminal_playground.ipynb
H = 10
MAX_TORQUE = 2.0
EPISODES = 4000
BUFFER_SIZE = 10
ALPHA_DQN = 0.01
GAMMA = 1.0

# Tensor method hyperparameters
K = 20
SCALE_max = 0.01
SCALE_true = 0.01
SCALE_true_er = 0.01
W_DECAY = 0.0
BUFFER_SIZE_TLR = 10

ALPHA_FHTLR_max = 0.01
ALPHA_FHTLR_true = 0.01
ALPHA_FHTLR_max_er = 0.01
ALPHA_FHTLR_true_er = 0.01
ALPHA_QL = 0.1
ALPHA_FHRBF = 0.1

EPS_DECAY = 0.9999

DISCRETIZER = Discretizer(
    min_points_states=[-1.0, -1.0, -5.0],
    max_points_states=[1.0, 1.0, 5.0],
    bucket_states=[10, 10, 10], 
    min_points_actions=[-MAX_TORQUE],
    max_points_actions=[MAX_TORQUE],
    bucket_actions=[11], 
)

# Ensure discretizer uses float64
DISCRETIZER.min_points_states = DISCRETIZER.min_points_states.astype(np.float64)
DISCRETIZER.max_points_states = DISCRETIZER.max_points_states.astype(np.float64)
DISCRETIZER.min_points_actions = DISCRETIZER.min_points_actions.astype(np.float64)
DISCRETIZER.max_points_actions = DISCRETIZER.max_points_actions.astype(np.float64)

def generate_env():
    # Parameters from notebook code execution
    return PendulumTerminalReward(T=H, max_torque=MAX_TORQUE, low_thdot=0.0, high_thdot=1.0)

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

    # Note: run_experiment defined in trainer.py returns Gs (list of returns)
    Gs = run_experiment(n=n_exp, E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=generate_env(), agent=agent)
    return Gs

def run_paralell(names, agents, n_exps, delta_exp=0):
    # Using multiprocessing to run multiple experiments
    with Pool() as pool:
        # starmap arguments: list of (agent_name, n_exp)
        # We run n_exps per agent
        tasks = []
        for i in range(len(agents)):
            for n in range(n_exps):
                tasks.append((agents[i], n + delta_exp))
        
        results_flat = pool.starmap(run_experiment_with_agent, tasks)

    # Process results and save
    # results_flat is a list of Gs arrays. We need to group them by agent.
    # The tasks were ordered by agent then experiment.
    
    results_per_agent = {} # agent_index -> list of results
    
    current_idx = 0
    for i in range(len(agents)):
        agent_results = results_flat[current_idx : current_idx + n_exps]
        current_idx += n_exps
        # Agent results is a list of arrays (one per experiment). Stack them?
        # Typically we want shape (n_exps, n_evaluations)
        # Let's check wireless.py how it does.
        # wireless.py: combined_data = results[idx:idx + n_exps] -> this is just a list of arrays if not np.array()
        # np.save saves the array. If inputs are different length, it saves as object.
        # Here episodes are fixed, so we can stack.
        
        combined_data = np.array(agent_results)
        
        name = names[i]
        file_path = f"results/pendulum_{name}.npy"
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            # Check if dimensions match. If not, maybe just overwrite or append strictly?
            # Assuming consistent dimensions
            combined_data = np.concatenate((existing_data, combined_data), axis=0)
        
        np.save(file_path, combined_data)


def run_pendulum_simulations(n_exps=100):
    # Running all agents
    agents_to_run = ['dqn', 'dfhqn', 'fhtlr_max', 'fhtlr_true', 'fhtlr_max_er', 'fhtlr_true_er', 'fhrbf', 'fhql']
    for agent in agents_to_run:
        print(f"Running experiments for {agent}...")
        run_paralell([agent], [agent], n_exps)

if __name__ == "__main__":
    # Default to a small number for testing if run directly, or use argument
    # But usually this is called by the user or script. 
    # Let's use 20 as default in the function but if main, maybe read args or just run default.
    run_pendulum_simulations(n_exps=100) # Using 4 for quick testing if user runs it
