import os
from multiprocessing import Pool

import random
import numpy as np
import torch

from src.environments import SourceChannelCodingEnv
from src.utils import Discretizer
from src.agents.dqn import DFHqn, Dqn
from src.agents.fhtlr import FHMaxTlr, FHTlr
from src.agents.ql import QLearning, FHQLearning
from src.agents.bf import FHLinear
from src.agents.rbf import FHRBF
from src.trainer import run_experiment


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

GAMMA = 1.0
H = 5
C = 4
Max_bits = 15.0

DISCRETIZER = Discretizer(
    min_points_states=[0, 0, 0, 0, 0],
    max_points_states=[Max_bits, 4, 4, 4, 4],
    bucket_states=[8, 8, 8, 8, 8],
    min_points_actions=[0, 0, 0, 0, 0],
    max_points_actions=[1, 1, 1, 1, 1],
    bucket_actions=[6, 6, 6, 6, 6],
)
# Ensure discretizer uses float64 for compatibility with agents
DISCRETIZER.min_points_states = DISCRETIZER.min_points_states.astype(np.float64)
DISCRETIZER.max_points_states = DISCRETIZER.max_points_states.astype(np.float64)
DISCRETIZER.min_points_actions = DISCRETIZER.min_points_actions.astype(np.float64)
DISCRETIZER.max_points_actions = DISCRETIZER.max_points_actions.astype(np.float64)


N_EXPS = 100
EPISODES = 30_000 
BUFFER_SIZE = 100
BUFFER_SIZE_TLR = 100

ALPHA_DQN = 0.1
ALPHA_DFHQN = 0.01

ALPHA_FHTLR = 0.01
ALPHA_FHTLR_MAX = 0.1

ALPHA_FHTLR_MAX_ER = 0.1
ALPHA_FHTLR_ER = 0.01

ALPHA_QL = 0.1
ALPHA_FHRBF = 0.1
ALPHA_LINEAR = 0.1

K = 20
SCALE = 0.1
SCALE_MAX = 1.0
W_DECAY = 0.0
EPS_DECAY = 0.9999


def generate_env():
    env = SourceChannelCodingEnv(
        T=H, 
        K=C, 
        data_initial=Max_bits, 
        rho=0.9, 
        sigma=0.2, 
        beta_success=2.0
    )
    return env


def run_experiment_with_agent(agent_name, n_exp):
    random.seed(n_exp)
    np.random.seed(n_exp)
    torch.manual_seed(n_exp)
    if agent_name == 'dqn':
        agent = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE, hidden_layers=[64, 64])
    if agent_name == 'dfhqn':
        agent = DFHqn(DISCRETIZER, ALPHA_DFHQN, H, BUFFER_SIZE, hidden_layers=[64, 64])
    if agent_name == "fhtlr_max":
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_MAX, H, K, SCALE_MAX, w_decay=W_DECAY, buffer_size=1)
    if agent_name == "fhtlr_true":
        agent = FHTlr(DISCRETIZER, ALPHA_FHTLR, H, K, SCALE, w_decay=W_DECAY, buffer_size=1)
    if agent_name == "fhtlr_max_er":
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_MAX_ER, H, K, SCALE_MAX, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    if agent_name == "fhtlr_true_er":
        agent = FHTlr(DISCRETIZER, ALPHA_FHTLR_ER, H, K, SCALE, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    if agent_name == "fhql":
        agent = FHQLearning(DISCRETIZER, ALPHA_QL, H, 0.1, 1)
    if agent_name == "fhbf":
        agent = FHLinear(DISCRETIZER, ALPHA_LINEAR, H, BUFFER_SIZE)
    if agent_name == "fhrbf":
        agent = FHRBF(DISCRETIZER, ALPHA_FHRBF, H, BUFFER_SIZE)
    if agent_name == "ql":
        agent = QLearning(DISCRETIZER, ALPHA_QL, GAMMA)
    
    Gs  = run_experiment(n=n_exp, E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=generate_env(), agent=agent)                               
    return Gs


def run_paralell(names, agents, n_exps, delta_exp=0):
    with Pool() as pool:
        results = pool.starmap(run_experiment_with_agent, [(agents[i], n_exp+delta_exp) for i in range(len(agents)) for n_exp in range(n_exps)])

    idx = 0
    for i, name in enumerate(names):
        file_path = f"results/channel_coding_{name}.npy"
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            combined_data = np.concatenate((existing_data, results[idx:idx + n_exps]), axis=0)
        else:
            combined_data = results[idx:idx + n_exps]
        
        np.save(file_path, combined_data)
        idx += n_exps


def run_channel_coding_simulations():
    run_paralell(['dqn'], ['dqn'], N_EXPS)
    run_paralell(['dfhqn'], ['dfhqn'], N_EXPS)
    run_paralell(['fhtlr_max'], ['fhtlr_max'], N_EXPS)
    run_paralell(['fhtlr_true'], ['fhtlr_true'], N_EXPS)
    run_paralell(['fhtlr_max_er'], ['fhtlr_max_er'], N_EXPS)
    run_paralell(['fhtlr_true_er'], ['fhtlr_true_er'], N_EXPS)
    run_paralell(['fhrbf'], ['fhrbf'], N_EXPS)
    run_paralell(['fhbf'], ['fhbf'], N_EXPS)
