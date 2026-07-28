import os
from multiprocessing import Pool
import random
import numpy as np
import torch


from src.environments import BatteryChargingEnv
from src.utils import Discretizer
from src.agents.dqn import DFHqn, Dqn
from src.agents.fhtlr import FHMaxTlr, FHTlr
from src.agents.ql import QLearning, FHQLearning
from src.agents.bf import FHLinear
from src.agents.rbf import FHRBF
from src.agents.pg import PG
from src.agents.ctrl_ucb import CTRLUCBM
from src.agents.lsvi_ucb import FHLSVIUCB
from src.trainer import run_ctrl_ucbm_experiment, run_experiment, run_pg_experiment


torch.set_num_threads(1)


GAMMA = 0.99
H = 5
 
ENV = BatteryChargingEnv(H=H)
 
DISCRETIZER = Discretizer(
    min_points_states=[0, 0, 0, 0],
    max_points_states=[1, 1, 1, 1],
    bucket_states=[10, 10, 10, 10],
    min_points_actions=[0, 0, 0],
    max_points_actions=[1, 1, 1],
    bucket_actions=[10, 10, 10],
)
 
EPISODES = 30_000
BUFFER_SIZE = 1_000
ALPHA_DQN = 0.1
ALPHA_FHTLR_max = 0.1
ALPHA_FHTLR_true = 0.05
ALPHA_FHTLR_max_er = 0.05
ALPHA_FHTLR_true_er = 0.01
ALPHA_QL = 10
ALPHA_FHRBF = 0.1
ALPHA_LINEAR = 0.1
ALPHA_PG = 3e-4
ALPHA_CTRL = 3e-4
K = 50
SCALE_max = 0.5
SCALE_true = 0.1
SCALE_QL = 0.01
W_DECAY = 0.0
EPS_DECAY = (0.9999)**(30_000/EPISODES)
BUFFER_SIZE_TLR = 5
N_EXPS = 100


def generate_env():
    env = BatteryChargingEnv(H=H)
    return env


def run_experiment_with_agent(agent_name, n_exp):
    random.seed(n_exp)
    np.random.seed(n_exp)
    torch.manual_seed(n_exp)
    if agent_name == 'dqn':
        agent = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE, hidden_layers=[32])
    if agent_name == 'dfhqn':
        agent = DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE, hidden_layers=[32])
    if agent_name == "fhtlr_max":
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_max, H, K, SCALE_max, w_decay=W_DECAY, buffer_size=1)
    if agent_name == "fhtlr_true":
        agent = FHTlr(DISCRETIZER, ALPHA_FHTLR_true, H, K, SCALE_true, w_decay=W_DECAY, buffer_size=1)
    if agent_name == "fhtlr_max_er":
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_max_er, H, K, SCALE_max, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    if agent_name == "fhtlr_true_er":
        agent = FHTlr(DISCRETIZER, ALPHA_FHTLR_true_er, H, K, SCALE_true, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    if agent_name == "fhql":
        agent = FHQLearning(DISCRETIZER, ALPHA_QL, H, SCALE_QL, 1000)
    if agent_name == "fhbf":
        agent = FHLinear(DISCRETIZER, ALPHA_LINEAR, H, BUFFER_SIZE)
    if agent_name == "fhrbf":
        agent = FHRBF(DISCRETIZER, ALPHA_FHRBF, H, BUFFER_SIZE)
    if agent_name == "ql":
        agent = QLearning(DISCRETIZER, ALPHA_LINEAR,0.99)
    if agent_name == "pg":
        agent = PG(DISCRETIZER, H, alpha=ALPHA_PG, gamma=GAMMA)
    if agent_name == "ctrl_ucbm":
        agent = CTRLUCBM(DISCRETIZER, H, gamma=GAMMA, actor_lr=ALPHA_CTRL)
    if agent_name == "lsvi_ucb":
        agent = FHLSVIUCB(DISCRETIZER, H, bonus_coef=5.0, feature_seed=n_exp)

    if agent_name == "pg":
        Gs = run_pg_experiment(
            n=n_exp, E=EPISODES, H=H, env=generate_env(), agent=agent
        )
    elif agent_name == "ctrl_ucbm":
        Gs = run_ctrl_ucbm_experiment(
            n=n_exp, E=EPISODES, H=H, env=generate_env(), agent=agent
        )
    else:
        Gs = run_experiment( n=n_exp,E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=generate_env(), agent=agent)
    return Gs


def run_paralell(names, agents, n_exps, delta_exp=0):
    with Pool() as pool:
        results = pool.starmap(run_experiment_with_agent, [(agents[i], n_exp+delta_exp) for i in range(len(agents)) for n_exp in range(n_exps)])


    idx = 0
    for _, name in enumerate(names):
        file_path = f"results/battery_{name}.npy"
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            combined_data = np.concatenate((existing_data, results[idx:idx + n_exps]), axis=0)
        else:
            combined_data = results[idx:idx + n_exps]
        
        np.save(file_path, combined_data)
        idx += n_exps


def run_pg_battery_simulations(n_exps=100, delta_exp=0):
    run_paralell(['pg'], ['pg'], n_exps, delta_exp)


def run_ctrl_ucbm_battery_simulations(n_exps=10, delta_exp=0):
    run_paralell(['ctrl_ucbm'], ['ctrl_ucbm'], n_exps, delta_exp)


def run_lsvi_ucb_battery_simulations(n_exps=10, delta_exp=0):
    run_paralell(['lsvi_ucb'], ['lsvi_ucb'], n_exps, delta_exp)


def run_battery_simulations():

    run_paralell(['dqn'], ['dqn'], N_EXPS)
    run_paralell(['dfhqn'], ['dfhqn'], N_EXPS)
    run_paralell(['fhtlr_max'], ['fhtlr_max'], N_EXPS)
    run_paralell(['fhtlr_true'], ['fhtlr_true'], N_EXPS)
    run_paralell(['fhtlr_max_er'], ['fhtlr_max_er'], N_EXPS)
    run_paralell(['fhtlr_true_er'], ['fhtlr_true_er'], N_EXPS)
    run_paralell(['fhrbf'], ['fhrbf'],N_EXPS)
