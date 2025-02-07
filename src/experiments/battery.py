import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import torch

from src.environments import BatteryChargingEnv
from src.utils import Discretizer
from src.agents.dqn import DFHqn, Dqn
from src.agents.fhtlr import FHMaxTlr, FHTlr
from src.agents.ql import QLearning, FHQLearning
from src.agents.bf import FHLinear
from src.agents.rbf import RBF, FHRBF
from src.trainer import run_test_episode,run_train_episode,run_experiment
from src.plots import plot_wireless
import os
import pickle

torch.set_num_threads(1)

#Enviroment
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

#Experiments
EPISODES = 20_000
BUFFER_SIZE = 1_000
ALPHA_DQN = 0.1
ALPHA_FHTLR_max = 0.1
ALPHA_FHTLR_true = 0.05
ALPHA_QL = 10
ALPHA_FHRBF = 0.1
ALPHA_LINEAR = 0.1
K = 50
SCALE_max = 0.5
SCALE_true = 0.1
SCALE_QL = 0.01
W_DECAY = 0.0
EPS_DECAY = (0.9999)**(30_000/EPISODES)
BUFFER_SIZE_TLR = 5
#N_EXPS = 100

def generate_env():
    env = BatteryChargingEnv(H=H)
    return env

def run_experiment_with_agent(agent_name, n_exp):
    if agent_name == 'dqn':
        agent = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE)
    if agent_name == 'dfhqn':
        agent = DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE)
    if agent_name == "fhtlr_max":
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_max, H, K, SCALE_max, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    if agent_name == "fhtlr_true":
        agent = FHTlr(DISCRETIZER, ALPHA_FHTLR_true, H, K, SCALE_true, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    if agent_name == "fhql":
        agent = FHQLearning(DISCRETIZER, ALPHA_QL, H, SCALE_QL, 1000)
    if agent_name == "fhbf":
        agent = FHLinear(DISCRETIZER, ALPHA_LINEAR, H, BUFFER_SIZE)
    if agent_name == "fhrbf":
        agent = FHRBF(DISCRETIZER, ALPHA_FHRBF, H, BUFFER_SIZE)
    if agent_name == "ql":
        agent = QLearning(DISCRETIZER, ALPHA_LINEAR,0.99)
    
    Gs  = run_experiment( n=n_exp,E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=generate_env(), agent=agent)
    """agent_file_path = f'results/agents/battery/{agent_name}/battery_{agent_name}_{n_exp}.pkl'
    with open(agent_file_path, 'wb') as f:
        pickle.dump(agent, f)"""                                
    return Gs

def run_paralell(names, agents,n_exps,delta_exp=0):
    with Pool() as pool:
        results = pool.starmap(run_experiment_with_agent, [(agents[i], n_exp+delta_exp) for i in range(len(agents)) for n_exp in range(n_exps)])

    idx = 0
    for i, name in enumerate(names):
        file_path = f"results/battery_{name}.npy"
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            combined_data = np.concatenate((existing_data, results[idx:idx + n_exps]), axis=0)
        else:
            combined_data = results[idx:idx + n_exps]
        
        np.save(file_path, combined_data)
        idx += n_exps

def run_battery_simulations():

    #agents = [dqn_learner,dfhqn_learner,fhtlr_max_learner,fhtlr_true_learner,fhql,bf]
    #run_paralell(['dqn2','dfhqn2',"fhtlr2_max","fhtlr2_true"], agents)

    N_EXPS = 100
    #run_paralell(['dqn-100'], ['dqn'],N_EXPS)
    #run_paralell(['dfhqn-100'], ['dfhqn'],N_EXPS)
    run_paralell(['fhtlr_max-explore'], ['fhtlr_max'],N_EXPS)
    run_paralell(['fhtlr_true-explore'], ['fhtlr_true'],N_EXPS)
    #run_paralell(['fhbf-explore'], ['fhbf'],N_EXPS)
    #run_paralell(['fhrbf-explore'], ['fhrbf'],N_EXPS)
    #run_paralell(['fhql-100'], ['fhql'],N_EXPS)
    
    """agents = ['fhql']
    names = ['fhql-100']
    total_episodes = N_EXPS
    episodes_per_run = 50 

    full_runs = total_episodes // episodes_per_run
    remaining_episodes = total_episodes % episodes_per_run

    for i in range(full_runs):
        print("Lote ", i)
        run_paralell(names, agents,episodes_per_run, i * episodes_per_run)
    
    if remaining_episodes > 0:
        run_paralell(names, agents,remaining_episodes, full_runs * episodes_per_run)"""

