import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

from src.environments import WirelessCommunicationsEnv
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

import random
import numpy as np
import torch
#Enviroment
GAMMA = 0.9
H = 5
C = 3

ENV = WirelessCommunicationsEnv(
    T=H,
    K=C,
    snr_max=8,
    snr_min=4,
    snr_autocorr=0.9,
    P_occ=np.array(
            [
                [0.5, 0.3],
                [0.5, 0.7],
            ]
        ),
        occ_initial=[1]*C,
        batt_harvest=2.0,
        P_harvest=0.5,
        batt_initial=10,
        batt_max_capacity=5,
        batt_weight=1.0,
        queue_initial=5,
        queue_weight=0.1,
        loss_busy=0.5,
)

DISCRETIZER = Discretizer(
    min_points_states=[0, 0, 0, 0, 0, 0, 0, 0],
    max_points_states=[10, 10, 10, 1, 1, 1, 10, 10],
    bucket_states=[10, 10, 10, 2, 2, 2, 10, 10],
    min_points_actions=[0, 0, 0],
    max_points_actions=[2, 2, 2],
    bucket_actions=[10, 10, 10],
)

#Experiments
EPISODES = 60_000
BUFFER_SIZE = 1_000
ALPHA_DQN = 0.01
ALPHA_FHRBF = 0.1
ALPHA_LINEAR = 0.1
ALPHA_FHTLR = 0.01
ALPHA_FHTLR_MAX = 0.05
ALPHA_QL = 10
K = 20
SCALE = 0.5
W_DECAY = 0.0
EPS_DECAY = (0.99999)**(20_000/40_000)
N_EXPS = 100

def generate_env():
    env = WirelessCommunicationsEnv(
        T=H,
        K=C,
        snr_max=8,
        snr_min=4,
        snr_autocorr=0.9,
        P_occ=np.array(
                [
                    [0.5, 0.3],
                    [0.5, 0.7],
                ]
            ),
            occ_initial=[1]*C,
            batt_harvest=2.0,
            P_harvest=0.5,
            batt_initial=10,
            batt_max_capacity=5,
            batt_weight=1.0,
            queue_initial=5,
            queue_weight=0.1,
            loss_busy=0.5,
        )
    return env

def run_experiment_with_agent(agent_name, n_exp):
    random.seed(n_exp)
    np.random.seed(n_exp)
    torch.manual_seed(n_exp)
    if agent_name == 'dqn':
        agent = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE)
    if agent_name == 'dfhqn':
        agent = DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE)
    if agent_name == "fhtlr_max":
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_MAX, H, K, SCALE, w_decay=W_DECAY)
    if agent_name == "fhtlr_true1":
        alpha=0.01
        agent = FHTlr(DISCRETIZER, alpha, H, K, SCALE, w_decay=W_DECAY)
    if agent_name == "fhtlr_true2":
        alpha=0.02
        agent = FHTlr(DISCRETIZER, alpha, H, K, SCALE, w_decay=W_DECAY)   
    if agent_name == "fhtlr_true5":
        alpha=0.05
        agent = FHTlr(DISCRETIZER, alpha, H, K, SCALE, w_decay=W_DECAY)
    if agent_name == "fhql":
        agent = FHQLearning(DISCRETIZER, ALPHA_QL, H, 0.1, 1)
    if agent_name == "fhbf":
        agent = FHLinear(DISCRETIZER, ALPHA_LINEAR, H, BUFFER_SIZE)
    if agent_name == "fhrbf":
        agent = FHRBF(DISCRETIZER, ALPHA_FHRBF, H, BUFFER_SIZE)
    if agent_name == "ql":
        agent = QLearning(DISCRETIZER, ALPHA_LINEAR,0.99)
    
    Gs  = run_experiment( n=n_exp,E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=generate_env(), agent=agent)
    """agent_file_path = f'results/agents/wireless/{agent_name}/battery_{agent_name}_{n_exp}.pkl'
    with open(agent_file_path, 'wb') as f:
        pickle.dump(agent, f)"""                                
    return Gs

def run_paralell(names, agents,n_exps,delta_exp=0):
    with Pool() as pool:
        results = pool.starmap(run_experiment_with_agent, [(agents[i], n_exp+delta_exp) for i in range(len(agents)) for n_exp in range(n_exps)])

    idx = 0
    for i, name in enumerate(names):
        file_path = f"results/wireless_{name}.npy"
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            combined_data = np.concatenate((existing_data, results[idx:idx + n_exps]), axis=0)
        else:
            combined_data = results[idx:idx + n_exps]
        
        np.save(file_path, combined_data)
        idx += n_exps

def run_wireless_simulations():

    #agents = [dqn_learner,dfhqn_learner,fhtlr_max_learner,fhtlr_true_learner,fhql,bf]
    #run_paralell(['dqn2','dfhqn2',"fhtlr2_max","fhtlr2_true"], agents)

    N_EXPS = 100
    #run_paralell(['dqn-explore'], ['dqn'],N_EXPS)
    #run_paralell(['dfhqn-explore'], ['dfhqn'],N_EXPS)
    run_paralell(['fhtlr_true-explore-lr01'], ['fhtlr_true1'],N_EXPS)
    run_paralell(['fhtlr_true-explore-lr02'], ['fhtlr_true2'],N_EXPS)
    run_paralell(['fhtlr_true-explore-lr05'], ['fhtlr_true5'],N_EXPS)
    #run_paralell(['fhtlr_true-explore'], ['fhtlr_true'],N_EXPS)
    #run_paralell(['fhbf'], ['fhbf'],N_EXPS)
    #run_paralell(['fhrbf-5'], ['fhrbf'],N_EXPS)
    #run_paralell(['fhql-100'], ['fhql'],N_EXPS)
    
    """agents = ['fhql']
    names = ['fhql-explore']
    total_episodes = N_EXPS
    episodes_per_run = 3 

    full_runs = total_episodes // episodes_per_run
    remaining_episodes = total_episodes % episodes_per_run

    for i in range(full_runs):
        print("Lote ", i)
        run_paralell(names, agents,episodes_per_run, i * episodes_per_run)
    
    if remaining_episodes > 0:
        run_paralell(names, agents,remaining_episodes, full_runs * episodes_per_run)"""

