import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

from src.environments import WirelessCommunicationsEnv
from src.utils import Discretizer
from src.agents.dqn import DFHqn, Dqn
from src.agents.fhtlr import FHMaxTlr, FHTlr
from src.agents.ql import QLearning, FHQLearning
from src.trainer import run_test_episode,run_train_episode,run_experiment
from src.plots import plot_wireless
import os

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
EPISODES = 20_000
BUFFER_SIZE = 1_000
ALPHA_DQN = 0.01
ALPHA_FHTLR = 0.1
ALPHA_QL = 10
K = 20
SCALE = 0.5
W_DECAY = 0.0
EPS_DECAY = 0.9999
N_EXPS = 7

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
    if agent_name == 'dqn2':
        agent = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE)
    if agent_name == 'dfhqn2':
        agent = DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE)
    if agent_name == "fhtlr2_max":
        agent = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR, H, K, SCALE, w_decay=W_DECAY)
    if agent_name == "fhtlr2_true":
        agent = FHTlr(DISCRETIZER, ALPHA_FHTLR, H, K, SCALE, w_decay=W_DECAY)
    if agent_name == "fhql":
        agent = FHQLearning(DISCRETIZER, ALPHA_QL, H, SCALE, 1)
                                   
    return run_experiment( n=n_exp,E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=generate_env(), agent=agent)

def run_paralell(names, agents,delta_exp=0):
    with Pool() as pool:
        results = pool.starmap(run_experiment_with_agent, [(agents[i], n_exp+delta_exp) for i in range(len(agents)) for n_exp in range(N_EXPS)])

    idx = 0
    for i, name in enumerate(names):
        file_path = f"results/{name}.npy"
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            combined_data = np.concatenate((existing_data, results[idx:idx + N_EXPS]), axis=0)
        else:
            combined_data = results[idx:idx + N_EXPS]
        
        np.save(file_path, combined_data)
        idx += N_EXPS

def run_wireless_simulations():

    #agents = [dqn_learner,dfhqn_learner,fhtlr_max_learner,fhtlr_true_learner]
    #run_paralell(['dqn2','dfhqn2',"fhtlr2_max","fhtlr2_true"], agents)
    
    agents = ['fhql']
    names = ['fhql-100']
    total_episodes = 100
    episodes_per_run = N_EXPS 

    full_runs = total_episodes // episodes_per_run
    remaining_episodes = total_episodes % episodes_per_run

    for i in range(full_runs):
        print("Lote ", i)
        run_paralell(names, agents, i * episodes_per_run)
    
    if remaining_episodes > 0:
        run_paralell(names, agents, full_runs * episodes_per_run)
    
    plot_wireless()
