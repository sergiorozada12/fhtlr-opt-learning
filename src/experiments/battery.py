import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

from src.environments import BatteryChargingEnv
from src.utils import Discretizer
from src.agents.dqn import DFHqn, Dqn
from src.agents.fhtlr import FHMaxTlr, FHTlr
from src.trainer import run_test_episode,run_train_episode,run_experiment
from src.plots import plot_wireless

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
EPISODES = 30_000
BUFFER_SIZE = 1_000
ALPHA_DQN = 0.1
ALPHA_FHTLR_max = 0.1
ALPHA_FHTLR_true = 0.1
K = 50
SCALE_max = 0.5
SCALE_true = 0.1
W_DECAY = 0.0
EPS_DECAY = 0.9999
N_EXPS = 20 

def run_experiment_with_agent(agent, n_exp):
    return run_experiment( n=n_exp,E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=ENV, agent=agent)

def run_paralell(names, agents):
    with Pool() as pool:
        results = pool.starmap(run_experiment_with_agent, [(agents[i], n_exp) for i in range(len(agents)) for n_exp in range(N_EXPS)])

    idx = 0
    for i, name in enumerate(names):
        # Recopilamos los resultados para el agente en la posición i
        np.save(f"results/{name}.npy", results[idx:idx + N_EXPS])
        idx += N_EXPS

def run_wireless_simulations():

    dqn_learner = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE)
    dfhqn_learner = DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE)
    fhtlr_max_learner = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR_max, H, K, SCALE_max, w_decay=W_DECAY)
    fhtlr_true_learner = FHTlr(DISCRETIZER, ALPHA_FHTLR_true, H, K, SCALE_true, w_decay=W_DECAY)

    agents = [dqn_learner,dfhqn_learner,fhtlr_max_learner,fhtlr_true_learner]

    run_paralell(['dqn2','dfhqn2',"fhtlr2_max","fhtlr2_true"], agents)
    plot_wireless()
