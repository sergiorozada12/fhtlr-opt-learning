import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

from src.environments import WirelessCommunicationsEnv
from src.utils import Discretizer
from src.agents.dqn import DFHqn, Dqn
from src.agents.fhtlr import FHMaxTlr, FHTlr
from src.trainer import run_test_episode,run_train_episode,run_experiment
from src.plots import plot_wireless

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
K = 20
SCALE = 0.5
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
    fhtlr_max_learner = FHMaxTlr(DISCRETIZER, ALPHA_FHTLR, H, K, SCALE, w_decay=W_DECAY)
    fhtlr_true_learner = FHTlr(DISCRETIZER, ALPHA_FHTLR, H, K, SCALE, w_decay=W_DECAY)

    agents = [dqn_learner,dfhqn_learner,fhtlr_max_learner,fhtlr_true_learner]

    run_paralell(['dqn2','dfhqn2',"fhtlr2_max","fhtlr2_true"], agents)
    plot_wireless()
