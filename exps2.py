import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

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
import pickle

GAMMA = 0.9
H = 4
C = 4
Max_bits = 100.0

DISCRETIZER = Discretizer(
    min_points_states=[0, 0, 0, 0, 0],
    max_points_states=[Max_bits, 4, 4, 4, 4],
    bucket_states=[20, 10, 10, 10, 10],
    min_points_actions=[0, 0, 0, 0, 0],
    max_points_actions=[1, 1, 1, 1, 1],
    bucket_actions=[10, 10, 10, 10, 10],
)

N_EXPS = 100
EPISODES = 100_000 
BUFFER_SIZE = 1_000
BUFFER_SIZE_TLR = 5
ALPHA_DQN = 0.01
ALPHA_FHRBF = 0.1
ALPHA_LINEAR = 0.1

ALPHA_FHTLR = 0.05
ALPHA_FHTLR_MAX = 0.25

ALPHA_FHTLR_MAX_ER = 0.01
ALPHA_FHTLR_ER = 0.005

ALPHA_QL = 10
K = 15
SCALE = 0.5
W_DECAY = 0.0
EPS_DECAY = (0.99995)**(30_000/EPISODES)

hidden_layers = [64, 64, 64]

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

def run_experiment_with_agent(agent_name, n_exp,alpha_exp=None,hidden_layers=None):
    random.seed(n_exp)
    np.random.seed(n_exp)
    torch.manual_seed(n_exp)
    if agent_name == 'dqn':
        agent = Dqn(DISCRETIZER, alpha_exp, GAMMA, BUFFER_SIZE, hidden_layers=hidden_layers)
    if agent_name == 'dfhqn':
        agent = DFHqn(DISCRETIZER, alpha_exp, H, BUFFER_SIZE, hidden_layers=hidden_layers)
    if agent_name == "fhtlr_max":
        agent = FHMaxTlr(DISCRETIZER, alpha_exp, H, K, SCALE, w_decay=W_DECAY, buffer_size=1)
    if agent_name == "fhtlr_true":
        agent = FHTlr(DISCRETIZER, alpha_exp, H, K, SCALE, w_decay=W_DECAY, buffer_size=1)
    if agent_name == "fhtlr_max_er":
        agent = FHMaxTlr(DISCRETIZER, alpha_exp, H, K, SCALE, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    if agent_name == "fhtlr_true_er":
        agent = FHTlr(DISCRETIZER, alpha_exp, H, K, SCALE, w_decay=W_DECAY, buffer_size=BUFFER_SIZE_TLR)
    if agent_name == "fhql":
        agent = FHQLearning(DISCRETIZER, alpha_exp, H, 0.1, 1)
    if agent_name == "fhbf":
        agent = FHLinear(DISCRETIZER, alpha_exp, H, BUFFER_SIZE)
    if agent_name == "fhrbf":
        agent = FHRBF(DISCRETIZER, alpha_exp, H, BUFFER_SIZE)
    if agent_name == "ql":
        agent = QLearning(DISCRETIZER, alpha_exp,0.99)
    
    Gs  = run_experiment( n=n_exp,E=EPISODES, H=H, eps=1.0, eps_decay=EPS_DECAY, env=generate_env(), agent=agent)                               
    return Gs

def run_experiment2(agent_name, alpha_exp=None, n_exps=1,hidden_layers=None):
    if alpha_exp is None:
        if agent_name == 'dqn':
            alpha_exp = ALPHA_DQN
        if agent_name == 'dfhqn':
            alpha_exp = ALPHA_DQN
        if agent_name == "fhtlr_max":
            alpha_exp = ALPHA_FHTLR_MAX
        if agent_name == "fhtlr_true":
            alpha_exp = ALPHA_FHTLR
        if agent_name == "fhtlr_max_er":
            alpha_exp = ALPHA_FHTLR_MAX_ER
        if agent_name == "fhtlr_true_er":
            alpha_exp = ALPHA_FHTLR_ER
        if agent_name == "fhql":
            alpha_exp = ALPHA_QL
        if agent_name == "fhbf":
            alpha_exp = ALPHA_LINEAR
        if agent_name == "fhrbf":
            alpha_exp = ALPHA_FHRBF
        if agent_name == "ql":
            alpha_exp = ALPHA_LINEAR

    all_rewards = []
    for seed in range(n_exps):
        print("Alpha:", alpha_exp, "Seed:", seed, "Agent:", agent_name)
        rewards = run_experiment_with_agent(agent_name, seed, alpha_exp=alpha_exp,hidden_layers=hidden_layers)
        all_rewards.append(rewards)
        
    # Save the results
    all_rewards = np.array(all_rewards)

    if agent_name == "dfhqn" or agent_name == "dqn":
        name = f"{agent_name}_alpha{alpha_exp}_epsdecay{EPS_DECAY:.8f}_hiden_{hidden_layers}"
    else:
        name = f"{agent_name}_alpha{alpha_exp}_k{K}_epsdecay{EPS_DECAY:.8f}"

    avg_reward = np.mean(all_rewards, axis=0)
    std_reward = np.std(all_rewards, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_reward, label=f'{agent_name} (media)', color='blue')
    plt.fill_between(range(len(avg_reward)), avg_reward - std_reward, avg_reward + std_reward, alpha=0.2, color='blue')
    # Configuración de los límites y marcas del eje Y
    plt.ylim(-30, 0)  # Limitar el eje Y entre -30 y 0
    plt.yticks(range(0, -31, -3))  # Marcas cada 3 unidades desde 0 hasta -30

    plt.title(f"Rendimiento promedio - {agent_name}")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figures/{name}.png")

    # Crear un diccionario con los datos
    datos = {
        'avg_reward': avg_reward,
        'std_reward': std_reward
    }

    # Ruta donde guardar el archivo
    save_path = f"./resultados/{name}_recompensas.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    # Guardar el string con pickle
    with open(save_path, 'wb') as f:
        pickle.dump(datos, f)

    print(f"Nombre guardado en: {save_path}")

#Siguiente posible cambioa implementar 
print("Running experiments with DFHqn agent 1")

EPS_DECAY = np.exp(np.log(0.1) / EPISODES)
run_experiment2("dqn", alpha_exp=0.5, n_exps=1, hidden_layers=[32,32])
run_experiment2("dqn", alpha_exp=10, n_exps=1,hidden_layers=[32,32])
run_experiment2("dqn", alpha_exp=1, n_exps=1,hidden_layers=[32,32])
"""print("Running experiments with DFHqn agent 2")
EPS_DECAY = np.exp(np.log(0.001) / EPISODES)
run_experiment2("dfhqn", alpha_exp=0.05, n_exps=1, hidden_layers=[128,64])
run_experiment2("dfhqn", alpha_exp=0.01, n_exps=1,hidden_layers=[128,128,64])
run_experiment2("dfhqn", alpha_exp=0.001, n_exps=1,hidden_layers=[128,64])
print("Running experiments with DFHqn agent 3")
EPS_DECAY = np.exp(np.log(0.2) / EPISODES)
run_experiment2("dfhqn", alpha_exp=0.05, n_exps=1, hidden_layers=[128,64])
run_experiment2("dfhqn", alpha_exp=0.1, n_exps=1,hidden_layers=[128,128,64])
run_experiment2("dfhqn", alpha_exp=0.1, n_exps=1,hidden_layers=[128,64])"""