import numpy as np
import pickle
from src.environments import GridWorldEnv
from src.agents.ql import QLearning, FHQLearning
from src.agents.dp import BackwardPropagation, FrontPolicyImprovement, BackPolicyImprovement
from src.utils import Discretizer
from src.trainer import run_experiment
from src.plots import plot_gridworld, plot_errors, plot_tensor_rank
from src.agents.bcd_grid import bcd, bcgd
from src.models import PARAFAC

Tamaño = 5

ENV = GridWorldEnv(nS =Tamaño*Tamaño,W = Tamaño, H = Tamaño, nA=5)

discretizer = Discretizer(
    min_points_states=[0, 0],
    max_points_states=[4, 4],
    bucket_states=[Tamaño,Tamaño],
    min_points_actions=[0],
    max_points_actions=[4],
    bucket_actions=[5],
)

#Hyperparameters

BCD_PE_k_list = [15,25,30]
BCD_PE_Q_scale = 1
BCD_PE_num_iter = 500

BCGD_PE_k_list = [15,25,30]
BCGD_PE_Q_scale = 0.5
BCGD_PE_num_iter = 2000
BCGD_PE_alpha = 10e-3

BCD_PI_k_list = [15, 25, 30]
BCD_PI_scale = 0.7
BCD_PI_bcd_num_iter = 5
BCD_PI_policy_num_iter = 100

BCGD_PI_k_list = [15, 25, 30]
BCGD_PI_scale = 0.5
BCGD_PI_bcd_num_iter = 50
BCGD_PI_policy_num_iter = 2000
BCGD_PI_alpha = 10e-3

def BCD_PE_exp(Q_opt, Pi):

    fo_list = []
    errors_list = []
    conv_list = []

    for k in BCD_PE_k_list:
        Q = PARAFAC(
                np.concatenate(
                    [[ENV.H], discretizer.bucket_states, discretizer.bucket_actions]
                ),
                k=k,
                scale= BCD_PE_Q_scale,
                nA=len(discretizer.bucket_actions),
            ).double()
        
        bcd_inv = bcd(Q,Pi,discretizer,ENV,k,Q_opt.reshape(ENV.H,ENV.W,ENV.W,ENV.nA))

        fo_values,errors,convs, Q = bcd_inv.run(BCD_PE_num_iter)
        fo_list.append(fo_values)
        errors_list.append(errors)
        conv_list.append(convs)
    
    data = [fo_list, errors_list, conv_list]

    with open('results/gridworld_bcd_pe.pkl', 'wb') as f:
        pickle.dump(data, f)  # serialize using dump()


def BCGD_PE_exp(Q_opt, Pi):

    fo_list = []
    errors_list = []
    conv_list = []

    for k in BCGD_PE_k_list:
        Q = PARAFAC(
                np.concatenate(
                    [[ENV.H], discretizer.bucket_states, discretizer.bucket_actions]
                ),
                k=k,
                scale= BCGD_PE_Q_scale,
                nA=len(discretizer.bucket_actions),
            ).double()
        
        bcd_grad = bcgd(Q,Pi,discretizer,ENV,k,Q_opt.reshape(ENV.H,ENV.W,ENV.W,ENV.nA),BCGD_PE_alpha)

        fo_values,errors,convs, Q = bcd_grad.run(BCGD_PE_num_iter)
        fo_list.append(fo_values)
        errors_list.append(errors)
        conv_list.append(convs)
    
        data = [fo_list, errors_list, conv_list]

    with open('results/gridworld_bcgd_pe.pkl', 'wb') as f:
        pickle.dump(data, f)  # serialize using dump()

def BCD_PI_exp(Q_opt, Pi):
    
    fo_list = []
    errors_list = []
    conv_list = []
    returns_mean_list = []
    returns_std_list = []

    for k in BCD_PI_k_list:
        Q = PARAFAC(
                    np.concatenate(
                        [[ENV.H], discretizer.bucket_states, discretizer.bucket_actions]
                    ),
                    k=k,
                    scale= BCD_PI_scale,
                    nA=len(discretizer.bucket_actions),
                ).double()

        bcd_inv = bcd(Q,Pi,discretizer,ENV,k,Q_opt.reshape(ENV.H,ENV.W,ENV.W,ENV.nA))

        fo_values,errors,convs,returns_mean,returns_std, Q = bcd_inv.bcd_policy_improvement(BCD_PI_policy_num_iter,BCD_PI_bcd_num_iter)
        fo_list.append(fo_values)
        errors_list.append(errors)
        conv_list.append(convs)
        returns_mean_list.append(returns_mean)
        returns_std_list.append(returns_std)

    data = [fo_list, errors_list, conv_list, returns_mean_list, returns_std_list]
    with open('results/gridworld_bcd_pi.pkl', 'wb') as f:
        pickle.dump(data, f)  # serialize using dump()

def BCGD_PI_exp(Q_opt, Pi):

    fo_list = []
    errors_list = []
    conv_list = []
    returns_mean_list = []
    returns_std_list = []

    for k in BCGD_PI_k_list:
        Q = PARAFAC(
                    np.concatenate(
                        [[ENV.H], discretizer.bucket_states, discretizer.bucket_actions]
                    ),
                    k=k,
                    scale= BCGD_PI_scale,
                    nA=len(discretizer.bucket_actions),
                ).double()

        bcd_grad = bcgd(Q,Pi,discretizer,ENV,k,Q_opt.reshape(ENV.H,ENV.W,ENV.W,ENV.nA),BCGD_PI_alpha)
        fo_values,errors,convs,returns_mean,returns_std, Q = bcd_grad.bcgd_policy_improvement(BCGD_PI_policy_num_iter,BCGD_PI_bcd_num_iter)
        fo_list.append(fo_values)
        errors_list.append(errors)
        conv_list.append(convs)
        returns_mean_list.append(returns_mean)
        returns_std_list.append(returns_std)

    data = [fo_list, errors_list, conv_list, returns_mean_list, returns_std_list]
    with open('results/gridworld_bcgd_pi.pkl', 'wb') as f:
        pickle.dump(data, f)  # serialize using dump()

def run_gridworld_simulations():

    # RANK ANALISYS FOR Q*
    bp_learner = BackwardPropagation(ENV.H,ENV.nS,ENV.nA,ENV.R,ENV.P)
    _ = bp_learner.run()
    Q_opt =  bp_learner.Q
    Pi_opt = np.zeros((ENV.H,ENV.nS, ENV.nA))
    for h in range(ENV.H):
        for s in range(ENV.nS):
            #a = np.argmax(Q.forward(np.array([h, s])).detach().numpy())
            a = np.argmax(Q_opt[h,s,:])
            Pi_opt[h,s, a] = 1

    # POLICY EVALUATION WITH BCD
    BCD_PE_exp(Q_opt,Pi_opt)

    #POLICY EVALUATION WITH BCGD
    BCGD_PE_exp(Q_opt,Pi_opt)

    # POLICY EVALUATION WITH BCD
    BCD_PI_exp(Q_opt,Pi_opt)

    #POLICY EVALUATION WITH BCGD
    BCGD_PI_exp(Q_opt,Pi_opt)