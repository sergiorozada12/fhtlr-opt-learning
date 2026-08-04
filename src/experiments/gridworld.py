import numpy as np
from multiprocessing import Pool
import pickle
import torch
from src.environments import GridWorldEnv
from src.agents.ql import QLearning, FHQLearning
from src.agents.dp import BackwardPropagation, FrontPolicyImprovement, BackPolicyImprovement
from src.utils import Discretizer
from src.trainer import run_experiment
from src.plots import plot_gridworld, plot_errors, plot_tensor_rank
from src.agents.bcd_grid import bcd, bcgd
from src.models import PARAFAC

Tamaño = 5
ENV = None
discretizer = None

def init_env():
    global ENV, discretizer
    ENV = GridWorldEnv(nS =Tamaño*Tamaño,W = Tamaño, H = Tamaño, nA=5)
    
    discretizer = Discretizer(
        min_points_states=[0, 0],
        max_points_states=[4, 4],
        bucket_states=[Tamaño,Tamaño],
        min_points_actions=[0],
        max_points_actions=[4],
        bucket_actions=[5],
    )


def run_rank_grid(seed=42):
    """Generate the final GridWorld CP-rank figure and its source data."""
    np.random.seed(seed)
    init_env()
    learner = BackwardPropagation(ENV.H, ENV.nS, ENV.nA, ENV.R, ENV.P)
    learner.run()
    q_grid = learner.Q.reshape(ENV.H, ENV.W, ENV.W, ENV.nA)
    return plot_tensor_rank(q_grid, random_state=seed)

#Hyperparameters

BCD_PE_k_list = [15,25,30]
BCD_PE_Q_scale = 1
BCD_PE_num_iter = 2000

BCGD_PE_k_list = [15,25,30]
BCGD_PE_Q_scale = 0.5
BCGD_PE_num_iter = 100000
BCGD_PE_alpha = 10e-3

BCD_PI_k_list = [15, 25, 30]
BCD_PI_scale = 0.7
BCD_PI_bcd_num_iter = 5
BCD_PI_policy_num_iter = 400

BCGD_PI_k_list = [15, 25, 30]
BCGD_PI_scale = 0.5
BCGD_PI_bcd_num_iter = 50
BCGD_PI_policy_num_iter = 2000
BCGD_PI_alpha = 10e-3
GRIDWORLD_FIGURE_SEEDS = {
    "bcd_pe": {15: 1015, 25: 1025, 30: 1030},
    "bcd_pi": {15: 101, 25: 27, 30: 101},
    "bcgd_pe": {15: 1015, 25: 1025, 30: 1030},
    "bcgd_pi": {15: 1015, 25: 1025, 30: 1030},
}

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
    init_env()
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

def _run_gridworld_job(kind, k, seed=None):
    """Run one independent rank configuration for the convergence figure."""
    seed = GRIDWORLD_FIGURE_SEEDS[kind][k] if seed is None else seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    init_env()
    bp_learner = BackwardPropagation(ENV.H, ENV.nS, ENV.nA, ENV.R, ENV.P)
    bp_learner.run()
    q_opt = bp_learner.Q
    pi_opt = np.zeros((ENV.H, ENV.nS, ENV.nA))
    for h in range(ENV.H):
        for state in range(ENV.nS):
            pi_opt[h, state, np.argmax(q_opt[h, state])] = 1
    q_model = PARAFAC(
        np.concatenate([[ENV.H], discretizer.bucket_states, discretizer.bucket_actions]),
        k=k, scale={"bcd_pe": BCD_PE_Q_scale, "bcgd_pe": BCGD_PE_Q_scale,
                   "bcd_pi": BCD_PI_scale, "bcgd_pi": BCGD_PI_scale}[kind],
        nA=len(discretizer.bucket_actions),
    ).double()
    shaped_q_opt = q_opt.reshape(ENV.H, ENV.W, ENV.W, ENV.nA)
    if kind == "bcd_pe":
        values = bcd(q_model, pi_opt, discretizer, ENV, k, shaped_q_opt).run(BCD_PE_num_iter)[:3]
    elif kind == "bcgd_pe":
        values = bcgd(q_model, pi_opt, discretizer, ENV, k, shaped_q_opt, BCGD_PE_alpha).run(BCGD_PE_num_iter)[:3]
    elif kind == "bcd_pi":
        values = bcd(q_model, pi_opt, discretizer, ENV, k, shaped_q_opt).bcd_policy_improvement(
            BCD_PI_policy_num_iter, BCD_PI_bcd_num_iter
        )[:5]
    elif kind == "bcgd_pi":
        values = bcgd(q_model, pi_opt, discretizer, ENV, k, shaped_q_opt, BCGD_PI_alpha).bcgd_policy_improvement(
            BCGD_PI_policy_num_iter, BCGD_PI_bcd_num_iter
        )[:5]
    else:
        raise ValueError(f"Unknown GridWorld job: {kind}")
    return kind, k, values


def run_gridworld_figure_experiments(processes=12):
    """Create the four GridWorld pickles, parallelizing independent ranks."""
    rank_map = {
        "bcd_pe": BCD_PE_k_list, "bcgd_pe": BCGD_PE_k_list,
        "bcd_pi": BCD_PI_k_list, "bcgd_pi": BCGD_PI_k_list,
    }
    jobs = [(kind, rank) for kind, ranks in rank_map.items() for rank in ranks]
    with Pool(processes=min(processes, len(jobs))) as pool:
        completed = pool.starmap(_run_gridworld_job, jobs)
    by_kind = {kind: {} for kind in rank_map}
    for kind, rank, values in completed:
        by_kind[kind][rank] = values
    for kind, ranks in rank_map.items():
        ordered = [by_kind[kind][rank] for rank in ranks]
        data = [list(field) for field in zip(*ordered)]
        path = f"results/gridworld_{kind}.pkl"
        with open(path, "wb") as handle:
            pickle.dump(data, handle)
        print(f"Saved {path}")
