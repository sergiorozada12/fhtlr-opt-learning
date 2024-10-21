import numpy as np

from src.environments import GridWorldEnv
from src.agents.ql import QLearning, FHQLearning
from src.agents.dp import BackwardPropagation, FrontPolicyImprovement, BackPolicyImprovement
from src.utils import Discretizer
from src.trainer import run_experiment
from src.plots import plot_gridworld, plot_errors, plot_tensor_rank


ALPHA_Q = 0.9
ALPHA_TLR = 0.001
GAMMA = 0.99
E = 50_000
H = 5
EPS = 1.0
EPS_DECAY = 0.9999
K = 8
SCALE = 0.1

ENV = GridWorldEnv()

DISCRETIZER = Discretizer(
    min_points_states=[0, 0],
    max_points_states=[4, 4],
    bucket_states=[5, 5],
    min_points_actions=[0],
    max_points_actions=[3],
    bucket_actions=[4],
)


def run_gridworld_simulations():

    fhq_learner = FHQLearning(DISCRETIZER, ALPHA_Q, H)
    bp_learner = BackwardPropagation(H,ENV.nS,ENV.nA,ENV.R,ENV.P)
    fpi_learner = FrontPolicyImprovement(H,ENV.nS,ENV.nA,ENV.R,ENV.P)
    bpi_learner = BackPolicyImprovement(H,ENV.nS,ENV.nA,ENV.R,ENV.P)

    _ = bp_learner.run()
    _, error_fpi = fpi_learner.run()
    _, error_bpi = bpi_learner.run()
    _ = run_experiment(0, E, H, EPS, EPS_DECAY, ENV, fhq_learner)

    mat_q = np.max(fhq_learner.Q, axis=3)[0].reshape(ENV.W, ENV.W)

    print(f"La diferencia entre fhq y bp es {np.linalg.norm(fhq_learner.Q-bp_learner.q_reshape(DISCRETIZER))}")
    print(f"La diferencia entre fhq y fpi es {np.linalg.norm(fhq_learner.Q-fpi_learner.q_reshape(DISCRETIZER))}")
    print(f"La diferencia entre fhq y bpi es {np.linalg.norm(fhq_learner.Q-bpi_learner.q_reshape(DISCRETIZER))}")

    plot_errors(error_fpi,"Front Policy Iteration")
    plot_errors(error_bpi, "Back Policy Iteration")
    plot_tensor_rank(fhq_learner.Q, "FHQ_Learner")



