import random
import numpy as np
import torch
from gymnasium import Env


def run_train_episode(env, agent, eps, eps_decay, H):
    s, _ = env.reset()
    for h in range(H):
        a = agent.select_action(s, h, eps)
        sp, r, d, _, _ = env.step(a)
        agent.buffer.append(h, s, a, sp, r, d)
        agent.update()

        if d:
            break

        s = sp
        eps *= eps_decay
    return eps


def run_test_episode(env, agent, H):
    G = 0
    s, _ = env.reset()
    for h in range(H):
        a = agent.select_greedy_action(s, h)
        s, r, d, _, _ = env.step(a)
        G += r

        if d:
            break
    return G


def run_experiment(
    n: int, E: int, H: int, eps: float, eps_decay: float, env: Env, agent
):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    Gs = []
    for _ in range(E):
        eps = run_train_episode(env, agent, eps, eps_decay, H)
        G = run_test_episode(env, agent, H)
        Gs.append(G)

        if hasattr(agent, "scheduler"):
            agent.scheduler.step()
    return Gs
