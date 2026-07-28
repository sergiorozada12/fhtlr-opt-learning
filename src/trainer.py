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
    for e in range(E):
        eps = run_train_episode(env, agent, eps, eps_decay, H)
        if e % 10 == 0:
            G = run_test_episode(env, agent, H)
            Gs.append(G)
        if e % 100 == 0:
            print(f'\r Episodio: {e}/{E}', end='')

    return Gs


def run_pg_experiment(n: int, E: int, H: int, env: Env, agent):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    Gs = []
    for e in range(E):
        s, _ = env.reset()
        for h in range(H):
            a, a_idx, value = agent.select_action(s, h)
            sp, r, d, _, _ = env.step(a)
            agent.store(h, s, a_idx, r, d, value)
            s = sp
            if d:
                break
        agent.maybe_update()
        if e % 10 == 0:
            Gs.append(run_test_episode(env, agent, H))
        if e % 100 == 0:
            print(f'\r Episodio FH-PG: {e}/{E}', end='')
    agent.maybe_update(force=True)
    return Gs


def run_ctrl_ucbm_experiment(n: int, E: int, H: int, env: Env, agent):
    """Run FH-CTRL-UCBM with the repository's every-10-episode evaluation."""
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    Gs = []
    for e in range(E):
        s, _ = env.reset()
        for h in range(H):
            a, action_idx = agent.select_action(s, h)
            sp, r, d, _, _ = env.step(a)
            agent.store(h, s, action_idx, sp, r, d)
            agent.update()
            s = sp
            if d:
                break
        if e % 10 == 0:
            Gs.append(run_test_episode(env, agent, H))
        if e % 100 == 0:
            print(f'\r Episodio FH-CTRL-UCBM: {e}/{E}', end='')
    return Gs
