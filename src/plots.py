import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import ScalarFormatter


import tensorly as tl
from tensorly.decomposition import parafac


def plot_gridworld(W, mat_q_stationary, mat_q, mat_tlr):
    mat_r = np.zeros((W, W))
    mat_r[0, 0] = 0.5
    mat_r[-1, -1] = 1

    with plt.style.context(["science"], ["ieee"]):
        matplotlib.rcParams.update({"font.size": 14})

        fig, axarr = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)

        vmin = 0.0
        vmax = 100.0

        cax1 = axarr[0, 0].imshow(mat_r, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_r[i, j], 1)
                axarr[0, 0].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[0, 0].set_xlabel("(a)")

        axarr[0, 1].imshow(mat_q_stationary, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_q_stationary[i, j], 1)
                axarr[0, 1].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[0, 1].set_xlabel("(b)")

        axarr[1, 0].imshow(mat_q, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_q[i, j], 1)
                axarr[1, 0].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[1, 0].set_xlabel("(c)")

        axarr[1, 1].imshow(mat_tlr, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_tlr[i, j], 1)
                axarr[1, 1].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[1, 1].set_xlabel("(d)")

        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        gap = 0.0
        width = 0.45
        height = 0.45

        axarr[0, 0].set_position([0, 0.5 + gap / 2, width, height])
        axarr[0, 1].set_position([0.5 + gap, 0.5 + gap / 2, width, height])
        axarr[1, 0].set_position([0, 0, width, height])
        axarr[1, 1].set_position([0.5 + gap, 0, width, height])

        fig.savefig("figures/fig_1.jpg", dpi=300)


def plot_wireless():
    dqn = np.load("results/dqn.npy")
    dfhqn = np.load("results/dfhqn.npy")
    fhtlr = np.load("results/fhtlr.npy")
    ql = np.load("results/ql.npy")

    mu_dqn = np.mean(dqn, axis=0)
    mu_dfhqn = np.mean(dfhqn, axis=0)
    mu_fhtlr = np.mean(fhtlr, axis=0)
    mu_ql = np.mean(ql, axis=0)

    w = 100

    mu_dqn_smt = [np.mean(mu_dqn[i - w : i]) for i in range(w, len(mu_dqn))]
    mu_dfhqn_smt = [np.mean(mu_dfhqn[i - w : i]) for i in range(w, len(mu_dfhqn))]
    mu_fhtlr_smt = [np.mean(mu_fhtlr[i - w : i]) for i in range(w, len(mu_fhtlr))]
    mu_ql_smt = [np.mean(mu_ql[i - w : i]) for i in range(w, len(mu_ql))]

    with plt.style.context(["science"], ["ieee"]):
        matplotlib.rcParams.update({"font.size": 16})

        fig = plt.figure(figsize=[8, 3])
        plt.plot(mu_dqn_smt, c="b", label="DQN")
        plt.plot(mu_dqn, alpha=0.2, c="b")
        plt.plot(mu_dfhqn_smt, c="orange", label="DFHQN")
        plt.plot(mu_dfhqn, alpha=0.2, c="orange")
        plt.plot(mu_fhtlr_smt, c="g", label="FHTLR")
        plt.plot(mu_fhtlr, alpha=0.2, c="g")
        plt.plot(mu_ql_smt, c="k", label="FHQ-learning")
        plt.plot(mu_ql, alpha=0.2, c="k")
        plt.xlim(0, 100_000)
        plt.ylim(-3, 1.6)
        plt.grid()
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Return")
        fig.savefig("figures/fig_2.jpg", dpi=300)

def plot_errors(errors,name):
    plt.stem(errors, basefmt=" ")
    plt.title("Error"+ name + "Iteration")
    plt.xlabel("Iteracion")
    plt.ylabel("Error")
    plt.grid(True)
    plt.savefig("figures/"+name+"errors")
    plt.clf()

def plot_tensor_rank(Q_to_plot,name):
    
    tensor = Q_to_plot - np.mean(Q_to_plot)
    norm_frobenius_original_tensor = np.linalg.norm(tensor)
    factors = []
    normlaized_errors = []
    max_rank =  25
    for i in range(1,max_rank):

        factor =  parafac(tensor, rank=i)
        factors.append(factor)
        reconstructed_tensor = tl.cp_to_tensor(factor)
        normlaized_errors.append(np.linalg.norm(tensor-reconstructed_tensor)/(norm_frobenius_original_tensor/100))
        if normlaized_errors[-1] < 10e-9:
            break

    rangos = np.arange(1, len(normlaized_errors)+1)  # Rango del 1 al 10
    error = np.array(normlaized_errors)  # Errores aleatorios para cada rango


    with plt.style.context(["science"], ["ieee"]):
        matplotlib.rcParams.update({"font.size": 16})

        fig = plt.figure(figsize=[5, 4])
        plt.plot(rangos, error, marker='o')

        # Configurar notación científica en los ejes
        ax = plt.gca()  # Obtener el objeto del eje actual
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        plt.locator_params(axis='y', nbins=5)  # 5 líneas en el eje Y

        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))  # Habilitar notación científica

        plt.ylim(0, 100)
        plt.xlim(0,len(error)+1)
        plt.xlabel("Rank")
        plt.ylabel("NFE(\%) - GridWord")
        plt.grid(True,axis='y')
        plt.tight_layout()
        if name is None:
            plt.show()
        else:
            plt.savefig("figures/"+name+"errors")
        plt.clf()