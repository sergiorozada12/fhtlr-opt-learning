import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import ScalarFormatter
from PIL import Image
from matplotlib import cm

import tensorly as tl
from tensorly.decomposition import parafac

def plot_gridworld(W, mat_q_stationary, mat_q, mat_tlr,Q_list):

    plot_gridworld_Q(W, mat_q_stationary, mat_q, mat_tlr)
    plot_gridwolrd_R(W)
    draw_grid_with_arrows(Q_list)
    # Lista de imágenes
    fotos = ["figures/fig_R.jpg", "figures/fig_Q_right.jpg", "figures/arrows.jpg"]
    output="figures/Q.jpg"
    # Cargar las imágenes
    imagenes = [Image.open(foto) for foto in fotos]

    # Obtener el tamaño máximo de cada fila y columna
    ancho_total = imagenes[0].size[0] + imagenes[1].size[0]
    alto_total = imagenes[0].size[1]

    # Crear un lienzo vacío para el mosaico
    mosaic = Image.new("RGB", (ancho_total, alto_total))

    # Colocar las imágenes en el lienzo
    mosaic.paste(imagenes[0], (0, 0))  # Esquina superior izquierda
    mosaic.paste(imagenes[1], (imagenes[0].size[0], 0))  # Esquina superior derecha
    mosaic.paste(imagenes[2], (imagenes[0].size[0] - 35, imagenes[1].size[1]))  # Esquina inferior derecha

    # Guardar o mostrar el mosaico
    mosaic.save(output)
    mosaic.show()

def plot_gridwolrd_R(W):
    mat_r = np.zeros((W, W))
    mat_r[0, 0] = 1
    mat_r[0, -1] = 1
    mat_r[-1, 0] = 1
    mat_r[-1, -1] = 1

    # Usar estilo
    with plt.style.context(["science", "ieee"]):
        matplotlib.rcParams.update({"font.size": 14})

        # Crear una fila de subplots
        fig, axarr = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

        vmin = 0.0
        vmax = 1.0

        # Primer gráfico (mat_r)
        cax1 = axarr.imshow(mat_r, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_r[i, j], 3)
                axarr.text(j, i, v, ha="center", va="bottom", color="silver")
        axarr.set_title("(a)", fontsize=14)

        axarr.set_xticks([])
        axarr.set_yticks([])

    fig.savefig("figures/fig_R.jpg", dpi=300)
    plt.clf()

def plot_gridworld_Q(W, mat_q_stationary, mat_q, mat_tlr):
    mat_r = np.zeros((W, W))
    mat_r[0, 0] = 1
    mat_r[0, -1] = 1
    mat_r[-1, 0] = 1
    mat_r[-1, -1] = 1

    # Usar estilo
    with plt.style.context(["science", "ieee"]):
        matplotlib.rcParams.update({"font.size": 14})

        # Crear una fila de subplots
        fig, axarr = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        vmin = 0.0
        vmax = 1.0

        # Primer gráfico (mat_r)
        """cax1 = axarr[0].imshow(mat_r, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_r[i, j], 3)
                axarr[0].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[0].set_title("(a)", fontsize=14)"""

        # Segundo gráfico (mat_q_stationary)
        axarr[0].imshow(mat_q_stationary, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_q_stationary[i, j], 3)
                axarr[0].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[0].set_title("(b)", fontsize=14)

        # Tercer gráfico (mat_q)
        axarr[1].imshow(mat_q, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_q[i, j], 3)
                axarr[1].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[1].set_title("(c)", fontsize=14)

        # Cuarto gráfico (mat_tlr)
        axarr[2].imshow(mat_tlr, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_tlr[i, j], 3)
                axarr[2].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[2].set_title("(d)", fontsize=14)

        # Ajustar los ejes
        for ax in axarr:
            ax.set_xticks([])
            ax.set_yticks([])

        # Guardar y mostrar la figura
        fig.savefig("figures/fig_Q_right.jpg", dpi=300)
        plt.clf()


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

def plot_tensor_rank(Q_to_plot,name,max_rank =  25):
    
    tensor = Q_to_plot - np.mean(Q_to_plot)
    norm_frobenius_original_tensor = np.linalg.norm(tensor)
    factors = []
    normlaized_errors = []
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

        fig = plt.figure(figsize=[5, 3])
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

def find_max_positions(array):
    # Encontrar el valor máximo
    max_value = np.max(array)
    
    # Encontrar las posiciones donde el array es igual al valor máximo
    posiciones = np.where(array == max_value)[0]  # Devuelve un array con índices

    # Convertir las posiciones a una lista
    return posiciones

def draw_grid_with_arrows(Q_list):
    cmap = cm.get_cmap('Reds')

    # Extraer un rojo oscuro del colormap (el más intenso, valor 1.0)
    red_from_cmap = cmap(1.0)  # RGBA correspondiente al rojo oscuro más inte
    white_from_cmap = cmap(0)  # RGBA correspondiente al rojo oscuro más inte
    # Configuración de la figura
    fig, ax = plt.subplots(1, len(Q_list), figsize=(12, 4))

    # Coordenadas de las flechas para la cuadrícula 5x5
    grid_size = 5
    x = np.arange(grid_size)
    y = np.arange(grid_size)

    # Flechas en todas las direcciones
    directions = [(0, 0.3), (-0.3, 0), (0, -0.3), (0.3, 0)]  # (dx, dy) arriba, abajo, derecha, izquierda


    # Dibujar las 3 imágenes
    for idx, ax_idx in enumerate(ax):
        Q2print = Q_list[idx]
        ax_idx.set_xlim(-0.5, grid_size - 0.5)
        ax_idx.set_ylim(-0.5, grid_size - 0.5)
        #ax_idx.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        #ax_idx.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        #ax_idx.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax_idx.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Dibujar flechas en cada celda
        for i in range(grid_size):
            for j in range(grid_size):
                if not((i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4) or (i == 0 and j == 0)):
                    #print(i,4-j,Q2print[i,4-j,:])
                    if np.max(Q2print[i,4-j,:]) > 0.5:
                        rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=red_from_cmap, alpha=1)
                        ax_idx.add_patch(rect)
                    else:
                        rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=white_from_cmap, alpha=1)
                        ax_idx.add_patch(rect)
                    direction_ids = find_max_positions(Q2print[i,4-j,:])
                    for dir_id in direction_ids:
                        if dir_id != 4 :
                            dx, dy = directions[dir_id]
                            ax_idx.arrow(i, j, dx, dy, head_width=0.05, head_length=0.05, fc='grey', ec='grey',alpha=0.7)
                        else:
                            ax_idx.plot(i, j, 'o', color='grey', markersize=6,alpha=0.7)                     
                else:
                    rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=white_from_cmap, alpha=1)
                    ax_idx.add_patch(rect)
                    ax_idx.plot(i, j, 'o', color='grey', markersize=6,alpha=0.7) 

    plt.tight_layout()
    plt.savefig("figures/arrows.jpg",dpi = 300)