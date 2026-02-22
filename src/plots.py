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
    draw_cmap()
    # Lista de imágenes
    fotos = ["figures/fig_R.jpg", "figures/fig_Q_right.jpg", "figures/arrows.jpg", "figures/cmap.jpg"]
    output="figures/paper_Q.jpg"
    # Cargar las imágenes
    imagenes = [Image.open(foto) for foto in fotos]

    # Obtener el tamaño máximo de cada fila y columna
    ancho_total = int(imagenes[0].size[0] + imagenes[1].size[0] + imagenes[3].size[0])
    alto_total = imagenes[0].size[1]

    # Crear un lienzo vacío para el mosaico
    mosaic = Image.new("RGB", (ancho_total, alto_total))

    # Colocar las imágenes en el lienzo
    mosaic.paste(imagenes[0], (0, 0))  # Esquina superior izquierda
    mosaic.paste(imagenes[2], (imagenes[0].size[0], 0))  # Esquina superior derecha
    mosaic.paste(imagenes[3], (imagenes[0].size[0] + imagenes[2].size[0], 0))  # Esquina inferior derecha
    

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
        matplotlib.rcParams.update({"font.size": 16})

        # Crear una fila de subplots
        fig, axarr = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)

        vmin = 0.0
        vmax = 1.0

        # Primer gráfico (mat_r)
        cax1 = axarr.imshow(mat_r, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_r[i, j], 3)
                axarr.text(j, i, v, ha="center", va="bottom", color="silver")
        axarr.set_xlabel("(a)", fontsize=16)

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
        matplotlib.rcParams.update({"font.size": 16})

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
        axarr[0].set_title("(b)", fontsize=16)

        # Tercer gráfico (mat_q)
        axarr[1].imshow(mat_q, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_q[i, j], 3)
                axarr[1].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[1].set_title("(c)", fontsize=16)

        # Cuarto gráfico (mat_tlr)
        axarr[2].imshow(mat_tlr, vmin=vmin, vmax=vmax, cmap="Reds")
        for i in range(W):
            for j in range(W):
                v = np.around(mat_tlr[i, j], 3)
                axarr[2].text(j, i, v, ha="center", va="bottom", color="silver")
        axarr[2].set_title("(d)", fontsize=16)

        # Ajustar los ejes
        for ax in axarr:
            ax.set_xticks([])
            ax.set_yticks([])

        # Guardar y mostrar la figura
        fig.savefig("figures/fig_Q_right.jpg", dpi=300)
        plt.clf()


def plot_wireless():
    try:
        dqn3 = np.load("results/wireless_dqn.npy")
        dfhqn = np.load("results/wireless_dfhqn.npy")
        fhtlr_max = np.load("results/wireless_fhtlr_max.npy")
        fhtlr_true = np.load("results/wireless_fhtlr_true.npy")
        fhtlr_max_er = np.load("results/wireless_fhtlr_max_er.npy")
        fhtlr_true_er = np.load("results/wireless_fhtlr_true_er.npy")
        # fhql = np.load("results/wireless_fhql.npy")
        fhrbf = np.load("results/wireless_fhrbf.npy")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Compute median
    mu_dqn3 = np.median(dqn3, axis=0)
    mu_dfhqn = np.median(dfhqn, axis=0)
    mu_fhtlr_max = np.median(fhtlr_max, axis=0)
    mu_fhtlr_true = np.median(fhtlr_true, axis=0)
    mu_fhtlr_max_er = np.median(fhtlr_max_er, axis=0)
    mu_fhtlr_true_er = np.median(fhtlr_true_er, axis=0)
    #mu_fhql = np.median(fhql, axis=0)
    mu_fhrbf = np.median(fhrbf, axis=0)

    p25 = 40
    p75 = 60

    # Compute P25 and P75
    p25_dqn3, p75_dqn3 = np.percentile(dqn3, [p25, p75], axis=0)
    p25_dfhqn, p75_dfhqn = np.percentile(dfhqn, [p25, p75], axis=0)
    p25_fhtlr_max, p75_fhtlr_max = np.percentile(fhtlr_max, [p25, p75], axis=0)
    p25_fhtlr_true, p75_fhtlr_true = np.percentile(fhtlr_true, [p25, p75], axis=0)
    p25_fhtlr_max_er, p75_fhtlr_max_er = np.percentile(fhtlr_max_er, [p25, p75], axis=0)
    p25_fhtlr_true_er, p75_fhtlr_true_er = np.percentile(fhtlr_true_er, [p25, p75], axis=0)
    #p25_fhql, p75_fhql = np.percentile(fhql, [p25, p75], axis=0)
    p25_fhrbf, p75_fhrbf = np.percentile(fhrbf, [p25, p75], axis=0)

    # Apply moving average for smoothing
    def smooth(series, window=100):
        return np.convolve(series, np.ones(window)/window, mode='valid')

    # Smooth the median and IQR bounds
    smoothed_mu_dqn3 = smooth(mu_dqn3)
    smoothed_p25_dqn3 = smooth(p25_dqn3)
    smoothed_p75_dqn3 = smooth(p75_dqn3)

    smoothed_mu_dfhqn = smooth(mu_dfhqn)
    smoothed_p25_dfhqn = smooth(p25_dfhqn)
    smoothed_p75_dfhqn = smooth(p75_dfhqn)

    smoothed_mu_fhtlr_max = smooth(mu_fhtlr_max)
    smoothed_p25_fhtlr_max = smooth(p25_fhtlr_max)
    smoothed_p75_fhtlr_max = smooth(p75_fhtlr_max)

    smoothed_mu_fhtlr_true = smooth(mu_fhtlr_true)
    smoothed_p25_fhtlr_true = smooth(p25_fhtlr_true)
    smoothed_p75_fhtlr_true = smooth(p75_fhtlr_true)

    smoothed_mu_fhtlr_max_er = smooth(mu_fhtlr_max_er)
    smoothed_p25_fhtlr_max_er = smooth(p25_fhtlr_max_er)
    smoothed_p75_fhtlr_max_er = smooth(p75_fhtlr_max_er)

    smoothed_mu_fhtlr_true_er = smooth(mu_fhtlr_true_er)
    smoothed_p25_fhtlr_true_er = smooth(p25_fhtlr_true_er)
    smoothed_p75_fhtlr_true_er = smooth(p75_fhtlr_true_er)
    """
    smoothed_mu_fhql = smooth(mu_fhql)
    smoothed_p25_fhql = smooth(p25_fhql)
    smoothed_p75_fhql = smooth(p75_fhql)
    """
    smoothed_mu_fhrbf = smooth(mu_fhrbf)
    smoothed_p25_fhrbf = smooth(p25_fhrbf)
    smoothed_p75_fhrbf = smooth(p75_fhrbf)

    # Adjust X-axis length for smoothed series
    x_smoothed = np.arange(0, len(smoothed_mu_fhtlr_max) * 10, 10)
    num_params = ["3,492", "13,392", "2,040", "2,040", "2,040", "2,040", "4000M", "20,000"]

    import matplotlib.ticker as ticker
    # Set up plot style
    with plt.style.context(["science", "ieee"]):
        matplotlib.rcParams.update({"font.size": 16})
        fig, ax = plt.subplots(figsize=[5, 3])
        
        # List of models for plotting
        models = [
            ("DQN", smoothed_mu_dqn3, smoothed_p25_dqn3, smoothed_p75_dqn3, "k", num_params[0]),
            ("DFHQN", smoothed_mu_dfhqn, smoothed_p25_dfhqn, smoothed_p75_dfhqn, "b", num_params[1]),
            ("BCTD-PI", smoothed_mu_fhtlr_max, smoothed_p25_fhtlr_max, smoothed_p75_fhtlr_max, "r", num_params[2]),
            ("S-BCGD-PI", smoothed_mu_fhtlr_true, smoothed_p25_fhtlr_true, smoothed_p75_fhtlr_true, "orange", num_params[3]),
            ("BCTD-PI (ER)", smoothed_mu_fhtlr_max_er, smoothed_p25_fhtlr_max_er, smoothed_p75_fhtlr_max_er, "g", num_params[4]),
            ("S-BCGD-PI (ER)", smoothed_mu_fhtlr_true_er, smoothed_p25_fhtlr_true_er, smoothed_p75_fhtlr_true_er, "y", num_params[5]),
            ("LFHQL", smoothed_mu_fhrbf, smoothed_p25_fhrbf, smoothed_p75_fhrbf, "purple", num_params[7]),
        ]
        
        for label, smoothed_median, smoothed_p25, smoothed_p75, color, params in models:
            ax.plot(x_smoothed[::100], smoothed_median[::100], c=color, label=f"{label} - {params} params.", linewidth=1)
            ax.fill_between(x_smoothed, smoothed_p25, smoothed_p75, color=color, alpha=0.05)
        
        ax.set_xlim(0, 140000)
        ax.set_ylim(4.5, 5.8)
        ax.grid()
        ax.set_xlabel("(a) Episodes")
        ax.set_ylabel("Return")
        ax.set_xticks([0, 40_000, 80_000, 120_000])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        
        output_file = "figures/wireless.jpg"
        plt.savefig(output_file, dpi=300)
        print(f"Figure saved to {output_file}")


def plot_battery():
    try:
        dqn3 = np.load("results/battery_dqn.npy")
        dfhqn = np.load("results/battery_dfhqn.npy")
        fhtlr_max = np.load("results/battery_fhtlr_max.npy")
        fhtlr_true = np.load("results/battery_fhtlr_true.npy")
        fhtlr_max_er = np.load("results/battery_fhtlr_max_er.npy")
        fhtlr_true_er = np.load("results/battery_fhtlr_true_er.npy")
        fhql = np.load("results/battery_fhql.npy")
        fhrbf = np.load("results/battery_fhrbf.npy")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Compute median
    mu_dqn3 = np.median(dqn3, axis=0)
    mu_dfhqn = np.median(dfhqn, axis=0)
    mu_fhtlr_max = np.median(fhtlr_max, axis=0)
    mu_fhtlr_true = np.median(fhtlr_true, axis=0)
    mu_fhtlr_max_er = np.median(fhtlr_max_er, axis=0)
    mu_fhtlr_true_er = np.median(fhtlr_true_er, axis=0)
    mu_fhql = np.median(fhql, axis=0)
    mu_fhrbf = np.median(fhrbf, axis=0)

    p25 = 40
    p75 = 60

    # Compute P25 and P75
    p25_dqn3, p75_dqn3 = np.percentile(dqn3, [p25, p75], axis=0)
    p25_dfhqn, p75_dfhqn = np.percentile(dfhqn, [p25, p75], axis=0)
    p25_fhtlr_max, p75_fhtlr_max = np.percentile(fhtlr_max, [p25, p75], axis=0)
    p25_fhtlr_true, p75_fhtlr_true = np.percentile(fhtlr_true, [p25, p75], axis=0)
    p25_fhtlr_max_er, p75_fhtlr_max_er = np.percentile(fhtlr_max_er, [p25, p75], axis=0)
    p25_fhtlr_true_er, p75_fhtlr_true_er = np.percentile(fhtlr_true_er, [p25, p75], axis=0)
    p25_fhql, p75_fhql = np.percentile(fhql, [p25, p75], axis=0)
    p25_fhrbf, p75_fhrbf = np.percentile(fhrbf, [p25, p75], axis=0)

    # Apply moving average for smoothing
    def smooth(series, window=50):
        return np.convolve(series, np.ones(window)/window, mode='valid')

    # Smooth the median and IQR bounds
    smoothed_mu_dqn3 = smooth(mu_dqn3)
    smoothed_p25_dqn3 = smooth(p25_dqn3)
    smoothed_p75_dqn3 = smooth(p75_dqn3)

    smoothed_mu_dfhqn = smooth(mu_dfhqn)
    smoothed_p25_dfhqn = smooth(p25_dfhqn)
    smoothed_p75_dfhqn = smooth(p75_dfhqn)

    smoothed_mu_fhtlr_max = smooth(mu_fhtlr_max)
    smoothed_p25_fhtlr_max = smooth(p25_fhtlr_max)
    smoothed_p75_fhtlr_max = smooth(p75_fhtlr_max)

    smoothed_mu_fhtlr_true = smooth(mu_fhtlr_true)
    smoothed_p25_fhtlr_true = smooth(p25_fhtlr_true)
    smoothed_p75_fhtlr_true = smooth(p75_fhtlr_true)

    smoothed_mu_fhtlr_max_er = smooth(mu_fhtlr_max_er)
    smoothed_p25_fhtlr_max_er = smooth(p25_fhtlr_max_er)
    smoothed_p75_fhtlr_max_er = smooth(p75_fhtlr_max_er)

    smoothed_mu_fhtlr_true_er = smooth(mu_fhtlr_true_er)
    smoothed_p25_fhtlr_true_er = smooth(p25_fhtlr_true_er)
    smoothed_p75_fhtlr_true_er = smooth(p75_fhtlr_true_er)

    smoothed_mu_fhql = smooth(mu_fhql)
    smoothed_p25_fhql = smooth(p25_fhql)
    smoothed_p75_fhql = smooth(p75_fhql)

    smoothed_mu_fhrbf = smooth(mu_fhrbf)
    smoothed_p25_fhrbf = smooth(p25_fhrbf)
    smoothed_p75_fhrbf = smooth(p75_fhrbf)

    # Adjust X-axis length for smoothed series
    x_smoothed = np.arange(0, len(smoothed_mu_fhtlr_max) * 10, 10)

    import matplotlib.ticker as ticker
    # Set up plot style
    with plt.style.context(["science", "ieee"]):
        matplotlib.rcParams.update({"font.size": 16})

        fig, ax = plt.subplots(figsize=[5, 3])

        # List of models for plotting
        models = [
            ("DQN", smoothed_mu_dqn3, smoothed_p25_dqn3, smoothed_p75_dqn3, "k", "33,160"),
            ("DFHQN", smoothed_mu_dfhqn, smoothed_p25_dfhqn, smoothed_p75_dfhqn, "b", "165,160"),
            ("BCTD-PI", smoothed_mu_fhtlr_max, smoothed_p25_fhtlr_max, smoothed_p75_fhtlr_max, "r", "3,750"),
            ("S-BCGD-PI", smoothed_mu_fhtlr_true, smoothed_p25_fhtlr_true, smoothed_p75_fhtlr_true, "orange", "3,750"),
            ("BCTD-PI (ER)", smoothed_mu_fhtlr_max_er, smoothed_p25_fhtlr_max_er, smoothed_p75_fhtlr_max_er, "g", "3,750"),
            ("S-BCGD-PI (ER)", smoothed_mu_fhtlr_true_er, smoothed_p25_fhtlr_true_er, smoothed_p75_fhtlr_true_er, "y", "3,750"),
            #("FHQL", smoothed_mu_fhql, smoothed_p25_fhql, smoothed_p75_fhql, "r", "50 M"),
            ("LFHQL", smoothed_mu_fhrbf, smoothed_p25_fhrbf, smoothed_p75_fhrbf, "purple", "30,000"),
        ]

        # Plot each model's smoothed median and IQR
        for label, smoothed_median, smoothed_p25, smoothed_p75, color, params in models:
            ax.plot(x_smoothed, smoothed_median, c=color, label=f"{label} - {params} params.", linewidth=1)  # Smoothed Median
            ax.fill_between(x_smoothed, smoothed_p25, smoothed_p75, color=color, alpha=0.05)  # Smoothed IQR shading

        # Formatting
        ax.set_xlim(0, 22000)
        ax.set_ylim(-50, -5)
        ax.grid()
        ax.set_xlabel("(b) Episodes")
        ax.set_ylabel("Return")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        ax.set_xticks([0, 6_000, 12_000, 18_000])
        ax.set_yticks([-50, -30, -10])

        # Scientific notation for Y-axis
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        output_file = "figures/battery.jpg"
        plt.savefig(output_file, dpi=300)
        print(f"Figure saved to {output_file}")

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

def find_max_positions(array):
    # Encontrar el valor máximo
    max_value = np.max(array)
    
    # Encontrar las posiciones donde el array es igual al valor máximo
    posiciones = np.where(array == max_value)[0]  # Devuelve un array con índices

    # Convertir las posiciones a una lista
    return posiciones

def draw_grid_with_arrows(Q_list):
    cmap = cm.get_cmap('Reds')

    # Extraer colores del colormap
    red_from_cmap = cmap(1.0)  # Rojo oscuro
    white_from_cmap = cmap(0)  # Blanco o color claro

    # Configuración de la figura
    fig, ax = plt.subplots(1, len(Q_list), figsize=(12, 4), constrained_layout=True)

    # Coordenadas de las flechas para la cuadrícula 5x5
    grid_size = 5
    x = np.arange(grid_size)
    y = np.arange(grid_size)

    # Flechas en todas las direcciones
    directions = [(0, 0.3), (-0.3, 0), (0, -0.3), (0.3, 0)]  # (dx, dy) arriba, abajo, derecha, izquierda
    xlabel = ["b", "c", "d"]
    # Dibujar las 3 imágenes
    for idx, ax_idx in enumerate(ax):
        Q2print = Q_list[idx]
        ax_idx.set_xlim(-0.5, grid_size - 0.5)
        ax_idx.set_ylim(-0.5, grid_size - 0.5)
        ax_idx.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_idx.set_xlabel(f"({xlabel[idx]})", fontsize=16)

        # Dibujar flechas en cada celda
        for i in range(grid_size):
            for j in range(grid_size):
                if not((i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4) or (i == 0 and j == 0)):
                    intensity = np.max(Q2print[i, 4 - j, :])
                    rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=cmap(intensity), alpha=1)
                    ax_idx.add_patch(rect)

                    direction_ids = find_max_positions(Q2print[i, 4 - j, :])
                    for dir_id in direction_ids:
                        if dir_id != 4:
                            dx, dy = directions[dir_id]
                            ax_idx.arrow(i, j, dx, dy, head_width=0.05, head_length=0.05, fc='grey', ec='grey', alpha=0.7)
                        else:
                            ax_idx.plot(i, j, 'o', color='grey', markersize=6, alpha=0.7)
                else:
                    rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=white_from_cmap, alpha=1)
                    ax_idx.add_patch(rect)
                    ax_idx.plot(i, j, 'o', color='grey', markersize=6, alpha=0.7)


    plt.savefig("figures/arrows.jpg", dpi=300)
    plt.clf()

def draw_cmap():  
    cmap = cm.get_cmap('Reds')
    fig, axarr = plt.subplots(1, 1, figsize=(1, 4), constrained_layout=True)
    axarr.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in axarr.spines.values():
        spine.set_visible(False)
    # Usar estilo
    with plt.style.context(["science", "ieee"]):
        plt.rcParams.update({'font.size': 14})  # Cambiar globalmente el tamaño de la fuente de la figura
        # Crear un eje específico para la barra de colores
        cax = fig.add_axes([0, 0.2, 0.3, 0.87])  # [left, bottom, width, height]
        
        # Añadir la barra de colores
        norm = plt.Normalize(vmin=0, vmax=1)  # Normalizar el rango de intensidades (0 a 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Se requiere para crear la barra de colores
        cbar = fig.colorbar(sm, cax=cax)

        # Etiqueta de la barra de colores
        cbar.set_label("Expecetd Return", fontsize=16)
        fig.savefig("figures/cmap.jpg", dpi=300)
        plt.clf()


def crear_mosaico(fotos, output="figures/PI-PI-final.png"):
    # Cargar las imágenes
    imagenes = [Image.open(foto) for foto in fotos]

    # Obtener el tamaño máximo de cada fila y columna
    ancho_total = max(img.size[0] for img in imagenes[:2]) + max(img.size[0] for img in imagenes[2:])
    alto_total = max(img.size[1] for img in imagenes[0::2]) + max(img.size[1] for img in imagenes[1::2])

    # Crear un lienzo vacío para el mosaico
    mosaic = Image.new("RGB", (ancho_total, alto_total))

    # Colocar las imágenes en el lienzo
    mosaic.paste(imagenes[0], (0, 0))  # Esquina superior izquierda
    mosaic.paste(imagenes[1], (max(img.size[0] for img in imagenes[:2]), 0))  # Esquina superior derecha
    mosaic.paste(imagenes[2], (0, max(img.size[1] for img in imagenes[0::2])))  # Esquina inferior izquierda
    mosaic.paste(imagenes[3], (max(img.size[0] for img in imagenes[:2]), max(img.size[1] for img in imagenes[0::2])))  # Esquina inferior derecha

    # Guardar o mostrar el mosaico
    mosaic.save(output)
    mosaic.show()

def plot_pendulum():
    try:
        dqn = np.load("results/pendulum_dqn.npy")
        dfhqn = np.load("results/pendulum_dfhqn.npy")
        fhtlr_max = np.load("results/pendulum_fhtlr_max.npy")
        fhtlr_true = np.load("results/pendulum_fhtlr_true.npy")
        fhtlr_max_er = np.load("results/pendulum_fhtlr_max_er.npy")
        fhtlr_true_er = np.load("results/pendulum_fhtlr_true_er.npy")
        fhrbf = np.load("results/pendulum_fhrbf.npy")
        fhql = np.load("results/pendulum_fhql.npy")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Calculate median
    mu_dqn = np.median(dqn, axis=0)
    mu_dfhqn = np.median(dfhqn, axis=0)
    mu_fhtlr_max = np.median(fhtlr_max, axis=0)
    mu_fhtlr_true = np.median(fhtlr_true, axis=0)
    mu_fhtlr_max_er = np.median(fhtlr_max_er, axis=0)
    mu_fhtlr_true_er = np.median(fhtlr_true_er, axis=0)
    mu_fhrbf = np.median(fhrbf, axis=0)
    mu_fhql = np.median(fhql, axis=0)

    p25 = 40
    p75 = 60

    p25_dqn, p75_dqn = np.percentile(dqn, [p25, p75], axis=0)
    p25_dfhqn, p75_dfhqn = np.percentile(dfhqn, [p25, p75], axis=0)
    p25_fhtlr_max, p75_fhtlr_max = np.percentile(fhtlr_max, [p25, p75], axis=0)
    p25_fhtlr_true, p75_fhtlr_true = np.percentile(fhtlr_true, [p25, p75], axis=0)
    p25_fhtlr_max_er, p75_fhtlr_max_er = np.percentile(fhtlr_max_er, [p25, p75], axis=0)
    p25_fhtlr_true_er, p75_fhtlr_true_er = np.percentile(fhtlr_true_er, [p25, p75], axis=0)
    p25_fhrbf, p75_fhrbf = np.percentile(fhrbf, [p25, p75], axis=0)
    p25_fhql, p75_fhql = np.percentile(fhql, [p25, p75], axis=0)

    # Apply moving average for smoothing
    def smooth(series, window=50):
        return np.convolve(series, np.ones(window)/window, mode='valid')

    smoothed_mu_dqn = smooth(mu_dqn)
    smoothed_p25_dqn = smooth(p25_dqn)
    smoothed_p75_dqn = smooth(p75_dqn)

    smoothed_mu_dfhqn = smooth(mu_dfhqn)
    smoothed_p25_dfhqn = smooth(p25_dfhqn)
    smoothed_p75_dfhqn = smooth(p75_dfhqn)

    smoothed_mu_fhtlr_max = smooth(mu_fhtlr_max)
    smoothed_p25_fhtlr_max = smooth(p25_fhtlr_max)
    smoothed_p75_fhtlr_max = smooth(p75_fhtlr_max)

    smoothed_mu_fhtlr_true = smooth(mu_fhtlr_true)
    smoothed_p25_fhtlr_true = smooth(p25_fhtlr_true)
    smoothed_p75_fhtlr_true = smooth(p75_fhtlr_true)

    smoothed_mu_fhtlr_max_er = smooth(mu_fhtlr_max_er)
    smoothed_p25_fhtlr_max_er = smooth(p25_fhtlr_max_er)
    smoothed_p75_fhtlr_max_er = smooth(p75_fhtlr_max_er)

    smoothed_mu_fhtlr_true_er = smooth(mu_fhtlr_true_er)
    smoothed_p25_fhtlr_true_er = smooth(p25_fhtlr_true_er)
    smoothed_p75_fhtlr_true_er = smooth(p75_fhtlr_true_er)

    smoothed_mu_fhrbf = smooth(mu_fhrbf)
    smoothed_p25_fhrbf = smooth(p25_fhrbf)
    smoothed_p75_fhrbf = smooth(p75_fhrbf)

    smoothed_mu_fhql = smooth(mu_fhql)
    smoothed_p25_fhql = smooth(p25_fhql)
    smoothed_p75_fhql = smooth(p75_fhql)

    x_smoothed = np.arange(0, len(smoothed_mu_fhtlr_max) * 10, 10)

    import matplotlib.ticker as ticker
    with plt.style.context(["science", "ieee"]):
        matplotlib.rcParams.update({"font.size": 16})

        fig, ax = plt.subplots(figsize=[5, 3])
        
        models = [
            ("DQN", smoothed_mu_dqn, smoothed_p25_dqn, smoothed_p75_dqn, "k"),
            ("DFHQN", smoothed_mu_dfhqn, smoothed_p25_dfhqn, smoothed_p75_dfhqn, "b"),
            ("BCTD-PI", smoothed_mu_fhtlr_max, smoothed_p25_fhtlr_max, smoothed_p75_fhtlr_max, "r"),
            ("S-BCGD-PI", smoothed_mu_fhtlr_true, smoothed_p25_fhtlr_true, smoothed_p75_fhtlr_true, "orange"),
            ("BCTD-PI (ER)", smoothed_mu_fhtlr_max_er, smoothed_p25_fhtlr_max_er, smoothed_p75_fhtlr_max_er, "g"),
            ("S-BCGD-PI (ER)", smoothed_mu_fhtlr_true_er, smoothed_p25_fhtlr_true_er, smoothed_p75_fhtlr_true_er, "y"),
            ("LFHQL", smoothed_mu_fhrbf, smoothed_p25_fhrbf, smoothed_p75_fhrbf, "purple"),
        ]

        for label, smoothed_median, smoothed_p25, smoothed_p75, color in models:
            ax.plot(x_smoothed, smoothed_median, c=color, label=f"{label}", linewidth=1)
            ax.fill_between(x_smoothed, smoothed_p25, smoothed_p75, color=color, alpha=0.05)

        ax.grid()
        ax.set_xlabel("(c) Episodes")
        ax.set_ylabel("Return")
        ax.set_ylim(-0.6, 0.05)
        ax.set_xlim(0, 3500)
        ax.set_xticks([0, 1000, 2000, 3000])
        ax.set_yticks([-0.6, -0.4, -0.2, 0])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        output_file = "figures/pendulum.jpg"
        plt.savefig(output_file, dpi=300)
        print(f"Figure saved to {output_file}")

def plot_cartpole():
    try:
        dqn = np.load("results/cartpole_dqn.npy")
        dfhqn = np.load("results/cartpole_dfhqn.npy")
        fhtlr_max = np.load("results/cartpole_fhtlr_max.npy")
        fhtlr_true = np.load("results/cartpole_fhtlr_true.npy")
        fhtlr_max_er = np.load("results/cartpole_fhtlr_max_er.npy")
        fhtlr_true_er = np.load("results/cartpole_fhtlr_true_er.npy")
        fhrbf = np.load("results/cartpole_fhrbf.npy")
        fhql = np.load("results/cartpole_fhql.npy")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Calculate median
    mu_dqn = np.median(dqn, axis=0)
    mu_dfhqn = np.median(dfhqn, axis=0)
    mu_fhtlr_max = np.median(fhtlr_max, axis=0)
    mu_fhtlr_true = np.median(fhtlr_true, axis=0)
    mu_fhtlr_max_er = np.median(fhtlr_max_er, axis=0)
    mu_fhtlr_true_er = np.median(fhtlr_true_er, axis=0)
    mu_fhrbf = np.median(fhrbf, axis=0)
    mu_fhql = np.median(fhql, axis=0)

    p25 = 40
    p75 = 60

    p25_dqn, p75_dqn = np.percentile(dqn, [p25, p75], axis=0)
    p25_dfhqn, p75_dfhqn = np.percentile(dfhqn, [p25, p75], axis=0)
    p25_fhtlr_max, p75_fhtlr_max = np.percentile(fhtlr_max, [p25, p75], axis=0)
    p25_fhtlr_true, p75_fhtlr_true = np.percentile(fhtlr_true, [p25, p75], axis=0)
    p25_fhtlr_max_er, p75_fhtlr_max_er = np.percentile(fhtlr_max_er, [p25, p75], axis=0)
    p25_fhtlr_true_er, p75_fhtlr_true_er = np.percentile(fhtlr_true_er, [p25, p75], axis=0)
    p25_fhrbf, p75_fhrbf = np.percentile(fhrbf, [p25, p75], axis=0)
    p25_fhql, p75_fhql = np.percentile(fhql, [p25, p75], axis=0)

    # Apply moving average for smoothing
    def smooth(series, window=50):
        return np.convolve(series, np.ones(window)/window, mode='valid')

    smoothed_mu_dqn = smooth(mu_dqn)
    smoothed_p25_dqn = smooth(p25_dqn)
    smoothed_p75_dqn = smooth(p75_dqn)

    smoothed_mu_dfhqn = smooth(mu_dfhqn)
    smoothed_p25_dfhqn = smooth(p25_dfhqn)
    smoothed_p75_dfhqn = smooth(p75_dfhqn)

    smoothed_mu_fhtlr_max = smooth(mu_fhtlr_max)
    smoothed_p25_fhtlr_max = smooth(p25_fhtlr_max)
    smoothed_p75_fhtlr_max = smooth(p75_fhtlr_max)

    smoothed_mu_fhtlr_true = smooth(mu_fhtlr_true)
    smoothed_p25_fhtlr_true = smooth(p25_fhtlr_true)
    smoothed_p75_fhtlr_true = smooth(p75_fhtlr_true)

    smoothed_mu_fhtlr_max_er = smooth(mu_fhtlr_max_er)
    smoothed_p25_fhtlr_max_er = smooth(p25_fhtlr_max_er)
    smoothed_p75_fhtlr_max_er = smooth(p75_fhtlr_max_er)

    smoothed_mu_fhtlr_true_er = smooth(mu_fhtlr_true_er)
    smoothed_p25_fhtlr_true_er = smooth(p25_fhtlr_true_er)
    smoothed_p75_fhtlr_true_er = smooth(p75_fhtlr_true_er)

    smoothed_mu_fhrbf = smooth(mu_fhrbf)
    smoothed_p25_fhrbf = smooth(p25_fhrbf)
    smoothed_p75_fhrbf = smooth(p75_fhrbf)

    smoothed_mu_fhql = smooth(mu_fhql)
    smoothed_p25_fhql = smooth(p25_fhql)
    smoothed_p75_fhql = smooth(p75_fhql)

    x_smoothed = np.arange(0, len(smoothed_mu_fhtlr_max) * 10, 10)

    import matplotlib.ticker as ticker
    with plt.style.context(["science", "ieee"]):
        matplotlib.rcParams.update({"font.size": 16})

        fig, ax = plt.subplots(figsize=[5, 3])
        
        models = [
            ("DQN", smoothed_mu_dqn, smoothed_p25_dqn, smoothed_p75_dqn, "k"),
            ("DFHQN", smoothed_mu_dfhqn, smoothed_p25_dfhqn, smoothed_p75_dfhqn, "b"),
            ("BCTD-PI", smoothed_mu_fhtlr_max, smoothed_p25_fhtlr_max, smoothed_p75_fhtlr_max, "r"),
            ("S-BCGD-PI", smoothed_mu_fhtlr_true, smoothed_p25_fhtlr_true, smoothed_p75_fhtlr_true, "orange"),
            ("BCTD-PI (ER)", smoothed_mu_fhtlr_max_er, smoothed_p25_fhtlr_max_er, smoothed_p75_fhtlr_max_er, "g"),
            ("S-BCGD-PI (ER)", smoothed_mu_fhtlr_true_er, smoothed_p25_fhtlr_true_er, smoothed_p75_fhtlr_true_er, "y"),
            ("LFHQL", smoothed_mu_fhrbf, smoothed_p25_fhrbf, smoothed_p75_fhrbf, "purple"),
        ]

        for label, smoothed_median, smoothed_p25, smoothed_p75, color in models:
            ax.plot(x_smoothed, smoothed_median, c=color, label=f"{label}", linewidth=1)
            ax.fill_between(x_smoothed, smoothed_p25, smoothed_p75, color=color, alpha=0.05)

        ax.grid()
        ax.set_xlabel("(d) Episodes")
        ax.set_ylabel("Return")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        ax.set_ylim(-0.125, 0.01)
        ax.set_xlim(0, 3500)
        ax.set_xticks([0, 1000, 2000, 3000])
        ax.set_yticks([-0.125, -0.0625, 0])
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        output_file = "figures/cartpole.jpg"
        plt.savefig(output_file, dpi=300)
        print(f"Figure saved to {output_file}")
