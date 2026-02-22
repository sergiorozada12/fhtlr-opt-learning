from src.experiments.gridworld import run_gridworld_simulations
from src.experiments.wireless import run_wireless_simulations
from src.experiments.battery import run_battery_simulations
from src.experiments.channel_coding import run_channel_coding_simulations
from src.experiments.pendulum import run_pendulum_simulations
from src.experiments.cartpole import run_cartpole_simulations
from src.plots import plot_wireless, plot_battery, plot_pendulum, plot_cartpole


if __name__ == "__main__":
    #run_channelcoding_simulations()
    #run_gridworld_simulations()
    #run_wireless_simulations()
    #run_battery_simulations()
    #run_pendulum_simulations()
    #run_cartpole_simulations()

    # Generate plots
    plot_wireless()
    plot_battery()
    plot_pendulum()
    plot_cartpole()