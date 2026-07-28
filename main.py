from src.experiments.gridworld import run_gridworld_simulations
from src.experiments.wireless import run_wireless_simulations
from src.experiments.battery import run_battery_simulations
from src.experiments.channel_coding import run_channel_coding_simulations
from src.experiments.pendulum import run_pendulum_simulations
from src.experiments.cartpole import run_cartpole_simulations
from src.experiments.gym_emp_lowrank import run_gym_simulations
from src.experiments.real_emp_lowrank import run_real_emp_lowrank
from src.plots import plot_wireless, plot_battery, plot_pendulum, plot_cartpole, plot_channel_coding, plot_gym_parafac, plot_gym_returns, plot_real_cases, plot_real_parafac


if __name__ == "__main__":
    """run_gridworld_simulations()
    run_wireless_simulations()
    run_battery_simulations()
    run_pendulum_simulations()
    run_cartpole_simulations()
    run_channel_coding_simulations()
    run_gym_simulations()
    run_real_emp_lowrank()
    
    plot_wireless()
    plot_battery()
    plot_pendulum()
    plot_cartpole()
    plot_channel_coding()
    plot_gym_parafac()
    plot_gym_returns()
    plot_real_cases()"""
    plot_real_parafac()
