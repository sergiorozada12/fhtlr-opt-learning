from src.experiments.gridworld import run_gridworld_simulations
from src.experiments.wireless import run_wireless_simulations
from src.experiments.battery import run_battery_simulations
from src.experiments.ChannelCoding import run_channelcoding_simulations


if __name__ == "__main__":
    run_channelcoding_simulations()
    #run_gridworld_simulations()
    #run_wireless_simulations()
    #run_battery_simulations()