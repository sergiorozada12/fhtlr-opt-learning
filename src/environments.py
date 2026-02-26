from typing import List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
try:
    import matrix_mdp
except ImportError:
    matrix_mdp = None


class GridWorldEnv:
    def __init__(
        self,
        nS: int = 25,
        nA: int = 4,
        W: int = 5,
        H: int = 5,
    ) -> None:
        if matrix_mdp is None:
            raise ImportError("matrix_mdp module is not installed. Please install it to use GridWorldEnv.")
        self.nS = nS
        self.nA = nA
        self.W = W
        self.H = H

        incs = {
            0: (0, -1),  # UP
            1: (-1, 0),  # LEFT
            2: (0, +1),  # DOWN
            3: (+1, 0),  # RIGHT
            4: ( 0, 0),  # NOTHING
        }

        P = np.zeros((nS, nS, nA))
        #R = -0.1*np.ones((nS, nS, nA))
        R = np.zeros((nS, nS, nA))
        P_0 = np.ones(nS) / (nS - 4)
        P_0[0] = 0
        P_0[nS - 1] = 0
        P_0[4 * W + 0] = 0
        P_0[0 * W + 4] = 0
        #P_0[4 * W + 2] = 0

        for s in range(nS):
            for a in range(nA):
                x, y = s % W, s // W
                dx, dy = incs[a]
                x = np.clip(x + dx, 0, W - 1)
                y = np.clip(y + dy, 0, W - 1)

                sp = y * W + x

                # Transition
                if s != 0 and s != nS - 1 and s != (4 * W + 0) and s != (0 * W + 4):
                #if s != (4 * W + 2):
                    P[sp, s, a] = 1
                if a == 4:
                    R[sp, s, a] = 0
                # Reward
                if sp == 0 and s != 0:
                    R[sp, s, a] = 1
                elif sp == nS - 1 and s != nS - 1:
                    R[sp, s, a] = 1
                elif sp == (4 * W + 0) and s != (4 * W + 0):
                    R[:, s, a] = 1
                elif sp == (0 * W + 4) and s != (0 * W + 4):
                    R[:, s, a] = 1
        
        self.P = P
        self.R = R
        self.env = gym.make("matrix_mdp/MatrixMDP-v0", p_0=P_0, p=P, r=R)

        self.H = H
        self.h = 0

    def reset(self):
        s, _ = self.env.reset()
        return np.array([s % self.W, s // self.W]), None

    def step(self, a):
        sp, r, d, _, _ = self.env.step(int(a[0]))
        self.h += 1

        if self.h == self.H:
            d = True

        if d:
            self.h = 0
        return np.array([sp % self.W, sp // self.W]), r, d, None, None


class WirelessCommunicationsEnv:
    """
    SETUP DESCRIPTION
    - Wireless communication setup, focus in one user sharing the media with other users
    - Finite horizon transmission (T slots)
    - User is equipped with a battery and queue where info is stored
    - K orthogonal channels, user can select power for each time instant
    - Rate given by Shannon's capacity formula

    STATES
    - Amount of energy in the battery: bt (real and positive)
    - Amount of packets in the queue: queuet (real and positive)
    - Normalized SNR (aka channel gain) for each of the channels: gkt (real and positive)
    - Channel being currently occupied: ok (binary)

    ACTIONS
    - Accessing or not each of the K channels pkt
    - Tx power for each of the K channels
    - We can merge both (by setting pkt=0)
    """

    def __init__(
        self,
        T: int = 10,  # Number of time slots
        K: int = 3,  # Number of channels
        snr_max: float = 10,  # Max SNR
        snr_min: float = 2,  # Min SNR
        snr_autocorr: float = 0.7,  # Autocorrelation coefficient of SNR
        P_occ: np.ndarray = np.array(
            [  # Prob. of transition of occupancy
                [0.3, 0.5],
                [0.7, 0.5],
            ]
        ),
        occ_initial: List[int] = [1, 1, 1],  # Initial occupancy state
        batt_harvest: float = 3,  # Battery to harvest following a Bernoulli
        P_harvest: float = 0.5,  # Probability of harvest energy
        batt_initial: float = 5,  # Initial battery
        batt_max_capacity: float = 50,  # Maximum capacity of the battery
        batt_weight: float = 1.0,  # Weight for the reward function
        queue_initial: float = 20,  # Initial size of the queue
        queue_weight: float = 1e-1,  # Weight for the reward function
        loss_busy: float = 0.80,  # Loss in the channel when busy
    ) -> None:
        self.T = T
        self.K = K

        self.snr = np.linspace(snr_max, snr_min, K)
        self.snr_autocorr = snr_autocorr

        self.occ_initial = occ_initial
        self.P_occ = P_occ

        self.batt_harvest = batt_harvest
        self.batt_initial = batt_initial
        self.P_harvest = P_harvest
        self.batt_max_capacity = batt_max_capacity
        self.batt_weight = batt_weight

        self.queue_initial = queue_initial
        self.queue_weight = queue_weight

        self.loss_busy = loss_busy

    def step(self, p: np.ndarray):
        if np.sum(p) > self.batt[self.t]:
            p = self.batt[self.t] * p / np.sum(p)

        self.c[:, self.t] = np.log2(1 + self.g[:, self.t] * p)
        self.c[:, self.t] *= (1 - self.loss_busy) * self.occ[:, self.t] + (
            1 - self.occ[:, self.t]
        )

        self.t += 1

        self.h[:, self.t] = np.sqrt(0.5 * self.snr) * (
            np.random.randn(self.K) + 1j * np.random.randn(self.K)
        )
        self.h[:, self.t] *= np.sqrt(1 - self.snr_autocorr)
        self.h[:, self.t] += np.sqrt(self.snr_autocorr) * self.h[:, self.t - 1]
        self.g[:, self.t] = np.abs(self.h[:, self.t]) ** 2

        self.occ[:, self.t] += (np.random.rand(self.K) > self.P_occ[1, 1]) * self.occ[
            :, self.t - 1
        ]
        self.occ[:, self.t] += (np.random.rand(self.K) > self.P_occ[0, 0]) * (
            1 - self.occ[:, self.t - 1]
        )

        energy_harv = self.batt_harvest * (self.P_harvest > np.random.rand())
        self.batt[self.t] = self.batt[self.t - 1] - np.sum(p) + energy_harv
        self.batt[self.t] = np.clip(self.batt[self.t], 0, self.batt_max_capacity)

        packets = 0
        if self.batt[self.t - 1] > 0:
            packets = np.sum(self.c[:, self.t - 1])
        self.queue[self.t] = self.queue[self.t - 1] - packets
        self.queue[self.t] = max(0, self.queue[self.t])

        r = (
            self.batt_weight * np.log(1 + self.batt[self.t])
            - self.queue_weight * self.queue[self.t]
        )
        done = self.t == self.T

        return self._get_obs(self.t), r, done, None, None

    def reset(self):
        self.t = 0
        self.h = np.zeros((self.K, self.T + 1), dtype=np.complex64)
        self.g = np.zeros((self.K, self.T + 1))
        self.c = np.zeros((self.K, self.T + 1))
        self.occ = np.zeros((self.K, self.T + 1))
        self.queue = np.zeros(self.T + 1)
        self.batt = np.zeros(self.T + 1)

        self.h[:, 0] = np.sqrt(0.5 * self.snr) * 0.5
        self.g[:, 0] = np.abs(self.h[:, 0]) ** 2
        self.occ[:, 0] = self.occ_initial
        self.queue[0] = self.queue_initial
        self.batt[0] = self.batt_initial

        return self._get_obs(0), None

    def _get_obs(self, t):
        return np.concatenate(
            [self.g[:, t], self.occ[:, t], [self.queue[t], self.batt[t]]]
        )


class BatteryChargingEnv(gym.Env):
    """
    SETUP DESCRIPTION
    - Battery management setup for optimal charging control with a finite horizon of T time steps.
    - Battery connected to various energy sources with varying costs and availability (e.g., solar, wind, grid power).
    - Decision-making at each time step involves selecting the charging power level, considering both immediate energy needs and long-term battery durability.
    - Durability term in the reward function penalizes charging behaviors that reduce battery lifespan, balancing immediate rewards with battery health.

    STATES
    - Battery Energy Level b_t: Amount of energy currently stored in the battery (real and positive).
    - Energy Source Cost c_t: Cost associated with charging from each available energy source (real and positive, potentially varying by source and time).
    - Battery Health h_t: Indicator of battery degradation level or capacity fade, representing battery wear due to previous charging decisions (real and positive).
    - Energy Demand d_t: Amount of energy required at the current time step (real and positive, representing power load or demand fluctuations).

    ACTIONS
    - Charging Power p_t: The power level selected to charge the battery at each time step (real and non-negative).
    - Energy Source Selection s_t: The energy source chosen for charging (e.g., solar, wind, grid); each source has different costs and potentially different impacts on battery health.
    """
    def __init__(self, H=5):
        super(BatteryChargingEnv, self).__init__()

        self.H = H
        self.current_step = 0

        # State space: [SoC, Solar Availability, Wind Availability, Grid Price]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),  # SoC, Solar, Wind, Grid Price min values
            high=np.array([1.0, 1.0, 1.0, 1.0]),  # SoC, Solar, Wind, Grid Price max values
            dtype=np.float64
        )

        # Action space: charging rates for [Solar, Wind, Grid]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,  # Assuming max charging rate of 1 for each source
            shape=(3,),
            dtype=np.float64
        )

        self.soc_target = 1.0  # Target SoC is 100%
        self.lambda_penalty = 5.0  # Penalty weight for not reaching the target SoC by end
        self.lambda_durability = 20.0

    def reset(self):
        self.soc = 0.0  # Initial state of charge (SoC), can vary as desired
        self.solar_availability = np.random.uniform(0.3, 0.7)  # Random initial solar availability
        self.wind_availability = np.random.uniform(0.3, 0.7)   # Random initial wind availability
        self.grid_price = np.random.uniform(0.1, 0.3)          # Random initial grid price

        self.current_step = 0

        return self._get_obs(), None

    def _get_obs(self):
        return np.array(
            [
                self.soc,
                self.solar_availability,
                self.wind_availability,
                self.grid_price
            ], dtype=np.float64)

    def step(self, action):
        solar_rate, wind_rate, grid_rate = action

        solar_efficiency = 0.9 if self.solar_availability > 0.5 else 0.2
        wind_efficiency = 0.8 if self.wind_availability > 0.5 else 0.2

        soc_increase = (
            solar_efficiency * solar_rate * self.solar_availability +
            wind_efficiency * wind_rate * self.wind_availability +
            grid_rate
        )

        self.soc = min(1.0, self.soc + soc_increase)

        degradation_cost = self.lambda_durability * (self.soc ** 2) * (soc_increase ** 2)
        charging_cost = grid_rate * self.grid_price
        unmet_target_penalty = self.lambda_penalty * max(0, self.soc_target - self.soc)
        reward = - (charging_cost + unmet_target_penalty + degradation_cost)

        self.solar_availability = np.clip(self.solar_availability + np.random.normal(0, 0.1), 0.0, 1.0)
        self.wind_availability = np.clip(self.wind_availability + np.random.normal(0, 0.1), 0.0, 1.0)
        self.grid_price = np.clip(self.grid_price + np.random.normal(0, 0.05), 0.1, 0.5)

        self.current_step += 1
        done = self.current_step >= self.H

        return self._get_obs(), reward, done, None, None

class SourceChannelCodingEnv(gym.Env):
    """
    Finite Horizon Source-Channel Coding Environment with Costs, Transmission Errors, and Dynamic Compression

    - T time steps
    - K parallel channels with stochastic noise (channel gains evolve over time)
    - At each step, choose compression and coding allocation
    - Compression shrinks remaining data dynamically
    """

    def __init__(self, T=10, K=3, data_initial=15.0, rho=0.9, sigma=0.2, beta_success=2.0):
        super(SourceChannelCodingEnv, self).__init__()

        self.T = T
        self.K = K
        self.data_initial = data_initial  # Initial amount of source bits to send
        self.rho = rho  # Autocorrelation coefficient for channel evolution
        self.sigma = sigma  # Noise strength for channel gain evolution
        self.beta_success = beta_success  # Sensitivity of success probability to channel gain

        self.current_step = 0

        # State: [Remaining data] + [Channel gains]
        low_state = np.array([0.0] + [0.0] * K)
        high_state = np.array([np.inf] + [np.inf] * K)
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

        # Action: [Compression level (0-1)] + [Power allocation per channel (0-1)]
        low_action = np.array([0.0] + [0.0] * K)
        high_action = np.array([1.0] + [1.0] * K)
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        # Compression-distortion model parameters
        self.alpha_distortion = 10.0  # Controls distortion curve

        # Final penalty weight
        self.final_penalty_weight = 50.0

        # controls the effectiveness of compression
        self.alpha_compression = 5

        # Cost per bit transmitted per channel (randomized at init)
        self.channel_costs = np.random.uniform(0.5, 1.5, size=self.K)

    def reset(self):
        self.current_step = 0
        self.remaining_data = self.data_initial
        self.gains = np.abs(np.random.normal(1.0, 0.5, size=self.K))  # Initial random channel gains

        return self._get_obs(), None

    def _get_obs(self):
        return np.concatenate([[self.remaining_data], self.gains])

    def step(self, action):
        compression_level = np.clip(action[0], 0.0, 1.0)
        allocations = np.clip(action[1:], 0.0, 1.0)
        allocations /= np.sum(allocations) + 1e-8  # Normalize allocations to sum to 1

        # Apply compression dynamically to the remaining data
        self.remaining_data *= (1.0 / (1.0 + self.alpha_compression * compression_level))

        # Channel capacities: simple model log(1 + SNR)
        capacities = np.log2(1 + self.gains)

        # Transmission success probabilities
        success_probs = 1 - np.exp(-self.beta_success * self.gains)
        success_mask = np.random.rand(self.K) < success_probs

        # Bits transmitted considering success/failure
        bits_transmitted = np.sum(allocations * capacities * success_mask)

        # Update remaining data after transmission
        self.remaining_data = max(0.0, self.remaining_data - bits_transmitted)

        # Transmission cost
        transmission_cost = np.sum(allocations * capacities * self.channel_costs)

        # Compute distortion: quadratic in compression level
        distortion = self.alpha_distortion * (compression_level ** 2)

        # Reward: negative distortion, negative transmission cost
        reward = - distortion - transmission_cost

        done = False
        self.current_step += 1

        # Update channel gains (Gauss-Markov process)
        noise = np.random.normal(0.0, self.sigma, size=self.K)
        self.gains = self.rho * self.gains + np.sqrt(1 - self.rho ** 2) * noise
        self.gains = np.clip(self.gains, 0.0, np.inf)

        if self.current_step >= self.T:
            done = True
            reward -= self.final_penalty_weight * self.remaining_data / self.data_initial

        return self._get_obs(), reward, done, None, None


class PendulumTerminalReward(gym.Env):
    """
    Pendulum environment where reward is 0 everywhere except at the final step T.
    The agent starts upright but with a high velocity, forcing it to brake immediately.
    """
    def __init__(self, T=10, max_torque=2.0, low_thdot=2.0, high_thdot=4.0):
        super(PendulumTerminalReward, self).__init__()
        self.T = T
        self.max_torque = max_torque
        self.low_thdot = low_thdot
        self.high_thdot = high_thdot
        
        self.max_speed = 8.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        
        self.current_step = 0
        self.state = None

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        reward = 0.0 # Default reward

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        
        self.current_step += 1
        terminated = False
        truncated = False
        
        if self.current_step >= self.T:
            terminated = True
            # Terminal reward
            reward = -(angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2))
            
        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Start upright (theta approx 0) but with high velocity
        high_theta = 0.05
        init_theta = np.random.uniform(low=-high_theta, high=high_theta)
        
        init_thdot = np.random.uniform(low=self.low_thdot, high=self.high_thdot)
        if np.random.rand() < 0.5:
            init_thdot = -init_thdot
            
        self.state = np.array([init_theta, init_thdot])
        self.last_u = None
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        th, thdot = self.state
        return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float64)

    def render(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        canvas = FigureCanvas(fig)
        
        # Pendulum parameters
        l = self.l
        
        x = l * np.sin(self.state[0])
        y = l * np.cos(self.state[0])
        
        # Draw track (reference circle)
        track = plt.Circle((0, 0), l, color='k', fill=False, linestyle='--', linewidth=1, alpha=0.2)
        ax.add_artist(track)

        # Draw pivot
        ax.plot([0], [0], 'ko', markersize=8)
        # Draw rod
        ax.plot([0, x], [0, y], 'b-', linewidth=6)
        # Draw bob
        ax.plot([x], [y], 'ro', markersize=16)
        
        ax.set_xlim([-l*1.3, l*1.3])
        ax.set_ylim([-l*1.3, l*1.3])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Render to numpy array
        canvas.draw()
        try:
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        except AttributeError:
            image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        
        w, h = fig.canvas.get_width_height()
        image = image.reshape((h, w, -1))
        if image.shape[2] == 4:
            image = image[:, :, :3]
            
        plt.close(fig)
        return image


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class CartPoleTerminalReward(gym.Env):
    """
    CartPole environment with terminal reward.
    Goal: Force the agent to balance the pole at the final step T.
    """
    def __init__(self, T=10, max_force=10.0, low_thdot=0.0, high_thdot=0.5):
        super(CartPoleTerminalReward, self).__init__()
        self.T = T
        self.max_force = max_force
        self.low_thdot = low_thdot
        self.high_thdot = high_thdot
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.tau = 0.02  # seconds between state updates
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float64).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float64).max],
            dtype=np.float64)
            
        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force,
            shape=(1,),
            dtype=np.float64
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)

        self.current_step = 0
        self.state = None

    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = np.clip(action, -self.max_force, self.max_force)[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)
        
        self.current_step += 1
        terminated = False
        truncated = False
        reward = 0.0
        
        if self.current_step >= self.T:
            terminated = True
            # Terminal reward: 1 if upright, 0 otherwise (or continuous)
            # Let's make it continuous: -(theta^2 + 0.1*theta_dot^2)
            # Maybe restrict x? -x^2? 
            # Primary goal: balance.
            reward = -(theta**2 + 0.1*theta_dot**2)
            
            # Penalize if out of bounds (though usually episode ends)
            if x < -self.x_threshold or x > self.x_threshold:
               reward -= 10.0
            if theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians:
               reward -= 10.0

        return np.array(self.state, dtype=np.float64), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # Start somewhat unstable
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        
        # Initialize angular velocity based on parameters
        init_thdot = np.random.uniform(low=self.low_thdot, high=self.high_thdot)
        if np.random.rand() < 0.5:
            init_thdot = -init_thdot
        self.state[3] = init_thdot
        
        self.current_step = 0
        return np.array(self.state, dtype=np.float64), {}

    def render(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        canvas = FigureCanvas(fig)

        # State
        x, _, theta, _ = self.state
        
        # Parameters for rendering
        screen_width = 6.0
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 0.0 # Cart vertical position
        cartwidth = 1.0
        cartheight = 0.6
        polewidth = 0.2
        polelen = self.length * 2 # Visual length
        
        # Draw track
        ax.plot([-self.x_threshold, self.x_threshold], [carty, carty], 'k-', linewidth=1)
        
        # Draw cart
        cart_rect = Rectangle((x - cartwidth / 2.0, carty - cartheight / 2.0), cartwidth, cartheight, color='black')
        ax.add_patch(cart_rect)
        
        # Draw pole
        # Pole start (center of cart)
        pole_start_x = x
        pole_start_y = carty
        # Pole end
        pole_end_x = x + polelen * np.sin(theta)
        pole_end_y = carty + polelen * np.cos(theta)
        
        ax.plot([pole_start_x, pole_end_x], [pole_start_y, pole_end_y], 'r-', linewidth=6)
        
        # Settings
        ax.set_xlim([-self.x_threshold * 2, self.x_threshold * 2])
        ax.set_ylim([-1, 2])
        ax.set_aspect('equal')
        ax.axis('off')

        # Render to numpy array
        canvas.draw()
        try:
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        except AttributeError:
             image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        
        w, h = fig.canvas.get_width_height()
        image = image.reshape((h, w, -1))
        # Drop alpha channel if present
        if image.shape[2] == 4:
            image = image[:, :, :3]
            
        plt.close(fig)
        return image


class MountainCarFiniteHorizon(gym.Env):
    def __init__(self, T=200, reward_fn=None):
        
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = 0
        
        self.force = 0.0015
        self.gravity = 0.0025
        
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low,
            high=self.high,
            dtype=np.float32
        )
        
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn
        self.current_step = 0
        self.state = None

    def _default_reward_fn(self, position, velocity, action):
        # Default continuous mountain car reward
        reward = 0
        if position >= self.goal_position:
            reward += 100.0
        reward -= np.power(action[0], 2) * 0.1
        return reward

    def step(self, action):
        position, velocity = self.state
        
        force = np.clip(action, -1.0, 1.0)[0]
        
        velocity += force * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        if position == self.min_position and velocity < 0:
            velocity = 0
            
        self.state = np.array([position, velocity], dtype=np.float64)
        
        reward = self.reward_fn(position, velocity, np.array([force]))
        
        self.current_step += 1
        terminated = False
        truncated = False
        
        if position >= self.goal_position: # Goal reached
            terminated = True
            
        if self.current_step >= self.T:
            terminated = True
            truncated = False # Handled as termination for the agent
            
        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0], dtype=np.float64)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.state, dtype=np.float64)

    def render(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        canvas = FigureCanvas(fig)
        
        # Mountain car landscape
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = np.sin(3 * xs) * 0.45 + 0.55
        
        ax.plot(xs, ys, 'k-', linewidth=1)
        
        # Car position
        pos = self.state[0]
        y_car = np.sin(3 * pos) * 0.45 + 0.55
        
        ax.plot([pos], [y_car], 'go', markersize=10) # Car
        ax.plot([self.goal_position], [np.sin(3*self.goal_position)*0.45+0.55], 'ys', markersize=8) # Goal
        
        ax.set_xlim([self.min_position, self.max_position])
        ax.set_ylim([-0.1, 1.1])
        ax.axis('off')
        
        # Render to numpy array
        # Render to numpy array
        canvas.draw()
        try:
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        except AttributeError:
             image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        
        # Reshape and handle potentially different channel counts (RGB vs RGBA)
        w, h = fig.canvas.get_width_height()
        image = image.reshape((h, w, -1))
        if image.shape[2] == 4:
            image = image[:, :, :3]
            
        plt.close(fig)
        return image