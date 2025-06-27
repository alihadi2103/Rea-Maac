import numpy as np
from scipy.integrate import solve_ivp






class OrnsteinUhlenbeckNoise:
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2, dt=0.1):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.state += dx
        return self.state













class ReactorEnv:
    def __init__(self, config):
        self.A = config['A']
        self.rho_a = config['rho_a']
        self.rho_b = config['rho_b']
        self.rho_w = config['rho_w']
        self.r = config['r']
        self.ko = config['ko']
        self.E = config['E']
        self.R = config['R']
        self.cp_w = config['cp_w']
        self.cp_a = config['cp_a']
        self.cp_b = config['cp_b']
        self.dHr = config['dHr']
        self.Tref = config['Tref']

        self.Cai = config['Cai']
        self.Cbi = config['Cbi']

        self.h_sp = config['h_sp']
        self.Ca_sp = config['Ca_sp']
        self.Cb_sp = config['Cb_sp']
        self.Cc_sp = config['Cc_sp']
        self.T_sp = config['T_sp']

        self.h_min = config['h_min']
        self.h_max = config['h_max']
        self.T_min = config['T_min']
        self.T_max = config['T_max']

        self.dt = config['dt']
        self.max_steps = config['max_steps']
        self.initial_state = np.array(config['initial_state'], dtype=np.float64)

        self.min_height = 0.01
        self.min_concentration = 1e-6
        self.max_reaction_rate = 1e4
        self.precision_tolerance = 1e-4

        self.n_agents = 4
        self.obs_dim = 2
        self.action_dim = 1

        self.reward_type = config.get("reward_type", "basic")
        
        self.ou_noise_Fai = OrnsteinUhlenbeckNoise(mu=0.0, theta=0.3, sigma=0.01, dt=self.dt)
        self.ou_noise_Fbi = OrnsteinUhlenbeckNoise(mu=0.0, theta=0.3, sigma=0.01, dt=self.dt)
        self.ou_noise_Tai = OrnsteinUhlenbeckNoise(mu=0.0, theta=0.2, sigma=0.5, dt=self.dt)
        self.ou_noise_Tbi = OrnsteinUhlenbeckNoise(mu=0.0, theta=0.2, sigma=0.5, dt=self.dt)

        self.reset()

    def reset(self):
        self.h, self.Ca, self.Cb, self.Cc, self.T = self.initial_state
        self.current_step = 0
        self.last_state = self.initial_state.copy()
        self.prev_action = None
        self.current_action = None
        self.prev_error = None
        
        self.ou_noise_Fai.reset()
        self.ou_noise_Fbi.reset()
        self.ou_noise_Tai.reset()
        self.ou_noise_Tbi.reset()
        
        return self.get_obs()

    def get_obs(self):
        return np.array([
            [self.Cc, self.h],
            [self.Cb, self.T],
            [self.T, self.Cc],
            [self.T, self.Ca]
        ], dtype=np.float32)

    def step(self, actions):
        if isinstance(actions, list) and len(actions) == 1 and isinstance(actions[0], np.ndarray):
            actions = actions[0].flatten()
        elif isinstance(actions, np.ndarray) and actions.ndim > 1:
            actions = actions.flatten()

        actions = [
            a.detach().cpu().numpy() if hasattr(a, 'detach') else a
            for a in actions
        ]
        actions = [a.item() if isinstance(a, np.ndarray) and a.size == 1 else a for a in actions]

        if isinstance(actions, list) and len(actions) > 1 and not isinstance(actions[0], np.ndarray):
            Fai = np.clip(actions[0], 0, 10000) + self.ou_noise_Fai.sample()
            Fbi = np.clip(actions[1], 0, 10000) + self.ou_noise_Fbi.sample()
            Tai = np.clip(actions[2], 0, 1000) + self.ou_noise_Tai.sample()
            Tbi = np.clip(actions[3], 0, 1000) + self.ou_noise_Tbi.sample()
        else:
            Fai = np.clip(actions[0][0, 0], 0, 10000) + self.ou_noise_Fai.sample()
            Fbi = np.clip(actions[0][1, 0], 0, 10000) + self.ou_noise_Fbi.sample()
            Tai = np.clip(actions[0][2, 0], 0, 1000) + self.ou_noise_Tai.sample()
            Tbi = np.clip(actions[0][3, 0], 0, 1000) + self.ou_noise_Tbi.sample()
            
        self.current_action = [Fai, Fbi, Tai, Tbi]

        try:
            initial_state = np.maximum([
                self.min_height, self.min_concentration,
                self.min_concentration, self.min_concentration, 0
            ], self.last_state)

            sol = solve_ivp(
                fun=lambda t, y: self.reactor_odes(t, y, Fai, Fbi, Tai, Tbi),
                t_span=[0, self.dt],
                y0=initial_state,
                method='RK45', rtol=1e-5, atol=1e-8
            )

            new_state = sol.y[:, -1]
            new_state[0] = np.clip(new_state[0], self.min_height, 10000)
            new_state[1:4] = np.clip(new_state[1:4], self.min_concentration, 10000)
            new_state[4] = np.clip(new_state[4], 0, 1000)

        except Exception:
            new_state = self.last_state

        self.h, self.Ca, self.Cb, self.Cc, self.T = new_state
        self.last_state = new_state.copy()

        obs = self.get_obs()
        reward = self.calculate_reward()
        self.prev_action = self.current_action

        h_viol = max(self.h_min - self.h, 0) + max(self.h - self.h_max, 0)
        T_viol = max(self.T_min - self.T, 0) + max(self.T - self.T_max, 0)
        cost = np.array([h_viol, T_viol])

        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = {
            'step': self.current_step,
            'h_violation': self.h < self.h_min or self.h > self.h_max,
            'T_violation': self.T < self.T_min or self.T > self.T_max,
            'precision_achieved': self.is_at_precision()
        }

        return obs, reward, done, info, cost

    def calculate_reward(self):
        if self.reward_type == "nmpc":
            return self.calculate_nmpc_reward()
        elif self.reward_type == "derivative":
            return self.calculate_derivative_reward()
        else:
            return self.calculate_basic_reward()

    def calculate_basic_reward(self):
        tol = self.precision_tolerance
        err_Cc = abs(self.Cc - self.Cc_sp)/self.Cc_sp
        err_T = abs(self.T - self.T_sp)/self.T_sp
        err_h = abs(self.h - self.h_sp)/self.h_sp

        reward = 0.0
        for err in [err_Cc, err_T, err_h]:
            if err < tol:
                reward += 1000.0 * (1 - err / tol)

        penalty = 0.0
        if self.T < self.T_min:
            penalty += 10.0 * (self.T_min - self.T)
        elif self.T > self.T_max:
            penalty += 10.0 * (self.T - self.T_max)

        if self.h < self.h_min:
            penalty += 10.0 * (self.h_min - self.h)
        elif self.h > self.h_max:
            penalty += 10.0 * (self.h - self.h_max)

        penalty += 100.0 * (err_Cc + err_T + err_h)

        return reward - penalty

    def calculate_nmpc_reward(self):
        y = np.array([self.Cc, self.T, self.h])
        r = np.array([self.Cc_sp, self.T_sp, self.h_sp])
        e = (y - r) / r

        Q = np.diag([1.0, 1.0, 1.0])
        S = np.diag([0.1, 0.1, 0.1, 0.1])
        P = 1000

        tracking_error = e.T @ Q @ e

        if self.prev_action is not None and self.current_action is not None:
            delta_u = np.array(self.current_action) - np.array(self.prev_action)
            rate_penalty = delta_u.T @ S @ delta_u
        else:
            rate_penalty = 0.0

        penalty = 0.0
        if self.h < self.h_min:
            penalty += (self.h_min - self.h) ** 2
        elif self.h > self.h_max:
            penalty += (self.h - self.h_max) ** 2

        if self.T < self.T_min:
            penalty += (self.T_min - self.T) ** 2
        elif self.T > self.T_max:
            penalty += (self.T - self.T_max) ** 2

        constraint_penalty = P * penalty
        
        print("rate_penalty", rate_penalty)

        total_cost = tracking_error + rate_penalty + constraint_penalty
        return -total_cost

    def calculate_derivative_reward(self):
        y = np.array([self.Cc, self.T, self.h])
        r = np.array([self.Cc_sp, self.T_sp, self.h_sp])
        
        e = (y - r) / r
        
        print("Current action:", self.current_action)
        print("Previous action:", self.prev_action)
      
        
        print("y", y)
        print("r", r)
        print("e", e)
        
       

        if self.prev_error is not None:
            de = (e - self.prev_error) 
        else:
            de = np.zeros_like(e)

        self.prev_error = e.copy()

        Q = np.diag([1.0, 1.0, 1.0])
        D = np.diag([1.0, 1.0, 1.0])
        P = 1000

        error_cost = e.T @ Q @ e
        derivative_cost = de.T @ D @ de

        penalty = 0.0
        if self.h < self.h_min:
            penalty += (self.h_min - self.h) ** 2
        elif self.h > self.h_max:
            penalty += (self.h - self.h_max) ** 2

        if self.T < self.T_min:
            penalty += (self.T_min - self.T) ** 2
        elif self.T > self.T_max:
            penalty += (self.T - self.T_max) ** 2

        constraint_penalty = P * penalty

        total_cost = error_cost + derivative_cost + constraint_penalty
        return -total_cost

    def is_at_precision(self):
        return (
            abs(self.Cc - self.Cc_sp) < self.precision_tolerance and
            abs(self.T - self.T_sp) < self.precision_tolerance and
            abs(self.h - self.h_sp) < self.precision_tolerance and
            self.h_min <= self.h <= self.h_max and
            self.T_min <= self.T <= self.T_max
        )




