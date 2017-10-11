import numpy as np
from cost_functions import trajectory_cost_fn
import time


class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        Controller.__init__(self)
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        Controller.__init__(self)
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.traj_costs = np.array([])

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        all_obs = np.zeros([self.horizon + 1, self.num_simulated_paths, self.ob_dim])
        acs = [self.env.action_space.sample() for _ in range(self.horizon * self.num_simulated_paths)]
        acs = np.array(acs).reshape([self.horizon, self.num_simulated_paths, self.ac_dim])

        # broadcast initial state
        all_obs[0, :, :] = state

        # step through horizons
        for step in range(self.horizon):
            all_obs[step + 1, :, :] = self.dyn_model.predict(all_obs[step, :, :], acs[step, :, :])

        # obs and next_obs
        obs = all_obs[:-1]
        next_obs = all_obs[1:]

        # compute cost along each trajectory
        traj_costs = []
        for i in range(self.num_simulated_paths):
            traj_costs.append(trajectory_cost_fn(self.cost_fn, obs[:, i, :], acs[:, i, :], next_obs[:, i, :]))

        j_best = np.argmax(np.array(traj_costs))
        self.traj_costs = traj_costs

        return acs[0, j_best, :]

    def get_traj_costs(self):
        return self.traj_costs