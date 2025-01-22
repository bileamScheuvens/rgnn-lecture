import einops
import torch


class Planner:
    def predict(self, model, actions, observation):
        observations = []
        for t in range(self._horizon):
            if len(actions[t].shape) > 1:
                inp = torch.cat([observation, actions[t]], dim=1)
            else:
                inp = torch.cat([observation, actions[t]])
            observation = model.forward(inp)
            observations.append(observation)
        observations = torch.stack(observations)
        return observations

    def __call__(self, model, observation):
        pass


class RandomPlanner(Planner):
    def __init__(
        self,
        action_size=2,
        horizon=20,
    ):
        Planner.__init__(self)
        self._horizon = horizon
        self._action_size = action_size

    def __call__(self, model, observation):
        action = torch.rand([self._action_size]) * 2 - 1
        return action


class CrossEntropyMethod(Planner):
    def __init__(
        self,
        action_size=2,
        horizon=50,
        num_inference_cycles=2,
        num_predictions=50,
        num_elites=5,
        num_keep_elites=2,
        criterion=torch.nn.MSELoss(),
        policy_handler=lambda x: x,
        var=0.2,
    ):
        Planner.__init__(self)
        self._action_size = action_size
        self._horizon = horizon
        self._num_inference_cycles = num_inference_cycles
        self._num_predictions = num_predictions
        self._num_elites = num_elites
        self._num_keep_elites = num_keep_elites
        self._criterion = criterion
        self._policy_handler = policy_handler

        self._mu = torch.zeros([self._horizon, self._action_size])
        self._var_init = var * torch.ones([self._horizon, self._action_size])
        self._var = self._var_init.clone().detach()
        self._dist = torch.distributions.MultivariateNormal(
            torch.zeros(self._action_size), torch.eye(self._action_size)
        )
        self._last_actions = None
        self._momentum = 0.1

    def __call__(self, model, observation):

        for _ in range(self._num_inference_cycles):

            # translation table:
            # N = num_predictions
            # H = horizon
            # A = action_size
            # K = num_elites
            
            samples = []
            for _ in range(self._num_predictions):
                samples.append(self._dist.sample([self._horizon]))
            samples = torch.stack(samples) # [N, H, A]
            samples += self._mu.unsqueeze(0)
            samples *= self._var.unsqueeze(0)

            
            costs = []
            for sample in samples:
                actions = self._policy_handler(sample)
                observations = self.predict(model, actions, observation)
                cost = self._criterion(observations.unsqueeze(0)).mean()
                costs.append(cost)
            costs = torch.stack(costs) # [N]
            
            elite_idx = torch.topk(costs, self._num_elites, largest=False).indices
            elites = samples[elite_idx] # [K, H, A]
            self._mu = torch.mean(elites, dim=0) # [H, A]
            self._var = torch.var(elites, dim=0) # [H, A]

            actions = self._policy_handler(self._mu)
        
        self._last_actions = actions

        with torch.no_grad():
            # Shift means by one time step
            self._mu[:-1] = self._mu[1:].clone()
            # Reset the variance
            self._var = self._var_init.clone()
            # Shift elites to keep by one time step
            self._last_actions[:-1] = self._last_actions[1:].clone()

        # return actions[0,0,:]
        return actions[0, :]
