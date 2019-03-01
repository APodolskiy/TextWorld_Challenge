from typing import Dict

from agents.utils.params import Params


class EpsScheduler:
    def __init__(self, config: Params):
        self.init_eps = config.pop("init_eps", 1.0)
        self.gamma = config.pop("gamma", 0.95)
        self.step_size = config.pop("step_size", 1)
        self.min_eps = config.pop("min_eps", 0.1)

    def eps(self, current_step):
        return max(self.min_eps, self.init_eps * self.gamma ** (current_step // self.step_size))


class LinearScheduler:
    def __init__(self, config: Dict):
        self.init_eps = config.pop("init_eps", 1.0)
        self.final_eps = config.pop("final_eps", 0.2)
        self.steps = config.pop("steps", 1000)
        self.step = (self.init_eps - self.final_eps) / self.steps

    def __call__(self, current_step):
        return max(self.final_eps, self.init_eps - self.step * current_step)
