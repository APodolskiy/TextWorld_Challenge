from typing import Dict

from agents.utils.params import Params


class EpsScheduler:
    def __init__(self, config: Params):
        self.init_eps = config.pop("init_eps")
        self.gamma = config.pop("gamma")
        self.step_size = config.pop("step_size")
        self.min_eps = config.pop("min_eps")
        self.current_step = 1

    @property
    def eps(self):
        return max(
            self.min_eps,
            self.init_eps * self.gamma ** (self.current_step // self.step_size),
        )

    def increase_step(self):
        self.current_step += 1


class DeterministicEpsScheduler:
    def __init__(self):
        pass

    @property
    def eps(self):
        return 0.0


class LinearScheduler:
    def __init__(self, config: Dict):
        self.init_eps = config.pop("init_eps", 1.0)
        self.final_eps = config.pop("final_eps", 0.2)
        self.steps = config.pop("steps", 1000)
        self.step = (self.init_eps - self.final_eps) / self.steps

    def __call__(self, current_step):
        return max(self.final_eps, self.init_eps - self.step * current_step)
