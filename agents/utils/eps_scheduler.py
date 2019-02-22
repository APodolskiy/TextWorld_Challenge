from agents.utils.params import Params


class EpsScheduler:
    def __init__(self, config: Params):
        self.init_eps = config.pop("init_eps", 1.0)
        self.gamma = config.pop("gamma", 0.95)
        self.step_size = config.pop("step_size", 1)

    def get_eps(self, current_step):
        return self.init_eps * self.gamma ** (current_step // self.step_size)
