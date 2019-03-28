import torch
from torch.nn.functional import softmax
from numpy.random import choice, rand


class GreedyPolicy:
    def __call__(self, q_values_batch):
        return [q_values.argmax().item() for q_values in q_values_batch]


class BoltzmannPolicy:
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, q_values_batch):
        return [
            softmax(q_values / self.temperature, dim=0).multinomial(1).item()
            for q_values in q_values_batch
        ]


class RandomPolicy:
    def __call__(self, q_values_batch):
        return [choice(len(q_values)) for q_values in q_values_batch]


class MixingPolicy:
    def __init__(self, policy_1, policy_2, epsilon):
        self.epsilon = epsilon
        self.policy_1 = policy_1
        self.policy_2 = policy_2

    def __call__(self, q_values_batch):
        if rand() < self.epsilon:
            return self.policy_1(q_values_batch)
        return self.policy_2(q_values_batch)


if __name__ == "__main__":
    greedy_policy = GreedyPolicy()
    boltzmann_policy = BoltzmannPolicy(temperature=1.0)
    random_policy = RandomPolicy()
    q_vals = [
        torch.tensor([1, 0, 0.4]),
        torch.tensor([0.1, 10, 0.4]),
        torch.arange(10).to(torch.float32),
    ]
    print(greedy_policy(q_vals))
    print(boltzmann_policy(q_vals))
    print(random_policy(q_vals))
