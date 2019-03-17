from logging import warning

from overrides import overrides
import random
from typing import NamedTuple, List

from tests.multiprocessing_example import Transition


class AbstractReplayMemory:
    """
    Abstract class that describes basic API for Replay Memory
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity

    def push(self, transition: NamedTuple):
        """
        Add transition to the replay memory.
        :param transition: transition performed by an agent, regularly: state, action, reward, next_state.
        """
        raise NotImplementedError("Push method should be implemented!")

    def sample(self, batch_size: int) -> List[NamedTuple]:
        """
        Sample batch of transitions from the replay memory.
        :param batch_size: size of batch
        :return: sampled batch of transitions
        """
        raise NotImplementedError("Sample method should be implemented!")


class ExperienceReplay(AbstractReplayMemory):
    """
    Simple Experience Replay memory.
    """

    def __init__(self, capacity: int = 100_000):
        super(ExperienceReplay, self).__init__(capacity=capacity)
        self.buffer = []
        self.position = 0

    @overrides
    def push(self, transition: NamedTuple):
        """
        Add transition to the replay memory.
        :param transition: transition performed by an agent, regularly: state, action, reward, next_state.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    @overrides
    def sample(self, batch_size: int) -> List[NamedTuple]:
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Can't sample batch of size {batch_size} from the buffer!\n"
                f"There are only {len(self.buffer)} elements in the buffer."
            )
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# TODO: create sum tree data structure
# TODO: continue creation of Prioritized Experience replay
class PrioritizedReplayMemory(AbstractReplayMemory):
    """
    Experience Replay memory that stores transitions with priorities and samples
    elements according to these priorities.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_init: float = 0.4,
        beta_anneal_steps: int = 10000,
    ):
        super(PrioritizedReplayMemory, self).__init__(capacity=capacity)
        self.alpha_smoothing = alpha
        self.buffer = []
        self.beta_init = beta_init
        self.beta_anneal_steps = beta_anneal_steps
        self.beta_steps = 0

    @overrides
    def push(self, transition: NamedTuple):
        pass

    @overrides
    def sample(self, batch_size: int) -> List[NamedTuple]:
        pass

    def _beta_by_step(self, step: int):
        beta_ang = (1 - self.beta_init) / self.beta_anneal_steps
        beta = beta_ang * step + self.beta_init
        return min(beta, 1.0)

    def __len__(self):
        return len(self.buffer)


class BinaryPrioritizeReplayMemory(AbstractReplayMemory):
    def __init__(self, capacity: int = 100_000, priority_fraction: float = 0.0):
        super(BinaryPrioritizeReplayMemory, self).__init__(capacity=capacity)
        self.prior_buffer = {}
        self.prior_position = 0
        self.prior_capacity = int(self.capacity * priority_fraction)
        self.secondary_buffer = {}
        self.secondary_position = 0
        self.secondary_capacity = self.capacity - self.prior_capacity
        self.priority_fraction = priority_fraction

    def push(self, transition: NamedTuple, is_prior: bool = False):
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            self.prior_position = self._push(transition, self.prior_buffer, self.prior_position)
        else:
            self.secondary_position = self._push(transition, self.secondary_buffer, self.secondary_position)

    def _push(self, transition: Transition, buffer, position):
        # if len(buffer) < self.prior_capacity:
        #     buffer.append(None)
        buffer[(transition.previous_state, transition.action)] = transition
        # buffer[position] = transition
        # position = (position + 1) % self.prior_capacity
        return position

    def sample(self, batch_size: int) -> List[NamedTuple]:
        if self.priority_fraction == 0.0:
            samples = random.sample(self.secondary_buffer, batch_size)
        else:
            prior_size = min(
                int(batch_size * self.priority_fraction), len(self.prior_buffer)
            )
            secondary_size = min(batch_size - prior_size, len(self.secondary_buffer))
            prior_samples = random.sample(list(self.prior_buffer.values()), prior_size)
            secondary_samples = random.sample(list(self.secondary_buffer.values()), secondary_size)
            samples = prior_samples + secondary_samples
            # random.shuffle(samples)
            if len(samples) != batch_size:
                warning(
                    f"Not enough elements in buffer: {len(prior_samples), len(secondary_samples)}"
                )

        # if len(self.secondary_buffer) < batch_size:
        #     raise ValueError(
        #         f"Can't sample batch of size {batch_size} from the buffer!\n"
        #         f"There are only {len(self.secondary_buffer)} elements in the buffer."
        #     )
        return samples

    def __len__(self) -> int:
        return len(self.prior_buffer) + len(self.secondary_buffer)

    def get_len_prior(self) -> int:
        return len(self.prior_buffer)

    def get_len_secondary(self) -> int:
        return len(self.secondary_buffer)
