from logging import warning, info

from overrides import overrides
import random
from typing import NamedTuple, List

from agents.utils.types import Transition


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
        self.prior_buffer = []
        self.prior_position = 0
        self.prior_capacity = int(self.capacity * priority_fraction)
        self.secondary_buffer = []
        self.secondary_position = 0
        self.secondary_capacity = self.capacity - self.prior_capacity
        self.priority_fraction = priority_fraction

    def push(self, transition: NamedTuple, is_prior: bool = False):
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            self.prior_position = self._push(
                transition, self.prior_buffer, self.prior_position
            )
        else:
            self.secondary_position = self._push(
                transition, self.secondary_buffer, self.secondary_position
            )

    def _push(self, transition: Transition, buffer, position):
        if len(buffer) < self.prior_capacity:
            buffer.append(None)
        buffer[position] = transition
        position = (position + 1) % self.prior_capacity
        return position

    def sample(self, batch_size: int) -> List[NamedTuple]:
        if self.priority_fraction == 0.0:
            samples = random.sample(self.secondary_buffer, batch_size)
        else:
            prior_size = min(
                int(batch_size * self.priority_fraction), len(self.prior_buffer)
            )
            secondary_size = min(batch_size - prior_size, len(self.secondary_buffer))
            prior_samples = random.sample(self.prior_buffer, prior_size)
            secondary_samples = random.sample(self.secondary_buffer, secondary_size)
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


class SeqTernaryPrioritizeReplayMemory(AbstractReplayMemory):
    def __init__(
        self,
        capacity: int = 100_000,
        good_samples_fraction: float = 0.0,
        bad_samples_fraction: float = 0.0,
    ):
        super().__init__(capacity=capacity)
        self.good_samples_fraction = good_samples_fraction
        self.bad_samples_fraction = bad_samples_fraction
        self.good_seqs_buffer = []
        self.good_buffer_position = 0
        self.bad_seqs_buffer = []
        self.bad_buffer_position = 0
        self.neutral_seqs_buffer = []
        self.neutral_buffer_position = 0
        self.good_samples_capacity = int(self.capacity * self.good_samples_fraction)
        self.bad_samples_capacity = int(self.capacity * self.bad_samples_fraction)
        self.secondary_capacity = (
            self.capacity - self.good_samples_capacity - self.bad_samples_capacity
        )

    def push(self, transitions: List[Transition]):
        is_good = False
        is_bad = False
        for transition in transitions:
            if transition.reward > 0.5:
                is_good = True
            elif transition.reward < -0.5:
                is_bad = True
        if is_good:
            self.good_buffer_position = self._push(
                transitions=transitions,
                position=self.good_buffer_position,
                buffer=self.good_seqs_buffer,
                capacity=self.good_samples_capacity,
            )
        if is_bad:
            self.bad_buffer_position = self._push(
                transitions=transitions,
                position=self.bad_buffer_position,
                buffer=self.bad_seqs_buffer,
                capacity=self.bad_samples_capacity,
            )
        if not is_good and not is_bad:
            self.neutral_buffer_position = self._push(
                transitions=transitions,
                position=self.neutral_buffer_position,
                buffer=self.neutral_seqs_buffer,
                capacity=self.secondary_capacity,
            )

    def _push(self, transitions, buffer, position, capacity):
        if len(buffer) < capacity:
            buffer.append(None)
        buffer[position] = transitions
        return (position + 1) % capacity

    def sample(self, batch_size: int):
        if self.good_samples_fraction == 0.0:
            batch_size = min(batch_size, len(self.neutral_seqs_buffer))
            return random.sample(self.neutral_seqs_buffer, batch_size)
        else:
            good_size = min(
                int(self.good_samples_fraction * batch_size), len(self.good_seqs_buffer)
            )
            bad_size = min(
                int(self.bad_samples_fraction * batch_size), len(self.bad_seqs_buffer)
            )
            secondary_size = min(
                batch_size - bad_size - good_size, len(self.neutral_seqs_buffer)
            )
            pos_prior_samples = random.sample(self.good_seqs_buffer, good_size)
            neg_prior_samples = random.sample(self.bad_seqs_buffer, bad_size)
            secondary_samples = random.sample(self.neutral_seqs_buffer, secondary_size)
            samples = pos_prior_samples + neg_prior_samples + secondary_samples
            info(
                f"Good samples: {len(pos_prior_samples)}; bad samples: {len(neg_prior_samples)} rest samples: {len(secondary_samples)}"
            )
            return samples


class TernaryPrioritizeReplayMemory(AbstractReplayMemory):
    def __init__(self, capacity: int = 100_000, priority_fraction: float = 0.0):
        super(TernaryPrioritizeReplayMemory, self).__init__(capacity=capacity)
        self.priority_fraction = priority_fraction
        self.pos_prior_buffer = []
        self.pos_prior_position = 0
        self.neg_prior_buffer = []
        self.neg_prior_position = 0
        self.prior_capacity = int(self.capacity * self.priority_fraction) // 2
        self.secondary_buffer = []
        self.secondary_position = 0
        self.secondary_capacity = self.capacity - 2 * self.prior_capacity

    def push(self, transition: NamedTuple):
        if self.priority_fraction == 0.0:
            is_prior = False
        else:
            is_prior = transition.reward != 0.0
        if is_prior:
            self._push_prior(transition)
        else:
            self._push_secondary(transition)

    def sample(self, batch_size: int):
        if self.priority_fraction == 0.0:
            batch_size = min(batch_size, len(self.secondary_buffer))
            return random.sample(self.secondary_buffer, batch_size)
        else:
            prior_size = min(
                int(batch_size * self.priority_fraction) // 2,
                min(len(self.pos_prior_buffer), len(self.neg_prior_buffer)),
            )
            secondary_size = min(batch_size - prior_size, len(self.secondary_buffer))
            pos_prior_samples = random.sample(self.pos_prior_buffer, prior_size)
            neg_prior_samples = random.sample(self.neg_prior_buffer, prior_size)
            secondary_samples = random.sample(self.secondary_buffer, secondary_size)
            samples = pos_prior_samples + neg_prior_samples + secondary_samples
            return samples

    def _push_prior(self, transition: NamedTuple):
        if transition.reward > 0.0:
            if len(self.pos_prior_buffer) < self.prior_capacity:
                self.pos_prior_buffer.append(None)
            self.pos_prior_buffer[self.pos_prior_position] = transition
            self.pos_prior_position = (
                self.pos_prior_position + 1
            ) % self.prior_capacity
        else:
            if len(self.neg_prior_buffer) < self.prior_capacity:
                self.neg_prior_buffer.append(None)
            self.neg_prior_buffer[self.neg_prior_position] = transition
            self.neg_prior_position = (
                self.neg_prior_position + 1
            ) % self.prior_capacity

    def _push_secondary(self, transition: NamedTuple):
        if len(self.secondary_buffer) < self.secondary_capacity:
            self.secondary_buffer.append(None)
        self.secondary_buffer[self.secondary_position] = transition
        self.secondary_position = (
            self.secondary_position + 1
        ) % self.secondary_capacity

    def __len__(self) -> int:
        return (
            len(self.pos_prior_buffer)
            + len(self.neg_prior_buffer)
            + len(self.secondary_buffer)
        )

    def get_len_prior(self) -> int:
        return len(self.pos_prior_buffer) + len(self.neg_prior_buffer)

    def get_len_secondary(self) -> int:
        return len(self.secondary_buffer)
