from overrides import overrides
import random
from typing import NamedTuple, List, Iterable


class AbstractReplayMemory:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition: NamedTuple):
        """
        Add transition to the replay memory.
        :param transition: transition performed by an agent, regularly: state, action, reward, next_state.
        """
        raise NotImplementedError("Push method should be implemented!")

    def push_batch(self, transitions: Iterable[NamedTuple]):
        for transition in transitions:
            self.push(transition)

    def sample(self, batch_size: int) -> List[NamedTuple]:
        raise NotImplementedError("Sample method should be implemented!")

    def __len__(self) -> int:
        return len(self.buffer)

    def is_empty(self) -> bool:
        return self.__len__() == 0

    @property
    def last_observation(self):
        return self.buffer[-1]

    # def


class ExperienceReplay(AbstractReplayMemory):
    """
    Simple Experience Replay memory.
    """

    def __init__(self, capacity: int = 100_000):
        super(ExperienceReplay, self).__init__(capacity=capacity)

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


class PrioritizedReplayMemory(AbstractReplayMemory):
    """
    Experience Replay memory that stores transitions with priorities and samples
    elements according to these priorities.
    """

    def __init__(self, capacity: int = 100_000):
        super(PrioritizedReplayMemory, self).__init__(capacity=capacity)

    @overrides
    def push(self, transition: NamedTuple):
        pass

    @overrides
    def sample(self, batch_size: int) -> List[NamedTuple]:
        pass


class BinaryPrioritizeReplayMemory(AbstractReplayMemory):
    def __init__(self, capacity: int = 100_000):
        super(BinaryPrioritizeReplayMemory, self).__init__(capacity=capacity)

    def push(self, transition: NamedTuple):
        pass

    def sample(self, batch_size: int) -> List[NamedTuple]:
        pass
