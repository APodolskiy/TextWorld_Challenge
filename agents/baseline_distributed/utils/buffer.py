from typing import Any, Iterable


class CircularBuffer:
    """
    Implementation of circular buffer
    """
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.start, self.end = 0, 0
        self.full = False
        self.buffer = []

    def __len__(self) -> int:
        """
        Returns current buffer size
        :return: circular buffer size
        """
        if self.end > self.start:
            return self.end - self.start
        else:
            return self.max_size - (self.start - self.end)

    def push(self, x: Any) -> None:
        if self.full:
            if self.end == self.start:
                self.start = self.end + 1
            self.buffer[self.end] = x
            self.end = (self.end + 1) % self.max_size
        else:
            self.buffer.append(x)
            self.end += 1
            if self.end == self.max_size:
                self.end = 0
                self.full = True

    def extend(self, elems: Iterable[Any]) -> None:
        for elem in elems:
            self.push(elem)

    def pop(self, num_elems: int = 1) -> Any:
        if len(self) == 0:
            return []
        num_elems = min(num_elems, len(self))
        res = self.buffer[self.start:self.start + num_elems]
        if self.start + num_elems > self.max_size:
            res += self.buffer[:num_elems - self.max_size + self.start]
        self.start = (self.start + num_elems) % self.max_size
        return res

    def __str__(self):
        if self.end > self.start:
            return str(self.buffer[self.start:self.end])
        else:
            return str(self.buffer[self.start:] + self.buffer[:self.end])

    def __repr__(self):
        return str(self)


if __name__ == '__main__':
    buf = CircularBuffer(max_size=6)
    buf.extend(range(10))
    assert str(buf) == str([4, 5, 6, 7, 8, 9])
