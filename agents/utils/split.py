from typing import Iterable


def get_chunks(sequence: Iterable, chunk_size: int):
    seq_len = len(sequence)
    if seq_len < chunk_size:
        return [sequence]
    chunks = []
    for start_idx in range(0, seq_len - chunk_size + 1):
        chunks.append(sequence[start_idx : start_idx + chunk_size])
    return chunks


if __name__ == "__main__":
    seq1 = [1, 2, 3]
    assert get_chunks(seq1, 10) == seq1
    assert get_chunks(seq1, 2) == [[1, 2], [2, 3]]
    assert get_chunks(seq1, 1) == [[1], [2], [3]]
