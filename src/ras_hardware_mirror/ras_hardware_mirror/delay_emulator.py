"""Transport-independent deterministic source-stamped delay queue."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np


@dataclass(frozen=True)
class DelayedSample:
    packet_id: int
    source_time_s: float
    requested_arrival_time_s: float
    measurement: np.ndarray
    source_truth: np.ndarray
    dropped: bool


class DelayQueue:
    def __init__(self, delay_s: float, position_sigma_m: float = 0.0, velocity_sigma_mps: float = 0.0, dropout_probability: float = 0.0, seed: int = 0) -> None:
        if delay_s < 0.0 or not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("invalid delay configuration")
        self.delay_s = float(delay_s)
        self.position_sigma_m = float(position_sigma_m)
        self.velocity_sigma_mps = float(velocity_sigma_mps)
        self.dropout_probability = float(dropout_probability)
        self.seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._heap: list[tuple[float, int, DelayedSample]] = []
        self._next_id = 0

    def enqueue(self, source_time_s: float, state6: np.ndarray) -> DelayedSample:
        truth = np.asarray(state6, dtype=float).copy()
        if truth.shape != (6,) or not np.all(np.isfinite(truth)):
            raise ValueError("state must be a finite six-vector")
        noise = np.concatenate((self._rng.normal(0.0, self.position_sigma_m, 3), self._rng.normal(0.0, self.velocity_sigma_mps, 3)))
        sample = DelayedSample(
            packet_id=self._next_id,
            source_time_s=float(source_time_s),
            requested_arrival_time_s=float(source_time_s) + self.delay_s,
            measurement=truth + noise,
            source_truth=truth,
            dropped=bool(self._rng.random() < self.dropout_probability),
        )
        heapq.heappush(self._heap, (sample.requested_arrival_time_s, sample.packet_id, sample))
        self._next_id += 1
        return sample

    def pop_ready(self, now_s: float) -> list[DelayedSample]:
        ready: list[DelayedSample] = []
        while self._heap and self._heap[0][0] <= float(now_s) + 1e-12:
            ready.append(heapq.heappop(self._heap)[2])
        return ready

    @property
    def pending(self) -> int:
        return len(self._heap)
