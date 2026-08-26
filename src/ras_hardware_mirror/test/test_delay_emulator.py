import numpy as np

from ras_hardware_mirror.delay_emulator import DelayQueue


def test_source_time_preserved_and_release_ordered_after_delay():
    queue = DelayQueue(0.12, seed=4)
    samples = [queue.enqueue(t, np.arange(6, dtype=float) + index) for index, t in enumerate((10.0, 10.02, 10.04))]
    assert queue.pop_ready(10.1199) == []
    first = queue.pop_ready(10.1201)
    assert [item.packet_id for item in first] == [0]
    assert first[0].source_time_s == 10.0
    rest = queue.pop_ready(10.17)
    assert [item.packet_id for item in rest] == [1, 2]
    assert all(abs((item.requested_arrival_time_s - item.source_time_s) - 0.12) < 1e-12 for item in samples)


def test_same_seed_produces_identical_schedule():
    def make(seed):
        queue = DelayQueue(0.05, 0.1, 0.2, 0.25, seed)
        return [queue.enqueue(i * 0.02, np.zeros(6)) for i in range(30)]
    left, right = make(55), make(55)
    assert [item.dropped for item in left] == [item.dropped for item in right]
    assert all(np.array_equal(a.measurement, b.measurement) for a, b in zip(left, right))
