"""Mean-belief deterministic capture-opportunity selector for M2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belief_rollout import BeliefHorizon
from .reachability_proxy import kinematic_reachable_distance


@dataclass(frozen=True)
class DeterministicSelection:
    selected_candidate_index: int
    selected_node: int
    selected_time_s: float
    target_position: np.ndarray
    target_velocity: np.ndarray
    margins_m: np.ndarray
    reachable_distances_m: np.ndarray


def select_deterministic_capture(
    belief: BeliefHorizon,
    candidate_nodes: np.ndarray,
    candidate_times_s: np.ndarray,
    pursuer_position: np.ndarray,
    pursuer_velocity: np.ndarray,
    vmax: float,
    amax: float,
    capture_radius_m: float,
) -> DeterministicSelection:
    mixture = belief.mixture_means()
    indices = np.asarray(candidate_nodes, dtype=int) - 1
    margins, reachable = [], []
    for node_index, candidate_time in zip(indices, candidate_times_s):
        target = mixture[node_index, :3]
        proxy = kinematic_reachable_distance(
            pursuer_position, pursuer_velocity, target, vmax, amax, candidate_time
        )
        reachable.append(proxy)
        margins.append(
            proxy + capture_radius_m - float(np.linalg.norm(target - pursuer_position))
        )
    margins_array = np.asarray(margins)
    feasible = np.flatnonzero(margins_array >= 0.0)
    selected = int(feasible[0]) if len(feasible) else int(np.argmax(margins_array))
    node_index = indices[selected]
    return DeterministicSelection(
        selected, int(candidate_nodes[selected]), float(candidate_times_s[selected]),
        mixture[node_index, :3].copy(), mixture[node_index, 3:6].copy(),
        margins_array, np.asarray(reachable),
    )
