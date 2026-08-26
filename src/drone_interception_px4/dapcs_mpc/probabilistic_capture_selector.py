"""Soft/conservative probabilistic capture-opportunity approximation for M3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belief_rollout import BeliefHorizon
from .reachability_proxy import kinematic_reachable_distance


CHI2_3_90 = 6.251388631170325


@dataclass(frozen=True)
class ProbabilisticSelection:
    selected_candidate_index: int
    selected_node: int
    selected_time_s: float
    target_position: np.ndarray
    target_velocity: np.ndarray
    mode_margins_m: np.ndarray
    coverages: np.ndarray
    weighted_margins_m: np.ndarray
    confidence_radii_m: np.ndarray
    selected_confidence_radius_m: float
    selected_coverage: float
    target_rule: str


def select_probabilistic_capture(
    belief: BeliefHorizon,
    candidate_nodes: np.ndarray,
    candidate_times_s: np.ndarray,
    pursuer_position: np.ndarray,
    pursuer_velocity: np.ndarray,
    vmax: float,
    amax: float,
    capture_radius_m: float,
    epsilon: float = .10,
    required_coverage: float = .90,
    beta_margin: float = .20,
    beta_time: float = .05,
) -> ProbabilisticSelection:
    if not np.isclose(epsilon, .10):
        raise ValueError("the frozen v1 selector requires epsilon=0.10")
    node_indices = np.asarray(candidate_nodes, dtype=int) - 1
    mode_margins = np.zeros((3, len(node_indices)))
    confidence_radii = np.zeros_like(mode_margins)
    coverages = np.zeros(len(node_indices))
    weighted_margins = np.zeros(len(node_indices))
    for candidate, (node_index, candidate_time) in enumerate(zip(node_indices, candidate_times_s)):
        probabilities = belief.mode_probabilities[:, node_index]
        for mode in range(3):
            target = belief.means[mode, node_index, :3]
            covariance = belief.covariances[mode, node_index, :3, :3]
            confidence_radii[mode, candidate] = np.sqrt(
                CHI2_3_90 * max(0.0, float(np.linalg.eigvalsh(covariance).max()))
            )
            proxy = kinematic_reachable_distance(
                pursuer_position, pursuer_velocity, target, vmax, amax, candidate_time
            )
            mode_margins[mode, candidate] = (
                proxy + capture_radius_m - float(np.linalg.norm(target - pursuer_position))
                - confidence_radii[mode, candidate]
            )
        reachable_modes = mode_margins[:, candidate] >= 0.0
        coverages[candidate] = float(probabilities[reachable_modes].sum())
        weighted_margins[candidate] = float(
            np.dot(probabilities, mode_margins[:, candidate])
        )
    feasible = np.flatnonzero(coverages >= required_coverage)
    if len(feasible):
        selected = int(feasible[0])
    else:
        span = float(np.ptp(weighted_margins))
        normalized_margin = (
            (weighted_margins - weighted_margins.min()) / span
            if span > 1e-12 else np.zeros_like(weighted_margins)
        )
        normalized_time = np.asarray(candidate_times_s, dtype=float) / max(candidate_times_s)
        scores = coverages + beta_margin * normalized_margin - beta_time * normalized_time
        selected = int(np.argmax(scores))
    node_index = node_indices[selected]
    probabilities = belief.mode_probabilities[:, node_index]
    active = mode_margins[:, selected] >= 0.0
    if np.any(active):
        weights = probabilities * active
        weights /= weights.sum()
        rule = "nonnegative-margin modes"
    else:
        weights = probabilities / probabilities.sum()
        rule = "full mixture fallback"
    target_position = np.einsum("m,md->d", weights, belief.means[:, node_index, :3])
    target_velocity = np.einsum("m,md->d", weights, belief.means[:, node_index, 3:6])
    active_radii = confidence_radii[active, selected]
    selected_radius = float(active_radii.max()) if len(active_radii) else float(
        confidence_radii[:, selected].max()
    )
    return ProbabilisticSelection(
        selected, int(candidate_nodes[selected]), float(candidate_times_s[selected]),
        target_position, target_velocity, mode_margins, coverages, weighted_margins,
        confidence_radii, selected_radius, float(coverages[selected]), rule,
    )
