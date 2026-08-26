"""Causal, mode-preserving future rollout for the delayed IMM posterior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .delay_aware_imm import DelayAwareIMM, MODE_NAMES, STATE_DIM


@dataclass(frozen=True)
class BeliefHorizon:
    times_s: np.ndarray
    mode_probabilities: np.ndarray  # [M, N]
    means: np.ndarray  # [M, N, state_dim]
    covariances: np.ndarray  # [M, N, state_dim, state_dim]

    def mixture_means(self) -> np.ndarray:
        return np.einsum("mn,mnd->nd", self.mode_probabilities, self.means)


def rollout_belief(estimator: DelayAwareIMM, times_s: np.ndarray) -> BeliefHorizon:
    """Roll out only the estimator posterior; no future telemetry is consumed."""
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0 or np.any(np.diff(times) <= 0.0) or times[0] <= 0.0:
        raise ValueError("future times must be a strictly increasing positive vector")
    means = estimator.means.copy()
    covariances = estimator.covariances.copy()
    probabilities = estimator.probabilities.copy()
    stored_means = np.zeros((3, len(times), STATE_DIM))
    stored_covariances = np.zeros((3, len(times), STATE_DIM, STATE_DIM))
    stored_probabilities = np.zeros((3, len(times)))
    elapsed = 0.0
    transition_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for node, target_time in enumerate(times):
        remaining = float(target_time - elapsed)
        while remaining > 1e-12:
            dt = min(estimator.config.nominal_dt_s, remaining)
            transition = estimator.config.transition_matrix
            predicted_probabilities = probabilities @ transition
            predicted_probabilities = np.maximum(predicted_probabilities, 1e-15)
            mixing = probabilities[:, None] * transition / predicted_probabilities[None, :]
            mixed_means = mixing.T @ means
            deltas = means[:, None, :] - mixed_means[None, :, :]
            outer = np.einsum("sdi,sdj->sdij", deltas, deltas)
            mixed_covariances = np.einsum(
                "sd,sij->dij", mixing, covariances
            ) + np.einsum("sd,sdij->dij", mixing, outer)
            cache_key = round(dt, 12)
            if cache_key not in transition_cache:
                transitions = np.asarray([
                    estimator.state_transition(mode, dt, estimator.config) for mode in MODE_NAMES
                ])
                process = np.asarray([
                    estimator.process_covariance(mode, dt) for mode in MODE_NAMES
                ])
                transition_cache[cache_key] = (transitions, process)
            transitions, process = transition_cache[cache_key]
            next_means = np.einsum("mij,mj->mi", transitions, mixed_means)
            next_covariances = transitions @ mixed_covariances @ np.swapaxes(
                transitions, 1, 2
            ) + process
            # Prediction from PSD inputs is PSD analytically. Symmetrization is
            # sufficient here and avoids hundreds of 9x9 eigendecompositions
            # in every 50 Hz rollout; the delayed update retains its PSD floor.
            next_covariances = .5 * (
                next_covariances + np.swapaxes(next_covariances, 1, 2)
            )
            means, covariances = next_means, next_covariances
            probabilities = predicted_probabilities / predicted_probabilities.sum()
            remaining -= dt
        elapsed = float(target_time)
        stored_means[:, node] = means
        stored_covariances[:, node] = covariances
        stored_probabilities[:, node] = probabilities
    return BeliefHorizon(times, stored_probabilities, stored_means, stored_covariances)
