#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Value-driven controller for mixed omnidirectional and directional sources.

Reuses the measurement, geometry, routing, and client behavior from problem3.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import problem3 as q3
except ModuleNotFoundError as exc:
    if exc.name != "problem3_fused_final":
        raise
    raise SystemExit(
        "缺少同目录 problem3_fused_final.py，无法复用问题 3 的接口与几何实现。"
    ) from exc


DEFERRED = "DEFERRED"
SEARCH_DEFERRED = "SEARCH_DEFERRED"


PARTICLE_COUNT = 1500
PARTICLE_REBUILD_FACTOR = 5
PARTICLE_SOFT_LIKELIHOOD = 1.0e-6
PARTICLE_RESAMPLE_ESS_RATIO = 0.42
INITIAL_PRESENCE_PRIOR = 13.0 / 20.0
INITIAL_DIRECTIONAL_PRIOR = 0.50
RANDOM_SEED = 20260911


BEARING_BIN_DEG = 4.0
DIRECTION_UNCERTAINTY_SCALE_M = 260.0
TYPE_UNCERTAINTY_SCALE_M = 120.0


OBJECTIVE_ALPHA = 0.86
AVERAGE_TIME_REFERENCE_S = 600.0
INFORMATION_REWARD = 0.28
DISCOVERY_REWARD = 0.90
BASE_PROGRESS_BONUS = 0.20
CLEAR_ACTION_BONUS = 0.80


MAX_DIRECTIONAL_CANDIDATES = 20
MAX_ACTIVE_MEASUREMENTS = 6
MAX_ACTIVE_AFTER_DISCOVERY = 6
EXACT_CHANNELS_PER_ROUND = 5
PLAN_TIME_BUDGET_S = 15.0
FAST_FINISH_REMAINING_S = 60.0
FINAL_EXIT_REMAINING_S = 20.0
MIN_MEASUREMENTS_BEFORE_DEFER = 5
MIN_INFORMATION_GAIN = 0.015
MIN_ACTIVE_VALUE_RATE = 2.5e-5
REACTIVATE_DISTANCE_M = 420.0
FOLLOWUP_ACTION_TIME_S = 90.0
MAX_OPPORTUNISTIC_AT_BASE = 2
ACTIVE_Q2_NEAR_OPT_REL = 0.16
SOFT_INSERTION_LIMIT_M = 1000.0
HARD_INSERTION_LIMIT_M = 1800.0
MIN_MEC_REDUCTION_RATIO = 0.005
MAX_STAGNANT_ACTIVE_STEPS = 4


MIN_REQUIRED_DISCOVERIES = 10
MAX_POSSIBLE_SOURCES = 16


DISCOVERY_COVERAGE_TARGET = 0.975
HIGH_RISK_COVERAGE_TARGET = 0.985
HIGH_RISK_PRESENCE_THRESHOLD = 0.10
EXPECTED_UNSEEN_STOP_THRESHOLD = 0.30
MIN_DISTINCT_BEARING_SECTORS = 4
BEARING_SECTOR_COUNT = 8


MAX_BACKBONE_PROBE_ACTIONS = 3
MAX_POST_BACKBONE_PROBE_ACTIONS = 8
MAX_EMERGENCY_PROBE_ACTIONS = 6
MAX_SUPPLEMENT_ACTIONS = (
    MAX_BACKBONE_PROBE_ACTIONS
    + MAX_POST_BACKBONE_PROBE_ACTIONS
    + MAX_EMERGENCY_PROBE_ACTIONS
)
BACKBONE_PROBE_INSERTION_LIMIT_M = 400.0
POST_PROBE_INSERTION_LIMIT_M = 2200.0
EMERGENCY_PROBE_INSERTION_LIMIT_M = 3600.0
MAX_CHANNELS_PER_SUPPLEMENT = 20
MIN_SUPPLEMENT_VALUE_RATE = 0.0


UNSEEN_PRESENCE_FLOOR = 0.06
SUPPLEMENT_SENSE_BUDGET_S = 125.0
SUPPLEMENT_TOTAL_BUDGET_S = 3600.0
SUPPLEMENT_DETOUR_BUDGET_M = 12000.0
SUPPLEMENT_CLEAR_PROBABILITY = 0.78


EMERGENCY_EXPECTED_UNSEEN_THRESHOLD = 0.45

DEFAULT_LOG = "problem4_robot_log.jsonl"


def circular_difference_deg(a: np.ndarray, b: float) -> np.ndarray:
    """Return the signed angular difference in [-180, 180)."""
    return (a - float(b) + 180.0) % 360.0 - 180.0


def binary_entropy(probability: float) -> float:
    p = min(1.0, max(0.0, float(probability)))
    if p <= 1.0e-12 or p >= 1.0 - 1.0e-12:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
    output_count: Optional[int] = None,
) -> np.ndarray:
    n = len(weights) if output_count is None else int(output_count)
    positions = (rng.random() + np.arange(n, dtype=float)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left")


def weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(points * weights[:, None], axis=0)


def weighted_rms_radius(points: np.ndarray, weights: np.ndarray) -> float:
    if len(points) == 0:
        return float("inf")
    center = weighted_mean(points, weights)
    squared = np.sum((points - center) ** 2, axis=1)
    return float(math.sqrt(max(0.0, float(np.dot(weights, squared)))))


def mixed_position_region(observations: list[dict[str, Any]]) -> Any:
    """Build the deterministic position envelope.

    A no-signal result updates particles only because a directional source may face away.
    """
    region = q3.initial_arena_region()
    for obs in observations:
        point = np.asarray(obs["position"], dtype=float)
        result = obs["result"]
        if result == "direction":
            region = q3.update_region_direction(region, point, float(obs["angle_deg"]))
        elif result == "near":


            near_disk = q3.disk(point, q3.NEAR_RADIUS + q3.ARENA_MODEL_MARGIN)
            region = q3.clean_geometry(region.intersection(near_disk))
        elif result == "clear_failure":
            region = q3.update_region_clear_failure(region, point)
        elif result == "no_signal":
            continue
        else:
            raise ValueError(f"未知历史观测类型：{result!r}")
        if region.is_empty:
            break
    return q3.clean_geometry(region)


def guarantees_clearance(state: q3.ChannelState, point: np.ndarray) -> bool:
    """Verify that every feasible source position is within the clearance threshold."""
    if state.region.is_empty:
        return False
    return (
        q3.max_distance_to_region(np.asarray(point, dtype=float), state.region)
        <= q3.CLEAR_MEC_TRIGGER + 1.0e-7
    )


@dataclass(frozen=True)
class BeliefPrediction:
    signal_probability: float
    information_gain: float
    spatial_gain: float
    expected_spatial_radius: float
    outcome_entropy: float


class MixedSourceBelief:
    """Particle posterior conditional on source presence, plus its presence probability."""

    def __init__(
        self,
        channel: int,
        rng: np.random.Generator,
        count: int = PARTICLE_COUNT,
    ) -> None:
        self.channel = int(channel)
        self.rng = rng
        self.count = int(count)
        self.present_probability = INITIAL_PRESENCE_PRIOR
        self.positions = np.empty((0, 2), dtype=float)
        self.ranges = np.empty(0, dtype=float)
        self.is_directional = np.empty(0, dtype=bool)
        self.directions = np.empty(0, dtype=float)
        self.weights = np.empty(0, dtype=float)
        self._initialize_prior()

    def _initialize_prior(self) -> None:
        radius = q3.TARGET_RADIUS * np.sqrt(self.rng.random(self.count))
        angle = self.rng.uniform(0.0, 2.0 * math.pi, self.count)
        self.positions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
        self._draw_latent(self.count)
        self.weights = np.full(self.count, 1.0 / self.count, dtype=float)

    def _draw_latent(self, count: int) -> None:
        self.ranges = self.rng.uniform(
            q3.SIGNAL_RADIUS_MIN, q3.SIGNAL_RADIUS_MAX, count
        )
        self.is_directional = self.rng.random(count) < INITIAL_DIRECTIONAL_PRIOR
        self.directions = self.rng.uniform(0.0, 2.0 * math.pi, count)

    def _signal_mask(self, station: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        station = np.asarray(station, dtype=float)
        delta = station[None, :] - self.positions
        distances = np.linalg.norm(delta, axis=1)
        front_dot = (
            delta[:, 0] * np.cos(self.directions)
            + delta[:, 1] * np.sin(self.directions)
        )
        angular_visible = (~self.is_directional) | (front_dot >= 0.0)
        return (distances <= self.ranges) & angular_visible, distances

    def _likelihood(self, observation: dict[str, Any]) -> np.ndarray:
        station = np.asarray(observation["position"], dtype=float)
        signal, distances = self._signal_mask(station)
        result = observation["result"]

        if result == "no_signal":
            valid = ~signal
        elif result == "near":
            valid = signal & (distances <= q3.NEAR_RADIUS + q3.ARENA_MODEL_MARGIN)
        elif result == "clear_failure":
            valid = distances > q3.CLEAR_FAILURE_REMOVE_RADIUS
        elif result == "direction":
            predicted = (
                np.degrees(
                    np.arctan2(
                        self.positions[:, 1] - station[1],
                        self.positions[:, 0] - station[0],
                    )
                )
                % 360.0
            )
            angular_ok = np.abs(
                circular_difference_deg(predicted, float(observation["angle_deg"]))
            ) <= q3.BEARING_ERROR_DEG + q3.ANGLE_MODEL_MARGIN_DEG
            valid = signal & (distances > q3.NEAR_RADIUS) & angular_ok
        else:
            raise ValueError(f"未知观测 {result!r}")

        return np.where(valid, 1.0, PARTICLE_SOFT_LIKELIHOOD)

    def _normalize_and_resample(self) -> None:
        total = float(np.sum(self.weights))
        if not math.isfinite(total) or total <= 0.0:
            self.weights = np.full(self.count, 1.0 / self.count, dtype=float)
            return
        self.weights /= total
        ess = 1.0 / max(float(np.dot(self.weights, self.weights)), 1.0e-15)
        if ess < PARTICLE_RESAMPLE_ESS_RATIO * self.count:
            indices = systematic_resample(self.weights, self.rng)
            self.positions = self.positions[indices].copy()
            self.ranges = self.ranges[indices].copy()
            self.is_directional = self.is_directional[indices].copy()
            self.directions = self.directions[indices].copy()
            self.weights = np.full(self.count, 1.0 / self.count, dtype=float)

    def _sample_positions_in_region(self, region: Any, wanted: int) -> np.ndarray:
        if region.is_empty:
            raise ValueError("不能从空位置区域重建粒子")
        min_x, min_y, max_x, max_y = region.bounds
        accepted: list[tuple[float, float]] = []
        attempts = 0
        maximum_attempts = max(50000, 220 * wanted)
        batch = max(2000, 5 * wanted)

        while len(accepted) < wanted and attempts < maximum_attempts:
            xs = self.rng.uniform(min_x, max_x, batch)
            ys = self.rng.uniform(min_y, max_y, batch)
            attempts += batch
            for x, y in zip(xs, ys):
                if region.covers(q3.Point(float(x), float(y))):
                    accepted.append((float(x), float(y)))
                    if len(accepted) >= wanted:
                        break

        if not accepted:
            representative = region.representative_point()
            accepted = [(float(representative.x), float(representative.y))]
        if len(accepted) < wanted:
            boundary = q3.sample_region_points(region, min(wanted, 240))
            if len(boundary):
                accepted.extend((float(x), float(y)) for x, y in boundary)
        values = np.asarray(accepted, dtype=float)
        if len(values) < wanted:
            indices = self.rng.integers(0, len(values), size=wanted - len(values))
            values = np.vstack((values, values[indices]))
        return values[:wanted]

    def rebuild(
        self,
        region: Any,
        observations: list[dict[str, Any]],
    ) -> None:
        """Resample inside the positive-observation envelope to avoid particle depletion."""
        candidate_count = max(self.count, PARTICLE_REBUILD_FACTOR * self.count)
        self.positions = self._sample_positions_in_region(region, candidate_count)
        self._draw_latent(candidate_count)
        self.weights = np.full(candidate_count, 1.0 / candidate_count, dtype=float)
        for obs in observations:
            self.weights *= self._likelihood(obs)
        total = float(np.sum(self.weights))
        if not math.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / candidate_count)
        else:
            self.weights /= total
        indices = systematic_resample(self.weights, self.rng, self.count)
        self.positions = self.positions[indices].copy()
        self.ranges = self.ranges[indices].copy()
        self.is_directional = self.is_directional[indices].copy()
        self.directions = self.directions[indices].copy()
        self.weights = np.full(self.count, 1.0 / self.count, dtype=float)

    def observe(
        self,
        observation: dict[str, Any],
        known_present: bool,
    ) -> None:
        likelihood = self._likelihood(observation)
        if observation["result"] == "no_signal" and not known_present:
            probability_no_signal_if_present = float(np.dot(self.weights, likelihood))
            prior = self.present_probability
            denominator = (1.0 - prior) + prior * probability_no_signal_if_present
            if denominator > 0.0:
                self.present_probability = (
                    prior * probability_no_signal_if_present / denominator
                )
        elif observation["result"] in ("direction", "near"):
            self.present_probability = 1.0

        self.weights *= likelihood
        self._normalize_and_resample()

    @property
    def directional_probability(self) -> float:
        return float(np.dot(self.weights, self.is_directional.astype(float)))

    @property
    def effective_sample_size(self) -> float:
        return 1.0 / max(float(np.dot(self.weights, self.weights)), 1.0e-15)

    def direction_estimate_deg(self) -> Optional[float]:
        mask = self.is_directional
        directional_mass = float(np.sum(self.weights[mask]))
        if directional_mass <= 1.0e-9:
            return None
        weights = self.weights[mask] / directional_mass
        z = np.dot(weights, np.exp(1j * self.directions[mask]))
        if abs(z) <= 1.0e-9:
            return None
        return float(math.degrees(math.atan2(z.imag, z.real)) % 360.0)

    def signal_probability(self, station: np.ndarray) -> float:
        signal, _ = self._signal_mask(np.asarray(station, dtype=float))
        return float(np.dot(self.weights, signal.astype(float)))

    def directional_signal_probability(self, station: np.ndarray) -> float:
        """Estimate detection probability conditional on a directional source.

        Mixing in omnidirectional particles would overstate directional coverage.
        """
        signal, _ = self._signal_mask(np.asarray(station, dtype=float))
        directional_mass = float(np.sum(self.weights[self.is_directional]))
        if directional_mass <= 1.0e-12:
            return 0.0
        detected_mass = float(
            np.sum(self.weights[self.is_directional & signal])
        )
        return min(1.0, max(0.0, detected_mass / directional_mass))

    def position_center(self) -> np.ndarray:
        return weighted_mean(self.positions, self.weights)

    def spatial_radius(self) -> float:
        return weighted_rms_radius(self.positions, self.weights)

    def _joint_uncertainty(self, indices: np.ndarray, weights: np.ndarray) -> float:
        conditional = weights / max(float(np.sum(weights)), 1.0e-15)
        spatial = weighted_rms_radius(self.positions[indices], conditional)
        is_dir = self.is_directional[indices]
        q_dir = float(np.dot(conditional, is_dir.astype(float)))
        type_term = TYPE_UNCERTAINTY_SCALE_M * binary_entropy(q_dir)
        direction_term = 0.0
        if q_dir > 1.0e-8 and np.any(is_dir):
            dweights = conditional[is_dir]
            dweights /= np.sum(dweights)
            resultant = abs(np.dot(dweights, np.exp(1j * self.directions[indices][is_dir])))
            circular_variance = max(0.0, 1.0 - float(resultant))
            direction_term = DIRECTION_UNCERTAINTY_SCALE_M * q_dir * circular_variance
        return spatial + type_term + direction_term

    def predict(self, station: np.ndarray) -> BeliefPrediction:
        station = np.asarray(station, dtype=float)
        signal, distances = self._signal_mask(station)
        bearings = (
            np.degrees(
                np.arctan2(
                    self.positions[:, 1] - station[1],
                    self.positions[:, 0] - station[0],
                )
            )
            % 360.0
        )
        labels = np.full(self.count, -1, dtype=int)
        labels[signal & (distances <= q3.NEAR_RADIUS)] = 1000
        directional_signal = signal & (distances > q3.NEAR_RADIUS)
        labels[directional_signal] = np.floor(
            bearings[directional_signal] / BEARING_BIN_DEG
        ).astype(int)

        all_indices = np.arange(self.count)
        prior_joint = self._joint_uncertainty(all_indices, self.weights)
        prior_spatial = self.spatial_radius()
        expected_joint = 0.0
        expected_spatial = 0.0
        outcome_entropy = 0.0
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            mass = float(np.sum(self.weights[indices]))
            if mass <= 0.0:
                continue
            outcome_entropy -= mass * math.log(mass)
            expected_joint += mass * self._joint_uncertainty(indices, self.weights[indices])
            conditional = self.weights[indices] / mass
            expected_spatial += mass * weighted_rms_radius(
                self.positions[indices], conditional
            )

        information = max(0.0, 1.0 - expected_joint / max(prior_joint, 1.0e-9))
        spatial_gain = max(0.0, 1.0 - expected_spatial / max(prior_spatial, 1.0e-9))
        return BeliefPrediction(
            signal_probability=float(np.dot(self.weights, signal.astype(float))),
            information_gain=min(1.0, information),
            spatial_gain=min(1.0, spatial_gain),
            expected_spatial_radius=expected_spatial,
            outcome_entropy=outcome_entropy,
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    channel: int
    point: np.ndarray
    prediction: BeliefPrediction
    action_time_s: float
    expected_clear_probability: float
    expected_remaining_cost_s: float
    delta_utility: float
    value_per_second: float
    insertion_distance_m: float


@dataclass(frozen=True)
class P4Action:
    kind: str
    point: np.ndarray
    score: float
    channel: Optional[int] = None
    channels: tuple[int, ...] = ()
    base_id: Optional[int] = None
    note: str = ""
    insertion_distance_m: float = 0.0
    estimated_time_s: float = 0.0
    probe_phase: str = ""


def as_q3_route_nodes(actions: list[P4Action]) -> list[q3.RouteNode]:
    return [
        q3.RouteNode(
            kind=action.kind,
            point=np.asarray(action.point, dtype=float),
            base_id=action.base_id,
            channel=action.channel,
        )
        for action in actions
    ]


def route_insertion(
    current: np.ndarray,
    route: list[P4Action],
    point: np.ndarray,
) -> tuple[float, int]:
    return q3.best_insertion(
        np.asarray(current, dtype=float),
        as_q3_route_nodes(route),
        np.asarray(point, dtype=float),
    )


def insert_actions_by_minimum_increment(
    current: np.ndarray,
    base_route: list[P4Action],
    tasks: list[P4Action],
) -> list[P4Action]:
    route = list(base_route)
    pending = list(tasks)
    priority = {"CLEAR": 0, "ACTIVE": 1, "PROBE": 2, "BASE": 3}
    while pending:
        best: Optional[tuple[tuple[float, int, float], int, int]] = None
        for task_index, task in enumerate(pending):
            extra, insert_at = route_insertion(current, route, task.point)
            key = (extra, priority.get(task.kind, 9), -task.score)
            if best is None or key < best[0]:
                best = (key, task_index, insert_at)
        assert best is not None
        _, task_index, insert_at = best
        route.insert(insert_at, pending.pop(task_index))
    return route


def expected_delta_utility(
    cleared: int,
    elapsed: float,
    estimated_total: float,
    success_probability: float,
    added_cost: float,
) -> float:
    """Estimate the marginal utility of continuing through clearance."""
    alpha = OBJECTIVE_ALPHA
    beta = 1.0 - alpha
    n = int(cleared)
    p = min(1.0, max(0.0, float(success_probability)))
    cost = max(0.0, float(added_cost))
    total = max(1.0, float(estimated_total))
    if n <= 0:
        return alpha * p / total - beta * cost / AVERAGE_TIME_REFERENCE_S
    average = elapsed / n
    average_term = p * (cost - average) / (n + 1.0) + (1.0 - p) * cost / n
    return alpha * p / total - beta * average_term / AVERAGE_TIME_REFERENCE_S


def estimate_followup(
    state: q3.ChannelState,
    prediction: BeliefPrediction,
    first_action_time: float,
) -> tuple[float, float]:
    current_radius = max(state.mec_radius, q3.CLEAR_RADIUS + 1.0e-6)
    combined_gain = max(
        0.04,
        min(0.85, 0.62 * prediction.spatial_gain + 0.38 * prediction.information_gain),
    )
    if current_radius <= q3.CLEAR_MEC_TRIGGER:
        steps = 0
    else:
        ratio = q3.CLEAR_MEC_TRIGGER / current_radius
        steps = int(
            math.ceil(math.log(max(ratio, 1.0e-6)) / math.log(1.0 - combined_gain))
        )
        steps = min(MAX_ACTIVE_MEASUREMENTS, max(1, steps))


    p_vis = prediction.signal_probability
    eventual = 1.0 - (1.0 - p_vis) ** max(1, steps)
    information_rescue = (1.0 - eventual) * prediction.information_gain
    p_clear = min(0.995, max(0.02, eventual + 0.55 * information_rescue))
    remaining_cost = (
        first_action_time
        + max(0, steps - 1) * FOLLOWUP_ACTION_TIME_S
        + 5.0
    )
    return p_clear, remaining_cost


class Problem4Controller(q3.Problem3Controller):
    def __init__(
        self,
        client: q3.RobotClient,
        outer_count: int,
        outer_radius: float,
        rotation_deg: float,
        seed: int = RANDOM_SEED,
    ) -> None:
        super().__init__(client, outer_count, outer_radius, rotation_deg)
        root_rng = np.random.default_rng(seed)
        self.beliefs = {
            channel: MixedSourceBelief(
                channel, np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
            )
            for channel in self.channels
        }
        self.supplement_points = self._make_supplement_points(outer_radius)
        self.emergency_points = self._make_emergency_points()
        self.supplement_actions_done = 0
        self.backbone_probe_actions_done = 0
        self.post_backbone_probe_actions_done = 0
        self.emergency_probe_actions_done = 0
        self.supplement_virtual_spent_s = 0.0
        self.supplement_detour_spent_m = 0.0


        self.discovery_survival_probability = {
            channel: 1.0 for channel in self.channels
        }
        self.exploration_exhausted = False
        self.deferred_reasons: dict[int, str] = {}
        self.active_attempts = {channel: 0 for channel in self.channels}
        self.stagnant_active_steps = {channel: 0 for channel in self.channels}
        self.last_active_mec_radius = {
            channel: float("inf") for channel in self.channels
        }
        self.exact_age = {channel: 0 for channel in self.channels}
        self.reactivation_count = {channel: 0 for channel in self.channels}
        self.plan_durations_s: list[float] = []
        self.fast_finish_mode = False

    @staticmethod
    def _make_supplement_points(outer_radius: float) -> list[np.ndarray]:
        """Build inner, middle, and outer probe candidates for adaptive selection."""
        points: list[np.ndarray] = []
        inner_radius = min(650.0, 0.58 * float(outer_radius))
        middle_radius = float(outer_radius)
        outer_ring_radius = q3.TARGET_RADIUS
        for radius, count, offset in (
            (inner_radius, 6, 30.0),
            (middle_radius, 4, 45.0),
            (outer_ring_radius, 6, 0.0),
        ):
            for index in range(count):
                theta = math.radians(offset + 360.0 * index / count)
                points.append(np.array([
                    radius * math.cos(theta),
                    radius * math.sin(theta),
                ]))
        return q3.deduplicate_points(points, tol=1.0e-5)

    @staticmethod
    def _make_emergency_points() -> list[np.ndarray]:
        points: list[np.ndarray] = []
        for index in range(12):
            theta = math.radians(15.0 + 30.0 * index)
            points.append(np.array([
                q3.TARGET_RADIUS * math.cos(theta),
                q3.TARGET_RADIUS * math.sin(theta),
            ]))
        return points

    def _backbone_supplement_candidates(
        self,
        route: list[P4Action],
    ) -> list[np.ndarray]:
        base_points = [
            np.asarray(action.point, dtype=float)
            for action in route
            if action.kind == "BASE"
        ]
        if not base_points:
            return []

        chain = [np.asarray(self.client.position, dtype=float)] + base_points
        candidates: list[np.ndarray] = []

        lateral_offset = 220.0
        for start, end in zip(chain[:-1], chain[1:]):
            segment = end - start
            length = float(np.linalg.norm(segment))
            if length < 320.0:
                continue
            midpoint = 0.5 * (start + end)
            normal = np.array([-segment[1], segment[0]], dtype=float) / length
            for sign in (-1.0, 1.0):
                point = midpoint + sign * lateral_offset * normal
                norm = float(np.linalg.norm(point))
                if norm > 2100.0:
                    point = point * (2100.0 / norm)
                candidates.append(point)

        candidates.extend(self.supplement_points)
        return q3.deduplicate_points(candidates, tol=1.0e-5)

    def _opposite_probe_candidates(self, channels: list[int]) -> list[np.ndarray]:
        """Generate opposite-side probes after repeated no-signal observations."""
        points: list[np.ndarray] = []
        for channel in channels:
            state = self.channels[channel]
            last_no_signal: Optional[np.ndarray] = None
            for obs in reversed(state.observations):
                if obs.get("result") == "no_signal":
                    last_no_signal = np.asarray(obs["position"], dtype=float)
                    break
            if last_no_signal is None:
                continue
            center = self.beliefs[channel].position_center()
            point = 2.0 * center - last_no_signal
            radius = float(np.linalg.norm(point))
            if radius > 2100.0:
                point = point * (2100.0 / radius)
            points.append(point)
        return q3.deduplicate_points(points, tol=1.0e-5)

    def _post_supplement_candidates(self, pending: list[int]) -> list[np.ndarray]:
        points = list(self.supplement_points)
        points.extend(self._opposite_probe_candidates(pending))
        return q3.deduplicate_points(points, tol=1.0e-5)

    def _emergency_supplement_candidates(self, pending: list[int]) -> list[np.ndarray]:
        points = self._make_emergency_points()
        points.extend(self._opposite_probe_candidates(pending))
        return q3.deduplicate_points(points, tol=1.0e-5)


    def estimated_total(self) -> float:
        discovered = float(self.discovered_count())
        unseen = sum(
            self.beliefs[ch].present_probability
            for ch, state in self.channels.items()
            if state.status in (q3.SEARCHING, SEARCH_DEFERRED)
        )
        return min(16.0, max(10.0, discovered + unseen, float(self.cleared_count)))

    def deferred_channels(self) -> list[int]:
        return [ch for ch, state in self.channels.items() if state.status == DEFERRED]

    def search_deferred_channels(self) -> list[int]:
        return [
            ch for ch, state in self.channels.items()
            if state.status == SEARCH_DEFERRED
        ]

    def discovery_coverage(self, channel: int) -> float:
        return 1.0 - self.discovery_survival_probability[channel]

    def corrected_presence_probabilities(self) -> dict[int, float]:
        """Adjust unseen-channel probabilities to respect the allowed source-count range.

        The raw particle probabilities remain unchanged.
        """
        unseen = self._unseen_channels()
        corrected = {ch: float(self.beliefs[ch].present_probability) for ch in unseen}
        required = max(0.0, float(MIN_REQUIRED_DISCOVERIES - self.discovered_count()))
        current = sum(corrected.values())
        if required <= current + 1.0e-12 or not unseen:
            return corrected

        lo, hi = 0.0, 1.0
        for _ in range(60):
            tau = 0.5 * (lo + hi)
            total = sum(max(corrected[ch], tau) for ch in unseen)
            if total < required:
                lo = tau
            else:
                hi = tau
        tau = hi
        return {ch: max(corrected[ch], tau) for ch in unseen}

    def expected_unseen_sources(self) -> float:
        return float(sum(self.corrected_presence_probabilities().values()))

    def _bearing_sector_count(self, channel: int) -> int:
        state = self.channels[channel]
        if not state.measured_positions:
            return 0
        center = self.beliefs[channel].position_center()
        occupied: set[int] = set()
        for raw in state.measured_positions:
            point = np.asarray(raw, dtype=float)
            delta = point - center
            if float(np.linalg.norm(delta)) < 1.0e-6:
                continue
            angle = (math.degrees(math.atan2(delta[1], delta[0])) + 360.0) % 360.0
            sector = int(math.floor(angle / (360.0 / BEARING_SECTOR_COUNT)))
            occupied.add(min(BEARING_SECTOR_COUNT - 1, sector))
        return len(occupied)

    def _required_coverage(self, channel: int, corrected_presence: float) -> float:
        if corrected_presence >= HIGH_RISK_PRESENCE_THRESHOLD:
            return HIGH_RISK_COVERAGE_TARGET
        return DISCOVERY_COVERAGE_TARGET

    def _channel_search_satisfied(
        self, channel: int, corrected_presence: float
    ) -> bool:
        coverage_ok = (
            self.discovery_coverage(channel)
            >= self._required_coverage(channel, corrected_presence) - 1.0e-9
        )


        sectors_ok = (
            corrected_presence < 0.04
            or self._bearing_sector_count(channel) >= MIN_DISTINCT_BEARING_SECTORS
        )
        return coverage_ok and sectors_ok

    def _record_discovery_outcome(
        self,
        channel: int,
        position: np.ndarray,
        result: str,
    ) -> None:
        """Record cumulative detection coverage using sequential conditional probabilities."""
        probability = self.beliefs[channel].directional_signal_probability(position)
        if result == "no_signal":
            survival = self.discovery_survival_probability[channel]
            self.discovery_survival_probability[channel] = min(
                1.0, max(0.0, survival * (1.0 - probability))
            )
        elif result in ("direction", "near"):
            self.discovery_survival_probability[channel] = 0.0
        print(
            f"[coverage-dir] ch={channel:02d} conditional_p={probability:.3f} "
            f"cumulative={self.discovery_coverage(channel):.3f}"
        )

    def done(self) -> bool:
        if self.cleared_count >= 16:
            return True
        if self.remaining_base_ids:
            return False
        if self.clearable_channels():
            return False
        if self.localizing_channels():
            return False


        if not self.exploration_exhausted and self._unseen_channels():
            return False
        return True

    def mark_search_complete_by_upper_bound(self) -> None:
        if self.discovered_count() >= 16:
            self.remaining_base_ids.clear()
            self.exploration_exhausted = True
            for state in self.channels.values():
                if state.status in (q3.SEARCHING, SEARCH_DEFERRED):
                    state.status = q3.ABSENT


    def _rebuild_region(self, state: q3.ChannelState, reason: str) -> None:
        self._set_region(state, mixed_position_region(state.observations), reason)

    def _latest_observation(self, state: q3.ChannelState) -> dict[str, Any]:
        if not state.observations:
            raise RuntimeError("缺少刚记录的观测")
        return state.observations[-1]

    def _update_belief_after_measurement(
        self,
        state: q3.ChannelState,
        positive: bool,
    ) -> None:
        belief = self.beliefs[state.channel]
        if positive:
            belief.present_probability = 1.0
            latest = self._latest_observation(state)
            positive_count = sum(
                obs["result"] in ("direction", "near")
                for obs in state.observations
            )
            if latest["result"] == "near":

                pass
            elif positive_count == 1:

                belief.rebuild(state.region, state.observations)
            else:

                belief.observe(latest, known_present=True)
        else:
            belief.observe(self._latest_observation(state), known_present=state.ever_detected)
        print(
            f"[belief] ch={state.channel:02d} present={belief.present_probability:.3f} "
            f"directional={belief.directional_probability:.3f} "
            f"phi={belief.direction_estimate_deg()}"
        )

    def handle_measure_response(
        self,
        channel: int,
        position: np.ndarray,
        response: dict[str, Any],
    ) -> None:
        state = self.channels[channel]
        result = response["measure_result"]
        was_undiscovered = not state.ever_detected
        if was_undiscovered and result in ("no_signal", "direction", "near"):
            self._record_discovery_outcome(channel, position, result)

        if result == "no_signal":
            self._record_measurement(state, position, "no_signal")

            state.region_version += 1
            state.invalidate_plan_cache()
            self._update_belief_after_measurement(state, positive=False)

        elif result == "direction":
            angle = float(response["svd_deg"])
            self._record_measurement(state, position, "direction", angle)
            self._mark_detected(state)
            self._rebuild_region(state, "direction")
            self._update_belief_after_measurement(state, positive=True)

        elif result == "near":
            self._record_measurement(state, position, "near")
            self._mark_detected(state)
            self._rebuild_region(state, "near")
            self._update_belief_after_measurement(state, positive=True)
            clear_response = self.client.clear(position, channel)
            self.handle_clear_response(channel, position, clear_response)

        else:
            raise RuntimeError(f"未知 measure_result={result!r}")

        self.mark_search_complete_by_upper_bound()

    def handle_clear_response(
        self,
        channel: int,
        position: np.ndarray,
        response: dict[str, Any],
    ) -> None:
        state = self.channels[channel]
        result = response["clear_result"]

        if result == "success":
            if state.status != q3.CLEARED:
                self.cleared_count += 1
            state.status = q3.CLEARED
            state.ever_detected = True
            self.discovered_channels.add(channel)
            print(f"[state] ch={channel:02d} CLEARED ({self.cleared_count} total)")

        elif result == "no_target_in_range":

            state.observations.append({
                "position": [float(position[0]), float(position[1])],
                "result": "clear_failure",
            })
            self._rebuild_region(state, "clear_failure")
            state.status = q3.LOCALIZING
            self.beliefs[channel].rebuild(state.region, state.observations)
            print(f"[state] ch={channel:02d} clear failed -> LOCALIZING")

        else:
            raise RuntimeError(f"未知 clear_result={result!r}")

        self.mark_search_complete_by_upper_bound()


    def _cached_q2_options(self, state: q3.ChannelState) -> list[q3.ActiveOption]:
        if state.active_cache_version == state.region_version:
            return state.active_cache
        try:
            options = q3.compute_active_options(
                region=state.region,
                observations=state.observations,
                measured_positions=state.measured_positions,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"[q2-p4] ch={state.channel:02d} no option: {exc}")
            options = []
        state.active_cache_version = state.region_version
        state.active_cache = options
        return options

    def _directional_candidates(self, channel: int) -> list[np.ndarray]:
        state = self.channels[channel]
        belief = self.beliefs[channel]
        points: list[np.ndarray] = []
        protected_q2: list[np.ndarray] = []


        options = self._cached_q2_options(state)
        if options:
            threshold = options[0].score * (1.0 + ACTIVE_Q2_NEAR_OPT_REL) + 1.0e-9
            near_optimal = [option for option in options if option.score <= threshold]
            protected_q2 = [option.point for option in near_optimal[:8]]
            points.extend(protected_q2)

        center = (
            state.mec_center.copy()
            if state.mec_center is not None
            else belief.position_center()
        )
        points.append(center)

        phi = belief.direction_estimate_deg()
        angles: list[float]
        if phi is None or belief.directional_probability < 0.58:
            angles = [float(v) for v in range(0, 360, 45)]
        else:

            angles = [phi + value for value in (0, -75, 75, 150, 180, 210)]
        for radius in (320.0, 620.0, 900.0):
            for angle in angles:
                theta = math.radians(angle)
                points.append(center + radius * np.array([math.cos(theta), math.sin(theta)]))

        points.extend(self.base_points[index] for index in self.remaining_base_ids)
        points.append(self.client.position.copy())
        unique = q3.deduplicate_points(points, tol=1.0e-4)
        unique = [
            point for point in unique
            if not q3.measured_before(state.measured_positions, point)
            and np.all(np.abs(point) <= 2_000_000.0)
        ]
        protected_q2 = [
            point for point in q3.deduplicate_points(protected_q2, tol=1.0e-4)
            if not q3.measured_before(state.measured_positions, point)
        ]
        other = [
            point for point in unique
            if not any(q3.dist(point, q2_point) < 1.0e-4 for q2_point in protected_q2)
        ]
        other.sort(key=lambda point: q3.dist(self.client.position, point))
        return (protected_q2 + other)[:MAX_DIRECTIONAL_CANDIDATES]

    def _evaluate_candidate(
        self,
        channel: int,
        point: np.ndarray,
        route: list[P4Action],
    ) -> CandidateEvaluation:
        state = self.channels[channel]
        belief = self.beliefs[channel]
        prediction = belief.predict(point)
        switch = 0.0 if self.client.current_channel == channel else 1.0
        insertion_distance, _ = route_insertion(
            self.client.position, route, point
        )
        action_time = insertion_distance / q3.DOG_SPEED + 5.0 + switch
        p_clear, remaining = estimate_followup(state, prediction, action_time)
        delta = expected_delta_utility(
            self.cleared_count,
            self.client.virtual_time,
            self.estimated_total(),
            p_clear,
            remaining,
        )
        information_value = INFORMATION_REWARD * (
            0.55 * prediction.information_gain + 0.45 * prediction.spatial_gain
        )
        value = max(0.0, delta) + information_value
        return CandidateEvaluation(
            channel=channel,
            point=np.asarray(point, dtype=float).copy(),
            prediction=prediction,
            action_time_s=action_time,
            expected_clear_probability=p_clear,
            expected_remaining_cost_s=remaining,
            delta_utility=delta,
            value_per_second=value / max(action_time, 1.0e-9),
            insertion_distance_m=insertion_distance,
        )

    def _best_candidate(
        self,
        channel: int,
        route: list[P4Action],
    ) -> Optional[CandidateEvaluation]:
        candidates = self._directional_candidates(channel)
        if not candidates:
            return None
        evaluations = [
            self._evaluate_candidate(channel, point, route)
            for point in candidates
        ]
        return max(
            evaluations,
            key=lambda item: (
                item.value_per_second,
                item.prediction.information_gain,
                item.prediction.signal_probability,
                -item.action_time_s,
            ),
        )

    def _measurement_count(self, channel: int) -> int:
        return sum(
            obs["result"] in ("direction", "no_signal", "near")
            for obs in self.channels[channel].observations
        )

    def _channel_center(self, channel: int) -> np.ndarray:
        state = self.channels[channel]
        if state.mec_center is not None:
            return state.mec_center.copy()
        return self.beliefs[channel].position_center()

    def _cheap_channel_priority(
        self,
        channel: int,
        route: list[P4Action],
    ) -> float:
        state = self.channels[channel]
        center = self._channel_center(channel)
        detour, _ = route_insertion(self.client.position, route, center)
        radius_term = min(1.0, max(0.0, state.mec_radius) / q3.TARGET_RADIUS)
        uncertainty = self.beliefs[channel].directional_probability + radius_term
        age_bonus = 0.10 * min(self.exact_age[channel], 6)
        return (uncertainty + age_bonus) / (1.0 + detour / 1000.0)

    def _set_deferred(self, channel: int, reason: str) -> None:
        self.channels[channel].status = DEFERRED
        self.deferred_reasons[channel] = reason
        print(f"[defer] ch={channel:02d} {reason}")

    def _defer_or_reactivate(
        self,
        route: list[P4Action],
        planning_deadline: float,
    ) -> dict[int, CandidateEvaluation]:
        best: dict[int, CandidateEvaluation] = {}


        for channel in self.deferred_channels():
            if time.monotonic() >= planning_deadline:
                break
            state = self.channels[channel]
            if not state.ever_detected or self.reactivation_count[channel] >= 1:
                continue
            center_detour, _ = route_insertion(
                self.client.position, route, self._channel_center(channel)
            )
            if center_detour > REACTIVATE_DISTANCE_M:
                continue
            evaluation = self._best_candidate(channel, route)
            if evaluation is None:
                continue
            if (
                evaluation.delta_utility > 0.0
                and evaluation.insertion_distance_m <= REACTIVATE_DISTANCE_M
            ):
                state.status = q3.LOCALIZING
                self.reactivation_count[channel] += 1
                self.active_attempts[channel] = min(
                    self.active_attempts[channel], MAX_ACTIVE_AFTER_DISCOVERY - 1
                )
                self.stagnant_active_steps[channel] = 0
                self.deferred_reasons.pop(channel, None)
                best[channel] = evaluation
                print(
                    f"[reactivate] ch={channel:02d} "
                    f"deltaU={evaluation.delta_utility:.5f} "
                    f"detour={evaluation.insertion_distance_m:.1f}m"
                )

        localizing = list(self.localizing_channels())
        for channel in localizing:
            self.exact_age[channel] += 1
            if self.active_attempts[channel] >= MAX_ACTIVE_AFTER_DISCOVERY:
                self._set_deferred(channel, "active_measurement_budget_exhausted")

        eligible = [
            channel for channel in localizing
            if self.channels[channel].status == q3.LOCALIZING
        ]
        eligible.sort(
            key=lambda channel: self._cheap_channel_priority(channel, route),
            reverse=True,
        )

        for channel in eligible[:EXACT_CHANNELS_PER_ROUND]:
            if time.monotonic() >= planning_deadline:
                print("[plan-budget] stop exact evaluation at soft deadline")
                break
            self.exact_age[channel] = 0
            evaluation = self._best_candidate(channel, route)
            if evaluation is None:
                if self._measurement_count(channel) >= MIN_MEASUREMENTS_BEFORE_DEFER:
                    self._set_deferred(channel, "no_candidate")
                continue

            hard_detour = (
                self.cleared_count >= 14
                and evaluation.insertion_distance_m > HARD_INSERTION_LIMIT_M
            )
            if hard_detour:
                self._set_deferred(
                    channel,
                    f"hard_detour={evaluation.insertion_distance_m:.1f}m",
                )
                continue

            enough_trials = self._measurement_count(channel) >= MIN_MEASUREMENTS_BEFORE_DEFER
            low_information = (
                evaluation.prediction.information_gain < MIN_INFORMATION_GAIN
                and evaluation.prediction.spatial_gain < MIN_INFORMATION_GAIN
            )
            soft_stop = (
                self.cleared_count >= 12
                and enough_trials
                and evaluation.delta_utility < 0.0
                and evaluation.insertion_distance_m > SOFT_INSERTION_LIMIT_M
                and (
                    low_information
                    or evaluation.value_per_second < MIN_ACTIVE_VALUE_RATE
                )
            )
            if soft_stop:
                self._set_deferred(
                    channel,
                    f"deltaU={evaluation.delta_utility:.6f}, "
                    f"detour={evaluation.insertion_distance_m:.1f}m",
                )
                continue
            best[channel] = evaluation
        return best


    def _base_route_actions(self) -> list[P4Action]:
        route_ids = self._remaining_base_route()
        unseen = self.searching_channels()
        actions: list[P4Action] = []
        previous = self.client.position
        for base_id in route_ids:
            point = self.base_points[base_id]
            expected_discoveries = sum(
                self.beliefs[ch].present_probability
                * self.beliefs[ch].signal_probability(point)
                for ch in unseen
            )
            travel = q3.dist(previous, point) / q3.DOG_SPEED
            service = 5.0 * len(unseen) + max(0, len(unseen) - 1)
            score = (
                DISCOVERY_REWARD * expected_discoveries + BASE_PROGRESS_BONUS
            ) / max(travel + service, 1.0)
            actions.append(P4Action(
                kind="BASE", point=point.copy(), base_id=base_id, score=score,
                channels=tuple(unseen), note=f"E[new]={expected_discoveries:.3f}",
            ))
            previous = point
        return actions

    def _clear_actions(self, route: list[P4Action]) -> list[P4Action]:
        actions: list[P4Action] = []
        for channel in self.clearable_channels():
            state = self.channels[channel]
            state.recompute_geometry()
            if state.mec_center is None or not guarantees_clearance(state, state.mec_center):
                continue
            insertion_distance, _ = route_insertion(
                self.client.position, route, state.mec_center
            )
            cost = insertion_distance / q3.DOG_SPEED + 5.0
            delta = expected_delta_utility(
                self.cleared_count,
                self.client.virtual_time,
                self.estimated_total(),
                1.0,
                cost,
            )
            score = (CLEAR_ACTION_BONUS + max(0.0, delta)) / max(cost, 1.0)
            actions.append(P4Action(
                kind="CLEAR", point=state.mec_center.copy(), channel=channel,
                score=score,
                note=(
                    f"certified MEC={state.mec_radius:.3f}m, "
                    f"detour={insertion_distance:.1f}m"
                ),
                insertion_distance_m=insertion_distance,
                estimated_time_s=cost,
            ))
        return actions

    def _active_actions(
        self,
        best: dict[int, CandidateEvaluation],
    ) -> list[P4Action]:
        actions: list[P4Action] = []
        for channel, evaluation in best.items():
            if self.channels[channel].status != q3.LOCALIZING:
                continue
            actions.append(P4Action(
                kind="ACTIVE",
                point=evaluation.point,
                channel=channel,
                channels=(channel,),
                score=evaluation.value_per_second,
                note=(
                    f"pvis={evaluation.prediction.signal_probability:.3f}, "
                    f"IG={evaluation.prediction.information_gain:.3f}, "
                    f"deltaU={evaluation.delta_utility:.5f}, "
                    f"detour={evaluation.insertion_distance_m:.1f}m"
                ),
                insertion_distance_m=evaluation.insertion_distance_m,
                estimated_time_s=evaluation.action_time_s,
            ))
        return actions

    def _unseen_channels(self) -> list[int]:
        return [
            channel for channel, state in self.channels.items()
            if state.status in (q3.SEARCHING, SEARCH_DEFERRED)
        ]

    def _coverage_pending_channels(self) -> list[int]:
        unseen = self._unseen_channels()
        if not unseen:
            return []
        corrected = self.corrected_presence_probabilities()


        if self.discovered_count() < MIN_REQUIRED_DISCOVERIES:
            return unseen

        expected_unseen = sum(corrected.values())
        pending = [
            channel for channel in unseen
            if not self._channel_search_satisfied(channel, corrected[channel])
        ]


        if expected_unseen >= EXPECTED_UNSEEN_STOP_THRESHOLD and not pending:
            ranked = sorted(unseen, key=lambda ch: corrected[ch], reverse=True)
            running = 0.0
            for channel in ranked:
                if corrected[channel] <= 0.01:
                    continue
                pending.append(channel)
                running += corrected[channel]
                if running >= min(expected_unseen, 0.60):
                    break
        return pending

    def _finish_supplement_search(self, reason: str) -> None:
        """Finish supplemental search only after the backbone is complete."""
        if self.remaining_base_ids:
            return
        self.exploration_exhausted = True
        self._defer_unseen(reason)

    def _supplement_action(
        self,
        route: list[P4Action],
        phase: str,
        may_exhaust: bool,
    ) -> Optional[P4Action]:
        if phase not in ("backbone", "post", "emergency"):
            raise ValueError(f"未知补盲阶段 {phase!r}")

        if phase == "backbone":
            if self.backbone_probe_actions_done >= MAX_BACKBONE_PROBE_ACTIONS:
                return None
            insertion_limit = BACKBONE_PROBE_INSERTION_LIMIT_M
        elif phase == "post":
            if self.post_backbone_probe_actions_done >= MAX_POST_BACKBONE_PROBE_ACTIONS:
                return None
            insertion_limit = POST_PROBE_INSERTION_LIMIT_M
        else:
            if self.emergency_probe_actions_done >= MAX_EMERGENCY_PROBE_ACTIONS:
                if may_exhaust:
                    self._finish_supplement_search("emergency_probe_count_limit")
                return None
            insertion_limit = EMERGENCY_PROBE_INSERTION_LIMIT_M

        budget_exhausted = (
            self.supplement_actions_done >= MAX_SUPPLEMENT_ACTIONS
            or self.supplement_virtual_spent_s >= SUPPLEMENT_TOTAL_BUDGET_S
            or self.supplement_detour_spent_m >= SUPPLEMENT_DETOUR_BUDGET_M
        )
        if budget_exhausted:
            if phase != "backbone" and may_exhaust:
                self._finish_supplement_search("supplement_budget_exhausted")
            return None
        if self.exploration_exhausted:
            return None

        pending = self._coverage_pending_channels()
        if not pending:
            if phase != "backbone":
                self._finish_supplement_search("aggressive_search_criteria_reached")
            return None

        corrected = self.corrected_presence_probabilities()
        remaining_time_budget = SUPPLEMENT_TOTAL_BUDGET_S - self.supplement_virtual_spent_s
        remaining_detour_budget = SUPPLEMENT_DETOUR_BUDGET_M - self.supplement_detour_spent_m

        if phase == "backbone":
            candidate_points = self._backbone_supplement_candidates(route)
        elif phase == "post":
            candidate_points = self._post_supplement_candidates(pending)
        else:
            candidate_points = self._emergency_supplement_candidates(pending)

        best: Optional[P4Action] = None
        for point in candidate_points:
            insertion_distance, _ = route_insertion(self.client.position, route, point)
            if insertion_distance > min(insertion_limit, remaining_detour_budget):
                continue

            channel_values: list[tuple[float, float, float, int]] = []
            for channel in pending:
                state = self.channels[channel]
                if q3.measured_before(state.measured_positions, point):
                    continue
                belief = self.beliefs[channel]
                presence = max(
                    corrected.get(channel, belief.present_probability),
                    UNSEEN_PRESENCE_FLOOR,
                )
                marginal_coverage = (
                    self.discovery_survival_probability[channel]
                    * belief.directional_signal_probability(point)
                )
                if marginal_coverage <= 1.0e-10:
                    continue


                sectors = self._bearing_sector_count(channel)
                fairness = 1.0 + 0.18 * max(0, MIN_DISTINCT_BEARING_SECTORS - sectors)
                if presence >= HIGH_RISK_PRESENCE_THRESHOLD:
                    fairness *= 1.18
                expected_discover_and_clear = (
                    presence * marginal_coverage * SUPPLEMENT_CLEAR_PROBABILITY
                )
                channel_values.append((
                    expected_discover_and_clear * fairness,
                    marginal_coverage,
                    presence,
                    channel,
                ))

            channel_values.sort(reverse=True)
            chosen = [
                channel for _, marginal, _, channel
                in channel_values[:MAX_CHANNELS_PER_SUPPLEMENT]
                if marginal > 1.0e-8
            ]
            if not chosen:
                continue

            expected_by_channel = {channel: expected for expected, _, _, channel in channel_values}
            marginal_by_channel = {channel: marginal for _, marginal, _, channel in channel_values}
            expected = sum(expected_by_channel[ch] for ch in chosen)
            total_coverage_gain = sum(marginal_by_channel[ch] for ch in chosen)

            switches = len(chosen) - (1 if self.client.current_channel in chosen else 0)
            service = 5.0 * len(chosen) + float(switches)
            projected_time = insertion_distance / q3.DOG_SPEED + service
            if (
                service > SUPPLEMENT_SENSE_BUDGET_S + 1.0e-9
                or projected_time > remaining_time_budget + 1.0e-9
            ):
                continue


            phase_bonus = 1.12 if phase == "emergency" else 1.0
            score = phase_bonus * DISCOVERY_REWARD * (
                expected + 0.35 * total_coverage_gain
            ) / max(math.sqrt(projected_time), 1.0)

            projected_coverages = [
                min(1.0, self.discovery_coverage(ch) + marginal_by_channel[ch])
                for ch in chosen
            ]
            action = P4Action(
                kind="PROBE",
                point=np.asarray(point, dtype=float).copy(),
                channels=tuple(chosen),
                score=score,
                note=(
                    f"phase={phase}, marginal_coverage={total_coverage_gain:.3f}, "
                    f"projected_min={min(projected_coverages):.3f}, "
                    f"expected_unseen={self.expected_unseen_sources():.3f}, "
                    f"detour={insertion_distance:.1f}m"
                ),
                insertion_distance_m=insertion_distance,
                estimated_time_s=projected_time,
                probe_phase=phase,
            )
            if best is None or action.score > best.score:
                best = action

        if best is None or best.score < MIN_SUPPLEMENT_VALUE_RATE:
            if phase == "emergency" and may_exhaust:
                self._finish_supplement_search("no_positive_emergency_probe")
            return None
        return best

    def _defer_unseen(self, reason: str) -> None:
        """Keep deferred unseen channels distinct from confirmed absences."""
        for channel in self._unseen_channels():
            self.channels[channel].status = SEARCH_DEFERRED
            self.deferred_reasons[channel] = reason

    def plan(self) -> list[P4Action]:
        started = time.monotonic()
        planning_deadline = started + PLAN_TIME_BUDGET_S
        try:
            self.mark_search_complete_by_upper_bound()
            self.fast_finish_mode = (
                self.client.real_time_remaining() <= FAST_FINISH_REMAINING_S
                and self.discovered_count() >= MIN_REQUIRED_DISCOVERIES
            )
            for channel in self.localizing_channels():
                self.channels[channel].recompute_geometry()


            base_route = self._base_route_actions()
            tasks = self._clear_actions(base_route)

            if not self.fast_finish_mode:
                best_by_channel = self._defer_or_reactivate(
                    base_route, planning_deadline
                )
                tasks.extend(self._active_actions(best_by_channel))

            preliminary_route = insert_actions_by_minimum_increment(
                self.client.position, base_route, tasks
            )


            if not self.fast_finish_mode:
                if base_route:
                    phase = "backbone"
                else:
                    post_exhausted = (
                        self.post_backbone_probe_actions_done
                        >= MAX_POST_BACKBONE_PROBE_ACTIONS
                    )
                    still_risky = (
                        self.discovered_count() < MIN_REQUIRED_DISCOVERIES
                        or self.expected_unseen_sources()
                        >= EMERGENCY_EXPECTED_UNSEEN_THRESHOLD
                        or bool(self._coverage_pending_channels())
                    )
                    phase = "emergency" if post_exhausted and still_risky else "post"
                supplement = self._supplement_action(
                    preliminary_route,
                    phase=phase,
                    may_exhaust=(phase == "emergency" and not preliminary_route),
                )


                if (
                    supplement is None
                    and phase == "post"
                    and not preliminary_route
                    and self._coverage_pending_channels()
                ):
                    supplement = self._supplement_action(
                        preliminary_route,
                        phase="emergency",
                        may_exhaust=True,
                    )
                if supplement is not None:
                    tasks.append(supplement)

            route = insert_actions_by_minimum_increment(
                self.client.position, base_route, tasks
            )

            if self.fast_finish_mode and not route:
                for channel in list(self.localizing_channels()):
                    self._set_deferred(channel, "fast_finish_no_safe_cached_action")
                self.exploration_exhausted = True
                self._defer_unseen("fast_finish")
            return route
        finally:
            elapsed = time.monotonic() - started
            self.plan_durations_s.append(elapsed)
            if elapsed > PLAN_TIME_BUDGET_S:
                print(
                    f"[plan-budget] soft limit exceeded: "
                    f"{elapsed:.3f}s > {PLAN_TIME_BUDGET_S:.1f}s"
                )


    def _opportunistic_at_point(self, point: np.ndarray) -> None:
        choices: list[tuple[float, int]] = []
        for channel in self.localizing_channels():
            state = self.channels[channel]
            if q3.measured_before(state.measured_positions, point):
                continue
            prediction = self.beliefs[channel].predict(point)
            value = prediction.information_gain + prediction.spatial_gain
            if value >= MIN_INFORMATION_GAIN:
                choices.append((value, channel))
        choices.sort(reverse=True)
        for _, channel in choices[:MAX_OPPORTUNISTIC_AT_BASE]:
            response = self.client.measure(point, channel)
            self.handle_measure_response(channel, point, response)

    def visit_base(self, base_id: int) -> None:
        point = self.base_points[base_id]
        search_tasks = self._unseen_channels()
        print(f"[base-p4] visit B{base_id}, search={len(search_tasks)}")
        for channel in self._measurement_order(search_tasks):
            if self.discovered_count() >= 16:
                break
            if self.channels[channel].status not in (q3.SEARCHING, SEARCH_DEFERRED):
                continue
            self.channels[channel].status = q3.SEARCHING
            self.deferred_reasons.pop(channel, None)
            response = self.client.measure(point, channel)
            self.handle_measure_response(channel, point, response)
        self._opportunistic_at_point(point)
        self.remaining_base_ids.discard(base_id)
        self.visited_base_ids.add(base_id)


    def execute_active(self, action: P4Action) -> None:
        if action.channel is None:
            return
        state = self.channels[action.channel]
        if state.status != q3.LOCALIZING:
            return
        if q3.measured_before(state.measured_positions, action.point):
            self._set_deferred(action.channel, "planner_reselected_measured_point")
            return
        if self.active_attempts[action.channel] >= MAX_ACTIVE_AFTER_DISCOVERY:
            self._set_deferred(action.channel, "active_measurement_budget_exhausted")
            return

        before_radius = state.mec_radius
        self.active_attempts[action.channel] += 1
        response = self.client.measure(action.point, action.channel)
        self.handle_measure_response(action.channel, action.point, response)

        state = self.channels[action.channel]
        if state.status == q3.LOCALIZING:
            after_radius = state.mec_radius
            if (
                math.isfinite(before_radius)
                and before_radius > 1.0e-9
                and math.isfinite(after_radius)
            ):
                reduction = max(0.0, (before_radius - after_radius) / before_radius)
            else:
                reduction = 1.0
            if reduction < MIN_MEC_REDUCTION_RATIO:
                self.stagnant_active_steps[action.channel] += 1
            else:
                self.stagnant_active_steps[action.channel] = 0
            self.last_active_mec_radius[action.channel] = after_radius

            if self.stagnant_active_steps[action.channel] >= MAX_STAGNANT_ACTIVE_STEPS:
                self._set_deferred(
                    action.channel,
                    f"mec_stagnation={self.stagnant_active_steps[action.channel]}",
                )
            elif self.active_attempts[action.channel] >= MAX_ACTIVE_AFTER_DISCOVERY:
                self._set_deferred(
                    action.channel, "active_measurement_budget_exhausted"
                )
        self._opportunistic_at_point(action.point)

    def execute_probe(self, action: P4Action) -> None:
        print(
            f"[probe] phase={action.probe_phase} "
            f"channels={list(action.channels)} {action.note}"
        )
        spent = 0.0
        measured_any = False
        for channel in self._measurement_order(action.channels):
            if self.channels[channel].status not in (q3.SEARCHING, SEARCH_DEFERRED):
                continue
            switch = 0.0 if self.client.current_channel == channel else 1.0
            if spent + switch + 5.0 > SUPPLEMENT_SENSE_BUDGET_S + 1.0e-9:
                break
            self.channels[channel].status = q3.SEARCHING
            self.deferred_reasons.pop(channel, None)
            response = self.client.measure(action.point, channel)
            self.handle_measure_response(channel, action.point, response)
            spent += switch + 5.0
            measured_any = True
        if measured_any:
            self.supplement_actions_done += 1
            if action.probe_phase == "backbone":
                self.backbone_probe_actions_done += 1
            elif action.probe_phase == "post":
                self.post_backbone_probe_actions_done += 1
            elif action.probe_phase == "emergency":
                self.emergency_probe_actions_done += 1


            self.supplement_virtual_spent_s += action.estimated_time_s
            self.supplement_detour_spent_m += action.insertion_distance_m
        if not self.remaining_base_ids:
            if not self._coverage_pending_channels():
                self._finish_supplement_search("aggressive_search_criteria_reached")
            elif (
                self.supplement_actions_done >= MAX_SUPPLEMENT_ACTIONS
                or self.emergency_probe_actions_done >= MAX_EMERGENCY_PROBE_ACTIONS
            ):
                self._finish_supplement_search("aggressive_probe_budget_exhausted")

    def execute_clear(self, action: P4Action) -> None:
        if action.channel is None:
            return
        state = self.channels[action.channel]
        if state.status != q3.CLEARABLE:
            return
        point = action.point
        if not guarantees_clearance(state, point):
            if state.mec_center is None or not guarantees_clearance(state, state.mec_center):
                raise RuntimeError(f"频道 {action.channel} 没有经过最远距离验证的清除点")
            point = state.mec_center
        response = self.client.clear(point, action.channel)
        self.handle_clear_response(action.channel, point, response)

    def execute_node(self, action: P4Action) -> None:
        if action.kind == "BASE":
            if action.base_id is None:
                raise ValueError("BASE 缺少 base_id")
            self.visit_base(action.base_id)
        elif action.kind == "ACTIVE":
            self.execute_active(action)
        elif action.kind == "CLEAR":
            self.execute_clear(action)
        elif action.kind == "PROBE":
            self.execute_probe(action)
        else:
            raise ValueError(f"未知动作类型 {action.kind!r}")

    def print_status(self) -> None:
        counts: dict[str, int] = {}
        for state in self.channels.values():
            counts[state.status] = counts.get(state.status, 0) + 1
        print(
            "[status-p4]", counts,
            f"discovered={self.discovered_count()}",
            f"cleared={self.cleared_count}",
            f"Nhat={self.estimated_total():.2f}",
            f"remaining_base={sorted(self.remaining_base_ids)}",
            f"probes={self.supplement_actions_done}"
            f"({self.backbone_probe_actions_done}+"
            f"{self.post_backbone_probe_actions_done}+"
            f"{self.emergency_probe_actions_done})",
            f"probe_budget={self.supplement_virtual_spent_s:.1f}s/"
            f"{self.supplement_detour_spent_m:.1f}m",
            f"coverage_pending={len(self._coverage_pending_channels())}",
            f"expected_unseen={self.expected_unseen_sources():.3f}",
            f"vt={self.client.virtual_time:.2f}",
            f"real_left={self.client.real_time_remaining():.1f}s",
        )

    def run(self) -> None:
        self.client.enter()
        try:
            while True:
                self.print_status()
                if self.done():
                    print("[done-p4] no positive-value executable action remains.")
                    break
                if self.client.real_time_remaining() <= FINAL_EXIT_REMAINING_S:
                    print("[stop] final exit safety margin reached.")
                    break
                actions = self.plan()
                if not actions:
                    if not self.exploration_exhausted:
                        self.exploration_exhausted = True
                        self._defer_unseen("no_positive_action")
                    for channel in list(self.localizing_channels()):
                        self._set_deferred(channel, "no_route_admissible_action")
                    print("[done-p4] planner has no route-admissible action.")
                    break
                action = actions[0]
                print(
                    f"[next-p4] {action.kind} ch={action.channel} "
                    f"base={action.base_id} point={tuple(np.round(action.point, 2))} "
                    f"score={action.score:.6g} {action.note}"
                )
                self.execute_node(action)
        finally:
            self.print_status()

            self.client.exit()
            print("\n=== PROBLEM 4 FINAL SUMMARY ===")
            print("cleared channels:", [ch for ch, st in self.channels.items() if st.status == q3.CLEARED])
            print("deferred channels:", self.deferred_channels())
            print("search-deferred channels:", self.search_deferred_channels())
            print("deferred reasons:", self.deferred_reasons)
            print("estimated_total_for_scheduling:", f"{self.estimated_total():.3f}")
            remaining_coverages = {
                channel: round(self.discovery_coverage(channel), 4)
                for channel in self._unseen_channels()
            }
            print("unseen_detection_coverage:", remaining_coverages)
            print("expected_unseen_sources:", f"{self.expected_unseen_sources():.3f}")
            print(
                "unseen_bearing_sectors:",
                {ch: self._bearing_sector_count(ch) for ch in self._unseen_channels()},
            )
            print(
                "probe_actions_backbone/post/emergency:",
                self.backbone_probe_actions_done,
                self.post_backbone_probe_actions_done,
                self.emergency_probe_actions_done,
            )
            print("virtual_time_s:", f"{self.client.virtual_time:.6f}")
            if self.plan_durations_s:
                print(
                    "planning_time_s: count/mean/p95/max =",
                    len(self.plan_durations_s),
                    f"{float(np.mean(self.plan_durations_s)):.3f}",
                    f"{float(np.quantile(self.plan_durations_s, 0.95)):.3f}",
                    f"{max(self.plan_durations_s):.3f}",
                )
            print("log:", self.client.log_path)


def synthetic_observation(
    source: np.ndarray,
    station: np.ndarray,
    result: str,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "position": [float(station[0]), float(station[1])],
        "result": result,
    }
    if result == "direction":
        observation["angle_deg"] = q3.angle_deg_from_to(station, source)
    return observation


def run_self_test(seed: int) -> None:
    source = np.array([400.0, 200.0])
    back = np.array([-500.0, 200.0])
    front = np.array([1000.0, 200.0])

    observations = [
        synthetic_observation(source, back, "no_signal"),
        synthetic_observation(source, front, "direction"),
    ]
    region = mixed_position_region(observations)
    belief = MixedSourceBelief(1, np.random.default_rng(seed), count=1200)
    belief.present_probability = 1.0
    belief.rebuild(region, observations)
    p_front = belief.signal_probability(front)
    p_back = belief.signal_probability(back)
    p_front_directional = belief.directional_signal_probability(front)
    p_back_directional = belief.directional_signal_probability(back)
    print(
        "[self-test directional]",
        f"q_dir={belief.directional_probability:.3f}",
        f"p_front={p_front:.3f}",
        f"p_back={p_back:.3f}",
        f"p_front_dir={p_front_directional:.3f}",
        f"p_back_dir={p_back_directional:.3f}",
        f"phi={belief.direction_estimate_deg()}",
    )
    if (
        belief.directional_probability < 0.75
        or p_front <= p_back
        or p_front_directional <= p_back_directional
    ):
        raise AssertionError("近距离背向 no_signal 未能形成预期的定向证据")

    omni_observations = [
        synthetic_observation(source, back, "direction"),
        synthetic_observation(source, front, "direction"),
    ]
    omni_region = mixed_position_region(omni_observations)
    omni = MixedSourceBelief(2, np.random.default_rng(seed + 1), count=1200)
    omni.present_probability = 1.0
    omni.rebuild(omni_region, omni_observations)
    print("[self-test omni]", f"q_dir={omni.directional_probability:.3f}")
    if omni.directional_probability > 0.35:
        raise AssertionError("相反来向均成功检测时，全向后验未按预期占优")

    prediction = belief.predict(source + np.array([700.0, 0.0]))
    if not 0.0 <= prediction.information_gain <= 1.0:
        raise AssertionError("信息增益越界")


    current = np.array([0.0, 0.0])
    base_route = [
        P4Action("BASE", np.array([1000.0, 0.0]), 1.0, base_id=1),
        P4Action("BASE", np.array([2000.0, 0.0]), 1.0, base_id=2),
    ]
    task = P4Action("ACTIVE", np.array([1000.0, 100.0]), 2.0, channel=1)
    expected_extra, expected_at = q3.best_insertion(
        current, as_q3_route_nodes(base_route), task.point
    )
    actual_extra, actual_at = route_insertion(current, base_route, task.point)
    if abs(actual_extra - expected_extra) > 1.0e-9 or actual_at != expected_at:
        raise AssertionError("第四问路线插入未严格复用第三问实现")
    route = insert_actions_by_minimum_increment(current, base_route, [task])
    base_ids = [action.base_id for action in route if action.kind == "BASE"]
    if base_ids != [1, 2] or route[expected_at] is not task:
        raise AssertionError("最小增量插入破坏了主骨架顺序")
    print(
        "[self-test route]",
        f"extra={actual_extra:.3f}m",
        f"insert_at={actual_at}",
    )
    print("[self-test] PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUMCM 2026 B Problem 4 mixed-source controller"
    )
    parser.add_argument(
        "--robot-id", default=os.getenv("CUMCM_ROBOT_ID", ""),
        help="当前登录参赛队号；也可设置环境变量 CUMCM_ROBOT_ID",
    )
    parser.add_argument(
        "--base-url", default=os.getenv("CUMCM_BASE_URL", q3.BASE_URL_DEFAULT),
        help="模拟器接口地址，默认 http://127.0.0.1:2026",
    )
    parser.add_argument("--log", default=DEFAULT_LOG, help="本地 JSONL 行为日志路径")
    parser.add_argument("--outer-count", type=int, default=q3.OUTER_COUNT)
    parser.add_argument("--outer-radius", type=float, default=q3.OUTER_RADIUS)
    parser.add_argument("--rotation-deg", type=float, default=q3.OUTER_ROTATION_DEG)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--check-geometry", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q3.validate_backbone_or_raise(args.outer_count, args.outer_radius)
    points = q3.regular_base_points(
        args.outer_count, args.outer_radius, args.rotation_deg
    )
    print("[geometry] inherited Q3 base points:")
    for index, point in enumerate(points):
        print(f"  B{index}: ({point[0]:.3f}, {point[1]:.3f})")
    if args.check_geometry:
        print("[geometry] PASS")
        return
    if args.self_test:
        run_self_test(args.seed)
        return
    if not args.robot_id:
        raise SystemExit(
            "请通过 --robot-id 或环境变量 CUMCM_ROBOT_ID 设置当前登录参赛队号。"
        )

    client = q3.RobotClient(
        base_url=args.base_url,
        robot_id=args.robot_id,
        log_path=Path(args.log),
    )
    controller = Problem4Controller(
        client=client,
        outer_count=args.outer_count,
        outer_radius=args.outer_radius,
        rotation_deg=args.rotation_deg,
        seed=args.seed,
    )
    controller.run()


if __name__ == "__main__":
    main()
