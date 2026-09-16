#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Adaptive routing and localization controller.

Combines a fixed search backbone with active measurements and deferred clearance tasks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import requests
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry


BASE_URL_DEFAULT = "http://127.0.0.1:2026"
ARENA_ID = "default"

TARGET_RADIUS = 1800.0
SIGNAL_RADIUS_MIN = 1000.0
SIGNAL_RADIUS_MAX = 1500.0
NEAR_RADIUS = 5.0
CLEAR_RADIUS = 20.0
BEARING_ERROR_DEG = 1.0

DOG_SPEED = 5.0


OUTER_COUNT = 6
OUTER_RADIUS = 1150.0
OUTER_ROTATION_DEG = 0.0


ARENA_MODEL_MARGIN = 0.8
SIGNAL_MAX_MODEL_MARGIN = 0.8
ANGLE_MODEL_MARGIN_DEG = 0.03

NO_SIGNAL_REMOVE_RADIUS = 999.0
CLEAR_FAILURE_REMOVE_RADIUS = 19.0


CLEAR_MEC_TRIGGER = 19.8

CIRCLE_QUAD_SEGS = 96


PHI0_DEG = 35.0
ROBUST_RX_LIMIT = 999.0
MIN_NEW_STATION_SEPARATION = 25.0


Q2_N_A = 25
Q2_N_B_PER_SIDE = 9
Q2_GENERIC_ANGLES = 24
Q2_GENERIC_RADII = (250.0, 450.0, 650.0, 850.0)


Q2_ROUGH_TARGETS = 8
Q2_EXACT_TARGETS = 22
Q2_EXACT_TOP_K = 14
Q2_ERROR_SCENARIOS_DEG = (-1.0, 0.0, 1.0)


ACTIVE_NEAR_OPT_REL = 0.12


BASE_REUSE_FACTOR = 1.35
BASE_REUSE_MIN_REDUCTION = 0.20


BASE_OPPORTUNISTIC_MIN_OVERLAP = 0.20
BASE_OPPORTUNISTIC_MAX_CHANNELS = 6


CLEAR_ROUTE_INSERT_MAX_M = 300.0

CLEAR_ESCAPE_CURRENT_MAX_M = 750.0
CLEAR_ESCAPE_FUTURE_MIN_M = 650.0

MAX_DEFERRED_CLEAR = 2

ACTIVE_CLEAR_BIAS = 0.15


HTTP_TIMEOUT_S = 5.0
NETWORK_RETRIES = 20
NETWORK_RETRY_SLEEP_S = 0.35
REAL_TIME_SAFETY_MARGIN_S = 12.0


DEFAULT_LOG = (
    f"problem3_robot_log_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.jsonl"
)


def dist(
    a: np.ndarray | tuple[float, float],
    b: np.ndarray | tuple[float, float],
) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return float(np.linalg.norm(aa - bb))


def angle_deg_from_to(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    return float(math.degrees(math.atan2(d[1], d[0])) % 360.0)


def regular_base_points(
    outer_count: int = OUTER_COUNT,
    outer_radius: float = OUTER_RADIUS,
    rotation_deg: float = OUTER_ROTATION_DEG,
) -> list[np.ndarray]:
    pts = [np.array([0.0, 0.0], dtype=float)]
    phi0 = math.radians(rotation_deg)
    for j in range(outer_count):
        phi = phi0 + 2.0 * math.pi * j / outer_count
        pts.append(
            np.array(
                [
                    outer_radius * math.cos(phi),
                    outer_radius * math.sin(phi),
                ],
                dtype=float,
            )
        )
    return pts


def regular_backbone_worst_distance(
    outer_count: int,
    outer_radius: float,
) -> float:
    """Return the worst covered distance for a center plus a regular n-gon backbone."""
    if outer_count < 1:
        return float("inf")

    alpha = math.pi / outer_count

    def d_at(rho: float) -> float:
        value = (
            rho * rho
            + outer_radius * outer_radius
            - 2.0 * rho * outer_radius * math.cos(alpha)
        )
        return math.sqrt(max(0.0, value))

    return max(d_at(1000.0), d_at(TARGET_RADIUS))


def validate_backbone_or_raise(
    outer_count: int,
    outer_radius: float,
) -> None:
    worst = regular_backbone_worst_distance(outer_count, outer_radius)
    print(
        f"[geometry] outer_count={outer_count}, outer_radius={outer_radius:.3f} m, "
        f"worst guaranteed source-to-outer-station distance={worst:.3f} m"
    )
    if worst > SIGNAL_RADIUS_MIN + 1e-9:
        raise ValueError(
            "当前基准点参数不能保证半径 1800 m 圆域的完备 1000 m 覆盖："
            f"最坏距离 {worst:.3f} > {SIGNAL_RADIUS_MIN:.3f} m"
        )


def disk(center: np.ndarray, radius: float) -> BaseGeometry:
    return Point(float(center[0]), float(center[1])).buffer(
        float(radius),
        quad_segs=CIRCLE_QUAD_SEGS,
    )


def initial_arena_region() -> BaseGeometry:
    return disk(
        np.array([0.0, 0.0]),
        TARGET_RADIUS + ARENA_MODEL_MARGIN,
    )


def make_wedge(
    station: np.ndarray,
    measured_deg: float,
    half_width_deg: float,
) -> Polygon:
    theta1 = math.radians(measured_deg - half_width_deg)
    theta2 = math.radians(measured_deg + half_width_deg)

    far = max(
        10000.0,
        float(np.linalg.norm(station)) + TARGET_RADIUS + 5000.0,
    )

    p1 = station + far * np.array([math.cos(theta1), math.sin(theta1)])
    p2 = station + far * np.array([math.cos(theta2), math.sin(theta2)])

    return Polygon(
        [
            (float(station[0]), float(station[1])),
            (float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1])),
        ]
    )


def clean_geometry(g: BaseGeometry) -> BaseGeometry:
    if g.is_empty:
        return g
    try:
        if not g.is_valid:
            g = g.buffer(0)
    except Exception:
        pass
    return g


def region_hull_points(region: BaseGeometry) -> np.ndarray:
    if region.is_empty:
        return np.empty((0, 2), dtype=float)

    hull = region.convex_hull

    if hull.geom_type == "Point":
        return np.array([[hull.x, hull.y]], dtype=float)

    if hull.geom_type == "LineString":
        return np.asarray(hull.coords, dtype=float)

    return np.asarray(hull.exterior.coords[:-1], dtype=float)


def region_diameter(region: BaseGeometry) -> float:
    pts = region_hull_points(region)
    n = len(pts)

    if n <= 1:
        return 0.0

    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(np.max(d2)))


def max_distance_to_region(
    point: np.ndarray,
    region: BaseGeometry,
) -> float:
    pts = region_hull_points(region)

    if len(pts) == 0:
        return 0.0

    return float(
        np.max(
            np.linalg.norm(
                pts - point[None, :],
                axis=1,
            )
        )
    )


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


def _is_in_circle(
    c: Circle,
    p: tuple[float, float],
) -> bool:
    return math.hypot(p[0] - c.x, p[1] - c.y) <= c.r + 1e-7


def _diameter_circle(
    a: tuple[float, float],
    b: tuple[float, float],
) -> Circle:
    cx = (a[0] + b[0]) / 2.0
    cy = (a[1] + b[1]) / 2.0
    return Circle(
        cx,
        cy,
        math.hypot(a[0] - b[0], a[1] - b[1]) / 2.0,
    )


def _circumcircle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> Optional[Circle]:
    ax, ay = a
    bx, by = b
    cx, cy = c

    d = 2.0 * (
        ax * (by - cy)
        + bx * (cy - ay)
        + cx * (ay - by)
    )

    if abs(d) < 1e-12:
        return None

    aa = ax * ax + ay * ay
    bb = bx * bx + by * by
    cc = cx * cx + cy * cy

    ux = (
        aa * (by - cy)
        + bb * (cy - ay)
        + cc * (ay - by)
    ) / d

    uy = (
        aa * (cx - bx)
        + bb * (ax - cx)
        + cc * (bx - ax)
    ) / d

    return Circle(
        ux,
        uy,
        math.hypot(ux - ax, uy - ay),
    )


def minimum_enclosing_circle(region: BaseGeometry) -> Circle:
    pts_np = region_hull_points(region)

    if len(pts_np) == 0:
        return Circle(0.0, 0.0, float("inf"))

    pts = [(float(x), float(y)) for x, y in pts_np]

    rng = random.Random(20260911)
    rng.shuffle(pts)

    c: Optional[Circle] = None

    for i, p in enumerate(pts):
        if c is not None and _is_in_circle(c, p):
            continue

        c = Circle(p[0], p[1], 0.0)

        for j in range(i):
            q = pts[j]

            if _is_in_circle(c, q):
                continue

            c = _diameter_circle(p, q)

            for k in range(j):
                r = pts[k]

                if _is_in_circle(c, r):
                    continue

                cc = _circumcircle(p, q, r)

                if cc is not None:
                    c = cc
                else:
                    candidates = [
                        _diameter_circle(p, q),
                        _diameter_circle(p, r),
                        _diameter_circle(q, r),
                    ]

                    valid = [
                        x
                        for x in candidates
                        if _is_in_circle(x, p)
                        and _is_in_circle(x, q)
                        and _is_in_circle(x, r)
                    ]

                    c = min(valid, key=lambda x: x.r)

    assert c is not None
    return c


def update_region_direction(
    region: BaseGeometry,
    station: np.ndarray,
    svd_deg: float,
) -> BaseGeometry:
    upper_disk = disk(
        station,
        SIGNAL_RADIUS_MAX + SIGNAL_MAX_MODEL_MARGIN,
    )

    wedge = make_wedge(
        station,
        svd_deg,
        BEARING_ERROR_DEG + ANGLE_MODEL_MARGIN_DEG,
    )

    out = (
        region
        .intersection(upper_disk)
        .intersection(wedge)
    )

    return clean_geometry(out)


def update_region_no_signal(
    region: BaseGeometry,
    station: np.ndarray,
) -> BaseGeometry:
    out = region.difference(
        disk(station, NO_SIGNAL_REMOVE_RADIUS)
    )
    return clean_geometry(out)


def update_region_clear_failure(
    region: BaseGeometry,
    station: np.ndarray,
) -> BaseGeometry:
    out = region.difference(
        disk(station, CLEAR_FAILURE_REMOVE_RADIUS)
    )
    return clean_geometry(out)


def sample_region_points(
    region: BaseGeometry,
    max_points: int,
) -> np.ndarray:
    """Sample convex-hull boundaries and one representative point per component."""
    if region.is_empty:
        return np.empty((0, 2), dtype=float)

    hull_pts = region_hull_points(region)

    if len(hull_pts) > max_points:
        idx = np.linspace(
            0,
            len(hull_pts) - 1,
            max_points,
            dtype=int,
        )
        hull_pts = hull_pts[idx]

    extras: list[tuple[float, float]] = []

    if region.geom_type == "MultiPolygon":
        for geom in region.geoms:
            p = geom.representative_point()
            extras.append((p.x, p.y))
    else:
        p = region.representative_point()
        extras.append((p.x, p.y))

    if extras:
        pts = np.vstack(
            [
                hull_pts,
                np.asarray(extras, dtype=float),
            ]
        )
    else:
        pts = hull_pts

    if len(pts) == 0:
        return pts

    rounded = np.round(pts, decimals=6)
    _, unique_idx = np.unique(
        rounded,
        axis=0,
        return_index=True,
    )

    return pts[np.sort(unique_idx)]


@dataclass(frozen=True)
class P2Point:
    x: float
    y: float

    def __add__(self, other: "P2Point") -> "P2Point":
        return P2Point(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other: "P2Point") -> "P2Point":
        return P2Point(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(self, value: float) -> "P2Point":
        return P2Point(
            self.x * value,
            self.y * value,
        )

    __rmul__ = __mul__

    def as_np(self) -> np.ndarray:
        return np.array(
            [self.x, self.y],
            dtype=float,
        )


def p2_distance(a: P2Point, b: P2Point) -> float:
    return math.hypot(
        a.x - b.x,
        a.y - b.y,
    )


@dataclass(frozen=True)
class Problem2Config:
    first_station: P2Point
    first_bearing_deg: float
    rho_minus: float
    rho_plus: float
    phi_0: float

    target_radius: float = 1800.0
    receive_radius_min: float = 1000.0
    receive_radius_max: float = 1500.0
    bearing_error_rad: float = math.radians(1.0)
    optical_radius: float = 20.0

    use_2d_receive_filter: bool = True
    limit_station_to_target_disk: bool = False
    use_optical_shortcut: bool = True

    n_a: int = Q2_N_A
    n_b_per_side: int = Q2_N_B_PER_SIDE
    n_rho: int = Q2_EXACT_TARGETS
    n_first_angle: int = 5
    n_second_error: int = len(Q2_ERROR_SCENARIOS_DEG)
    top_k_for_exact: int = Q2_EXACT_TOP_K


@dataclass(frozen=True)
class Candidate:
    a: float
    b: float
    xy: P2Point


@dataclass(frozen=True)
class SelectionResult:
    best: Candidate
    score: float
    score_name: str
    candidates: tuple[Candidate, ...]
    proxy_scores: tuple[float, ...]
    exact_scores: tuple[float | None, ...]
    targets: tuple[P2Point, ...]


def linspace(
    start: float,
    stop: float,
    count: int,
) -> list[float]:
    if count <= 0:
        raise ValueError("count 必须为正整数")

    if count == 1:
        return [(start + stop) / 2.0]

    step = (stop - start) / (count - 1)

    return [
        start + index * step
        for index in range(count)
    ]


def unit_basis(
    theta_deg: float,
) -> tuple[P2Point, P2Point]:
    theta = math.radians(theta_deg)

    return (
        P2Point(
            math.cos(theta),
            math.sin(theta),
        ),
        P2Point(
            -math.sin(theta),
            math.cos(theta),
        ),
    )


def local_to_global(
    config: Problem2Config,
    a: float,
    b: float,
) -> P2Point:
    u, v = unit_basis(
        config.first_bearing_deg
    )
    return config.first_station + a * u + b * v


def global_to_local(
    config: Problem2Config,
    point: P2Point,
) -> tuple[float, float]:
    u, v = unit_basis(
        config.first_bearing_deg
    )

    relative = point - config.first_station

    return (
        relative.x * u.x + relative.y * u.y,
        relative.x * v.x + relative.y * v.y,
    )


def wrap_deg(angle_deg: float) -> float:
    return angle_deg % 360.0


def validate_config(
    config: Problem2Config,
) -> tuple[float, float]:
    if not 0.0 <= config.rho_minus < config.rho_plus:
        raise ValueError(
            "必须满足 0 <= rho_minus < rho_plus"
        )

    if (
        config.rho_plus
        > config.receive_radius_max + 1.0e-9
    ):
        raise ValueError(
            "rho_plus 不能超过第一次有效接收距离上界"
        )

    if not 0.0 < config.phi_0 < math.pi / 2.0:
        raise ValueError(
            "phi_0 必须位于 (0, pi/2)"
        )

    m = 0.5 * (
        config.rho_minus + config.rho_plus
    )
    h = 0.5 * (
        config.rho_plus - config.rho_minus
    )

    if h > config.receive_radius_min + 1.0e-9:
        raise ValueError(
            "h > receive_radius_min，纯接收鲁棒区域为空"
        )

    if (
        h
        > config.receive_radius_min
        * math.cos(config.phi_0)
        + 1.0e-9
    ):
        raise ValueError(
            "候选区域为空：h > receive_radius_min*cos(phi_0)"
        )

    return m, h


def candidate_a_interval(
    config: Problem2Config,
) -> tuple[float, float]:
    m, h = validate_config(config)

    half_width = (
        config.receive_radius_min
        * math.cos(config.phi_0)
        - h
    )

    return (
        m - half_width,
        m + half_width,
    )


def candidate_b_bounds(
    config: Problem2Config,
    a: float,
) -> tuple[float, float]:
    m, h = validate_config(config)

    q = h + abs(a - m)

    if q > config.receive_radius_min:
        raise ValueError(
            "该 a 不满足鲁棒接收约束"
        )

    b_min = q * math.tan(config.phi_0)

    b_max = math.sqrt(
        max(
            0.0,
            config.receive_radius_min**2 - q**2,
        )
    )

    if b_min > b_max + 1.0e-10:
        raise ValueError(
            "该 a 下交会角约束与鲁棒接收约束不相容"
        )

    return b_min, b_max


def analytic_centers(
    config: Problem2Config,
) -> list[Candidate]:
    m, h = validate_config(config)

    b_star = math.sqrt(
        max(
            0.0,
            config.receive_radius_min**2 - h**2,
        )
    )

    return [
        Candidate(
            m,
            +b_star,
            local_to_global(
                config,
                m,
                +b_star,
            ),
        ),
        Candidate(
            m,
            -b_star,
            local_to_global(
                config,
                m,
                -b_star,
            ),
        ),
    ]


def candidate_grid(
    config: Problem2Config,
) -> list[Candidate]:
    m, h = validate_config(config)

    a_low, a_high = candidate_a_interval(
        config
    )

    candidates: list[Candidate] = []

    for a in linspace(
        a_low,
        a_high,
        config.n_a,
    ):
        b_min, b_max = candidate_b_bounds(
            config,
            a,
        )

        absolute_b_values = (
            [0.5 * (b_min + b_max)]
            if abs(b_max - b_min) <= 1.0e-12
            else linspace(
                b_min,
                b_max,
                config.n_b_per_side,
            )
        )

        for absolute_b in absolute_b_values:
            if absolute_b <= 1.0e-10:
                continue

            for sign in (-1.0, +1.0):
                b = sign * absolute_b
                q = h + abs(a - m)

                weak_physical_ok = (
                    abs(b)
                    <= config.receive_radius_min
                    + 1.0e-9
                )

                robust_axis_ok = (
                    q**2 + b**2
                    <= config.receive_radius_min**2
                    + 1.0e-7
                )

                angle_ok = (
                    abs(b) + 1.0e-9
                    >= q * math.tan(config.phi_0)
                )

                xy = local_to_global(
                    config,
                    a,
                    b,
                )

                disk_ok = (
                    not config.limit_station_to_target_disk
                    or math.hypot(xy.x, xy.y)
                    <= config.target_radius
                    + 1.0e-9
                )

                if (
                    weak_physical_ok
                    and robust_axis_ok
                    and angle_ok
                    and disk_ok
                ):
                    candidates.append(
                        Candidate(
                            a,
                            b,
                            xy,
                        )
                    )

    if not candidates:
        raise RuntimeError(
            "离散候选点为空"
        )

    return candidates


def possible_target_scenarios(
    config: Problem2Config,
    two_dimensional: bool | None = None,
) -> list[P2Point]:
    validate_config(config)

    if two_dimensional is None:
        two_dimensional = (
            config.use_2d_receive_filter
        )

    rhos = linspace(
        config.rho_minus,
        config.rho_plus,
        config.n_rho,
    )

    offsets = (
        linspace(
            -config.bearing_error_rad,
            config.bearing_error_rad,
            config.n_first_angle,
        )
        if two_dimensional
        else [0.0]
    )

    theta_center = math.radians(
        config.first_bearing_deg
    )

    targets: list[P2Point] = []

    for rho in rhos:
        for offset in offsets:
            radial_distance = (
                rho / math.cos(offset)
            )

            theta = (
                theta_center + offset
            )

            target = (
                config.first_station
                + P2Point(
                    radial_distance
                    * math.cos(theta),
                    radial_distance
                    * math.sin(theta),
                )
            )

            if (
                radial_distance
                <= config.receive_radius_max
                + 1.0e-9
                and math.hypot(
                    target.x,
                    target.y,
                )
                <= config.target_radius
                + 1.0e-9
            ):
                targets.append(target)

    if not targets:
        raise RuntimeError(
            "可能目标场景为空"
        )

    return targets


def passes_sampled_receive_check(
    config: Problem2Config,
    candidate: Candidate,
    targets: list[P2Point],
) -> bool:
    return (
        max(
            p2_distance(
                candidate.xy,
                target,
            )
            for target in targets
        )
        <= config.receive_radius_min
        + 1.0e-9
    )


def acute_intersection_angle(
    config: Problem2Config,
    second_station: P2Point,
    target: P2Point,
) -> float:
    first_vector = (
        target - config.first_station
    )

    second_vector = (
        target - second_station
    )

    first_norm = math.hypot(
        first_vector.x,
        first_vector.y,
    )

    second_norm = math.hypot(
        second_vector.x,
        second_vector.y,
    )

    if (
        first_norm <= 1.0e-12
        or second_norm <= 1.0e-12
    ):
        return math.pi / 2.0

    cosine = abs(
        (
            first_vector.x * second_vector.x
            + first_vector.y * second_vector.y
        )
        / (first_norm * second_norm)
    )

    cosine = min(
        1.0,
        max(
            0.0,
            cosine,
        ),
    )

    return math.atan2(
        math.sqrt(
            max(
                0.0,
                1.0 - cosine**2,
            )
        ),
        cosine,
    )


def linearized_diameter(
    config: Problem2Config,
    second_station: P2Point,
    target: P2Point,
) -> float:
    r1 = p2_distance(
        target,
        config.first_station,
    )

    r2 = p2_distance(
        target,
        second_station,
    )

    if (
        config.use_optical_shortcut
        and r2 <= config.optical_radius
    ):
        return 0.0

    phi = acute_intersection_angle(
        config,
        second_station,
        target,
    )

    sine = abs(math.sin(phi))

    if sine <= 1.0e-10:
        return float("inf")

    w1 = (
        r1
        * math.tan(
            config.bearing_error_rad
        )
    )

    w2 = (
        r2
        * math.tan(
            config.bearing_error_rad
        )
    )

    return (
        2.0
        / sine
        * math.sqrt(
            w1**2
            + w2**2
            + 2.0
            * abs(math.cos(phi))
            * w1
            * w2
        )
    )


def proxy_robust_score(
    config: Problem2Config,
    candidate: Candidate,
    targets: list[P2Point],
) -> float:
    return max(
        linearized_diameter(
            config,
            candidate.xy,
            target,
        )
        for target in targets
    )


def first_direction_observation(
    observations: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    for obs in observations:
        if obs.get("result") == "direction":
            return obs
    return None


def build_q2_config_from_current_region(
    region: BaseGeometry,
    observations: list[dict[str, Any]],
    phi0_deg: float = PHI0_DEG,
) -> Problem2Config:
    first = first_direction_observation(
        observations
    )

    if first is None:
        raise ValueError(
            "没有 direction 观测，不能构造问题二配置"
        )

    s_np = np.asarray(
        first["position"],
        dtype=float,
    )

    theta_deg = float(
        first["angle_deg"]
    )

    theta_rad = math.radians(
        theta_deg
    )

    u_np = np.array(
        [
            math.cos(theta_rad),
            math.sin(theta_rad),
        ]
    )

    pts = region_hull_points(
        region
    )

    if len(pts) == 0:
        raise ValueError(
            "当前候选区域为空"
        )

    proj = (
        pts - s_np[None, :]
    ) @ u_np

    rho_minus = max(
        0.0,
        float(np.min(proj)),
    )

    rho_plus = min(
        SIGNAL_RADIUS_MAX,
        float(np.max(proj)),
    )

    if rho_plus <= rho_minus + 1e-6:
        rho_plus = min(
            SIGNAL_RADIUS_MAX,
            rho_minus + 1.0,
        )

    h = 0.5 * (
        rho_plus - rho_minus
    )

    phi_candidates = [
        phi0_deg,
        30.0,
        25.0,
        20.0,
        15.0,
        10.0,
        5.0,
    ]

    feasible_phi = None

    for deg in phi_candidates:
        if (
            h
            <= SIGNAL_RADIUS_MIN
            * math.cos(
                math.radians(deg)
            )
            + 1e-9
        ):
            feasible_phi = deg
            break

    if feasible_phi is None:
        raise ValueError(
            "当前轴向区间过长，问题二鲁棒候选域为空"
        )

    return Problem2Config(
        first_station=P2Point(
            float(s_np[0]),
            float(s_np[1]),
        ),
        first_bearing_deg=theta_deg,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        phi_0=math.radians(
            feasible_phi
        ),
        target_radius=TARGET_RADIUS,
        receive_radius_min=SIGNAL_RADIUS_MIN,
        receive_radius_max=SIGNAL_RADIUS_MAX,
        bearing_error_rad=math.radians(
            BEARING_ERROR_DEG
        ),
        optical_radius=CLEAR_RADIUS,
        use_2d_receive_filter=True,
        limit_station_to_target_disk=False,
        use_optical_shortcut=True,
        n_a=Q2_N_A,
        n_b_per_side=Q2_N_B_PER_SIDE,
        n_rho=Q2_EXACT_TARGETS,
        n_first_angle=5,
        n_second_error=len(
            Q2_ERROR_SCENARIOS_DEG
        ),
        top_k_for_exact=Q2_EXACT_TOP_K,
    )


def target_inside_current_region(
    target: P2Point,
    region: BaseGeometry,
) -> bool:
    return region.buffer(1.0).covers(
        Point(
            target.x,
            target.y,
        )
    )


def _deduplicate_p2_targets(
    targets: Iterable[P2Point],
    decimals: int = 4,
) -> list[P2Point]:
    seen: set[tuple[float, float]] = set()
    result: list[P2Point] = []

    for t in targets:
        key = (
            round(t.x, decimals),
            round(t.y, decimals),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(t)

    return result


def _downsample_targets(
    targets: list[P2Point],
    max_count: int,
) -> list[P2Point]:
    if max_count <= 0:
        return targets

    if len(targets) <= max_count:
        return targets

    idx = np.linspace(
        0,
        len(targets) - 1,
        max_count,
        dtype=int,
    )

    return [
        targets[int(i)]
        for i in idx
    ]


def current_q2_targets(
    config: Problem2Config,
    region: BaseGeometry,
) -> list[P2Point]:
    """Combine first-bearing scenarios with samples from the current full region."""
    scenario_targets = (
        possible_target_scenarios(
            config
        )
    )

    filtered = [
        t
        for t in scenario_targets
        if target_inside_current_region(
            t,
            region,
        )
    ]

    boundary_pts = sample_region_points(
        region,
        Q2_EXACT_TARGETS,
    )

    merged: list[P2Point] = list(
        filtered
    )

    for x, y in boundary_pts:
        merged.append(
            P2Point(
                float(x),
                float(y),
            )
        )

    merged = _deduplicate_p2_targets(
        merged
    )

    if merged:
        return merged

    pts = sample_region_points(
        region,
        max(
            8,
            Q2_EXACT_TARGETS,
        ),
    )

    return [
        P2Point(
            float(x),
            float(y),
        )
        for x, y in pts
    ]


def exact_robust_score_current_region(
    config: Problem2Config,
    candidate: Candidate,
    targets: list[P2Point],
    current_region: BaseGeometry,
) -> float:
    """Evaluate the worst new-region diameter over targets and bearing errors."""
    worst = 0.0

    s_np = candidate.xy.as_np()

    for target in targets:
        r2 = p2_distance(
            target,
            candidate.xy,
        )

        if (
            config.use_optical_shortcut
            and r2 <= config.optical_radius
        ):
            scenario_value = 0.0

        else:
            true_second_deg = math.degrees(
                math.atan2(
                    target.y - candidate.xy.y,
                    target.x - candidate.xy.x,
                )
            )

            scenario_value = 0.0

            for error_deg in Q2_ERROR_SCENARIOS_DEG:
                measured_second_deg = wrap_deg(
                    true_second_deg
                    + error_deg
                )

                new_region = update_region_direction(
                    current_region,
                    s_np,
                    measured_second_deg,
                )

                if new_region.is_empty:
                    diameter = float("inf")
                else:
                    diameter = region_diameter(
                        new_region
                    )

                scenario_value = max(
                    scenario_value,
                    diameter,
                )

        worst = max(
            worst,
            scenario_value,
        )

    return worst


@dataclass
class ActiveOption:
    point: np.ndarray
    score: float


def measured_before(
    measured_positions: list[np.ndarray],
    candidate: np.ndarray,
) -> bool:
    return any(
        dist(candidate, p)
        < MIN_NEW_STATION_SEPARATION
        for p in measured_positions
    )


def problem2_score_existing_station(
    region: BaseGeometry,
    observations: list[dict[str, Any]],
    measured_positions: list[np.ndarray],
    station: np.ndarray,
) -> float:
    if measured_before(
        measured_positions,
        station,
    ):
        return float("inf")

    if (
        max_distance_to_region(
            station,
            region,
        )
        > ROBUST_RX_LIMIT
    ):
        return float("inf")

    try:
        config = (
            build_q2_config_from_current_region(
                region,
                observations,
            )
        )

        targets = current_q2_targets(
            config,
            region,
        )

    except Exception:
        return float("inf")

    a, b = global_to_local(
        config,
        P2Point(
            float(station[0]),
            float(station[1]),
        ),
    )

    c = Candidate(
        a,
        b,
        P2Point(
            float(station[0]),
            float(station[1]),
        ),
    )

    return exact_robust_score_current_region(
        config,
        c,
        targets,
        region,
    )


def generic_active_candidates(
    region: BaseGeometry,
    measured_positions: list[np.ndarray],
) -> list[np.ndarray]:
    mec = minimum_enclosing_circle(
        region
    )

    center = mec.center

    result: list[np.ndarray] = []

    radii = (
        0.0,
        *Q2_GENERIC_RADII,
    )

    for rr in radii:
        n_ang = (
            1
            if rr == 0.0
            else Q2_GENERIC_ANGLES
        )

        for j in range(n_ang):
            ang = (
                0.0
                if rr == 0.0
                else 2.0
                * math.pi
                * j
                / n_ang
            )

            p = (
                center
                + rr
                * np.array(
                    [
                        math.cos(ang),
                        math.sin(ang),
                    ]
                )
            )

            if measured_before(
                measured_positions,
                p,
            ):
                continue

            if (
                max_distance_to_region(
                    p,
                    region,
                )
                <= ROBUST_RX_LIMIT
            ):
                result.append(p)

    return deduplicate_points(
        result
    )


def deduplicate_points(
    points: Iterable[np.ndarray],
    tol: float = 1e-4,
) -> list[np.ndarray]:
    seen: set[tuple[int, int]] = set()
    out: list[np.ndarray] = []

    scale = 1.0 / tol

    for p in points:
        p = np.asarray(
            p,
            dtype=float,
        )

        key = (
            int(
                round(
                    float(p[0]) * scale
                )
            ),
            int(
                round(
                    float(p[1]) * scale
                )
            ),
        )

        if key not in seen:
            seen.add(key)
            out.append(p)

    return out


def compute_active_options(
    region: BaseGeometry,
    observations: list[dict[str, Any]],
    measured_positions: list[np.ndarray],
) -> list[ActiveOption]:
    try:
        config = (
            build_q2_config_from_current_region(
                region,
                observations,
            )
        )

        targets = current_q2_targets(
            config,
            region,
        )

        candidates = candidate_grid(
            config
        )

    except Exception as exc:
        print(
            f"[q2] analytic candidate construction fallback: {exc}"
        )

        candidates = []
        targets = []
        config = None

    valid: list[Candidate] = []

    if config is not None:
        for c in candidates:
            p = c.xy.as_np()

            if measured_before(
                measured_positions,
                p,
            ):
                continue

            if (
                config.use_2d_receive_filter
                and not passes_sampled_receive_check(
                    config,
                    c,
                    targets,
                )
            ):
                continue

            if (
                max_distance_to_region(
                    p,
                    region,
                )
                > ROBUST_RX_LIMIT
            ):
                continue

            valid.append(c)

    if (
        config is not None
        and valid
    ):
        rough_targets = _downsample_targets(
            targets,
            Q2_ROUGH_TARGETS,
        )

        proxy_scores = [
            proxy_robust_score(
                config,
                c,
                rough_targets,
            )
            for c in valid
        ]

        ranked = sorted(
            range(len(valid)),
            key=proxy_scores.__getitem__,
        )

        keep = ranked[
            : min(
                config.top_k_for_exact,
                len(valid),
            )
        ]

        exact: list[ActiveOption] = []

        for idx in keep:
            score = (
                exact_robust_score_current_region(
                    config,
                    valid[idx],
                    targets,
                    region,
                )
            )

            if math.isfinite(score):
                exact.append(
                    ActiveOption(
                        valid[idx].xy.as_np(),
                        float(score),
                    )
                )

        exact.sort(
            key=lambda x: x.score
        )

        if exact:
            return exact

    fallback_points = (
        generic_active_candidates(
            region,
            measured_positions,
        )
    )

    if not fallback_points:
        return []

    target_pts = sample_region_points(
        region,
        Q2_EXACT_TARGETS,
    )

    options: list[ActiveOption] = []

    for p in fallback_points:
        worst = 0.0

        for g in target_pts:
            if dist(p, g) <= CLEAR_RADIUS:
                scenario = 0.0

            else:
                true_deg = angle_deg_from_to(
                    p,
                    g,
                )

                scenario = 0.0

                for error_deg in Q2_ERROR_SCENARIOS_DEG:
                    nr = update_region_direction(
                        region,
                        p,
                        true_deg + error_deg,
                    )

                    if nr.is_empty:
                        scenario = float("inf")
                        break

                    scenario = max(
                        scenario,
                        region_diameter(nr),
                    )

            worst = max(
                worst,
                scenario,
            )

        if math.isfinite(worst):
            options.append(
                ActiveOption(
                    p,
                    worst,
                )
            )

    options.sort(
        key=lambda x: x.score
    )

    return options


SEARCHING = "SEARCHING"
LOCALIZING = "LOCALIZING"
CLEARABLE = "CLEARABLE"
CLEARED = "CLEARED"
ABSENT = "ABSENT"


@dataclass
class ChannelState:
    channel: int
    status: str = SEARCHING
    region: BaseGeometry = field(
        default_factory=initial_arena_region
    )
    observations: list[dict[str, Any]] = field(
        default_factory=list
    )
    measured_positions: list[np.ndarray] = field(
        default_factory=list
    )
    ever_detected: bool = False
    region_version: int = 0

    mec_center: Optional[np.ndarray] = None
    mec_radius: float = float("inf")
    diameter: float = float("inf")

    active_cache_version: int = -1
    active_cache: list[ActiveOption] = field(
        default_factory=list
    )

    def invalidate_plan_cache(self) -> None:
        self.active_cache_version = -1
        self.active_cache = []

    def recompute_geometry(self) -> None:
        if self.region.is_empty:
            self.mec_center = None
            self.mec_radius = float("inf")
            self.diameter = 0.0
            return

        self.diameter = region_diameter(
            self.region
        )

        mec = minimum_enclosing_circle(
            self.region
        )

        self.mec_center = mec.center
        self.mec_radius = mec.r

        if (
            self.ever_detected
            and self.status
            not in (CLEARED, ABSENT)
            and self.mec_radius
            <= CLEAR_MEC_TRIGGER
        ):
            self.status = CLEARABLE

        elif (
            self.ever_detected
            and self.status == CLEARABLE
            and self.mec_radius
            > CLEAR_MEC_TRIGGER
        ):
            self.status = LOCALIZING


class RobotClient:
    def __init__(
        self,
        base_url: str,
        robot_id: str,
        log_path: Path,
    ):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.log_path = log_path

        self.run_id = (
            f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
            f"-{uuid.uuid4().hex[:8]}"
        )

        self.counter = 0
        self.position = np.array(
            [0.0, 0.0],
            dtype=float,
        )
        self.current_channel = 1
        self.virtual_time = 0.0

        self.entered = False
        self.deadline_monotonic: Optional[float] = None

        self.session = requests.Session()

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _new_request_id(
        self,
        prefix: str,
    ) -> str:
        self.counter += 1
        token = uuid.uuid4().hex[:8]
        return (
            f"{prefix}-{self.counter}-{token}"
        )

    def _base_payload(
        self,
        request_id: str,
    ) -> dict[str, Any]:
        return {
            "arena_id": ARENA_ID,
            "robot_id": self.robot_id,
            "request_id": request_id,
        }

    def _action_payload(
        self,
        request_id: str,
        position: np.ndarray,
        channel: int,
    ) -> dict[str, Any]:
        payload = self._base_payload(
            request_id
        )

        payload["position"] = {
            "x": float(position[0]),
            "y": float(position[1]),
        }

        payload["channel"] = int(
            channel
        )

        return payload

    def _write_log(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        http_status: Optional[int],
        response: Optional[dict[str, Any]],
        error: Optional[str],
    ) -> None:
        record = {
            "run_id": self.run_id,
            "local_time": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(),
            ),
            "path": path,
            "payload": payload,
            "http_status": http_status,
            "response": response,
            "error": error,
            "client_position": (
                self.position.tolist()
            ),
            "client_current_channel": (
                self.current_channel
            ),
            "client_virtual_time": (
                self.virtual_time
            ),
        }

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def real_time_remaining(self) -> float:
        if self.deadline_monotonic is None:
            return float("inf")

        return (
            self.deadline_monotonic
            - time.monotonic()
        )

    def _post_same_action_with_retry(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Optional[str] = None

        for attempt in range(
            1,
            NETWORK_RETRIES + 1,
        ):
            if (
                self.entered
                and self.real_time_remaining()
                <= REAL_TIME_SAFETY_MARGIN_S
            ):
                raise TimeoutError(
                    "现实运行时间即将耗尽，停止发送新的 measure/clear 动作。"
                )

            try:
                r = self.session.post(
                    self.base_url + path,
                    json=payload,
                    timeout=HTTP_TIMEOUT_S,
                )

                try:
                    data = r.json()
                except Exception:
                    data = None

                self._write_log(
                    path=path,
                    payload=payload,
                    http_status=r.status_code,
                    response=data,
                    error=None,
                )

                if r.status_code != 200:
                    raise RuntimeError(
                        f"{path} HTTP {r.status_code}: "
                        f"{r.text[:500]}"
                    )

                if not isinstance(
                    data,
                    dict,
                ):
                    raise RuntimeError(
                        f"{path} 未返回合法 JSON 对象"
                    )

                if (
                    data.get("accepted")
                    is not True
                ):
                    raise RuntimeError(
                        f"{path} accepted=false，动作未执行：{data}"
                    )

                return data

            except (
                requests.Timeout,
                requests.ConnectionError,
            ) as exc:
                last_error = repr(exc)

                self._write_log(
                    path=path,
                    payload=payload,
                    http_status=None,
                    response=None,
                    error=last_error,
                )

                if attempt >= NETWORK_RETRIES:
                    break

                time.sleep(
                    NETWORK_RETRY_SLEEP_S
                )

        raise ConnectionError(
            f"{path} 网络重试失败，最后错误：{last_error}"
        )

    def enter(self) -> dict[str, Any]:
        rid = self._new_request_id(
            "enter"
        )

        payload = self._base_payload(
            rid
        )

        data = None
        last_error = None

        for attempt in range(
            1,
            NETWORK_RETRIES + 1,
        ):
            try:
                r = self.session.post(
                    self.base_url + "/enter",
                    json=payload,
                    timeout=HTTP_TIMEOUT_S,
                )

                try:
                    data = r.json()
                except Exception:
                    data = None

                self._write_log(
                    path="/enter",
                    payload=payload,
                    http_status=r.status_code,
                    response=data,
                    error=None,
                )

                if r.status_code != 200:
                    raise RuntimeError(
                        f"/enter HTTP {r.status_code}: "
                        f"{r.text[:500]}"
                    )

                if not isinstance(
                    data,
                    dict,
                ):
                    raise RuntimeError(
                        "/enter 未返回合法 JSON"
                    )

                if (
                    data.get("accepted")
                    is not True
                ):
                    raise RuntimeError(
                        f"/enter accepted=false：{data}"
                    )

                break

            except (
                requests.Timeout,
                requests.ConnectionError,
            ) as exc:
                last_error = repr(exc)

                self._write_log(
                    path="/enter",
                    payload=payload,
                    http_status=None,
                    response=None,
                    error=last_error,
                )

                if attempt >= NETWORK_RETRIES:
                    raise

                time.sleep(
                    NETWORK_RETRY_SLEEP_S
                )

        assert isinstance(
            data,
            dict,
        )

        self.entered = True

        self.position = np.array(
            [0.0, 0.0],
            dtype=float,
        )

        self.current_channel = 1

        self.virtual_time = float(
            data["virtual_time_s"]
        )

        remaining = float(
            data["remaining_real_duration_s"]
        )

        self.deadline_monotonic = (
            time.monotonic()
            + remaining
        )

        print(
            f"[enter] run_id={self.run_id}, "
            f"remaining_real_duration_s={remaining:.1f}, "
            f"virtual_time={self.virtual_time:.3f}"
        )

        return data

    def measure(
        self,
        position: np.ndarray,
        channel: int,
    ) -> dict[str, Any]:
        rid = self._new_request_id(
            "measure"
        )

        payload = self._action_payload(
            rid,
            position,
            channel,
        )

        data = (
            self._post_same_action_with_retry(
                "/measure",
                payload,
            )
        )

        self.position = np.asarray(
            position,
            dtype=float,
        ).copy()

        self.current_channel = int(
            channel
        )

        self.virtual_time = float(
            data["virtual_time_s"]
        )

        result = data[
            "measure_result"
        ]

        if result == "direction":
            print(
                f"[measure] ch={channel:02d} "
                f"pos={tuple(np.round(position, 2))} "
                f"-> direction {data['svd_deg']:.2f}°, "
                f"vt={self.virtual_time:.2f}"
            )

        else:
            print(
                f"[measure] ch={channel:02d} "
                f"pos={tuple(np.round(position, 2))} "
                f"-> {result}, "
                f"vt={self.virtual_time:.2f}"
            )

        return data

    def clear(
        self,
        position: np.ndarray,
        channel: int,
    ) -> dict[str, Any]:
        rid = self._new_request_id(
            "clear"
        )

        payload = self._action_payload(
            rid,
            position,
            channel,
        )

        data = (
            self._post_same_action_with_retry(
                "/clear",
                payload,
            )
        )

        self.position = np.asarray(
            position,
            dtype=float,
        ).copy()

        self.virtual_time = float(
            data["virtual_time_s"]
        )

        print(
            f"[clear] ch={channel:02d} "
            f"pos={tuple(np.round(position, 2))} "
            f"-> {data['clear_result']}, "
            f"vt={self.virtual_time:.2f}"
        )

        return data

    def exit(self) -> Optional[dict[str, Any]]:
        if not self.entered:
            return None

        rid = self._new_request_id(
            "exit"
        )

        payload = self._base_payload(
            rid
        )

        try:
            r = self.session.post(
                self.base_url + "/exit",
                json=payload,
                timeout=HTTP_TIMEOUT_S,
            )

            try:
                data = r.json()
            except Exception:
                data = None

            self._write_log(
                path="/exit",
                payload=payload,
                http_status=r.status_code,
                response=data,
                error=None,
            )

            if (
                r.status_code == 200
                and isinstance(
                    data,
                    dict,
                )
                and data.get("accepted")
                is True
            ):
                self.virtual_time = float(
                    data["virtual_time_s"]
                )

                print(
                    f"[exit] reason={data.get('exit_reason')}, "
                    f"virtual_time={self.virtual_time:.2f}"
                )

                return data

            print(
                "[exit] 未成功接受，可能测试已结束。"
            )

            return data

        except Exception as exc:
            print(
                f"[exit] 接口可能已经关闭，无法退出：{exc}"
            )

            return None


@dataclass
class RouteNode:
    kind: str
    point: np.ndarray
    base_id: Optional[int] = None
    channel: Optional[int] = None


def shortest_open_base_route(
    current: np.ndarray,
    remaining_base_ids: list[int],
    base_points: list[np.ndarray],
) -> list[int]:
    if not remaining_base_ids:
        return []

    if len(remaining_base_ids) <= 8:
        best_perm: Optional[
            tuple[int, ...]
        ] = None

        best_len = float("inf")

        for perm in itertools.permutations(
            remaining_base_ids
        ):
            total = dist(
                current,
                base_points[perm[0]],
            )

            for a, b in zip(
                perm[:-1],
                perm[1:],
            ):
                total += dist(
                    base_points[a],
                    base_points[b],
                )

            if total < best_len:
                best_len = total
                best_perm = perm

        assert best_perm is not None

        return list(
            best_perm
        )

    left = set(
        remaining_base_ids
    )

    route: list[int] = []

    p = current.copy()

    while left:
        nxt = min(
            left,
            key=lambda i: dist(
                p,
                base_points[i],
            ),
        )

        route.append(nxt)
        p = base_points[nxt]
        left.remove(nxt)

    return route


def route_points(
    route: list[RouteNode],
) -> list[np.ndarray]:
    return [
        n.point
        for n in route
    ]


def best_insertion(
    current: np.ndarray,
    route: list[RouteNode],
    point: np.ndarray,
) -> tuple[float, int]:
    if not route:
        return dist(
            current,
            point,
        ), 0

    best_cost = float("inf")
    best_pos = 0

    cost = (
        dist(
            current,
            point,
        )
        + dist(
            point,
            route[0].point,
        )
        - dist(
            current,
            route[0].point,
        )
    )

    if cost < best_cost:
        best_cost = cost
        best_pos = 0

    for i in range(
        len(route) - 1
    ):
        a = route[i].point
        b = route[i + 1].point

        cost = (
            dist(a, point)
            + dist(point, b)
            - dist(a, b)
        )

        if cost < best_cost:
            best_cost = cost
            best_pos = i + 1

    cost = dist(
        route[-1].point,
        point,
    )

    if cost < best_cost:
        best_cost = cost
        best_pos = len(route)

    return best_cost, best_pos


def insert_tasks_greedily(
    current: np.ndarray,
    route: list[RouteNode],
    tasks: list[RouteNode],
) -> list[RouteNode]:
    route = list(route)
    pending = list(tasks)

    while pending:
        best = None

        for ti, task in enumerate(
            pending
        ):
            cost, pos = best_insertion(
                current,
                route,
                task.point,
            )


            key = (
                cost,
                0
                if task.kind == "CLEAR"
                else 1,
            )

            if (
                best is None
                or key < best[0]
            ):
                best = (
                    key,
                    ti,
                    pos,
                )

        assert best is not None

        _, ti, pos = best
        task = pending.pop(ti)

        route.insert(
            pos,
            task,
        )

    return route


class Problem3Controller:
    def __init__(
        self,
        client: RobotClient,
        outer_count: int,
        outer_radius: float,
        rotation_deg: float,
    ):
        self.client = client

        validate_backbone_or_raise(
            outer_count,
            outer_radius,
        )

        self.base_points = (
            regular_base_points(
                outer_count=outer_count,
                outer_radius=outer_radius,
                rotation_deg=rotation_deg,
            )
        )

        self.remaining_base_ids: set[int] = set(
            range(
                len(
                    self.base_points
                )
            )
        )

        self.visited_base_ids: set[int] = set()

        self.channels = {
            k: ChannelState(
                channel=k
            )
            for k in range(
                1,
                21,
            )
        }

        self.cleared_count = 0

        self.discovered_channels: set[int] = set()

        self.base_assignments: dict[
            int,
            list[int],
        ] = {}


    def searching_channels(
        self,
    ) -> list[int]:
        return [
            k
            for k, st in self.channels.items()
            if st.status == SEARCHING
        ]

    def localizing_channels(
        self,
    ) -> list[int]:
        return [
            k
            for k, st in self.channels.items()
            if st.status == LOCALIZING
        ]

    def clearable_channels(
        self,
    ) -> list[int]:
        return [
            k
            for k, st in self.channels.items()
            if st.status == CLEARABLE
        ]

    def done(self) -> bool:
        if self.cleared_count >= 16:
            return True

        if self.remaining_base_ids:
            return False

        for st in self.channels.values():
            if st.status == SEARCHING:
                st.status = ABSENT

        return all(
            st.status
            in (
                CLEARED,
                ABSENT,
            )
            for st in self.channels.values()
        )

    def discovered_count(
        self,
    ) -> int:
        return len(
            self.discovered_channels
        )

    def mark_search_complete_by_upper_bound(
        self,
    ) -> None:
        if (
            self.discovered_count()
            >= 16
        ):
            self.remaining_base_ids.clear()

            for st in self.channels.values():
                if st.status == SEARCHING:
                    st.status = ABSENT


    def _record_measurement(
        self,
        st: ChannelState,
        position: np.ndarray,
        result: str,
        angle_deg: Optional[float] = None,
    ) -> None:
        obs = {
            "position": [
                float(position[0]),
                float(position[1]),
            ],
            "result": result,
        }

        if angle_deg is not None:
            obs["angle_deg"] = float(
                angle_deg
            )

        st.observations.append(
            obs
        )

        st.measured_positions.append(
            np.asarray(
                position,
                dtype=float,
            ).copy()
        )

    def _set_region(
        self,
        st: ChannelState,
        new_region: BaseGeometry,
        reason: str,
    ) -> None:
        new_region = clean_geometry(
            new_region
        )

        if (
            new_region.is_empty
            and st.ever_detected
        ):
            print(
                f"[warning] ch={st.channel:02d} "
                f"region became empty after {reason}; "
                "keep previous conservative region."
            )
            return

        st.region = new_region
        st.region_version += 1
        st.invalidate_plan_cache()

        if (
            st.status == SEARCHING
            and st.region.is_empty
            and not st.ever_detected
        ):
            st.status = ABSENT

        st.recompute_geometry()

    def _mark_detected(
        self,
        st: ChannelState,
    ) -> None:
        if not st.ever_detected:
            st.ever_detected = True

            self.discovered_channels.add(
                st.channel
            )

        if (
            st.status
            not in (
                CLEARED,
                CLEARABLE,
            )
        ):
            st.status = LOCALIZING

    def handle_measure_response(
        self,
        channel: int,
        position: np.ndarray,
        response: dict[str, Any],
    ) -> None:
        st = self.channels[
            channel
        ]

        result = response[
            "measure_result"
        ]

        if result == "no_signal":
            self._record_measurement(
                st,
                position,
                "no_signal",
            )

            new_region = (
                update_region_no_signal(
                    st.region,
                    position,
                )
            )

            self._set_region(
                st,
                new_region,
                "no_signal",
            )

        elif result == "direction":
            angle = float(
                response["svd_deg"]
            )

            self._record_measurement(
                st,
                position,
                "direction",
                angle,
            )

            self._mark_detected(
                st
            )

            new_region = (
                update_region_direction(
                    st.region,
                    position,
                    angle,
                )
            )

            self._set_region(
                st,
                new_region,
                "direction",
            )

        elif result == "near":
            self._record_measurement(
                st,
                position,
                "near",
            )

            self._mark_detected(
                st
            )

            clear_response = (
                self.client.clear(
                    position,
                    channel,
                )
            )

            self.handle_clear_response(
                channel,
                position,
                clear_response,
            )

        else:
            raise RuntimeError(
                f"未知 measure_result={result!r}"
            )

        self.mark_search_complete_by_upper_bound()

    def handle_clear_response(
        self,
        channel: int,
        position: np.ndarray,
        response: dict[str, Any],
    ) -> None:
        st = self.channels[
            channel
        ]

        result = response[
            "clear_result"
        ]

        if result == "success":
            if st.status != CLEARED:
                self.cleared_count += 1

            st.status = CLEARED
            st.ever_detected = True

            self.discovered_channels.add(
                channel
            )

            print(
                f"[state] ch={channel:02d} CLEARED "
                f"({self.cleared_count} total)"
            )

        elif (
            result
            == "no_target_in_range"
        ):
            new_region = (
                update_region_clear_failure(
                    st.region,
                    position,
                )
            )

            self._set_region(
                st,
                new_region,
                "clear_failure",
            )

            st.status = LOCALIZING

            print(
                f"[state] ch={channel:02d} "
                "clear failed -> LOCALIZING"
            )

        else:
            raise RuntimeError(
                f"未知 clear_result={result!r}"
            )

        self.mark_search_complete_by_upper_bound()


    def _remaining_base_route(
        self,
    ) -> list[int]:
        return shortest_open_base_route(
            current=self.client.position,
            remaining_base_ids=sorted(
                self.remaining_base_ids
            ),
            base_points=self.base_points,
        )

    def _base_score(
        self,
        st: ChannelState,
        base_id: int,
    ) -> float:
        p = self.base_points[
            base_id
        ]

        if measured_before(
            st.measured_positions,
            p,
        ):
            return float("inf")

        if (
            max_distance_to_region(
                p,
                st.region,
            )
            > ROBUST_RX_LIMIT
        ):
            return float("inf")

        return problem2_score_existing_station(
            region=st.region,
            observations=st.observations,
            measured_positions=st.measured_positions,
            station=p,
        )

    def _active_options(
        self,
        st: ChannelState,
    ) -> list[ActiveOption]:
        if (
            st.active_cache_version
            == st.region_version
            and st.active_cache
        ):
            return st.active_cache

        options = compute_active_options(
            region=st.region,
            observations=st.observations,
            measured_positions=st.measured_positions,
        )

        st.active_cache_version = (
            st.region_version
        )

        st.active_cache = options

        return options

    def _choose_active_point_for_route(
        self,
        st: ChannelState,
        base_route_nodes: list[RouteNode],
    ) -> Optional[ActiveOption]:
        """Choose among near-optimal sensing points using route cost as a tiebreaker.

        This preserves localization quality while favoring points useful for later clearance.
        """
        options = self._active_options(
            st
        )

        if not options:
            return None

        best_score = options[
            0
        ].score

        threshold = (
            best_score
            * (
                1.0
                + ACTIVE_NEAR_OPT_REL
            )
            + 1e-9
        )

        near_opt = [
            o
            for o in options
            if o.score <= threshold
        ]

        clear_points: list[np.ndarray] = []

        for clear_ch in self.clearable_channels():
            clear_st = self.channels[
                clear_ch
            ]
            clear_st.recompute_geometry()

            if (
                clear_st.mec_center is not None
                and clear_st.mec_radius
                <= CLEAR_MEC_TRIGGER
            ):
                clear_points.append(
                    clear_st.mec_center.copy()
                )

        best: Optional[
            tuple[
                float,
                float,
                ActiveOption,
            ]
        ] = None

        for opt in near_opt:
            insertion, _ = best_insertion(
                self.client.position,
                base_route_nodes,
                opt.point,
            )

            if clear_points:
                nearest_clear = min(
                    dist(
                        opt.point,
                        clear_point,
                    )
                    for clear_point in clear_points
                )
            else:
                nearest_clear = 0.0

            route_score = (
                insertion
                + ACTIVE_CLEAR_BIAS
                * nearest_clear
            )


            key = (
                route_score,
                opt.score,
                opt,
            )

            if (
                best is None
                or key[:2] < best[:2]
            ):
                best = key

        assert best is not None

        return best[2]


    def _base_overlap_ratio(
        self,
        st: ChannelState,
        point: np.ndarray,
    ) -> float:
        """Estimate geometric overlap with the reception disk; this is not a probability."""
        if st.region.is_empty:
            return 0.0

        area = float(
            st.region.area
        )

        if area <= 1e-9:

            return (
                1.0
                if max_distance_to_region(
                    point,
                    st.region,
                )
                <= ROBUST_RX_LIMIT
                else 0.0
            )

        inside = (
            st.region.intersection(
                disk(
                    point,
                    NO_SIGNAL_REMOVE_RADIUS,
                )
            )
        )

        return float(
            inside.area / area
        )

    def _opportunistic_base_channels(
        self,
        base_id: int,
        already_scheduled: Iterable[int],
    ) -> list[int]:
        """Select extra channel measurements only at backbone nodes."""
        p = self.base_points[
            base_id
        ]

        excluded = set(
            int(x)
            for x in already_scheduled
        )

        scored: list[
            tuple[
                float,
                float,
                int,
            ]
        ] = []

        for ch in self.localizing_channels():
            if ch in excluded:
                continue

            st = self.channels[
                ch
            ]

            if measured_before(
                st.measured_positions,
                p,
            ):
                continue

            ratio = self._base_overlap_ratio(
                st,
                p,
            )

            if (
                ratio
                < BASE_OPPORTUNISTIC_MIN_OVERLAP
            ):
                continue


            scored.append(
                (
                    -ratio,
                    st.mec_radius,
                    ch,
                )
            )

        scored.sort()

        selected = [
            ch
            for _, _, ch in scored[
                :BASE_OPPORTUNISTIC_MAX_CHANNELS
            ]
        ]

        return selected


    def plan(
        self,
    ) -> list[RouteNode]:
        """Plan the backbone and active measurements before inserting clearance tasks.

        Only the first node is executed before feedback triggers replanning.
        """
        self.mark_search_complete_by_upper_bound()

        base_ids = self._remaining_base_route()

        base_route = [
            RouteNode(
                kind="BASE",
                point=self.base_points[i],
                base_id=i,
            )
            for i in base_ids
        ]

        self.base_assignments = {
            i: []
            for i in base_ids
        }

        active_tasks: list[
            RouteNode
        ] = []


        for ch in self.localizing_channels():
            st = self.channels[
                ch
            ]

            st.recompute_geometry()

            if st.status == CLEARABLE:
                continue

            base_scores: list[
                tuple[
                    float,
                    int,
                ]
            ] = []

            for base_id in base_ids:
                score = self._base_score(
                    st,
                    base_id,
                )

                if math.isfinite(score):
                    base_scores.append(
                        (
                            score,
                            base_id,
                        )
                    )

            base_scores.sort(
                key=lambda x: x[0]
            )

            active_options = (
                self._active_options(
                    st
                )
            )

            active_best_score = (
                active_options[0].score
                if active_options
                else float("inf")
            )

            use_base = False
            selected_base: Optional[
                int
            ] = None

            if base_scores:
                best_base_score = (
                    base_scores[0][0]
                )

                comparable_to_active = (
                    not math.isfinite(
                        active_best_score
                    )
                    or best_base_score
                    <= BASE_REUSE_FACTOR
                    * active_best_score
                )

                current_d = max(
                    st.diameter,
                    1e-9,
                )

                meaningful_reduction = (
                    best_base_score
                    <= (
                        1.0
                        - BASE_REUSE_MIN_REDUCTION
                    )
                    * current_d
                )

                if (
                    comparable_to_active
                    or meaningful_reduction
                ):
                    best_value = (
                        best_base_score
                    )

                    candidate_ids = {
                        bid
                        for score, bid
                        in base_scores
                        if score
                        <= 1.12
                        * best_value
                        + 1e-9
                    }

                    for bid in base_ids:
                        if bid in candidate_ids:
                            selected_base = bid
                            break

                    if selected_base is not None:
                        use_base = True

            if (
                use_base
                and selected_base
                is not None
            ):
                self.base_assignments[
                    selected_base
                ].append(ch)

                print(
                    f"[plan] ch={ch:02d} "
                    f"reuse BASE {selected_base}"
                )

                continue

            opt = (
                self._choose_active_point_for_route(
                    st,
                    base_route,
                )
            )

            if opt is not None:
                active_tasks.append(
                    RouteNode(
                        kind="ACTIVE",
                        point=opt.point,
                        channel=ch,
                    )
                )

                print(
                    f"[plan] ch={ch:02d} "
                    f"ACTIVE score={opt.score:.2f}"
                )

            else:
                mec = minimum_enclosing_circle(
                    st.region
                )

                p = mec.center

                if not measured_before(
                    st.measured_positions,
                    p,
                ):
                    active_tasks.append(
                        RouteNode(
                            kind="ACTIVE",
                            point=p,
                            channel=ch,
                        )
                    )

                    print(
                        f"[plan] ch={ch:02d} "
                        "fallback ACTIVE at MEC center"
                    )


        clear_tasks: list[
            RouteNode
        ] = []

        for ch in self.clearable_channels():
            st = self.channels[
                ch
            ]

            st.recompute_geometry()

            if (
                st.mec_center is not None
                and st.mec_radius
                <= CLEAR_MEC_TRIGGER
            ):
                clear_tasks.append(
                    RouteNode(
                        kind="CLEAR",
                        point=st.mec_center,
                        channel=ch,
                    )
                )


        core_route = insert_tasks_greedily(
            current=self.client.position,
            route=base_route,
            tasks=active_tasks,
        )


        if core_route:
            insert_now: list[
                RouteNode
            ] = []


            deferred: list[
                tuple[
                    float,
                    float,
                    float,
                    RouteNode,
                ]
            ] = []

            for task in clear_tasks:
                extra_dist, _ = best_insertion(
                    self.client.position,
                    core_route,
                    task.point,
                )

                current_dist = dist(
                    self.client.position,
                    task.point,
                )

                future_nearest = min(
                    dist(
                        task.point,
                        node.point,
                    )
                    for node in core_route
                )

                on_route = (
                    extra_dist
                    <= CLEAR_ROUTE_INSERT_MAX_M
                )

                escaping_region = (
                    current_dist
                    <= CLEAR_ESCAPE_CURRENT_MAX_M
                    and future_nearest
                    >= CLEAR_ESCAPE_FUTURE_MIN_M
                )

                if on_route:
                    insert_now.append(
                        task
                    )

                    print(
                        f"[plan] ch={task.channel:02d} "
                        f"CLEAR on-route "
                        f"(extra={extra_dist:.1f}m, "
                        f"current={current_dist:.1f}m, "
                        f"future_nearest={future_nearest:.1f}m)"
                    )

                elif escaping_region:
                    insert_now.append(
                        task
                    )

                    print(
                        f"[plan] ch={task.channel:02d} "
                        f"CLEAR before-leaving-region "
                        f"(extra={extra_dist:.1f}m, "
                        f"current={current_dist:.1f}m, "
                        f"future_nearest={future_nearest:.1f}m)"
                    )

                else:
                    deferred.append(
                        (
                            extra_dist,
                            current_dist,
                            future_nearest,
                            task,
                        )
                    )


            if (
                len(deferred)
                > MAX_DEFERRED_CLEAR
            ):

                deferred.sort(
                    key=lambda x: (
                        x[0],
                        x[1],
                    )
                )

                force_count = (
                    len(deferred)
                    - MAX_DEFERRED_CLEAR
                )

                forced = deferred[
                    :force_count
                ]

                deferred = deferred[
                    force_count:
                ]

                for (
                    extra_dist,
                    current_dist,
                    future_nearest,
                    task,
                ) in forced:
                    insert_now.append(
                        task
                    )

                    print(
                        f"[plan] ch={task.channel:02d} "
                        f"CLEAR forced-by-backlog "
                        f"(extra={extra_dist:.1f}m, "
                        f"current={current_dist:.1f}m, "
                        f"future_nearest={future_nearest:.1f}m, "
                        f"max_deferred={MAX_DEFERRED_CLEAR})"
                    )

            for (
                extra_dist,
                current_dist,
                future_nearest,
                task,
            ) in deferred:
                print(
                    f"[plan] ch={task.channel:02d} "
                    f"CLEAR deferred "
                    f"(extra={extra_dist:.1f}m, "
                    f"current={current_dist:.1f}m, "
                    f"future_nearest={future_nearest:.1f}m)"
                )

            route = insert_tasks_greedily(
                current=self.client.position,
                route=core_route,
                tasks=insert_now,
            )

        else:


            route = insert_tasks_greedily(
                current=self.client.position,
                route=[],
                tasks=clear_tasks,
            )

        return route


    def _measurement_order(
        self,
        channels: Iterable[int],
    ) -> list[int]:
        unique = sorted(
            set(
                int(x)
                for x in channels
            )
        )

        if (
            self.client.current_channel
            in unique
        ):
            unique.remove(
                self.client.current_channel
            )

            return [
                self.client.current_channel
            ] + unique

        return unique

    def visit_base(
        self,
        base_id: int,
    ) -> None:
        p = self.base_points[
            base_id
        ]

        search_tasks = (
            self.searching_channels()
        )

        localize_tasks = [
            ch
            for ch in self.base_assignments.get(
                base_id,
                [],
            )
            if self.channels[ch].status
            == LOCALIZING
        ]


        opportunistic_tasks = (
            self._opportunistic_base_channels(
                base_id,
                already_scheduled=(
                    search_tasks
                    + localize_tasks
                ),
            )
        )

        tasks = self._measurement_order(
            search_tasks
            + localize_tasks
            + opportunistic_tasks
        )

        print(
            f"[base] visit B{base_id}, "
            f"search={len(search_tasks)}, "
            f"reuse_localize={localize_tasks}, "
            f"opportunistic={opportunistic_tasks}"
        )

        for ch in tasks:
            if (
                self.client.real_time_remaining()
                <= REAL_TIME_SAFETY_MARGIN_S
            ):
                raise TimeoutError(
                    "现实剩余时间不足，停止新的检测动作。"
                )

            st = self.channels[
                ch
            ]

            if (
                st.status
                in (
                    CLEARED,
                    ABSENT,
                )
            ):
                continue

            if (
                st.status == SEARCHING
                and self.discovered_count()
                >= 16
            ):
                st.status = ABSENT
                continue


            resp = self.client.measure(
                p,
                ch,
            )

            self.handle_measure_response(
                ch,
                p,
                resp,
            )

        self.remaining_base_ids.discard(
            base_id
        )

        self.visited_base_ids.add(
            base_id
        )

        if not self.remaining_base_ids:
            for st in self.channels.values():
                if st.status == SEARCHING:
                    st.status = ABSENT

    def execute_active(
        self,
        node: RouteNode,
    ) -> None:
        assert node.channel is not None

        ch = node.channel
        st = self.channels[
            ch
        ]

        if st.status != LOCALIZING:
            return


        resp = self.client.measure(
            node.point,
            ch,
        )

        self.handle_measure_response(
            ch,
            node.point,
            resp,
        )

    def execute_clear(
        self,
        node: RouteNode,
    ) -> None:
        assert node.channel is not None

        ch = node.channel
        st = self.channels[
            ch
        ]

        if st.status != CLEARABLE:
            return

        resp = self.client.clear(
            node.point,
            ch,
        )

        self.handle_clear_response(
            ch,
            node.point,
            resp,
        )

    def execute_node(
        self,
        node: RouteNode,
    ) -> None:
        if node.kind == "BASE":
            assert node.base_id is not None
            self.visit_base(
                node.base_id
            )

        elif node.kind == "ACTIVE":
            self.execute_active(
                node
            )

        elif node.kind == "CLEAR":
            self.execute_clear(
                node
            )

        else:
            raise ValueError(
                f"unknown node kind={node.kind}"
            )


    def print_status(
        self,
    ) -> None:
        counts: dict[
            str,
            int,
        ] = {}

        for st in self.channels.values():
            counts[st.status] = (
                counts.get(
                    st.status,
                    0,
                )
                + 1
            )

        print(
            "[status]",
            counts,
            f"discovered={self.discovered_count()}",
            f"cleared={self.cleared_count}",
            f"remaining_base={sorted(self.remaining_base_ids)}",
            f"vt={self.client.virtual_time:.2f}",
            f"real_left={self.client.real_time_remaining():.1f}s",
        )

    def run(
        self,
    ) -> None:
        self.client.enter()

        try:
            while True:
                self.print_status()

                if self.done():
                    print(
                        "[done] completion certificate satisfied."
                    )
                    break

                if (
                    self.client.real_time_remaining()
                    <= REAL_TIME_SAFETY_MARGIN_S
                ):
                    print(
                        "[stop] real-time safety margin reached."
                    )
                    break

                route = self.plan()

                if not route:
                    unresolved = [
                        k
                        for k, st
                        in self.channels.items()
                        if st.status
                        not in (
                            CLEARED,
                            ABSENT,
                        )
                    ]

                    raise RuntimeError(
                        "规划器没有生成可执行节点，未完成频道："
                        f"{unresolved}"
                    )

                next_node = route[
                    0
                ]

                print(
                    f"[next] {next_node.kind} "
                    f"channel={next_node.channel} "
                    f"base={next_node.base_id} "
                    f"point={tuple(np.round(next_node.point, 2))}"
                )

                self.execute_node(
                    next_node
                )

        finally:
            self.print_status()
            self.client.exit()

            print(
                "\n=== FINAL SUMMARY ==="
            )

            print(
                "cleared channels:",
                [
                    k
                    for k, st
                    in self.channels.items()
                    if st.status == CLEARED
                ],
            )

            print(
                "absent channels:",
                [
                    k
                    for k, st
                    in self.channels.items()
                    if st.status == ABSENT
                ],
            )

            print(
                "unresolved channels:",
                [
                    k
                    for k, st
                    in self.channels.items()
                    if st.status
                    not in (
                        CLEARED,
                        ABSENT,
                    )
                ],
            )

            print(
                f"virtual_time_s="
                f"{self.client.virtual_time:.6f}"
            )

            print(
                f"run_id={self.client.run_id}"
            )

            print(
                f"log={self.client.log_path}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CUMCM 2026 B Problem 3 optimized robot controller"
        )
    )

    p.add_argument(
        "--robot-id",
        default=os.getenv(
            "CUMCM_ROBOT_ID",
            "",
        ),
        help=(
            "当前登录参赛队号；也可设置环境变量 CUMCM_ROBOT_ID"
        ),
    )

    p.add_argument(
        "--base-url",
        default=os.getenv(
            "CUMCM_BASE_URL",
            BASE_URL_DEFAULT,
        ),
        help=(
            "模拟器接口地址，默认 http://127.0.0.1:2026"
        ),
    )

    p.add_argument(
        "--log",
        default=DEFAULT_LOG,
        help=(
            "本地 JSONL 行为日志路径；默认每次运行使用带时间戳的新文件"
        ),
    )

    p.add_argument(
        "--outer-count",
        type=int,
        default=OUTER_COUNT,
        help="外围基准点个数",
    )

    p.add_argument(
        "--outer-radius",
        type=float,
        default=OUTER_RADIUS,
        help="外围正多边形外接圆半径/m",
    )

    p.add_argument(
        "--rotation-deg",
        type=float,
        default=OUTER_ROTATION_DEG,
        help="外围正多边形整体旋转角/deg",
    )

    p.add_argument(
        "--check-geometry",
        action="store_true",
        help="只验证基准点覆盖，不连接模拟器",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    validate_backbone_or_raise(
        args.outer_count,
        args.outer_radius,
    )

    pts = regular_base_points(
        args.outer_count,
        args.outer_radius,
        args.rotation_deg,
    )

    print(
        "[geometry] base points:"
    )

    for i, p in enumerate(pts):
        print(
            f"  B{i}: "
            f"({p[0]:.3f}, {p[1]:.3f})"
        )

    if args.check_geometry:
        print(
            "[geometry] PASS"
        )
        return

    if not args.robot_id:
        raise SystemExit(
            "请通过 --robot-id 或环境变量 CUMCM_ROBOT_ID "
            "设置当前登录参赛队号。"
        )

    client = RobotClient(
        base_url=args.base_url,
        robot_id=args.robot_id,
        log_path=Path(args.log),
    )

    controller = Problem3Controller(
        client=client,
        outer_count=args.outer_count,
        outer_radius=args.outer_radius,
        rotation_deg=args.rotation_deg,
    )

    controller.run()


if __name__ == "__main__":
    main()
