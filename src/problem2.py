"""Robust selection of a second sensing station.

Angles use east as zero and increase counterclockwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, isfinite, pi, radians, sin, sqrt, tan
from typing import Sequence

from problem1 import Point, Region, TriangleMode, distance


@dataclass(frozen=True)
class Problem2Config:

    first_station: Point
    first_bearing_deg: float
    rho_minus: float
    rho_plus: float
    phi_0: float

    target_radius: float = 1800.0
    receive_radius_min: float = 1000.0
    receive_radius_max: float = 1500.0
    bearing_error_rad: float = radians(1.0)
    optical_radius: float = 20.0

    use_2d_receive_filter: bool = True
    limit_station_to_target_disk: bool = True
    use_optical_shortcut: bool = True
    triangle_mode: TriangleMode = "inscribed"

    n_a: int = 81
    n_b_per_side: int = 41
    n_rho: int = 31
    n_first_angle: int = 5
    n_second_error: int = 5
    top_k_for_exact: int = 40


@dataclass(frozen=True)
class Candidate:

    a: float
    b: float
    xy: Point


@dataclass(frozen=True)
class SelectionResult:

    best: Candidate
    score: float
    score_name: str
    candidates: tuple[Candidate, ...]
    proxy_scores: tuple[float, ...]
    exact_scores: tuple[float | None, ...]
    targets: tuple[Point, ...]


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("count 必须为正整数")
    if count == 1:
        return [(start + stop) / 2.0]
    step = (stop - start) / (count - 1)
    return [start + index * step for index in range(count)]


def unit_basis(theta_deg: float) -> tuple[Point, Point]:
    theta = radians(theta_deg)
    return Point(cos(theta), sin(theta)), Point(-sin(theta), cos(theta))


def local_to_global(config: Problem2Config, a: float, b: float) -> Point:
    u, v = unit_basis(config.first_bearing_deg)
    return config.first_station + a * u + b * v


def global_to_local(config: Problem2Config, point: Point) -> tuple[float, float]:
    u, v = unit_basis(config.first_bearing_deg)
    relative = point - config.first_station
    return (
        relative.x * u.x + relative.y * u.y,
        relative.x * v.x + relative.y * v.y,
    )


def wrap_deg(angle_deg: float) -> float:
    return angle_deg % 360.0


def validate_config(config: Problem2Config) -> tuple[float, float]:
    if not 0.0 <= config.rho_minus < config.rho_plus:
        raise ValueError("必须满足 0 <= rho_minus < rho_plus")
    if config.rho_plus > config.receive_radius_max + 1.0e-12:
        raise ValueError("rho_plus 不能超过第一次有效接收距离上界")
    if not 0.0 < config.phi_0 < pi / 2.0:
        raise ValueError("phi_0 必须位于 (0, pi/2) 内，单位为弧度")
    if config.target_radius <= 0.0:
        raise ValueError("target_radius 必须为正")
    if config.receive_radius_min <= 0.0:
        raise ValueError("receive_radius_min 必须为正")
    if config.receive_radius_max < config.receive_radius_min:
        raise ValueError("receive_radius_max 不能小于 receive_radius_min")
    if not 0.0 <= config.bearing_error_rad < pi / 2.0:
        raise ValueError("bearing_error_rad 必须位于 [0, pi/2) 内")
    if config.optical_radius < 0.0:
        raise ValueError("optical_radius 不能为负")
    if config.triangle_mode not in ("inscribed", "conservative"):
        raise ValueError("triangle_mode 只能是 'inscribed' 或 'conservative'")

    integer_parameters = {
        "n_a": config.n_a,
        "n_b_per_side": config.n_b_per_side,
        "n_rho": config.n_rho,
        "n_first_angle": config.n_first_angle,
        "n_second_error": config.n_second_error,
        "top_k_for_exact": config.top_k_for_exact,
    }
    for name, value in integer_parameters.items():
        if value <= 0:
            raise ValueError(f"{name} 必须为正整数")

    m = 0.5 * (config.rho_minus + config.rho_plus)
    h = 0.5 * (config.rho_plus - config.rho_minus)
    if h > config.receive_radius_min + 1.0e-12:
        raise ValueError("h > receive_radius_min，纯接收鲁棒区域为空")
    if h > config.receive_radius_min * cos(config.phi_0) + 1.0e-12:
        raise ValueError(
            "候选区域为空：h > receive_radius_min*cos(phi_0)。"
            "请降低 phi_0 或缩小 [rho_minus, rho_plus]。"
        )
    return m, h


def candidate_a_interval(config: Problem2Config) -> tuple[float, float]:
    m, h = validate_config(config)
    half_width = config.receive_radius_min * cos(config.phi_0) - h
    return m - half_width, m + half_width


def candidate_b_bounds(config: Problem2Config, a: float) -> tuple[float, float]:
    m, h = validate_config(config)
    q = h + abs(a - m)
    if q > config.receive_radius_min:
        raise ValueError("该 a 不满足鲁棒接收约束")
    b_min = q * tan(config.phi_0)
    b_max = sqrt(max(0.0, config.receive_radius_min**2 - q**2))
    if b_min > b_max + 1.0e-10:
        raise ValueError("该 a 下交会角约束与鲁棒接收约束不相容")
    return b_min, b_max


def analytic_centers(config: Problem2Config) -> list[Candidate]:
    """Return the two analytic centers on the robust reception boundary at a=m."""
    m, h = validate_config(config)
    b_star = sqrt(max(0.0, config.receive_radius_min**2 - h**2))
    return [
        Candidate(m, +b_star, local_to_global(config, m, +b_star)),
        Candidate(m, -b_star, local_to_global(config, m, -b_star)),
    ]


def maximum_guaranteed_angle(config: Problem2Config) -> float:
    """Return the maximum guaranteed worst-case intersection angle in radians."""
    _, h = validate_config(config)
    ratio = min(1.0, max(0.0, h / config.receive_radius_min))
    return atan2(sqrt(max(0.0, 1.0 - ratio**2)), ratio)


def candidate_grid(config: Problem2Config) -> list[Candidate]:
    """Generate candidate wings that satisfy reception and intersection-angle constraints."""
    m, h = validate_config(config)
    a_low, a_high = candidate_a_interval(config)
    candidates: list[Candidate] = []

    for a in linspace(a_low, a_high, config.n_a):
        b_min, b_max = candidate_b_bounds(config, a)
        if abs(b_max - b_min) <= 1.0e-12:
            absolute_b_values = [0.5 * (b_min + b_max)]
        else:
            absolute_b_values = linspace(b_min, b_max, config.n_b_per_side)

        for absolute_b in absolute_b_values:
            if absolute_b <= 1.0e-10:
                continue
            for sign in (-1.0, +1.0):
                b = sign * absolute_b
                q = h + abs(a - m)


                weak_physical_ok = abs(b) <= config.receive_radius_min + 1.0e-9


                robust_axis_ok = (
                    q**2 + b**2 <= config.receive_radius_min**2 + 1.0e-7
                )
                angle_ok = abs(b) + 1.0e-9 >= q * tan(config.phi_0)

                xy = local_to_global(config, a, b)
                disk_ok = (
                    not config.limit_station_to_target_disk
                    or hypot(xy.x, xy.y) <= config.target_radius + 1.0e-9
                )

                if weak_physical_ok and robust_axis_ok and angle_ok and disk_ok:
                    candidates.append(Candidate(a, b, xy))

    if not candidates:
        raise RuntimeError("离散候选点为空，请检查参数或提高网格分辨率")
    return candidates


def possible_target_scenarios(
    config: Problem2Config,
    two_dimensional: bool | None = None,
) -> list[Point]:
    """Sample target scenarios implied by the first bearing."""
    validate_config(config)
    if two_dimensional is None:
        two_dimensional = config.use_2d_receive_filter

    rhos = linspace(config.rho_minus, config.rho_plus, config.n_rho)
    offsets = (
        linspace(
            -config.bearing_error_rad,
            config.bearing_error_rad,
            config.n_first_angle,
        )
        if two_dimensional
        else [0.0]
    )

    theta_center = radians(config.first_bearing_deg)
    targets: list[Point] = []
    for rho in rhos:
        for offset in offsets:

            radial_distance = rho / cos(offset)
            theta = theta_center + offset
            target = config.first_station + Point(
                radial_distance * cos(theta),
                radial_distance * sin(theta),
            )
            if (
                radial_distance <= config.receive_radius_max + 1.0e-9
                and hypot(target.x, target.y) <= config.target_radius + 1.0e-9
            ):
                targets.append(target)

    if not targets:
        raise RuntimeError("可能目标场景为空，请检查首次检测点、示向度和 rho 区间")
    return targets


def passes_sampled_receive_check(
    config: Problem2Config,
    candidate: Candidate,
    targets: Sequence[Point],
) -> bool:
    """Check the reception limit against sampled two-dimensional target scenarios."""
    return max(distance(candidate.xy, target) for target in targets) <= (
        config.receive_radius_min + 1.0e-9
    )


def acute_intersection_angle(
    config: Problem2Config,
    second_station: Point,
    target: Point,
) -> float:
    first_vector = target - config.first_station
    second_vector = target - second_station
    first_norm = hypot(first_vector.x, first_vector.y)
    second_norm = hypot(second_vector.x, second_vector.y)
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        return pi / 2.0

    cosine = abs(
        (first_vector.x * second_vector.x + first_vector.y * second_vector.y)
        / (first_norm * second_norm)
    )
    cosine = min(1.0, max(0.0, cosine))
    return atan2(sqrt(max(0.0, 1.0 - cosine**2)), cosine)


def linearized_diameter(
    config: Problem2Config,
    second_station: Point,
    target: Point,
) -> float:
    """Approximate the two-bearing region diameter with a local error-band model."""
    r1 = distance(target, config.first_station)
    r2 = distance(target, second_station)
    if config.use_optical_shortcut and r2 <= config.optical_radius:
        return 0.0

    phi = acute_intersection_angle(config, second_station, target)
    sine = abs(sin(phi))
    if sine <= 1.0e-10:
        return float("inf")

    w1 = r1 * tan(config.bearing_error_rad)
    w2 = r2 * tan(config.bearing_error_rad)
    return 2.0 / sine * sqrt(
        w1**2 + w2**2 + 2.0 * abs(cos(phi)) * w1 * w2
    )


def proxy_robust_score(
    config: Problem2Config,
    candidate: Candidate,
    targets: Sequence[Point],
) -> float:
    return max(linearized_diameter(config, candidate.xy, target) for target in targets)


class Problem1Adapter:
    """Use problem1.py to compute the exact two-bearing region diameter."""

    def __init__(self, config: Problem2Config):
        self.config = config

    def diameter_from_measurements(
        self,
        measurements: Sequence[tuple[Point, float]],
    ) -> float:
        region = Region(radius=self.config.target_radius)
        for station, measured_deg in measurements:
            region.add_bearing_deg(
                station,
                measured_deg,
                error_deg=degrees(self.config.bearing_error_rad),
                max_range=self.config.receive_radius_max,
                triangle_mode=self.config.triangle_mode,
            )
        boundary = region.boundary()
        return region.diameter(boundary).distance


def exact_robust_score(
    config: Problem2Config,
    adapter: Problem1Adapter,
    candidate: Candidate,
    targets: Sequence[Point],
) -> float:
    """Evaluate J(S2) over sampled target positions and second-bearing errors."""
    second_errors = linspace(
        -config.bearing_error_rad,
        config.bearing_error_rad,
        config.n_second_error,
    )
    worst = 0.0

    for target in targets:
        r2 = distance(target, candidate.xy)
        if config.use_optical_shortcut and r2 <= config.optical_radius:
            scenario_value = 0.0
        else:
            true_second_deg = degrees(
                atan2(target.y - candidate.xy.y, target.x - candidate.xy.x)
            )
            scenario_value = 0.0
            for error_rad in second_errors:
                measured_second_deg = wrap_deg(true_second_deg + degrees(error_rad))
                measurements = [
                    (config.first_station, wrap_deg(config.first_bearing_deg)),
                    (candidate.xy, measured_second_deg),
                ]
                try:
                    diameter = adapter.diameter_from_measurements(measurements)
                except ValueError:
                    diameter = float("inf")
                scenario_value = max(scenario_value, diameter)
        worst = max(worst, scenario_value)
    return worst


def choose_second_station(
    config: Problem2Config,
    use_problem1: bool = False,
    adapter: Problem1Adapter | None = None,
) -> SelectionResult:
    """Choose the second station with the smallest worst-case localization diameter."""
    targets = possible_target_scenarios(config)
    candidates = candidate_grid(config)

    if config.use_2d_receive_filter:
        candidates = [
            candidate
            for candidate in candidates
            if passes_sampled_receive_check(config, candidate, targets)
        ]
        if not candidates:
            raise RuntimeError(
                "加入二维角域宽度后候选点为空。可降低 phi_0、缩小 rho 区间，"
                "或提高候选网格分辨率。"
            )

    proxy_scores = [
        proxy_robust_score(config, candidate, targets) for candidate in candidates
    ]
    exact_scores: list[float | None] = [None] * len(candidates)

    if not use_problem1:
        best_index = min(range(len(candidates)), key=proxy_scores.__getitem__)
        return SelectionResult(
            best=candidates[best_index],
            score=proxy_scores[best_index],
            score_name="线性化最坏直径",
            candidates=tuple(candidates),
            proxy_scores=tuple(proxy_scores),
            exact_scores=tuple(exact_scores),
            targets=tuple(targets),
        )

    if adapter is None:
        adapter = Problem1Adapter(config)

    ranked_indices = sorted(range(len(candidates)), key=proxy_scores.__getitem__)
    selected_indices = ranked_indices[: min(config.top_k_for_exact, len(candidates))]
    for index in selected_indices:
        exact_scores[index] = exact_robust_score(
            config,
            adapter,
            candidates[index],
            targets,
        )

    finite_indices = [
        index
        for index in selected_indices
        if exact_scores[index] is not None and isfinite(exact_scores[index])
    ]
    if not finite_indices:
        raise RuntimeError("问题一算法未得到有限直径，请检查角度约定和模型参数")

    best_index = min(
        finite_indices,
        key=lambda index: (
            exact_scores[index]
            if exact_scores[index] is not None
            else float("inf")
        ),
    )
    return SelectionResult(
        best=candidates[best_index],
        score=float(exact_scores[best_index]),
        score_name="问题一离散鲁棒直径",
        candidates=tuple(candidates),
        proxy_scores=tuple(proxy_scores),
        exact_scores=tuple(exact_scores),
        targets=tuple(targets),
    )


__all__ = [
    "Candidate",
    "Problem1Adapter",
    "Problem2Config",
    "SelectionResult",
    "acute_intersection_angle",
    "analytic_centers",
    "candidate_a_interval",
    "candidate_b_bounds",
    "candidate_grid",
    "choose_second_station",
    "exact_robust_score",
    "global_to_local",
    "linearized_diameter",
    "linspace",
    "local_to_global",
    "maximum_guaranteed_angle",
    "passes_sampled_receive_check",
    "possible_target_scenarios",
    "proxy_robust_score",
    "unit_basis",
    "validate_config",
    "wrap_deg",
]
