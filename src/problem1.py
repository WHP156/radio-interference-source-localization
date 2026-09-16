from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, radians, sin, sqrt
from typing import Iterable, Literal, Optional


TAU = 2.0 * pi
DEFAULT_MAX_RECEIVE_RANGE = 1500.0
TriangleMode = Literal["inscribed", "conservative"]


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Point":
        return Point(self.x * k, self.y * k)

    __rmul__ = __mul__


def dot(p: Point, q: Point) -> float:
    return p.x * q.x + p.y * q.y


def norm(p: Point) -> float:
    return hypot(p.x, p.y)


def distance(p: Point, q: Point) -> float:
    return norm(p - q)


def circle_point(radius: float, theta: float) -> Point:
    return Point(radius * cos(theta), radius * sin(theta))


def normalize_angle(theta: float) -> float:
    return theta % TAU


@dataclass(frozen=True)
class HalfPlane:
    """A normalized half-plane represented by a*x + b*y <= c."""

    a: float
    b: float
    c: float

    @staticmethod
    def make(a: float, b: float, c: float) -> "HalfPlane":
        length = hypot(a, b)
        if length == 0.0:
            raise ValueError("半平面法向量不能为零")
        return HalfPlane(a / length, b / length, c / length)

    def contains(self, p: Point, eps_len: float) -> bool:
        return self.a * p.x + self.b * p.y <= self.c + eps_len


@dataclass(frozen=True)
class Segment:
    p: Point
    q: Point


@dataclass(frozen=True)
class Arc:
    """A closed counterclockwise arc on the target circle. end may exceed 2*pi for wraparound."""

    start: float
    end: float

    def contains_angle(self, theta: float, eps_ang: float) -> bool:
        t = normalize_angle(theta)
        if t < self.start - eps_ang:
            t += TAU
        return t <= self.end + eps_ang


@dataclass
class Boundary:
    segments: list[Segment]
    arcs: list[Arc]
    points: list[Point]


@dataclass(frozen=True)
class DiameterResult:
    a: Point
    b: Point
    distance: float


@dataclass(frozen=True)
class CoverResult:
    covered: bool
    center: Point
    radius: float
    max_distance: float
    worst_point: Point
    margin: float


class Region:
    def __init__(self, radius: float = 1800.0):
        if radius <= 0.0:
            raise ValueError("目标圆半径必须为正")
        self.radius = radius
        self.halfplanes: list[HalfPlane] = []


        self.eps_len = 1.0e-9 * max(1.0, radius)
        self.eps_ang = 1.0e-12

    def add_halfplane(self, a: float, b: float, c: float) -> None:
        self.halfplanes.append(HalfPlane.make(a, b, c))

    def add_direction_wedge(
        self,
        station: Point,
        alpha_rad: float,
        error_rad: float = pi / 180.0,
    ) -> None:
        """Add the two bearing-wedge boundaries without a range limit."""
        beta_low = alpha_rad - error_rad
        beta_high = alpha_rad + error_rad

        a1 = sin(beta_low)
        b1 = -cos(beta_low)
        c1 = a1 * station.x + b1 * station.y

        a2 = -sin(beta_high)
        b2 = cos(beta_high)
        c2 = a2 * station.x + b2 * station.y

        self.add_halfplane(a1, b1, c1)
        self.add_halfplane(a2, b2, c2)

    def add_direction_wedge_deg(
        self,
        station: Point,
        alpha_deg: float,
        error_deg: float = 1.0,
    ) -> None:
        self.add_direction_wedge(station, radians(alpha_deg), radians(error_deg))

    def add_bearing(
        self,
        station: Point,
        alpha_rad: float,
        error_rad: float = pi / 180.0,
        max_range: float = DEFAULT_MAX_RECEIVE_RANGE,
        triangle_mode: TriangleMode = "inscribed",
    ) -> None:
        """Add the three half-planes for a bearing and its range limit.

        The inscribed mode uses sides of max_range; conservative mode encloses the exact sector.
        """
        if max_range <= 0.0:
            raise ValueError("有效接收距离上界必须为正")
        if not 0.0 <= error_rad < pi / 2.0:
            raise ValueError("测向误差角必须位于 [0, pi/2) 内")
        if triangle_mode not in ("inscribed", "conservative"):
            raise ValueError(
                "triangle_mode 只能是 'inscribed' 或 'conservative'"
            )

        self.add_direction_wedge(station, alpha_rad, error_rad)

        direction_x = cos(alpha_rad)
        direction_y = sin(alpha_rad)
        if triangle_mode == "inscribed":
            cutoff = max_range * cos(error_rad)
        else:
            cutoff = max_range


        c = (
            direction_x * station.x
            + direction_y * station.y
            + cutoff
        )
        self.add_halfplane(direction_x, direction_y, c)

    def add_bearing_deg(
        self,
        station: Point,
        alpha_deg: float,
        error_deg: float = 1.0,
        max_range: float = DEFAULT_MAX_RECEIVE_RANGE,
        triangle_mode: TriangleMode = "inscribed",
    ) -> None:
        self.add_bearing(
            station,
            radians(alpha_deg),
            radians(error_deg),
            max_range,
            triangle_mode,
        )

    def contains(self, p: Point) -> bool:
        if dot(p, p) > self.radius * self.radius + 2.0 * self.radius * self.eps_len:
            return False
        return all(h.contains(p, self.eps_len) for h in self.halfplanes)

    def _line_circle_points(self, h: HalfPlane) -> list[Point]:
        r = self.radius
        if abs(h.c) > r + self.eps_len:
            return []

        foot = Point(h.a * h.c, h.b * h.c)
        tangent_sq = max(0.0, r * r - h.c * h.c)
        offset = sqrt(tangent_sq)
        direction = Point(-h.b, h.a)

        p = foot + direction * offset
        if offset <= self.eps_len:
            return [p]
        q = foot - direction * offset
        return [p, q]

    def _extract_segments(self) -> tuple[list[Segment], list[Point]]:
        """Clip each half-plane boundary to its feasible one-dimensional interval."""
        r = self.radius
        segments: list[Segment] = []
        degenerate_points: list[Point] = []

        for h in self.halfplanes:
            if abs(h.c) > r + self.eps_len:
                continue

            p0 = Point(h.a * h.c, h.b * h.c)
            direction = Point(-h.b, h.a)
            extent = sqrt(max(0.0, r * r - h.c * h.c))
            low = -extent
            high = extent
            feasible = True

            for g in self.halfplanes:
                q = g.a * direction.x + g.b * direction.y
                rhs = g.c - g.a * p0.x - g.b * p0.y

                if abs(q) <= self.eps_ang:
                    if rhs < -self.eps_len:
                        feasible = False
                        break
                    continue

                bound = rhs / q
                if q > 0.0:
                    high = min(high, bound)
                else:
                    low = max(low, bound)

                if low > high + self.eps_len:
                    feasible = False
                    break

            if not feasible:
                continue

            if high - low <= self.eps_len:
                p = p0 + direction * ((low + high) / 2.0)
                if self.contains(p):
                    degenerate_points.append(p)
            else:
                p = p0 + direction * low
                q = p0 + direction * high
                if self.contains(p) and self.contains(q):
                    segments.append(Segment(p, q))

        return self._deduplicate_segments(segments), degenerate_points

    def _extract_arcs(self) -> list[Arc]:
        break_angles: list[float] = []
        for h in self.halfplanes:
            for p in self._line_circle_points(h):
                break_angles.append(normalize_angle(atan2(p.y, p.x)))

        break_angles = self._deduplicate_angles(break_angles)

        if not break_angles:
            if self.contains(Point(self.radius, 0.0)):
                return [Arc(0.0, TAU)]
            return []

        arcs: list[Arc] = []
        count = len(break_angles)
        for i, start in enumerate(break_angles):
            end = break_angles[(i + 1) % count]
            if i == count - 1:
                end += TAU
            elif end <= start:
                end += TAU

            middle = (start + end) / 2.0
            if self.contains(circle_point(self.radius, middle)):
                arcs.append(Arc(start, end))

        return arcs

    def boundary(self) -> Boundary:
        segments, degenerate = self._extract_segments()
        arcs = self._extract_arcs()

        points = list(degenerate)
        for s in segments:
            points.extend((s.p, s.q))
        for arc in arcs:
            points.append(circle_point(self.radius, arc.start))
            points.append(circle_point(self.radius, arc.end))

        points = self._deduplicate_points(points)
        if not segments and not arcs and not points:
            raise ValueError("定位区域为空：请检查测向数据、角度约定和误差模型")

        return Boundary(segments=segments, arcs=arcs, points=points)

    def _find_antipodal_arc_pair(
        self, arcs: Iterable[Arc]
    ) -> Optional[tuple[Point, Point]]:
        """Find two feasible arc points separated by pi, if they exist."""
        arc_list = list(arcs)
        for first in arc_list:
            for second in arc_list:


                for k in range(-2, 3):
                    shifted_low = second.start - pi + k * TAU
                    shifted_high = second.end - pi + k * TAU
                    low = max(first.start, shifted_low)
                    high = min(first.end, shifted_high)
                    if low <= high + self.eps_ang:
                        theta = (low + high) / 2.0
                        p = circle_point(self.radius, theta)
                        q = circle_point(self.radius, theta + pi)
                        if self.contains(p) and self.contains(q):
                            return p, q
        return None

    def diameter(self, boundary: Optional[Boundary] = None) -> DiameterResult:
        if boundary is None:
            boundary = self.boundary()

        antipodal = self._find_antipodal_arc_pair(boundary.arcs)
        if antipodal is not None:
            return DiameterResult(antipodal[0], antipodal[1], 2.0 * self.radius)

        candidates = list(boundary.points)
        if not candidates:
            raise ValueError("非空区域没有可用边界候选点")

        best_a = candidates[0]
        best_b = candidates[0]
        best_sq = 0.0

        def consider(p: Point, q: Point) -> None:
            nonlocal best_a, best_b, best_sq
            d = p - q
            value = dot(d, d)
            if value > best_sq:
                best_sq = value
                best_a, best_b = p, q


        for i, p in enumerate(candidates):
            for q in candidates[i + 1 :]:
                consider(p, q)


        for p in candidates:
            length = norm(p)
            if length <= self.eps_len:
                continue
            q = Point(-self.radius * p.x / length, -self.radius * p.y / length)
            if self.contains(q):
                consider(p, q)

        return DiameterResult(best_a, best_b, sqrt(best_sq))

    def check_diameter_circle(
        self,
        diameter: Optional[DiameterResult] = None,
        boundary: Optional[Boundary] = None,
    ) -> CoverResult:
        if boundary is None:
            boundary = self.boundary()
        if diameter is None:
            diameter = self.diameter(boundary)

        center = (diameter.a + diameter.b) * 0.5
        radius = diameter.distance * 0.5

        candidates = list(boundary.points)


        if not candidates and boundary.arcs:
            candidates.append(circle_point(self.radius, boundary.arcs[0].start))
        if not candidates:
            raise ValueError("非空区域没有可用覆盖候选点")


        center_norm = norm(center)
        if center_norm > self.eps_len:
            opposite = Point(
                -self.radius * center.x / center_norm,
                -self.radius * center.y / center_norm,
            )
            if self.contains(opposite):
                candidates.append(opposite)

        worst_point = candidates[0]
        max_distance = distance(center, worst_point)
        for p in candidates[1:]:
            current = distance(center, p)
            if current > max_distance:
                max_distance = current
                worst_point = p

        cover_eps = 10.0 * self.eps_len
        margin = radius - max_distance
        return CoverResult(
            covered=max_distance <= radius + cover_eps,
            center=center,
            radius=radius,
            max_distance=max_distance,
            worst_point=worst_point,
            margin=margin,
        )

    def _deduplicate_points(self, points: Iterable[Point]) -> list[Point]:
        result: list[Point] = []
        eps = 10.0 * self.eps_len
        for p in points:
            if not any(distance(p, q) <= eps for q in result):
                result.append(p)
        return result

    def _deduplicate_angles(self, angles: Iterable[float]) -> list[float]:
        values = sorted(normalize_angle(a) for a in angles)
        result: list[float] = []
        for value in values:
            if not result or abs(value - result[-1]) > self.eps_ang:
                result.append(value)


        if len(result) >= 2 and TAU - result[-1] + result[0] <= self.eps_ang:
            result[0] = 0.5 * (result[0] + result[-1] - TAU) % TAU
            result.pop()
            result.sort()
        return result

    def _deduplicate_segments(self, segments: Iterable[Segment]) -> list[Segment]:
        result: list[Segment] = []
        eps = 10.0 * self.eps_len
        for segment in segments:
            duplicate = False
            for old in result:
                same_order = (
                    distance(segment.p, old.p) <= eps
                    and distance(segment.q, old.q) <= eps
                )
                reverse_order = (
                    distance(segment.p, old.q) <= eps
                    and distance(segment.q, old.p) <= eps
                )
                if same_order or reverse_order:
                    duplicate = True
                    break
            if not duplicate:
                result.append(segment)
        return result
