"""Visualization and test cases for problem1.py."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, pi, radians, sin, sqrt
from pathlib import Path
from typing import Optional

try:
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.patches import Circle
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 matplotlib。请先运行：python3 -m pip install matplotlib"
    ) from exc

from problem1 import (
    Boundary,
    CoverResult,
    DEFAULT_MAX_RECEIVE_RANGE,
    DiameterResult,
    Point,
    Region,
    TriangleMode,
    circle_point,
)


@dataclass(frozen=True)
class BearingMeasurement:

    station: Point
    alpha_deg: float
    error_deg: float = 1.0
    max_range: float = DEFAULT_MAX_RECEIVE_RANGE
    triangle_mode: TriangleMode = "inscribed"
    name: str = ""


@dataclass
class TestCase:
    title: str
    region: Region
    measurements: list[BearingMeasurement]
    true_source: Optional[Point] = None


def bearing_deg(station: Point, target: Point) -> float:
    return degrees(atan2(target.y - station.y, target.x - station.x))


def region_from_measurements(
    radius: float,
    measurements: list[BearingMeasurement],
) -> Region:
    region = Region(radius)
    for measurement in measurements:
        region.add_bearing_deg(
            measurement.station,
            measurement.alpha_deg,
            measurement.error_deg,
            measurement.max_range,
            measurement.triangle_mode,
        )
    return region


def add_convex_polygon(region: Region, vertices_ccw: list[Point]) -> None:
    """Convert a counterclockwise convex polygon into half-plane constraints."""
    for index, start in enumerate(vertices_ccw):
        end = vertices_ccw[(index + 1) % len(vertices_ccw)]
        dx = end.x - start.x
        dy = end.y - start.y


        region.add_halfplane(
            dy,
            -dx,
            dy * start.x - dx * start.y,
        )


def case_boundary_source() -> TestCase:
    radius = 1800.0
    true_source = Point(1710.0, 250.0)
    stations = [
        Point(550.0, -550.0),
        Point(700.0, 1100.0),
    ]
    measurements = [
        BearingMeasurement(
            station,
            bearing_deg(station, true_source),
            error_deg=3.0,
            name=f"S{index + 1}",
        )
        for index, station in enumerate(stations)
    ]
    return TestCase(
        title="近圆周干扰源与圆弧边界",
        region=region_from_measurements(radius, measurements),
        measurements=measurements,
        true_source=true_source,
    )


def configure_chinese_font() -> None:
    """Select an installed font that supports Chinese labels."""
    preferred = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def sample_arc(radius: float, start: float, end: float) -> list[Point]:
    """Sample an arc for plotting; region calculations remain analytic."""
    count = max(2, int(ceil((end - start) / radians(0.5))) + 1)
    return [
        circle_point(radius, start + (end - start) * i / (count - 1))
        for i in range(count)
    ]


def sampled_boundary_points(boundary: Boundary, radius: float) -> list[Point]:
    """Combine exact segment endpoints and sampled arcs into a fill boundary."""
    points: list[Point] = []
    for segment in boundary.segments:
        points.extend((segment.p, segment.q))
    for arc in boundary.arcs:
        points.extend(sample_arc(radius, arc.start, arc.end))
    points.extend(boundary.points)

    unique: list[Point] = []
    tolerance = 1.0e-7 * max(1.0, radius)
    for point in points:
        if not any(
            (point.x - old.x) ** 2 + (point.y - old.y) ** 2 <= tolerance**2
            for old in unique
        ):
            unique.append(point)

    if len(unique) <= 2:
        return unique

    center_x = sum(point.x for point in unique) / len(unique)
    center_y = sum(point.y for point in unique) / len(unique)
    return sorted(
        unique,
        key=lambda point: atan2(point.y - center_y, point.x - center_x),
    )


def draw_target_circle(ax: Axes, radius: float) -> None:
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            color="#707070",
            linewidth=1.4,
            linestyle="--",
            label="目标圆域边界",
            zorder=1,
        )
    )
    ax.scatter([0.0], [0.0], marker="+", s=55, color="#555555", zorder=4)


def draw_halfplane_lines(ax: Axes, region: Region, length: float) -> None:
    first = True
    for halfplane in region.halfplanes:
        foot = Point(halfplane.a * halfplane.c, halfplane.b * halfplane.c)
        direction = Point(-halfplane.b, halfplane.a)
        p = foot - direction * length
        q = foot + direction * length
        ax.plot(
            [p.x, q.x],
            [p.y, q.y],
            color="#B7B7B7",
            linewidth=0.8,
            linestyle=":",
            alpha=0.75,
            label="半平面边界" if first else None,
            zorder=0,
        )
        first = False


def draw_measurements(
    ax: Axes,
    measurements: list[BearingMeasurement],
) -> None:
    for index, measurement in enumerate(measurements):
        station = measurement.station
        alpha = radians(measurement.alpha_deg)
        error = radians(measurement.error_deg)

        if measurement.triangle_mode == "inscribed":
            side_length = measurement.max_range
            center_length = measurement.max_range * cos(error)
        else:
            side_length = measurement.max_range / cos(error)
            center_length = measurement.max_range

        triangle_vertices = [station]

        for boundary_index, theta in enumerate((alpha - error, alpha + error)):
            endpoint = Point(
                station.x + side_length * cos(theta),
                station.y + side_length * sin(theta),
            )
            triangle_vertices.append(endpoint)
            ax.plot(
                [station.x, endpoint.x],
                [station.y, endpoint.y],
                color="#E69F00",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                label="测向三角形边界" if index == 0 and boundary_index == 0 else None,
                zorder=2,
            )

        center_endpoint = Point(
            station.x + center_length * cos(alpha),
            station.y + center_length * sin(alpha),
        )
        ax.plot(
            [station.x, center_endpoint.x],
            [station.y, center_endpoint.y],
            color="#D55E00",
            linewidth=1.1,
            alpha=0.75,
            label="示向中心线" if index == 0 else None,
            zorder=2,
        )
        ax.plot(
            [triangle_vertices[1].x, triangle_vertices[2].x],
            [triangle_vertices[1].y, triangle_vertices[2].y],
            color="#E69F00",
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
            zorder=2,
        )
        ax.fill(
            [point.x for point in triangle_vertices],
            [point.y for point in triangle_vertices],
            color="#E69F00",
            alpha=0.035,
            zorder=1,
        )
        ax.scatter(
            [station.x],
            [station.y],
            marker="^",
            s=65,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.7,
            label="测量点" if index == 0 else None,
            zorder=7,
        )
        name = measurement.name or f"S{index + 1}"
        ax.annotate(
            name,
            (station.x, station.y),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=9,
            color="#8A3500",
            zorder=8,
        )


def draw_region_geometry(
    ax: Axes,
    region: Region,
    boundary: Boundary,
    diameter: DiameterResult,
    cover: CoverResult,
    true_source: Optional[Point],
) -> list[Point]:
    outline = sampled_boundary_points(boundary, region.radius)

    if len(outline) >= 3:
        ax.fill(
            [point.x for point in outline],
            [point.y for point in outline],
            color="#56B4E9",
            alpha=0.22,
            label="可行定位区域",
            zorder=3,
        )

    first_segment = True
    for segment in boundary.segments:
        ax.plot(
            [segment.p.x, segment.q.x],
            [segment.p.y, segment.q.y],
            color="#0072B2",
            linewidth=2.5,
            label="解析边界" if first_segment else None,
            zorder=5,
        )
        first_segment = False

    first_arc = first_segment
    for arc in boundary.arcs:
        points = sample_arc(region.radius, arc.start, arc.end)
        ax.plot(
            [point.x for point in points],
            [point.y for point in points],
            color="#0072B2",
            linewidth=2.5,
            label="解析边界" if first_arc else None,
            zorder=5,
        )
        first_arc = False

    if boundary.points:
        ax.scatter(
            [point.x for point in boundary.points],
            [point.y for point in boundary.points],
            s=20,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.5,
            label="边界候选点",
            zorder=6,
        )

    ax.add_patch(
        Circle(
            (cover.center.x, cover.center.y),
            cover.radius,
            color="#CC79A7",
            fill=True,
            alpha=0.10,
            linewidth=0.0,
            zorder=2,
        )
    )
    ax.add_patch(
        Circle(
            (cover.center.x, cover.center.y),
            cover.radius,
            color="#AA3377",
            fill=False,
            linewidth=1.8,
            linestyle="--",
            label="最远点对直径圆",
            zorder=4,
        )
    )

    ax.plot(
        [diameter.a.x, diameter.b.x],
        [diameter.a.y, diameter.b.y],
        color="#6A3D9A",
        linewidth=2.4,
        label="最远点对",
        zorder=7,
    )
    ax.scatter(
        [diameter.a.x, diameter.b.x],
        [diameter.a.y, diameter.b.y],
        s=60,
        color="#6A3D9A",
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )
    ax.scatter(
        [cover.center.x],
        [cover.center.y],
        marker="x",
        s=70,
        color="#AA3377",
        linewidth=2.0,
        label="直径圆圆心",
        zorder=8,
    )
    ax.plot(
        [cover.center.x, cover.worst_point.x],
        [cover.center.y, cover.worst_point.y],
        color="#009E73",
        linewidth=1.2,
        linestyle=":",
        zorder=6,
    )
    ax.scatter(
        [cover.worst_point.x],
        [cover.worst_point.y],
        marker="*",
        s=135,
        color="#009E73",
        edgecolor="white",
        linewidth=0.7,
        label="覆盖最不利点",
        zorder=9,
    )

    if true_source is not None:
        ax.scatter(
            [true_source.x],
            [true_source.y],
            marker="X",
            s=85,
            color="#C00000",
            edgecolor="white",
            linewidth=0.8,
            label="测试用真实源",
            zorder=9,
        )

    return outline


def set_equal_limits(ax: Axes, points: list[Point], minimum_span: float) -> None:
    if not points:
        points = [Point(-minimum_span / 2.0, 0.0), Point(minimum_span / 2.0, 0.0)]

    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, minimum_span)
    half = 0.62 * span

    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y - half, center_y + half)
    ax.set_aspect("equal", adjustable="box")


def finish_axis(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, pad=9)
    ax.set_xlabel("x（m）")
    ax.set_ylabel("y（m）")
    ax.grid(True, color="#DDDDDD", linewidth=0.65, alpha=0.65)
    ax.axhline(0.0, color="#AAAAAA", linewidth=0.6, zorder=-2)
    ax.axvline(0.0, color="#AAAAAA", linewidth=0.6, zorder=-2)


def unique_legend(ax: Axes, keep: list[str] | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle

    if keep is not None:
        unique = {k: v for k, v in unique.items() if k in keep}

    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="best",
            fontsize=8,
            framealpha=0.90,
            ncol=1,
        )


def visualize_case(
    case: TestCase,
    global_save_path: Optional[str] = None,
    local_save_path: Optional[str] = None,
    show: bool = True,
) -> tuple[DiameterResult, CoverResult]:
    """Render separate global-constraint and local-region figures."""
    configure_chinese_font()

    boundary = case.region.boundary()
    diameter = case.region.diameter(boundary)
    cover = case.region.check_diameter_circle(diameter, boundary)


    fig1, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    draw_target_circle(ax, case.region.radius)
    draw_halfplane_lines(ax, case.region, 3.0 * case.region.radius)
    draw_measurements(ax, case.measurements)
    draw_region_geometry(ax, case.region, boundary, diameter, cover, case.true_source)

    global_points = [
        Point(-case.region.radius, -case.region.radius),
        Point(case.region.radius, case.region.radius),
        *(m.station for m in case.measurements),
    ]
    set_equal_limits(ax, global_points, 2.2 * case.region.radius)
    finish_axis(ax, "测向约束与可行定位区域")
    unique_legend(ax, ["目标圆域边界", "测量点", "测向中心线", "可行定位区域"])

    if global_save_path:
        out = Path(global_save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out, dpi=300, bbox_inches="tight")
        print(f"全局图已保存：{out}")


    fig2, ax2 = plt.subplots(figsize=(5.8, 5.8))
    outline = draw_region_geometry(ax2, case.region, boundary, diameter, cover, case.true_source)

    local_points = [
        *outline,
        diameter.a,
        diameter.b,
        cover.center,
        cover.worst_point,
        Point(cover.center.x-cover.radius, cover.center.y-cover.radius),
        Point(cover.center.x+cover.radius, cover.center.y+cover.radius),
    ]
    set_equal_limits(ax2, local_points, 0.05 * case.region.radius)

    ax2.set_title("")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.grid(False)
    ax2.set_axis_off()

    if local_save_path:
        out = Path(local_save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"局部图已保存：{out}")

    print(f"区域直径 D = {diameter.distance:.6f} m")
    print("覆盖结论：", "能够覆盖" if cover.covered else "不能覆盖")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    return diameter, cover


if __name__ == "__main__":

    case = case_boundary_source()

    visualize_case(
        case,
        global_save_path="problem1_global_view.png",
        local_save_path="problem1_local_view.png",
        show=True,
    )
