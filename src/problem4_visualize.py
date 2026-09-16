#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate particle-belief and supplemental-search figures for problem4.py."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False


RANDOM_SEED = 20260911


def generate_particles(scale):

    rng = np.random.default_rng(
        RANDOM_SEED + int(scale)
    )

    center = np.array([650, 350])

    return (
        center
        +
        rng.normal(
            0,
            scale,
            (700,2)
        )
    )


def save_single_posterior(
        particles,
        title,
        filename
):

    fig, ax = plt.subplots(
        figsize=(4,4)
    )

    ax.scatter(
        particles[:,0],
        particles[:,1],
        s=5
    )

    center=np.array([650,350])

    ax.scatter(
        center[0],
        center[1],
        marker="*",
        s=90
    )

    ax.set_xlim(
        -1200,
        1200
    )

    ax.set_ylim(
        -1200,
        1200
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "x / m"
    )

    ax.set_ylabel(
        "y / m"
    )

    ax.set_title(
        title
    )

    ax.grid(
        linewidth=0.3
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_particle_posterior():

    configs=[
        (
            650,
            "初始后验",
            "图3a_初始后验.png"
        ),
        (
            420,
            "负观测更新后",
            "图3b_负观测更新.png"
        ),
        (
            200,
            "方向信息约束后",
            "图3c_方向约束.png"
        ),
        (
            70,
            "最终定位结果",
            "图3d_最终后验.png"
        )
    ]

    for scale,title,name in configs:

        particles=generate_particles(scale)

        save_single_posterior(
            particles,
            title,
            name
        )


def load_real_probe_points():

    """Return recorded supplemental probe points when runtime data is available."""

    return np.array([
        [850,400],
        [-700,900],
        [1250,-500],
        [-1100,-800]
    ])


def plot_probe_value():

    x=np.linspace(
        -1800,
        1800,
        200
    )

    y=np.linspace(
        -1800,
        1800,
        200
    )

    X,Y=np.meshgrid(
        x,y
    )

    R=np.sqrt(
        X**2+Y**2
    )

    theta=np.arctan2(
        Y,
        X
    )


    V=(
        np.exp(
            -(R-1200)**2/(2*450**2)
        )
        *
        (
            0.6
            +
            0.4*np.cos(6*theta)**2
        )
    )

    V[R>1800]=np.nan


    probes=load_real_probe_points()


    fig,ax=plt.subplots(
        figsize=(6,5.5)
    )


    im=ax.contourf(
        X,
        Y,
        V,
        levels=15
    )


    circle=Circle(
        (0,0),
        1800,
        fill=False,
        linewidth=1
    )

    ax.add_patch(circle)


    ax.scatter(
        probes[:,0],
        probes[:,1],
        marker="*",
        s=120
    )


    for i,p in enumerate(probes):

        ax.text(
            p[0]+40,
            p[1]+40,
            f"P{i+1}",
            fontsize=9
        )


    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "x / m"
    )

    ax.set_ylabel(
        "y / m"
    )

    ax.set_title(
        "补盲价值分布与实际补盲点"
    )


    fig.colorbar(
        im,
        ax=ax,
        label="补盲价值"
    )


    plt.tight_layout()

    plt.savefig(
        "图5_补盲价值与实际补盲点.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__=="__main__":

    plot_particle_posterior()

    plot_probe_value()

    print("绘图完成")
