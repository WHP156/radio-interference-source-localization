import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


events = [

    (0,0,"start"),

    (575,-996,"measure"),
    (419,-1157,"clear"),

    (-575,-996,"measure"),
    (-414,-150,"clear"),
    (-768,56,"clear"),
    (-649,144,"clear"),

    (-575,996,"measure"),
    (-470,995,"clear"),

    (445,1318,"clear"),
    (394,1311,"clear"),
    (16,1263,"clear"),

    (1150,0,"measure"),
    (1142,-167,"clear"),
    (1216,-505,"clear"),

    (1216,-505,"exit")
]


x=np.array([e[0] for e in events])
y=np.array([e[1] for e in events])


fig,ax=plt.subplots(
    figsize=(7,7),
    dpi=300
)


arena_fill = plt.Circle(
    (0,0),
    1800,
    color="#f7fbff",
    alpha=1,
    zorder=0
)

ax.add_patch(arena_fill)


arena = plt.Circle(
    (0,0),
    1800,
    fill=False,
    linestyle="--",
    linewidth=1.5,
    color="#636363",
    zorder=5
)

ax.add_patch(arena)


R=1150

base_points=[]

for k in range(6):

    theta=np.deg2rad(k*60)

    base_points.append(
        (
            R*np.cos(theta),
            R*np.sin(theta)
        )
    )


hex_patch=Polygon(
    base_points,
    closed=True,
    facecolor="#9ecae1",
    alpha=0.22,
    edgecolor="#6baed6",
    linestyle="--",
    linewidth=1.5,
    zorder=1
)

ax.add_patch(hex_patch)


for i in range(6):

    p1=np.array(base_points[i])
    p2=np.array(base_points[(i+1)%6])

    poly=np.vstack([
        p1,
        p2,
        [0,0]
    ])

    patch=Polygon(
        poly,
        facecolor="#deebf7",
        alpha=0.18,
        edgecolor="none",
        zorder=0
    )

    ax.add_patch(patch)


for bx,by in base_points:

    ax.scatter(
        bx,
        by,
        s=45,
        color="#6baed6",
        edgecolors="white",
        zorder=3
    )


ax.scatter(
    0,
    0,
    s=45,
    color="#6baed6",
    edgecolors="white",
    zorder=3
)


clear_radius=50


for px,py,t in events:

    if t=="clear":

        c=plt.Circle(
            (px,py),
            clear_radius,
            color="#fb6a4a",
            alpha=0.18,
            zorder=2
        )

        ax.add_patch(c)


ax.plot(
    x,
    y,
    color="#2171b5",
    linewidth=2.3,
    zorder=4
)


for i in range(len(x)-1):

    dx=x[i+1]-x[i]
    dy=y[i+1]-y[i]

    ax.arrow(
        x[i],
        y[i],
        dx*0.55,
        dy*0.55,
        head_width=35,
        head_length=60,
        color="#2171b5",
        alpha=0.45,
        length_includes_head=True,
        zorder=4
    )


for px,py,t in events:


    if t=="measure":

        ax.scatter(
            px,
            py,
            s=90,
            marker="o",
            color="#41ab5d",
            edgecolors="white",
            zorder=6
        )


    elif t=="clear":

        ax.scatter(
            px,
            py,
            s=110,
            marker="^",
            color="#de2d26",
            edgecolors="white",
            zorder=6
        )


    elif t=="start":

        ax.scatter(
            px,
            py,
            s=150,
            marker="*",
            color="#08519c",
            zorder=6
        )


    elif t=="exit":

        ax.scatter(
            px,
            py,
            s=130,
            marker="X",
            color="#54278f",
            zorder=6
        )


ax.set_xlim(-1900,1900)
ax.set_ylim(-1900,1900)

ax.set_aspect("equal")


ax.set_xlabel("x / m")
ax.set_ylabel("y / m")


ax.grid(
    linestyle=":",
    alpha=0.25
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.tight_layout()


plt.savefig(
    "problem3_region_trajectory_map.png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()
