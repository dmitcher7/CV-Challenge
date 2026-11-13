"""Heatmaps and average-position (formation) views drawn on the core.pitch minimap."""

import cv2
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

from core.pitch import LENGTH, WIDTH, draw_minimap  # noqa: E402
from .zones import ZONE_KEYS, ZONE_LABELS  # noqa: E402

TEAM_BGR = {"A": (220, 120, 40), "B": (50, 50, 230)}  # blue-ish, red-ish (BGR)


def heatmap(xs, ys, sigma_m=2.5):
    """Occupancy grid [68, 105] (rows = y, cols = x), 1 m bins, gaussian-blurred.
    Positions outside the pitch are clipped to the border. Away from the edges, sum == n samples."""
    x = np.clip(np.asarray(xs, float), 0, LENGTH - 1e-6)
    y = np.clip(np.asarray(ys, float), 0, WIDTH - 1e-6)
    grid, _, _ = np.histogram2d(y, x, bins=[int(WIDTH), int(LENGTH)], range=[[0, WIDTH], [0, LENGTH]])
    return gaussian_filter(grid, sigma_m, mode="constant") if sigma_m > 0 else grid


ARROW_LEN_M = 30.0      # length of the attacking-direction arrow
ZONE_COLOURS = {"standing": "#9aa5b1", "walking": "#4c9be8", "jogging": "#35b779", "running": "#f2a93b",
                "sprinting": "#d64545"}   # cool -> warm with speed


def draw_attack_arrow(image, direction, scale=10, margin=4):
    """Arrow along the pitch towards the attacked goal: direction +1 = towards x=105, -1 = towards x=0."""
    y = WIDTH - 6.0
    x0, x1 = LENGTH / 2 - direction * ARROW_LEN_M / 2, LENGTH / 2 + direction * ARROW_LEN_M / 2
    p0, p1 = [(int(round((x + margin) * scale)), int(round((y + margin) * scale))) for x in (x0, x1)]
    cv2.arrowedLine(image, p0, p1, (0, 0, 0), 9, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(image, p0, p1, (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.25)


def draw_heatmap(grid, title, path, scale=10, margin=4, attack_direction=None):
    """Overlay a heatmap grid on the pitch and write a PNG.
    attack_direction (+1 towards x=105, -1 towards x=0, None = no arrow) draws the attacking direction."""
    image = draw_minimap(scale, margin)
    if grid.max() > 0:
        norm = grid / grid.max()
        size = (int(LENGTH * scale), int(WIDTH * scale))
        big = np.clip(cv2.resize(norm.astype(np.float32), size, interpolation=cv2.INTER_CUBIC), 0, 1)
        colour = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        o = margin * scale
        roi = image[o:o + size[1], o:o + size[0]].astype(np.float32)
        alpha = np.clip(big * 1.5, 0, 0.85)[..., None]
        image[o:o + size[1], o:o + size[0]] = (roi * (1 - alpha) + colour * alpha).astype(np.uint8)
    if attack_direction:
        draw_attack_arrow(image, 1 if attack_direction > 0 else -1, scale, margin)
    cv2.putText(image, title, (margin * scale, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def zone_figure(zones, team_row, team_name, title):
    """Matplotlib figure: stacked bars of time per speed zone per track, and the team's distance per zone.
    zones = zone_table rows of ONE team; team_row = its row of team_zone_summary."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 1.3]})
    ids = [str(i) for i in zones["track_id"]]
    left = np.zeros(len(zones))
    for k in ZONE_KEYS:
        v = zones[f"time_{k}_s"].to_numpy(float)
        ax1.barh(ids, v, left=left, color=ZONE_COLOURS[k], label=ZONE_LABELS[k])
        left += v
    ax1.set_xlabel("tijd (s)")
    ax1.set_ylabel("track-id")
    ax1.set_title(f"{team_name}: tijd per snelheidszone per track")
    ax1.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    bottom = 0.0
    for k in ZONE_KEYS:
        d = float(team_row[f"dist_{k}_m"]) / 1000
        ax2.bar(["team"], [d], bottom=bottom, color=ZONE_COLOURS[k])
        bottom += d
    ax2.set_ylabel("afstand (km)")
    ax2.set_title("afstand per zone")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def draw_zone_bars(zones, team_row, team_name, title, path):
    """Write the zone_figure of one team as PNG."""
    fig = zone_figure(zones, team_row, team_name, title)
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def average_positions(tracks, table, team):
    """Mean (x, y) per kept track of one team -> list of (track_id, x, y)."""
    keep = set(table.loc[table["team"] == team, "track_id"])
    g = tracks[tracks["track_id"].isin(keep)].groupby("track_id")[["x_m", "y_m"]].mean()
    return [(int(i), float(r.x_m), float(r.y_m)) for i, r in g.iterrows()]


def draw_average_positions(positions_by_team, title, path, scale=10, margin=4):
    image = draw_minimap(scale, margin)
    for team, positions in positions_by_team.items():
        for tid, x, y in positions:
            c = (int(round((x + margin) * scale)), int(round((y + margin) * scale)))
            cv2.circle(image, c, 11, TEAM_BGR[team], -1, cv2.LINE_AA)
            cv2.circle(image, c, 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(tid), (c[0] + 13, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, title, (margin * scale, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image)
