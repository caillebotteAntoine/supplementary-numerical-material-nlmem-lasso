import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =======================================#
def fill_bar(ax, x, y, bottom=None, hatch="+", number_hatch=2, color="black"):
    hatch_height = 0.03

    if bottom is None:
        bottom = np.zeros_like(y)

    for xi, yi, bi in zip(x, y, bottom):
        for k in range(int(yi // hatch_height)):
            hatch_shift = k % 2 * "    "
            ax.text(
                xi,
                k * hatch_height + bi + hatch_height / 2,
                hatch_shift + f"{hatch}  " * number_hatch,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )


def detect_groups(values):
    groups_values = [values[0]]
    groups = [0]
    group_id = 0

    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            group_id += 1
            groups_values.append(values[i])
        groups.append(group_id)

    return groups, groups_values


def add_group_frame_by(
    ax, x, values, name, height, space_between_frame=0.5, do_frame=True, **kwargs
):

    groups, groups_values = detect_groups(values)
    for group in np.unique(groups):
        idx = [x[i] for i, g in enumerate(groups) if g == group]
        if not idx:
            continue
        left = min(idx) - space_between_frame
        right = max(idx) + space_between_frame
        if do_frame:
            rect = plt.Rectangle(
                (left, 0),
                right - left,
                height,
                fill=False,
                edgecolor="black",
                linewidth=1,
                linestyle="-",
            )
            ax.add_patch(rect)

        ax.text(
            (left + right) / 2,
            kwargs.get("height_factor_group_name", 1.05) * height,
            f"{name}{groups_values[group]}",
            ha="center",
            va="top",
            fontsize=kwargs.get("fontsize_group", 20),
            fontweight="bold",
            transform=ax.transData,
        )
    return ax


def plot_selection_score(
    results,
    width=0.6,
    space_between_groups=0.2,
    space_between_methods=0.05,
    group_by=1,
    groups=[
        {"column": "N", "name": "N = ", "height": 1.1},
        {
            "column": "P",
            "name": "P = ",
            "height": 1.025,
            "do_frame": False,
            "space_between_frame": 0.5,
        },
    ],
    ax=None,
    **kwargs,
):
    fill_colors = kwargs.get(
        "fill_colors", {"under": "#f7a278", "exact": "#c8e9a0", "over": "#6dd3ce"}
    )

    assert all(
        [x in fill_colors for x in ["under", "exact", "over"]]
    ), "fill_colors must contain keys 'under', 'exact' and 'over'"

    legend_anchor = kwargs.get("legend_anchor", (0.5, -0.2))
    fontsize_labels = kwargs.get("fontsize_labels", 12)

    # === Data preparation === #
    exact = results["correct_support"].values
    over = results["overselection"].values
    under = 1 - (exact + over)

    n_group = len(results) // group_by

    x = (space_between_methods + width) * np.arange(len(results))
    group_pos = np.arange(n_group) * space_between_groups

    x += np.repeat(group_pos, group_by)

    # === Plotting === #
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

    _ = ax.bar(
        x,
        under,
        width,
        label="Under selection",
        edgecolor="black",
        color=fill_colors["under"],
    )
    fill_bar(ax, x, under, hatch="─", number_hatch=kwargs.get("number_hatch_under", 2))

    bars_exact = ax.bar(
        x,
        exact,
        width,
        bottom=under,
        label="Support recovery",
        edgecolor="black",
        color=fill_colors["exact"],
    )

    _ = ax.bar(
        x,
        over,
        width,
        bottom=under + exact,
        label="Over selection",
        edgecolor="black",
        color=fill_colors["over"],
    )
    fill_bar(
        ax,
        x,
        over,
        bottom=under + exact,
        hatch="+",
        number_hatch=kwargs.get("number_hatch_over", 2),
    )

    # === value of correct support on bars === #
    for bar, value in zip(bars_exact, exact):
        height = bar.get_height()
        if value > 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                f"{int(100 * value)}%",
                ha="center",
                va="center",
                fontsize=fontsize_labels,
                color="black",
            )

    # === Frame groups by N === #
    for g in groups:
        add_group_frame_by(
            ax,
            x,
            results[g["column"]].values,
            g.get("name", f"{g['column']} = "),
            height=g["height"],
            space_between_frame=g.get(
                "space_between_frame", width / 2 + space_between_methods
            ),
            do_frame=g.get("do_frame", True),
            **kwargs,
        )

    # === axis labels and ticks === #
    ax.set_ylabel("Proportion (in %)", fontsize=fontsize_labels)
    if len(groups) > 0:
        ax.set_ylim(0, 1.1)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks, (100 * yticks).astype(int))

    if len(yticks) > 1:
        ax.set_yticks(yticks[:-1])

    ax.set_xticks(x, x, fontsize=fontsize_labels)

    # === Legend outside the plot === #
    handles = [
        Patch(facecolor=fill_colors["over"], edgecolor="black", label="Overselection"),
        Patch(
            facecolor=fill_colors["exact"], edgecolor="black", label="Support recovery"
        ),
        Patch(
            facecolor=fill_colors["under"], edgecolor="black", label="Underselection"
        ),
    ]
    ymax = ax.get_ylim()[1]
    ax.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=legend_anchor,
        frameon=False,
        fontsize=fontsize_labels,
    )

    # Remove top and right lines of the plot frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax.figure, ax
