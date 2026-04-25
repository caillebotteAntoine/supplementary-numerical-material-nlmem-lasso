# ========================================== #
# Generation of Figure 2 in the paper:
# Proportion of underselection (orange), correct support selection (green) and overselection
# (blue)using both procedures NLMEM-LASSO and glmmLasso for N ∈ {100, 200} and for
# p ∈ {200, 1000} in a linear mixed-effects model.
# ========================================== #

# pylint: skip-file

import pandas as pd
import numpy as np

from utils import get_selection_results, plot_selection_score

true_theta = np.array([2, 5, 1, 0, 0, 4, 1, 8, -10, 20])

param_cols = [
    r"$\mu_1$",
    r"$\mu_2$",
    r"$\gamma^2_1$",
    r"$\gamma^2_{1,2}$",
    r"$\gamma^2_{2,1}$",
    r"$\gamma^2_2$",
    r"$\sigma^2$",
    r"$\beta_{1}$",
    r"$\beta_{2}$",
    r"$\beta_{3}$",
]


def get_results(N, P, method=""):
    return get_selection_results(f"results/LMEM_N{N}J10P{P}{method}.csv", true_theta, 7)


# =======================================#


def plot_score(results):
    results_df = [
        {k: res[k] for k in ("N", "P", "correct_support", "overselection")}
        for res in results
    ]
    results_df = pd.DataFrame(results_df)
    fig, ax = plot_selection_score(
        results_df,
        group_by=4,
        space_between_groups=0.2,
        space_between_methods=0.08,
        legend_anchor=(0.5, -0.25),
        fontsize_labels=12,
        fontsize_group=14,
        height_factor_group_name=1.08,
        figsize=(9, 5),
        groups=[
            {"column": "N", "name": "N = ", "height": 1.1},
            {
                "column": "P",
                "name": "P = ",
                "height": 0.99,
                "do_frame": False,
                "space_between_frame": 0.5,
            },
        ],
    )
    ax.text(
        2.4,
        -0.13,
        "Scenario: A : NLMEM-LASSO, B : GlmmLasso",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )

    xticks = ax.get_xticks()
    ax.set_xticks(xticks, ["A", "B"] * 4)
    return fig, ax


# =======================================#
if __name__ == "__main__":
    results = []

    for N in [100, 200]:
        for P in [200, 1000]:
            for method in ["", "_glmmlasso"]:
                out = get_results(N, P, method)
                name = "GlmmLasso" if method == "_glmmlasso" else "NLMEM-LASSO"
                results.append({"N": N, "P": P, "method": name} | out)

    fig, _ = plot_score(results)
    # fig.savefig("LMEM_HD_score.png", bbox_inches="tight")
