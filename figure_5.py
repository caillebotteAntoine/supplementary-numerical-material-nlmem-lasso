# ========================================== #
# Generation of Figure 5 in the paper:
#  Proportion of underselection (orange), correct support selection (green) and
# overselection (blue) in the nonlinear logistic mixed-effects model using the procedure
# NLMEM-LASSO for N ∈{100,200}, p ∈ {500,1000} and γ2^2 ∈ {15^2,30^2,45^2}.
# ========================================== #

# pylint: skip-file

import pandas as pd
import numpy as np

from utils import get_selection_results, plot_selection_score

true_theta = np.array(
    [200.0, 1200.0, 49.0, 0, 0, 900.0, 300.0, 30.0, 120.0, 70.0, 40.0]
)


def get_results(N, P, COV="iid", VAR=30):
    true_theta[5] = VAR**2

    return get_selection_results(
        f"results/figure_5_NLMEM_N{N}J10P{P}COV{COV}VAR{VAR}.csv", true_theta, 8
    )


# =======================================#


def plot_score(results):
    results_df = [
        {k: res[k] for k in ("N", "P", "VAR", "correct_support", "overselection")}
        for res in results
    ]
    results_df = pd.DataFrame(results_df)

    fig, ax = plot_selection_score(
        results_df,
        group_by=3,
        space_between_groups=0.2,
        space_between_methods=0.08,
        legend_anchor=(0.5, -0.25),
        fontsize_labels=12,
        fontsize_group=14,
        height_factor_group_name=1.06,
        figsize=(12, 5),
    )
    var = [f"${v}^2$" for v in results_df["VAR"].values]
    ax.set_xticks(ax.get_xticks(), var)
    ax.set_xlabel("Variance $\gamma_1^2$ value")

    return fig, ax


# =======================================#

# =======================================#

if __name__ == "__main__":
    results = []

    for N in (100, 200):
        for p in (500, 1000):
            for VAR in (15, 30, 45):
                out = get_results(N, p, VAR=VAR)
                results.append({"N": N, "P": p, "VAR": VAR} | out)

    fig, _ = plot_score(results)
    # fig.savefig("NLMEM_HD_score_VAR.png", bbox_inches="tight")
