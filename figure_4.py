# ========================================== #
# Generation of Figure 2 in the paper:
# Proportion of underselection (orange), correct support selection (green) and overselection
# (blue) in the non linear logistic mixed-effects model using the procedure NLMEM-LASSOfor
# N∈{100,200} and for p∈{500,1000,2000,6000}. The results are based on 100 simulated
# datasets.
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
        f"results/figure_4_NLMEM_N{N}J10P{P}COV{COV}VAR{VAR}.csv", true_theta, 8
    )


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
            {"column": "N", "name": "N = ", "height": 1.025},
        ],
    )
    xticks = ax.get_xticks()
    ax.set_xticks(xticks, ["500", "1000", "2000", "6000"] * 2)
    ax.set_xlabel("Number of covariates p", fontsize=14)
    return fig, ax


# =======================================#
if __name__ == "__main__":
    results = []

    for N in [100, 200]:
        for P in [500, 1000, 2000, 6000]:
            out = get_results(N, P)
            results.append({"N": N, "P": P} | out)

    fig, _ = plot_score(results)
    # fig.savefig("NLMEM_HD_score.png", bbox_inches="tight")
