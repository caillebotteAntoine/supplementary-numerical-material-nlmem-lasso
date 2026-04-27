# ========================================== #
# Generation of Figure 2 in the paper:
#  Proportion of underselection (orange), correct support selection (green) and
# overselection (blue) in the nonlinear logistic mixed-effects model using the procedure
# NLMEM-LASSO for N ∈{100,200}, p ∈ {500,1000} and γ2
# 2 ∈ {152,302,452}.
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
        f"results/figure_6_NLMEM_N{N}J10P{P}COV{COV}VAR{VAR}.csv", true_theta, 8
    )


# =======================================#


def plot_score(results):
    results_df = [
        {k: res[k] for k in ("N", "P", "COV", "correct_support", "overselection")}
        for res in results
    ]
    results_df = pd.DataFrame(results_df)
    cov_labels = {"iid": "A", "ar": "B", "ar8": "C"}
    results_df["COV"] = results_df["COV"].map(cov_labels)

    fig, ax = plot_selection_score(
        results_df,
        group_by=3,
        space_between_groups=0.2,
        space_between_methods=0.08,
        legend_anchor=(0.5, -0.25),
        fontsize_labels=12,
        fontsize_group=14,
        height_factor_group_name=1.06,
        figsize=(6, 5),
        groups=[
            {"column": "N", "name": "N = ", "height": 1.09},
            {
                "column": "P",
                "name": "P = ",
                "height": 1.01,
                "do_frame": False,
                "space_between_frame": 0.5,
            },
            # {"column": "COV", "name": "", "height": 1.01, "do_frame": False},
        ],
    )
    ax.text(
        1.75,
        -0.15,
        "Scenario: A : iid, B : AR($\\rho = 0.6$), C : AR($\\rho = 0.8$)",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )

    xticks = ax.get_xticks()
    ax.set_xticks(xticks, ["A", "B", "C"] * 2)
    return fig, ax


# =======================================#


if __name__ == "__main__":
    results = []

    for N in (200,):
        for P in [500, 1000]:
            for COV in ["iid", "ar", "ar8"]:
                out = get_results(N, P, COV=COV)
                results.append({"N": N, "P": P, "COV": COV} | out)

    fig, _ = plot_score(results)
    # fig.savefig("NLMEM_HD_score_COV.png", bbox_inches="tight")
