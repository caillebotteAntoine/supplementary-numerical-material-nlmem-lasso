# ========================================== #
# Generation of Figure 7 in the paper:
# Proportion of underselection (orange), correct support selection (green) and overselection
# (blue) for the nlm-glmnet, Saemix-glmnet and NLMEM-LASSOprocedures under increasing
# missing data rates in the pharmacological model.
# ========================================== #

# pylint: skip-file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import get_selection_results, plot_selection_score

true_theta = np.array(
    [200.0, 1200.0, 49.0, 0, 0, 900.0, 300.0, 30.0, 120.0, 70.0, 40.0]
)

# =======================================#


def plot_score_slide_full_data(results):
    results_df = results.rename(
        columns={"exact": "correct_support", "over": "overselection"}
    )

    results_df["overselection"] = (
        results_df["overselection"] - results_df["correct_support"]
    )

    results_df = results_df.sort_values(by=["para_ind", "partial"]).reset_index(
        drop=True
    )

    cov_labels = {"g": "A", "g_saemix": "B", "sgd": "C"}
    results_df["method"] = results_df["method"].map(cov_labels)

    fig, ax = plot_selection_score(
        results_df,
        group_by=3,
        space_between_groups=0.2,
        space_between_methods=0.0,
        legend_anchor=(0.5, -0.25),
        fontsize_labels=12,
        width=0.8,
        groups=[
            {"column": "method", "name": "", "height": 1.04, "do_frame": False},
        ],
        figsize=(6, 5),
        number_hatch_over=6,
    )
    ax.text(
        1,
        -0.15,
        "Methods: \nA : nlm-glmnet, B : Saemix-glmnet, C : NLMEM-LASSO",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="black",
    )

    # var = [f"${v*10}\%$" for v in [0, 1, 2, 3, 4]]
    # x = [0.8 * k + 0.2 * (k // 3) for k in [1, 4, 7, 10, 13]]
    ax.set_xticks([], [])
    # ax.set_xlabel("Percentage of partial observations")

    return fig, ax


def plot_score(results, ax=None):
    results_df = results.rename(
        columns={"exact": "correct_support", "over": "overselection"}
    )

    results_df["overselection"] = (
        results_df["overselection"] - results_df["correct_support"]
    )

    results_df = results_df.sort_values(by=["para_ind", "partial"]).reset_index(
        drop=True
    )

    cov_labels = {"g": "A", "g_saemix": "B", "sgd": "C"}
    results_df["method"] = results_df["method"].map(cov_labels)

    fig, ax = plot_selection_score(
        results_df,
        group_by=3,
        space_between_groups=0.2,
        space_between_methods=0.0,
        legend_anchor=(0.5, -0.25),
        fontsize_labels=13,
        width=0.8,
        groups=[
            {"column": "method", "name": "", "height": 1.01, "do_frame": False},
            {"column": "para_ind_name", "name": "", "height": 1.07, "do_frame": False},
        ],
        ax=ax,
        figsize=(10, 8),
    )
    ax.text(
        6,
        -0.15,
        "Methods: \nA : nlm-glmnet, B : Saemix-glmnet, C : NLMEM-LASSO",
        ha="center",
        va="center",
        fontsize=18,
        # fontweight="bold",
        color="black",
    )

    var = [f"${v*10}\%$" for v in [0, 1, 2, 3, 4]]
    x = [0.8 * k + 0.2 * (k // 3) for k in [1, 4, 7, 10, 13]]
    ax.set_xticks(x, var, size=16)
    ax.set_xlabel("Percentage of partial observations", size=16)

    return fig, ax


def plot_score_both_para(results):
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

    _, _ = plot_score(results.loc[results["para_ind"] == 1], ax=axes[0])
    _, _ = plot_score(results.loc[results["para_ind"] == 2], ax=axes[1])
    return fig, axes


# =======================================#

results = pd.read_csv(f"results/figure_7_two_step_comparaison.csv", sep=";")
results.loc[results["para_ind"] == 1, "para_ind_name"] = "First individual parameter"
results.loc[results["para_ind"] == 2, "para_ind_name"] = "Second individual parameter"

fig, _ = plot_score_both_para(results)
# fig.savefig("two_step_comparison.png", bbox_inches="tight", dpi=300)
