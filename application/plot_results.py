"""Plotting functions for senescence selection results."""

import jax.numpy as jnp

import sdg4varselect.plotting as sdgplt
from sdg4varselect.outputs import RegularizationPath

from . import model_params_names
from . import load_data

from . import RESULTS_DIR


def plot_chr_regpath(chr_name: str, folder: str, seed=None, save_fig: bool = False):
    """Plotting regularization path for a given chromosome."""
    data, _ = load_data(chr_name)
    _, p = data["cov"].shape

    regpath_name = (
        RESULTS_DIR
        / folder
        / "results"
        / (f"senescence_chr{chr_name}_res" + (f"_{seed}" if seed is not None else ""))
    )
    regpath = RegularizationPath.load(regpath_name.as_posix())

    regpath = regpath.standardize()
    fig = sdgplt.plot_regpath(regpath, P=p, fig=sdgplt.figure(6, 6))
    fig.suptitle(f"Senescence Model - chr{chr_name}")
    if save_fig:
        fig_name = (
            RESULTS_DIR / folder / "fig" / f"senescence_chr{chr_name}_regpath.png"
        )
        fig.savefig(fig_name.as_posix())

    sdgplt.plt.show()

    return fig


def do_plot(regpath, data, beta_init, model, chr_name, save_fig: bool = False):
    """Plotting results for senescence selection.

    Parameters
    ----------
    regpath : RegularizationPath
        The regularization path results.
    data : dict
        The data used for the model.
    beta_init : jnp.ndarray
        The initial beta values.
    model : SenescenceModel
        The senescence model used.
    save_fig : bool, optional
        Whether to save the figures, by default False.
    """

    fig = sdgplt.plot_regpath(regpath, P=model.P, fig=sdgplt.figure(6, 6))
    if save_fig:
        fig.savefig(f"results/senescence_chr{chr_name}_regpath.png")

    fig = sdgplt.figure(6, 5)
    _ = sdgplt.plot_theta(
        regpath,
        fig=[*fig.subfigures(1, 2)],
        params_names=model_params_names,
        id_to_plot=[
            [0, 1, 2, 5, 6],
            [7, 8, 9, 10, 11],
        ],
        log_scale=False,
    )
    fig.suptitle(f"Senescence Model - chr{chr_name}")
    if save_fig:
        fig.savefig(f"results/senescence_chr{chr_name}_theta1.png")

    fig = sdgplt.figure(6, 5)
    _ = sdgplt.plot_theta(
        regpath,
        fig=[*fig.subfigures(1, 2)],
        params_names=model_params_names,
        id_to_plot=[
            [12 + i for i in range(5)],
            [17 + i for i in range(5)],
        ],
        log_scale=False,
    )
    fig.suptitle(f"Senescence Model - chr{chr_name}")
    if save_fig:
        fig.savefig(f"results/senescence_chr{chr_name}_theta2.png")

    # ebic_argmin = jnp.argmin(regpath.ebic)
    # y_predict = model.mixed_effect_function(
    #     params=model.parametrization.reals1d_to_params(
    #         regpath[ebic_argmin].last_theta_reals1d
    #     ),
    #     times=data["mem_obs_time"],
    #     **data,
    #     **regpath[ebic_argmin].latent_variables,
    # )

    # ax = sdgplt.ax(5, 5)
    # _ = ax.plot(data["mem_obs_time"].T, data["Y"].T, "o")
    # _ = ax.plot(data["mem_obs_time"].T, y_predict.T, "-", alpha=0.8)
    # _ = ax.set_title(f"Prediction : Senescence - chr{chr_name}")
    # _ = ax.set_xlabel("Time")
    # _ = ax.set_ylabel("Senescence")

    know_n_qtl_indices = jnp.where(beta_init == 0.25)[0]

    for k in range(len(know_n_qtl_indices) // 10 + 1):
        ii = 12 + know_n_qtl_indices[k * 10 : (k + 1) * 10]

        fig = sdgplt.figure(6, 5)
        _ = sdgplt.plot_theta(
            regpath,
            fig=[*fig.subfigures(1, 2)],
            params_names=model_params_names,
            id_to_plot=[
                [ii[i] for i in range(5)],
                [ii[i] for i in range(5, 10)],
            ],
            log_scale=False,
        )
        fig.suptitle(f"Senescence Model - chr{chr_name} - known QTLs {k}")
        if save_fig:
            fig.savefig(f"results/senescence_chr{chr_name}_known_qtl_theta{k}.png")

    sdgplt.plt.show()
