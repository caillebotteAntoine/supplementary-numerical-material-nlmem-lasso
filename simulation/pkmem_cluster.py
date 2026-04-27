# pylint: skip-file
# pylint: skip-file

##############################

import os, sys

NTHREADS = 1
print(f"Nombre de threads alloués : {NTHREADS}")

os.environ["MKL_NUM_THREADS"] = f"{NTHREADS}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NTHREADS}"
os.environ["OMP_NUM_THREADS"] = f"{NTHREADS}"

# os.environ["NUM_INTER_THREADS"]=f"{NTHREADS}" # J'ai trouvé ces commandes,
# os.environ["NUM_INTRA_THREADS"]=f"{NTHREADS}" # mais je n'ai pas observé de changement en pratique

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " f"intra_op_parallelism_threads={NTHREADS} "
)

##############################

import numpy as np

import pandas as pd

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.outputs import MultiGDResults
from sdg4varselect._regularization_function import regularization_path
import sdg4varselect.plotting as sdgplt

from phd_code_results.chapter_3_mem_selection.models.estim_selection_fct import (
    algo_factory,
    strong_selection_estim,
    pkm_selection_estim,
)
from phd_code_results.chapter_3_mem_selection.models.pkmem import (
    PKMixedEffectsModel,
    p_star,
    p_names,
)


# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #

seed, N, P = 0, 200, 500
p_partial = 0
# seed = int(sys.argv[1]) - 1
# N = int(sys.argv[2])
# P = int(sys.argv[3]) * 10
# p_partial = int(sys.argv[4])


myModel = PKMixedEffectsModel(N=N, P=P * 2, D=100, V=30)

p_star = myModel.new_params(
    mean_latent={"ka": 6, "Cl": 8},
    cov_latent=jnp.diag(
        jnp.array([0.2, 0.1])
    ),  # jnp.array([[0.2, 0.05], [0.05, 0.1]]),
    var_residual=1e-3,
    beta1=jnp.concatenate(
        (
            jnp.array([3, 2, 1, 0, 0]),
            jnp.zeros(
                shape=P - 5,
            ),
        )
    ),
    beta2=jnp.concatenate(
        (
            jnp.array([0, 0, 3, 2, 1]),
            jnp.zeros(
                shape=P - 5,
            ),
        )
    ),
)

myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0))

ax = sdgplt.ax(4, 6)
_ = ax.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "-", alpha = 0.8)
ax.set_xlabel("Time")
ax.set_ylabel("Concentration")
# Remove top and right lines of the plot frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.figure.savefig(f"PKMEM_N{myModel.N}_J12_P{myModel.P}_sample.png", dpi=300)


p_names = np.array(
    ["$\\mu_1$", "$\\mu_2$"]
    + [f"$\\omega^2_{{{i//(2)},{i%(2)}}}$" for i in range(2**2)]
    + ["$\\sigma^2$"]
    + [f"$\\beta_{{1,{i}}}$" for i in range(myModel.P // 2)]
    + [f"$\\beta_{{2,{i}}}$" for i in range(myModel.P // 2)],
)


cov = pd.read_csv(
    f"simulation_files/PKMEM_N300_J12_P1000_cov.csv",
    sep=";",
    decimal=".",
)
cov = cov.loc[cov["seed"] == seed]
cov.head()

data = pd.read_csv(
    f"simulation_files/PKMEM_N300_J12_P1000_data.csv",
    sep=";",
    decimal=".",
)
data = data.loc[data["seed"] == seed]
data = data.loc[data["id"] < myModel.N]
data.head()

shape = (myModel.N, 12)
myobs = {
    "mem_obs_time": jnp.array(data.times).reshape(shape),
    "Y": jnp.array(data.Y).reshape(shape),
    "cov1": jnp.array(cov.iloc[:, 1 : (myModel.P // 2 + 1)]),
    "cov2": jnp.array(cov.iloc[:, 1 : (myModel.P // 2 + 1)]),
}

N1 = int((p_partial / 10) * myModel.N)
myobs["mem_obs_time"] = myobs["mem_obs_time"].at[:N1, 4:].set(jnp.nan)
myobs["Y"] = myobs["Y"].at[:N1, 4:].set(jnp.nan)

_ = sdgplt.ax(4, 4).plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
# sdgplt.plt.savefig(f"results_pkm/PKMEM_N{myModel.N}_J12_P{myModel.P}_C{p_partial}_sample.png", dpi=300)
# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #

algo = algo_factory(myModel, "adagrad", max_iter=1000)
algo.estimate_average_length = 1

algo_estim = algo_factory(myModel, "adagrad", max_iter=1000)

algo.save_all = True
algo_estim.save_all = True

if seed > 0:
    algo.save_all = False
    algo_estim.save_all = False

algo_estim._preconditioner._scale = jnp.concatenate(
    [
        1 * jnp.ones(shape=(2,)),
        2 * jnp.ones(shape=(2,)),
        1 * jnp.ones(shape=(1,)),
        0.4 * jnp.ones(myModel.P),
    ]
)
algo._preconditioner._scale = jnp.concatenate(
    [
        2 * jnp.ones(shape=(2,)),
        2 * jnp.ones(shape=(2,)),
        1 * jnp.ones(shape=(1,)),
        2 * jnp.ones(myModel.P),
    ]
)


def estim(prngkey, lbd):
    out = strong_selection_estim(
        selection_estim=pkm_selection_estim,
        algo_select=algo,
        algo_estim=algo_estim,
        prngkey=prngkey,
        model=myModel,
        data=myobs,
        lbd=lbd,
    )
    out.theta_star = p_star
    return out


from tests.phd_code_results.chapter_1_jm_estimation.models.estim_selection_fct import (
    _one_estim,
)

prngkey_list = jrd.split(jrd.PRNGKey(seed), num=1)
x = MultiGDResults(
    results=[
        estim(prngkey_list[i], lbd=2 * 1e-1) for i in range(0)
    ]  # len(prngkey_list))]
)


if __name__ == "__main___":
    lbd_set = jnp.logspace(-1, 1, num=5)

    # lbd_set = jnp.array([2 * 1e-2])
    x = regularization_path(
        strong_selection_estim,
        jrd.PRNGKey(seed),
        lbd_set,
        selection_estim=pkm_selection_estim,
        algo_select=algo,
        algo_estim=algo_estim,
        model=myModel,
        data=myobs,
        ntry=1,
    )
    x.theta_star = p_star
    # x.save(myModel.name, f"results_pkm/N{myModel.N}P{myModel.P}C{p_partial}_{seed}")

if __name__ == "__main___":

    y = x.standardize()
    y.update_bic(myModel)
    fig = sdgplt.figure(6, 6)
    fig = sdgplt.plot_regpath(y, P=myModel.P, fig=fig)
    _ = fig.axes[1].legend(loc="best")
    _ = fig.axes[1].scatter(y.lbd_set, y.ebic, marker="d", linewidths=5)

    # sdgplt.plt.savefig(f"results_pkm/PKMEM_N{myModel.N}_J12_P{myModel.P}C{p_partial}_reg.png")


if seed == 0:
    fig = sdgplt.figure(6 * 2, 2 * 16)
    _ = sdgplt.plot_theta(
        x,
        fig=[*fig.subfigures(1, 6)],
        params_names=p_names,
        id_to_plot=[
            [0, 1, 6],
            [2, 3, 4, 5],
            [7, 8, 9],
            [10, 11, 12],
            # [13, 14, 15],
            # [16, 17, 18],
            [9 + myModel.P // 2, 9 + myModel.P // 2 + 1, 9 + myModel.P // 2 + 2],
            [9 + myModel.P // 2 + 3, 9 + myModel.P // 2 + 4, 9 + myModel.P // 2 + 5],
        ],
        log_scale=False,
    )

    # sdgplt.plt.savefig(f"results_pkm/PKMEM_N{myModel.N}_J12_P{myModel.P}C{p_partial}_ld.png")

    ax = sdgplt.ax(6, 6)
    _ = ax.plot(x.theta[1, :, 10 : 9 + myModel.P // 2])
    # sdgplt.plt.savefig(f"results_pkm/PKMEM_N{myModel.N}_J12_P{myModel.P}C{p_partial}_hd_1.png")
    _ = ax.plot(x.theta[1, :, 9 + myModel.P // 2 + 3 :])
    # sdgplt.plt.savefig(f"results_pkm/PKMEM_N{myModel.N}_J12_P{myModel.P}C{p_partial}_hd_2.png")


# # Extraire les valeurs nécessaires
# iii = 1
# grad = x[iii].grad[1:,]  # shape: (n_iter, param_size)
# grad_precond = x[iii].grad_precond  # shape: (n_iter, param_size)
# theta = x[iii].theta  # shape: (n_iter, param_size)
# adagrad_past = jnp.array(algo._preconditioner._past)  # grad_precond / grad
# window_size = lbd_set[iii] / jnp.sqrt(
#     adagrad_past + 1e-8
# )  # shape: (n_iter, param_size)

# # On choisit les indices des paramètres à afficher (par exemple, les betas)
# param_indices = list(
#     range(myModel.parametrization_size)
# )  # ou [2, 3, 4, 5] pour les betas
# # param_indices = [999]


# fig, axes = sdgplt.plt.subplots(5, 1, figsize=(10, 16), sharex=True)

# for idx in param_indices:
#     axes[0].plot(grad[:, idx], label=f"grad[{idx}]")
#     axes[1].plot(grad_precond[:, idx], label=f"grad_precond[{idx}]")
#     axes[2].plot(theta[:, idx], label=f"theta[{idx}]")
#     axes[3].plot(adagrad_past[:, idx], label=f"adagrad_past[{idx}]")
#     axes[4].plot(
#         jnp.abs(theta[1:, idx] / window_size[1:, idx]), label=f"window_size[{idx}]"
#     )
#     # axes[3].plot((grad / grad_precond)[:, idx], label=f"adagrad_past[{idx}]")

#     # axes[4].errorbar(
#     #     range(window_size.shape[0]),
#     #     # np.zeros(window_size.shape[0]),
#     #     theta[1:, idx],
#     #     yerr=window_size[1:, idx],
#     #     fmt="o",
#     #     alpha=0.5,
#     #     label=f"window[{idx}]",
#     # )
# axes[0].set_title("Gradient")
# axes[1].set_title("Gradient préconditionné")
# axes[2].set_title("Valeur de theta")
# axes[3].set_title("Valeur de Adagrad (_past)")
# axes[4].set_title("Taille de la fenêtre de sélection")

# for ax in axes:
#     # ax.legend()
#     ax.grid(True)

# axes[-1].set_xlabel("Itération")
# sdgplt.plt.tight_layout()
# sdgplt.plt.show()
