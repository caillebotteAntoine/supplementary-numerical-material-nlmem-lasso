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
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NTHREADS}"


##############################

import jax.numpy as jnp
import jax.random as jrd

from sdg4varselect._regularization_function import regularization_path
import sdg4varselect.plotting as sdgplt

from phd_code_results.chapter_3_mem_selection.models.estim_selection_fct import (
    strong_selection_estim,
    lm_selection_estim,
)
from lmem import LinearMixedEffectsModel, p_star, p_names

CLUSTER = False
N = 100
P = 500
seed = 0

if CLUSTER:
    seed = int(sys.argv[1]) - 1
    N = int(sys.argv[2]) * 10
    P = int(sys.argv[3]) * 100

J = 10

all_arg_used = f"N{N}J{J}P{P}"

# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #

myModel = LinearMixedEffectsModel(N=N, J=J, P=P)


p_star = myModel.new_params(
    mean_latent=jnp.array([2, 5]),
    cov_latent=jnp.diag(jnp.array([1**2, 2**2])),
    var_residual=1,
    alpha=jnp.concatenate([jnp.array([8, -10, 20]), jnp.zeros(shape=(myModel.P - 3,))]),
)

p_names = (
    ["$\mu_a$", "$\mu_b$"]
    + ["$\\sigma_a^2$", "NA", "NA", "$\\sigma_b^2$"]
    + ["$\\sigma^2$"]
    + [f"$\\alpha_{{{1+i}}}$" for i in range(myModel.P)]
)

myobs, mysim = myModel.sample(
    p_star, jrd.PRNGKey(seed), Sigma=jnp.diag(jnp.ones(myModel.P))
)

if seed == 0:
    _ = sdgplt.ax(4, 4).plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
    sdgplt.plt.savefig(f"results_lm/images/LMEM_{all_arg_used}_sample.png")

# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #


def estim(algo_type, seed):
    out = lm_selection_estim(
        algo_type=algo_type,
        prngkey=jrd.PRNGKey(seed),
        model=myModel,
        data=myobs,
        lbd=1e-3,
    )
    out.theta_star = p_star
    return out


lbd_set = 10 ** jnp.linspace(0, 1.5, num=5)
# if P > 2000:
#     lbd_set = 10 ** jnp.linspace(-0.5, 0.7, num=5)

x = regularization_path(
    strong_selection_estim,
    jrd.PRNGKey(seed),
    lbd_set,
    selection_estim=lm_selection_estim,
    model=myModel,
    data=myobs,
    seed=seed,
    ntry=1,
)
x.theta_star = p_star

if seed == 0:

    fig = sdgplt.figure(6 * 2, 2 * 8)
    _ = sdgplt.plot_theta(
        x,
        fig=[*fig.subfigures(1, 4)],
        params_names=p_names,
        id_to_plot=[
            [0, 1, 6],
            [2, 5],
            [7, 8, 9],
            [10, 11, 12],
        ],
        log_scale=False,
    )
    sdgplt.plt.savefig(f"results_lm/images/LMEM_{all_arg_used}_iter.png")

    _ = sdgplt.ax(6, 6).plot(x.theta[jnp.argmin(x.ebic), :, 7:])
    sdgplt.plt.savefig(f"results_lm/images/LMEM_{all_arg_used}_iter_hd_bestebic.png")

    sdgplt.plot_regpath(x, P=myModel.P, fig=sdgplt.figure(6, 6))
    sdgplt.plt.savefig(f"results_lm/images/LMEM_{all_arg_used}_regpath.png")

    if not CLUSTER:
        res = x[0].theta_reals1d[1]
        xx = res.cumsum(axis=0) / jnp.arange(1, 1 + res.shape[0])[:, None]
        xxx = jnp.abs(xx[101:, :] - xx[100:-1, :])
        _ = sdgplt.ax(4, 4).plot(xx)
        _ = sdgplt.ax(4, 4).plot(xxx)
        print(xx[-1, :])

if CLUSTER:
    x.make_it_lighter()
    x.save(
        myModel.name,
        root=f"results_lm/{all_arg_used}",
        filename_add_on=f"{seed}",
    )
