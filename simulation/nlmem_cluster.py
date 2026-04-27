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

import os, sys

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.outputs import MultiGDResults
from sdg4varselect._regularization_function import regularization_path
import sdg4varselect.plotting as sdgplt

from tests.phd_code_results.chapter_1_jm_estimation.models.estim_selection_fct import (
    strong_selection_estim,
    nlm_selection_estim,
)
from nlmem import NonLinearMixedEffectsModel, p_star, p_names

CLUSTER = True
N = 200
P = 500
seed = 0
cov_type = ["iid", "ar", "ar8"][0]
var_type = [15, 30, 45][1]

if CLUSTER:
    seed = int(sys.argv[1]) - 1
    N = int(sys.argv[2]) * 10
    P = int(sys.argv[3]) * 100
    cov_type = str(sys.argv[4]).rstrip()
    var_type = int(sys.argv[5])

J = 10

all_arg_used = f"N{N}J{J}P{P}COV{cov_type}VAR{var_type}"

# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #

myModel = NonLinearMixedEffectsModel(N=N, J=J, P=P)


p_star = myModel.new_params(
    mean_latent={"mu_0": 200, "mu_1": 1200},
    cov_latent=jnp.diag(jnp.array([7**2, var_type**2])),
    tau=300,
    var_residual=30,
    beta=jnp.concatenate(
        [20 + jnp.array([100, 50, 20]), jnp.zeros(shape=(myModel.P - 3,))]
    ),
)

myobs, mysim = myModel.sample(
    p_star,
    jrd.PRNGKey(seed),
    Sigma=jnp.diag(jnp.ones(myModel.P)),
    cov_type=cov_type,
)
if seed == 0:
    _ = sdgplt.ax(4, 4).plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
    sdgplt.plt.savefig(f"results_nlm/images/NLMEM_{all_arg_used}_sample.png")

# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #


def estim(algo_type, seed):
    out = nlm_selection_estim(
        algo_type=algo_type,
        prngkey=jrd.PRNGKey(seed),
        model=myModel,
        data=myobs,
        lbd=5e-2 if algo_type == "adam" else 1.6e-1,
    )
    out.theta_star = p_star
    return out

lbd_set = 10 ** jnp.linspace(0, 1.5, num=20)
if P > 2000 :
    lbd_set = 10 ** jnp.linspace(-0.5, 0.7, num=20)

x = regularization_path(
    strong_selection_estim,
    jrd.PRNGKey(seed),
    lbd_set,
    selection_estim=nlm_selection_estim,
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
            [2, 5, 7],
            [8, 9, 10],
            [11, 12, 13],
        ],
        log_scale=False,
    )
    sdgplt.plt.savefig(f"results_nlm/images/NLMEM_{all_arg_used}_iter.png")

    _ = sdgplt.ax(6, 6).plot(x.theta[jnp.argmin(x.ebic), :, 8:])
    sdgplt.plt.savefig(f"results_nlm/images/NLMEM_{all_arg_used}_iter_hd_bestebic.png")

    sdgplt.plot_regpath(x, P=myModel.P, fig=sdgplt.figure(6, 6))
    sdgplt.plt.savefig(f"results_nlm/images/NLMEM_{all_arg_used}_regpath.png")

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
        root=f"results_nlm/{all_arg_used}",
        filename_add_on=f"{seed}",
    )
