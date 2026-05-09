# pylint: disable=all

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

import jax.numpy as jnp
import jax.random as jrd


from sdg4varselect._regularization_function import regularization_path


from . import (
    SenescenceModel,
    do_plot,
    load_data,
    strong_selection_estim,
    selection_and_estim,
)


seed, chr = 0, "2b"
# print(sys.argv[1])
if len(sys.argv) > 2:
    seed = int(sys.argv[1]) - 1
    chr = str(sys.argv[2]).rstrip()

data, beta_init = load_data(chr)

known_qtl = jnp.where(beta_init == 0.25)[0]
beta_init = beta_init.at[known_qtl].set(0.5)
beta_init = beta_init.at[known_qtl].add(
    0.2 * jrd.normal(jrd.PRNGKey(seed), shape=known_qtl.shape)
)


N, J = data["Y"].shape  # 220, 18
myModel = SenescenceModel(N=N, J=J, P=data["cov"].shape[1])


p_init = myModel.new_params(
    mean_latent={"mu": 20.0, "eta": 5.0},
    cov_latent=jnp.array([[50.0, 0.0], [0.0, 50.0]]),
    var_residual=80.0,
    alpha=0.5 * jnp.ones(shape=(5,)),
    beta=beta_init,
)
p_init = myModel.parametrization.reals1d_to_params(
    myModel.parametrization.params_to_reals1d(p_init)
    .at[: myModel.parametrization_size - 5 - myModel.P]
    .add(
        jrd.normal(
            jrd.PRNGKey(seed),
            shape=(myModel.parametrization_size - 5 - myModel.P,),
        )
        * 2
    )
)

lbd_set = 10 ** jnp.linspace(jnp.log10(20), jnp.log10(75), num=10)

regpath = regularization_path(
    strong_selection_estim,
    jrd.PRNGKey(seed),
    lbd_set,
    model=myModel,
    data=data,
    seed=seed,
    p_init=p_init,
    selection_estim=selection_and_estim,
    ntry=1,
    ntry_exc=5,
)

# ============== Plotting ================== #
try:
    do_plot(regpath, data, beta_init, myModel, chr, save_fig=seed == 0)
except Exception as e:
    print("Plotting failed:", e)
# ========================================== #

regpath.make_it_lighter()
regpath.save(f"results/senescence_chr{chr}_res_{seed}")
