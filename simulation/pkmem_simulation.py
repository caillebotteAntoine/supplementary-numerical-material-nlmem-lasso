# pylint: skip-file
import functools

import numpy as np
import jax.numpy as jnp
import jax.random as jrd


import pandas as pd
from sdg4varselect.outputs import _get_filename

from pkmem import PKMixedEffectsModel, p_star

myModel = PKMixedEffectsModel(N=300, P=500 * 2, D=100, V=30)


p_star = myModel.new_params(
    mean_latent={"ka": 6, "Cl": 8},
    cov_latent=jnp.diag(
        jnp.array([0.2, 0.1])
    ),  # jnp.array([[0.2, 0.05], [0.05, 0.1]]),
    var_residual=1e-3,
    beta1=jnp.concatenate(
        (
            # jnp.zeros(shape=(5,)),
            jnp.array([3, 2, 1, 0, 0]),
            jnp.zeros(
                shape=myModel.P // 2 - 5,  # - 5,
            ),
        )
    ),
    beta2=jnp.concatenate(
        (
            # jnp.zeros(shape=(5,)),
            jnp.array([0, 0, 3, 2, 1]),
            jnp.zeros(
                shape=myModel.P // 2 - 5,  # - 5,  #
            ),
        )
    ),
)


all_cov = []
all_data = []
phi_dfs = []

for seed in range(100):
    print(seed)
    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(seed))

    ii = jnp.repeat(jnp.arange(0, myModel.N), myModel.J).astype(int)
    tt = myobs["mem_obs_time"].reshape((myModel.N * myModel.J, 1))
    Y = myobs["Y"].reshape((myModel.N * myModel.J, 1))
    seed_col = jnp.full((myModel.N * myModel.J, 1), seed).astype(int)

    dt = pd.DataFrame(
        jnp.column_stack([ii, tt, Y, seed_col]),
        columns=["id", "times", "Y", "seed"],
    )
    all_data.append(dt)

    ii = jnp.arange(0, myModel.N).astype(int)
    seed_col = jnp.full((myModel.N, 1), seed).astype(int)

    all_cov.append(
        pd.DataFrame(
            jnp.column_stack([ii, myobs["cov1"], seed_col]),
            columns=["id"] + [f"{i}" for i in range(myobs["cov1"].shape[1])] + ["seed"],
        )
    )

    phi_dfs.append(
        pd.DataFrame(
            {
                "phi1": mysim["phi_1"],  # + myobs["cov1"] @ p_star.beta1,
                "phi2": mysim["phi_2"],  # + myobs["cov2"] @ p_star.beta2,
                "id": jnp.arange(myModel.N),
                "seed": seed,
            }
        )
    )


print("simulation ended")

final_df = pd.concat(all_data, ignore_index=True)
filename = _get_filename(myModel.name, "simulation_files", "data")
print("saving to", filename)
final_df.to_csv(f"{filename}.csv", sep=";", index=False)

final_df = pd.concat(all_cov, ignore_index=True)
filename = _get_filename(myModel.name, "simulation_files", "cov")
print("saving to", filename)
final_df.to_csv(f"{filename}.csv", sep=";", index=False)

final_df = pd.concat(phi_dfs, ignore_index=True)
filename = _get_filename(myModel.name, "simulation_files", "phi")
print("saving to", filename)
final_df.to_csv(f"{filename}.csv", sep=";", index=False)

(mysim["phi_1"]).mean()
(mysim["phi_2"]).mean()
