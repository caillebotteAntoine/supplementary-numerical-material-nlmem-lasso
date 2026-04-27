# pylint: skip-file
import functools

import numpy as np
import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import parametrization_cookbook.jax as pc
import sdg4varselect.plotting as sdgplt

sdgplt.FIGSIZE = 4

from sdg4varselect.models import AbstractMixedEffectsModel, AbstractHDModel


# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #

###############################
##### MODEL SPECIFICATION #####
###############################


def m(params, times: jnp.ndarray, a, b, cov: jnp.ndarray, **kwargs) -> jnp.ndarray:
    out = (cov @ params.alpha).reshape(times.shape) + a[:, None] + b[:, None] * times
    assert out.shape == times.shape
    return out


class LinearMixedEffectsModel(AbstractMixedEffectsModel, AbstractHDModel):

    def __init__(self, N=1, J=1, P=1, **kwargs):
        AbstractHDModel.__init__(self, P=P)
        AbstractMixedEffectsModel.__init__(self, N=N, J=J, **kwargs)
        self.add_latent_variables("a")
        self.add_latent_variables("b")

    @property
    def name(self):
        return f"LMEM_N{self.N}_J{self.J}_P{self.P}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.Real(shape=(2,), scale=(1,)),
            cov_latent=pc.MatrixDiagPosDef(dim=2, scale=(1,)),
            var_residual=pc.RealPositive(scale=1),
            alpha=pc.Real(shape=(self.P,), scale=1),
        )
        self.parametrization_size = self._parametrization.size

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(self, params, *args, **kwargs) -> jnp.ndarray:
        return m(params, *args, **kwargs)

    # ============================================================== #

    def sample(self, params_star, prngkey, **kwargs):

        (prngkey_mem, prngkey_cov) = jrd.split(prngkey, num=2)

        # === nlmem_simulation() === #
        time = jnp.linspace(0.01, 1, self.J)
        time = jnp.tile(time, (self.N, 1))

        cov = jrd.uniform(
            prngkey_cov, minval=-0.1, maxval=0.1, shape=(self.N * self.J, self.P)
        )  #
        obs, sim = AbstractMixedEffectsModel.sample(
            self, params_star, prngkey_mem, mem_obs_time=time, cov=cov
        )

        return {"mem_obs_time": time, "cov": cov} | obs, sim


############################
##### MODEL DEFINITION #####
############################
myModel = LinearMixedEffectsModel(N=100, J=10, P=10)

p_star = myModel.new_params(
    mean_latent=jnp.array([2, 5]),
    cov_latent=jnp.diag(jnp.array([1**2, 2**2])),
    var_residual=1e-8,
    alpha=0
    * jnp.concatenate([jnp.array([8, -10, 20]), jnp.zeros(shape=(myModel.P - 3,))]),
)

p_names = (
    ["$\mu_a$", "$\mu_b$"]
    + ["$\\sigma_a^2$", "NA", "NA", "$\\sigma_b^2$"]
    + ["$\\sigma^2$"]
    + [f"$\\alpha_{{{1+i}}}$" for i in range(2000)]
)

if __name__ == "__main__":
    myobs, mysim = myModel.sample(
        p_star,
        jrd.PRNGKey(2),  # Sigma=jnp.diag(jnp.ones(myModel.P))
    )
    _ = sdgplt.ax(4, 4).plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
    # ============================================================================================================================ #

    # Affichage de la répartition des colonnes de cov
    # fig, axes = sdgplt.plt.subplots(2, 5, figsize=(15, 6))
    # for i in range(10):
    #     ax = axes[i // 5, i % 5]
    #     ax.hist(cov[:, i], bins=30, alpha=0.7, color="C0")
    #     ax.set_title(f"Colonne {i}")
    # plt.tight_layout()
    # plt.show()
    myModel = LinearMixedEffectsModel(N=100, J=10, P=500)

    p_star = myModel.new_params(
        mean_latent=jnp.array([2, 5]),
        cov_latent=jnp.diag(jnp.array([1**2, 2**2])),
        var_residual=1,
        alpha=jnp.concatenate(
            [jnp.array([8, -10, 20]), jnp.zeros(shape=(myModel.P - 3,))]
        ),
    )

    for seed in range(0):
        print(seed)
        myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(seed))

        import pandas as pd
        from sdg4varselect.outputs import _get_filename

        myobs["Y"] = myobs["Y"].reshape((myobs["Y"].size, 1))
        myobs["mem_obs_time"] = myobs["mem_obs_time"].reshape(
            (myobs["mem_obs_time"].size, 1)
        )
        myobs["id"] = jnp.repeat(jnp.arange(0, myModel.N), myModel.J)

        # myobs["cov"] = jnp.repeat(myobs["cov"], myModel.J, axis=0)

        dt = pd.DataFrame(
            jnp.column_stack(list(myobs.values())),
            columns=["t"] + [f"X{i}" for i in range(myModel.P)] + ["Y", "id"],
        )

        filename = _get_filename(myModel.name, "simulation_files", f"data{seed}")

        dt.to_csv(f"{filename}.csv", sep=";")
