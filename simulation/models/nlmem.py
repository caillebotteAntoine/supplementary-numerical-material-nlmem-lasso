# pylint: skip-file
import functools

import numpy as np
import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import parametrization_cookbook.jax as pc
import sdg4varselect.plotting as sdgplt

sdgplt.FIGSIZE = 4

from sdg4varselect.algo import StochasticGradientDescentPrecond as SPG
import sdg4varselect.algo.preconditioner as preconditioner
from sdg4varselect.outputs import MultiGDResults

from sdg4varselect.models import AbstractMixedEffectsModel, AbstractHDModel


from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    _mean_formatting,
)

# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #


@jit
def m(params, times, psi, phi, **kwargs):
    # phi = cov @ params.beta + phi

    out = psi[:, None] / (1 + jnp.exp(-(times - phi[:, None]) / params.tau))
    assert out.shape == times.shape
    return out


class NonLinearMixedEffectsModel(AbstractMixedEffectsModel, AbstractHDModel):
    def __init__(self, N=1, J=1, P=1, **kwargs):
        AbstractHDModel.__init__(self, P=P)
        AbstractMixedEffectsModel.__init__(self, N=N, J=J, **kwargs)
        self.add_latent_variables("psi")
        self.add_latent_variables("phi")

    @property
    def name(self):
        return f"NLMEM_N{self.N}_J{self.J}_P{self.P}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                mu_0=pc.Real(scale=10, loc=200),
                mu_1=pc.Real(scale=10, loc=1200),
            ),
            cov_latent=pc.MatrixDiagPosDef(dim=2, scale=(10, 1000)),
            tau=pc.Real(scale=10, loc=300),
            var_residual=pc.RealPositive(scale=10),
            beta=pc.Real(scale=10, shape=(self.P,)),
        )
        self.parametrization_size = self._parametrization.size

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(self, params, *args, **kwargs) -> jnp.ndarray:
        return m(params, *args, **kwargs)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def get_mean_latent(self, params, **kwargs) -> jnp.ndarray:
        cov = kwargs["cov"]
        D = params.cov_latent.shape[0]

        mean = _mean_formatting(params.mean_latent, size=D)
        return mean + jnp.array([jnp.zeros(shape=(self.N,)), cov @ params.beta]).T

    # ============================================================== #

    def sample(self, params_star, prngkey, **kwargs):
        (prngkey_cov, prngkey_mem) = jrd.split(prngkey, num=2)

        time = jnp.linspace(150, 3000, self.J)
        time = jnp.tile(time, (self.N, 1))

        # Ajout d'un paramètre pour choisir le type de covariance
        cov_type = kwargs.get("cov_type", "iid")
        if cov_type == "iid":
            cov = jrd.multivariate_normal(
                prngkey_cov,
                mean=jnp.zeros(shape=(self.P,)),
                cov=jnp.diag(jnp.ones(self.P)),
                shape=(self.N,),
            )
        elif cov_type == "ar":
            rho = 0.6
            cov_latent = jnp.array(
                [[rho ** abs(i - j) for j in range(self.P)] for i in range(self.P)]
            )
            cov = jrd.multivariate_normal(
                prngkey_cov,
                mean=jnp.zeros(shape=(self.P,)),
                cov=cov_latent,
                shape=(self.N,),
            )
        elif cov_type == "ar8":
            rho = 0.8
            cov_latent = jnp.array(
                [[rho ** abs(i - j) for j in range(self.P)] for i in range(self.P)]
            )
            cov = jrd.multivariate_normal(
                prngkey_cov,
                mean=jnp.zeros(shape=(self.P,)),
                cov=cov_latent,
                shape=(self.N,),
            )
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")

        obs, sim = AbstractMixedEffectsModel.sample(
            self, params_star, prngkey_mem, mem_obs_time=time, cov=cov
        )
        return {"mem_obs_time": time, "cov": cov} | obs, sim


############################
##### MODEL DEFINITION #####
############################
myModel = NonLinearMixedEffectsModel(N=20, J=10, P=1000)

p_star = myModel.new_params(
    mean_latent={"mu_0": 200, "mu_1": 1200},
    cov_latent=jnp.diag(jnp.array([7**2, 30**2])),
    tau=300,
    var_residual=30,
    beta=jnp.concatenate(
        [25 + jnp.array([100, 50, 20]), jnp.zeros(shape=(myModel.P - 3,))]
    ),
)

p_names = (
    ["$\\mu_0$", "$\\mu_1$"]
    + [f"$\\omega^2_{{{i//(2)},{i%(2)}}}$" for i in range(2**2)]
    + ["$\\tau$", "$\\sigma^2$"]
    + [f"$\\beta_{{{i}}}$" for i in range(6000)]
)

if __name__ == "__main__":
    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(2), cov_type="iid")
    _ = sdgplt.ax(4, 4).plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(2), cov_type="ar")
    _ = sdgplt.ax(4, 4).plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(2), cov_type="ar8")
    ax = sdgplt.ax(4, 6)
    _ = ax.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
    ax.set_xlabel("Time")
    ax.set_ylabel("Observation")
    # Remove top and right lines of the plot frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # saving figure
    ax.figure.savefig("logistic_example.png", dpi=300)

    # ============================================================================================================================ #

    # Affichage de la répartition des colonnes de cov
    # fig, axes = sdgplt.plt.subplots(2, 5, figsize=(15, 6))
    # for i in range(10):
    #     ax = axes[i // 5, i % 5]
    #     ax.hist(cov[:, i], bins=30, alpha=0.7, color="C0")
    #     ax.set_title(f"Colonne {i}")
    # plt.tight_layout()
    # plt.show()
