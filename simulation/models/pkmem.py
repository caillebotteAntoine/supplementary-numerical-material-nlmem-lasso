# pylint: skip-file
import sys

import warnings
from copy import deepcopy
import functools

import numpy as np
import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import parametrization_cookbook.jax as pc
import sdg4varselect.plotting as sdgplt

sdgplt.FIGSIZE = 4

import pandas as pd

from sdg4varselect.algo import StochasticProximalGradientDescentPrecond as SPGD
import sdg4varselect.algo.preconditioner as preconditioner
from sdg4varselect.outputs import MultiGDResults, MultiRegularizationPath
from sdg4varselect._regularization_function import regularization_path
from sdg4varselect.exceptions import Sdg4vsException

from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    multivariate_normal_log_pdf,
)
from sdg4varselect.models import AbstractMixedEffectsModel, AbstractHDModel

from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    _mean_formatting,
)
from jax.scipy.stats import multivariate_normal

# ============================================================================================================================ #
# ============================================================================================================================ #
# ============================================================================================================================ #

###############################
##### MODEL SPECIFICATION #####
###############################


@jit
def m(params, times, D, V, phi_1, phi_2, **kwargs):
    out = (
        D
        * phi_1[:, None]
        / (V * phi_1[:, None] - phi_2[:, None])
        * (jnp.exp(-phi_2[:, None] / V * times) - jnp.exp(-phi_1[:, None] * times))
    )

    assert out.shape == times.shape
    return out


class PKMixedEffectsModel(AbstractMixedEffectsModel, AbstractHDModel):
    def __init__(self, N=1, P=1, **kwargs):
        AbstractHDModel.__init__(self, P=P)
        AbstractMixedEffectsModel.__init__(self, N=N, J=12, **kwargs)
        self.add_latent_variables("phi_1")
        self.add_latent_variables("phi_2")

    @property
    def name(self):
        return f"PKMEM_N{self.N}_J{self.J}_P{self.P}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                ka=pc.Real(loc=10, scale=1),
                Cl=pc.Real(loc=10, scale=1),
            ),
            # cov_latent=pc.MatrixDiagPosDef(dim=3, scale=1),
            cov_latent=pc.MatrixDiagPosDef(dim=2, scale=0.1),
            var_residual=pc.RealPositive(scale=0.01),
            beta1=pc.Real(scale=1, shape=(self.P // 2,)),
            beta2=pc.Real(scale=1, shape=(self.P // 2,)),
        )
        self.parametrization_size = self._parametrization.size

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(self, params, *args, **kwargs) -> jnp.ndarray:
        return m(params, *args, **kwargs)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def get_mean_latent(self, params, **kwargs) -> jnp.ndarray:
        """
        Compute the mean of the mixed effects model.

        Parameters
        ----------
        params : object
            Contains attributes `mean_latent`, `beta1`, and `beta2`.
        **kwargs : dict
            Additional data to be passed to the mixed_effect_function.

        Returns
        -------
        jnp.ndarray
            Mean of the mixed effects model.
        """
        cov1 = kwargs["cov1"]
        cov2 = kwargs["cov2"]
        D = params.cov_latent.shape[0]

        mean = _mean_formatting(params.mean_latent, size=D)
        return mean + jnp.array([cov1 @ params.beta1, cov2 @ params.beta2]).T

    # ============================================================== #

    def sample(self, params_star, prngkey, **kwargs):
        (prngkey_cov, prngkey_mem) = jrd.split(prngkey, num=2)

        # time = jnp.linspace(0, 30, self.J)
        time = jnp.array([0.05, 0.15, 0.25, 0.4, 0.5, 0.8, 1, 2, 7, 12, 24, 40])
        self._j = time.shape[0]
        time = jnp.tile(time, (self.N, 1))

        cov = jrd.bernoulli(prngkey_cov, p=0.2, shape=(self.N, self.P // 2))
        cov /= jnp.std(cov, axis=0)[None, :]
        cov -= jnp.mean(cov, axis=0)[None, :]

        obs, sim = AbstractMixedEffectsModel.sample(
            self,
            params_star,
            prngkey_mem,
            mem_obs_time=time,
            cov1=cov,
            cov2=cov[:, :],
        )

        return {"mem_obs_time": time, "cov1": cov, "cov2": cov[:, :]} | obs, sim


MultiRegularizationPath
############################
##### MODEL DEFINITION #####
############################
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


t0 = myModel.parametrization.params_to_reals1d(p_star)
# print(t0)
# print(
#     myModel.parametrization.reals1d_to_params(
#         t0.at[2:5].set([3.11902645, 4.46064732, 0.0])
#     )
# )
# print(
#     myModel.parametrization.reals1d_to_params(
#         t0.at[2:5].set([12.24744392, 12.64897567, 0.0])
#     )
# )
# print(myModel.parametrization.reals1d_to_params(t0))
p_names = np.array(
    ["$\\mu_1$", "$\\mu_2$"]
    + [f"$\\omega^2_{{{i//(2)},{i%(2)}}}$" for i in range(2**2)]
    + ["$\\sigma^2$"]
    + [f"$\\beta_{{1,{i}}}$" for i in range(myModel.P // 2)]
    + [f"$\\beta_{{2,{i}}}$" for i in range(myModel.P // 2)],
)

myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0))

# print(mysim["phi_1"] + myobs["cov1"] @ p_star.beta1)
# print(mysim["phi_2"] + myobs["cov2"] @ p_star.beta2)


if __name__ == "__main___":
    seed = 0
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
    data.head()

    shape = (myModel.N, 12)
    myobs = {
        "mem_obs_time": jnp.array(data.times).reshape(shape),
        "Y": jnp.array(data.Y).reshape(shape),
        "cov1": jnp.array(cov.iloc[:, 1 : (myModel.P // 2 + 1)]),
        "cov2": jnp.array(cov.iloc[:, 1 : (myModel.P // 2 + 1)]),
    }

    # myobs, mysim = myModel.sample(
    #     p_star, jrd.PRNGKey(0), Sigma=jnp.diag(jnp.ones(myModel.P))
    # )

    p_partial = 0.4
    N1 = int(p_partial * myModel.N)
    myobs["mem_obs_time"] = myobs["mem_obs_time"].at[:N1, 4:].set(jnp.nan)
    myobs["Y"] = myobs["Y"].at[:N1, 4:].set(jnp.nan)

    _ = sdgplt.ax(4, 7).plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
