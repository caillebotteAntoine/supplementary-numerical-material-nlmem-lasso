"""define senescence mixed effects model"""

import functools


import jax.numpy as jnp
from jax import jit

import parametrization_cookbook.jax as pc

from sdg4varselect.models import AbstractMixedEffectsModel, AbstractHDModel

import sdg4varselect.plotting as sdgplt
from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    _mean_formatting,
)

sdgplt.FIGSIZE = 4


def m(params, times, phi, psi, **kwargs) -> jnp.ndarray:
    """mixed effect function for senescence model
    Parameters
    ----------
    params : pc.NamedTuple
        model parameters
    times : jnp.ndarray
        observation times, shape (N, T)
    phi : jnp.ndarray
        individual parameter phi, shape (N,)
    psi : jnp.ndarray
        individual parameter psi, shape (N,)
    kwargs : dict
        additional arguments
    Returns
    -------
    jnp.ndarray
        mixed effect function evaluated at times, shape (N, T)
    """
    out = 100.0 / (1 + jnp.exp(-(times - phi[:, None]) / psi[:, None]))
    assert out.shape == times.shape
    return out


def individual_parameters(self, params, **kwargs) -> jnp.ndarray:
    """compute individual parameters for senescence model
    Parameters
    ----------
    params : pc.NamedTuple
        model parameters
    kwargs : dict
        additional arguments
        pca : jnp.ndarray
            principal components, shape (N, 5)
        cov : jnp.ndarray
            covariates, shape (N, P)
    """

    cov = kwargs.get("cov")
    pca = kwargs.get("pca")

    d = params.cov_latent.shape[0]

    mean = _mean_formatting(params.mean_latent, size=d)
    mean_phi = pca @ params.alpha + cov @ params.beta

    return mean + jnp.array([mean_phi, jnp.zeros(shape=(self.N,))]).T


class SenescenceModel(AbstractMixedEffectsModel, AbstractHDModel):
    """define a logistic mixed effects model"""

    def __init__(self, N=1, J=1, P=1, **kwargs):
        AbstractHDModel.__init__(self, P=P)
        AbstractMixedEffectsModel.__init__(self, N=N, J=J, **kwargs)
        self.add_latent_variables("phi")
        self.add_latent_variables("psi")

    @property
    def name(self):
        return f"SEN_N{self.N}_J{self.J}_P{self.P}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                mu=pc.Real(scale=1, loc=16), eta=pc.Real(scale=1, loc=3)
            ),
            cov_latent=pc.MatrixDiagPosDef(dim=2, scale=(1, 1)),
            var_residual=pc.RealPositive(scale=10),
            alpha=pc.Real(scale=1, shape=(5,)),
            beta=pc.Real(scale=1, shape=(self.P,)),
        )
        self.parametrization_size = self._parametrization.size

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(self, params, *args, **kwargs) -> jnp.ndarray:
        return m(params, *args, **kwargs)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def get_mean_latent(self, params, **kwargs) -> jnp.ndarray:
        return individual_parameters(self, params, **kwargs)

    # ============================================================== #

    def sample(self, params_star, prngkey, **kwargs):
        raise NotImplementedError(
            "This model is only define for estimation on real data"
        )


model_params_names = (
    ["$\\mu$", "$\\eta$"]
    + ["$\\Gamma^2$", "NA"]
    + ["NA", "$\\Omega^2$"]
    + ["$\\sigma^2$"]
    + [f"$\\alpha_{{{i}}}$" for i in range(5)]
    + [f"$\\beta_{{{i}}}$" for i in range(2000)]
)
