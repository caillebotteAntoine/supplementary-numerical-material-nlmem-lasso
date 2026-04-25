import functools

import numpy as np
import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from datetime import timedelta

import parametrization_cookbook.jax as pc

from sdg4varselect._fit_results import _get_filename
from sdg4varselect.models import AbstractMixedEffectsModel, AbstractHDModel
import sdg4varselect.plotting as sdgplt
from sdg4varselect.outputs import MultiGDResults, MultiRegularizationPath
from sdg4varselect._criterion_bic_ebic import compute_metrics
import pandas as pd


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


p = 1000
results = MultiRegularizationPath.load(
    f"results/NLMEM_N200_J10_P1000_N200J10P1000COViidVAR30_all_0_100"
)
x = results[10].standardize()


def axvline_argmin(ax, lbd_set, x, color, x_scale=None):
    lbd = lbd_set[jnp.argmin(x)]
    if x_scale is None:
        x_scale = x

    ax.axvline(x=lbd, color=color, linewidth=2, linestyle="--")
    ax.text(
        lbd,
        0.8 * x_scale.max() + 0.2 * x_scale.min(),
        rf"$\lambda$ = {lbd:.3e}",
        color=color,
        ha="center",
        va="center",
        rotation="vertical",
        backgroundcolor="white",
    )


fig = sdgplt.figure(5, 10)
axes = fig.subplots(1, 2)

lbd_set_shrink = 7

multi_theta_hd = x.last_theta[lbd_set_shrink:, -p:]
lbd_set = x.lbd_set[lbd_set_shrink:]
ebic_hd = x.ebic[lbd_set_shrink:]

axes[0].plot(lbd_set, multi_theta_hd)
axes[0].set_title("Regularization path")
axes[0].set_xlabel(r"Regularization penalty ($\lambda$)")
axes[0].set_ylabel(r"Parameter")
axes[0].set_xscale("log")
axvline_argmin(axes[0], lbd_set, ebic_hd, color="k", x_scale=multi_theta_hd[0, :])


axes[1].plot(lbd_set, ebic_hd, color="r")
axes[1].set_title("eBIC")
axes[1].set_xlabel(r"Regularization penalty ($\lambda$)")
axes[1].set_xscale("log")
axvline_argmin(axes[1], lbd_set, ebic_hd, color="k")
