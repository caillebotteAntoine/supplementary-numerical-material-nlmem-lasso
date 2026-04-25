# pylint: skip-file
# ========================================== #
# Generation of Figure 1 in the paper:
# Trajectories of 5 runs of the algorithm performed on the same dataset for the first
# four components of β with randomly chosen initializations and three regularization parameters
# λ leading to (a) over-selection, (b) exact selection of the support and (c) under-selection of the
# support.
# ========================================== #
import functools

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import parametrization_cookbook.jax as pc
import sdg4varselect.plotting as sdgplt
from sdg4varselect.outputs import MultiGDResults

from sdg4varselect.algo import StochasticProximalGradientDescentPrecond as SPGD
import sdg4varselect.algo.preconditioner as preconditioner

from sdg4varselect.models import AbstractMixedEffectsModel, AbstractHDModel


###############################
##### MODEL SPECIFICATION #####
###############################
class LinearMixedEffectsModel(AbstractMixedEffectsModel, AbstractHDModel):
    def __init__(self, N=1, J=1, P=1, **kwargs):
        AbstractHDModel.__init__(self, P=P)
        AbstractMixedEffectsModel.__init__(self, N=N, J=J, **kwargs)
        self.add_latent_variables("b")

    @property
    def name(self):
        return f"BLUE_N{self.N}_J{self.J}_P{self.P}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.Real(shape=(0,)),  # No latent mean in this model
            cov_latent=pc.MatrixDiag(dim=1, scale=1),
            var_residual=pc.Real(scale=1),
            beta=pc.Real(scale=1, shape=(self.P,)),
        )
        self.parametrization_size = self._parametrization.size

    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(self, params, times, b, **kwargs) -> jnp.ndarray:
        out = kwargs["X"] @ params.beta + jnp.repeat(b, self.J)

        return out.reshape(self.N, self.J)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def get_mean_latent(self, params, **kwargs) -> jnp.ndarray:
        return jnp.zeros((self.N, 1))  # No latent mean in this model

    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(self, theta_reals1d: jnp.ndarray, **kwargs):
        # params = self.parametrization.reals1d_to_params(theta_reals1d)

        X = kwargs["X"]
        Y = kwargs["Y"]
        b = kwargs["b"]
        # b = self._cst["_b"]
        var_residual = theta_reals1d[1]  # params.var_residual

        beta = theta_reals1d[2:]  # params.beta
        #
        pred = (X @ beta + jnp.repeat(b, self.J)).reshape(self.N, self.J)

        # Compute the log-likelihood for each observation
        # Log-likelihood of the observed data given b
        log_likelihoods = (
            # -0.5 * self.J * jnp.log(2 * jnp.pi * params.var_residual)
            -0.5
            * jnp.sum((Y - pred) ** 2, axis=1)
            / var_residual
        )

        # Log-density of the latent variable b (Gaussian)
        cov_latent = theta_reals1d[0]  # params.cov_latent[0, 0]

        log_density_b = (
            # -0.5 * jnp.log(2 * jnp.pi * cov_latent)
            -0.5
            * (b**2)
            / cov_latent
        )

        assert log_likelihoods.shape == (self.N,)
        assert log_density_b.shape == (self.N,)
        return log_likelihoods + log_density_b

    def sample(self, params_star, prngkey, **kwargs):
        (prngkey_cov, prngkey_mem) = jrd.split(prngkey, num=2)

        V = jnp.zeros((self.N * self.J, self.P))
        for i in range(self.N):
            prngkey_cov, prngkey_tmp = jrd.split(prngkey_cov)

            U = jrd.normal(prngkey_tmp, shape=(self.J, self.P))
            U -= U.mean(axis=0, keepdims=True)  # Centering the design matrix
            V = V.at[i * self.J : (i + 1) * self.J, :].set(U)

        # print(f"V.shape = {V.shape}")
        T = V.T @ V
        # print(f"T.shape = {T.shape}")
        L = jnp.linalg.cholesky(T, upper=False)
        # print(jnp.sum(jnp.abs(L @ L.T - T)))
        # print(f"L.shape = {L.shape}")
        X = V @ jnp.linalg.inv(L.T)
        # print(jnp.sum(jnp.abs(L.T @ jnp.linalg.inv(L.T) - jnp.eye(L.shape[0]))))
        # print(f"X.shape = {X.shape}")

        prngkey_mem, prngkey_tmp = jrd.split(prngkey_mem)
        b = jrd.normal(prngkey_tmp, shape=(self.N,))
        eps = jrd.normal(prngkey_mem, shape=(self.N * self.J,))

        Y = X @ params_star.beta + eps + jnp.repeat(b, self.J)

        Y = Y.reshape(self.N, self.J)
        return {"mem_obs_time": Y, "X": X, "Y": Y}, {"eps": eps, "_b": b}


myModel = LinearMixedEffectsModel(N=100, J=5, P=400)

# define the true parameters
p_star = myModel.new_params(
    mean_latent=jnp.zeros((0,)),  # No latent mean in this model
    cov_latent=jnp.array([[3**2]]),
    var_residual=1.5**2,
    beta=jnp.concatenate(
        [
            jnp.array([4.0, -3.0]),
            jnp.zeros(myModel.P - 2),
        ]
    ),
)

myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(1234 * 3))

# ========================================== #
#                   ESTIMATION               #
# ========================================== #
from utils.estim_selection_fct import (
    _one_estim,
)


p_names = [f"$\\omega^2$", "$\\sigma^2$"] + [
    f"$\\beta_{{{i}}}$" for i in range(myModel.P)
]

# Example usage:
Y = myobs["Y"].reshape(-1, 1)  # Ensure Y is a column vector
X = myobs["X"]
J = myModel.J
N = myModel.N
var_eps = float(p_star.var_residual)
var_b = float(p_star.cov_latent[0, 0])

beta_hat = jnp.linalg.solve(X.T @ X, X.T @ Y).reshape(-1)
print(jnp.abs(X.T @ X - jnp.eye(X.shape[1])).max())


def get_beta_lasso(beta_hat, lam, var_eps):
    """
    Compute the Lasso estimator (soft-thresholding) for beta given estimate and lambda.

    Parameters
    ----------
    beta_hat : jnp.ndarray
        estimate of beta, shape (P,)
    lam : float
        Regularization parameter (lambda)

    Returns
    -------
    beta_lasso : jnp.ndarray
        Lasso estimate of beta, shape (P,)
    """
    return jnp.sign(beta_hat) * jnp.maximum(jnp.abs(beta_hat) - var_eps * lam, 0.0)


def estim(prngkey, lbd):

    beta_lasso = get_beta_lasso(beta_hat, lambda_val, var_eps)

    prngkey_theta, prngkey_select, prngkey_estim = jrd.split(prngkey, 3)
    theta0 = jrd.normal(prngkey_theta, shape=(myModel.parametrization.size,))

    # The variance are assumed to be known here
    # We only estimate beta
    # We freeze the variance parameters to their true values
    frozen = (
        jnp.array([False] * myModel.parametrization.size)
        .at[0]
        .set(True)
        .at[1]
        .set(True)
    )

    theta0 = theta0.at[0].set(p_star.cov_latent[0, 0])  # type: ignore
    theta0 = theta0.at[1].set(p_star.var_residual)  # type: ignore

    theta0 = theta0.at[2:].set(theta0[2:] + beta_lasso)

    return _one_estim(
        algo, theta0, prngkey_select, myModel, data=myobs, lbd=lbd, freeze=frozen
    )


AdaGrad = preconditioner.AdaGrad(scale=None, regularization=1e-8)

algo = SPGD(AdaGrad, partial_fit=False)
algo.init_mcmc(myModel, adaptative_sd=True, sd={"b": 0.05})

algo.max_iter = 80
algo.step_size.heating.step = None
algo.step_size.max = 1

algo.estimate_average_length = 1
algo.save_all = True
algo._preconditioner._scale = 0.8 * jnp.ones((myModel.parametrization_size,))

for i, lambda_val in enumerate([0.6, 1, 2]):
    beta_lasso = get_beta_lasso(beta_hat, lambda_val, var_eps)
    print(f"beta_lasso = {beta_lasso[:4]}")

    out = MultiGDResults(
        results=[estim(jrd.PRNGKey(i + 50), lambda_val) for i in range(5)]
    )
    # out.save("N100P400J5lbd0_3bis")
    # _ = sdgplt.plot_mcmc(algo)

    print(f"beta_sgd = {out.last_theta[:,2:6].mean(axis = 0)}")

    p_blue = myModel.new_params(
        mean_latent=jnp.zeros((0,)),  # No latent mean in this model
        cov_latent=jnp.array([[4.0**2]]),
        var_residual=1.0**2,
        beta=beta_lasso,
    )

    out.theta_star = p_blue
    fig = sdgplt.figure(4, 10)
    _ = sdgplt.plot_theta(
        out,
        fig=[*fig.subfigures(1, 2)],
        params_names=p_names,
        log_scale=False,
        id_to_plot=[[2, 3], [4, 5]],
    )
    fig.suptitle(
        f"Estimation pénalisé avec lambda = {lambda_val}, (N={myModel.N}, J={myModel.J}, P={myModel.P})"
    )
    # fig.savefig(
    #     f"LME_exact_2col_{['surselection', 'selection', 'sousselection'][i]}_{myModel.N}_J{myModel.J}_P{myModel.P}_lbd{lambda_val}.png"
    # )
    sdgplt.plt.show()

    # ========================================== #
    # CONVERGENCE CRITIQUE VALUE
    # ========================================== #

    x = out[0]
    grad = x.grad  # shape: (n_iter, param_size)

    gamma_0 = algo._preconditioner._scale
    K_critique = jnp.round((lambda_val * gamma_0) ** 2 / grad[-2, :] ** 4)
    K_critique[2:6]

    I_J = jnp.eye(myModel.J)
    W = jnp.ones((myModel.J, 1))  # Design matrix for the random effects

    Gamma = p_star.var_residual * I_J + p_star.cov_latent[0, 0] * (W @ W.T)
    Gamma.shape
    gamma_inv = jnp.linalg.inv(Gamma)

    Y = myobs["Y"]
    beta_lasso = get_beta_lasso(beta_hat, lambda_val, var_eps)

    def get_crit(prngkey):

        beta = beta_lasso + jrd.uniform(
            prngkey, shape=(myModel.P,), minval=-0.0002, maxval=0.0002
        )

        crit = (
            jnp.array(
                [
                    (Y - (X @ beta).reshape(Y.shape))[i]
                    @ gamma_inv
                    @ (Y - (X @ beta).reshape(Y.shape))[i]
                    for i in range(myModel.N)
                ]
            ).sum()
            / 2
            + lambda_val * jnp.abs(beta).sum()
        )
        return crit

    all_crit = jnp.array([get_crit(jrd.PRNGKey(i)) for i in range(100)])

    crit_lasso = (
        jnp.array(
            [
                (Y - (X @ beta_lasso).reshape(Y.shape))[i]
                @ gamma_inv
                @ (Y - (X @ beta_lasso).reshape(Y.shape))[i]
                for i in range(myModel.N)
            ]
        ).sum()
        / 2
        + lambda_val * jnp.abs(beta_lasso).sum()
    )

    beta = out.theta[0, 25:, 2:].mean(axis=0)
    crit_sgd_lasso = (
        jnp.array(
            [
                (Y - (X @ beta).reshape(Y.shape))[i]
                @ gamma_inv
                @ (Y - (X @ beta).reshape(Y.shape))[i]
                for i in range(myModel.N)
            ]
        ).sum()
        / 2
        + lambda_val * jnp.abs(beta).sum()
    )

    print(
        f"crit_lasso = {crit_lasso}, min crit = {jnp.min(all_crit)}, crit_sgd_lasso = {crit_sgd_lasso}"
    )
