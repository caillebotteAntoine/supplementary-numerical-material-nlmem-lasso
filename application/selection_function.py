from copy import deepcopy

import jax.numpy as jnp
import jax.random as jrd

from sdg4varselect.algo import StochasticProximalGradientDescentPrecond as SPGD
import sdg4varselect.algo.preconditioner as preconditioner

from sdg4varselect.algo import StochasticProximalGradientDescentPrecond as SPGD
import sdg4varselect.algo.preconditioner as preconditioner
from sdg4varselect.exceptions import Sdg4vsException

import sdg4varselect.algo.preconditioner as preconditioner

# pylint: disable=missing-function-docstring
from . import DEBUG_FLAG


def expand_mask(x, y, P):
    supp = (y.last_theta != 0).at[:-P].set(True)
    row, _ = x.theta.shape
    D = supp.shape[0] - P
    theta_expand = jnp.zeros(shape=(row, supp.shape[0]))
    x.theta = theta_expand.at[:, jnp.where(supp)[0]].set(x.theta)

    D = y.grad.shape[1] - P
    grad_expand = jnp.zeros(shape=(x.grad.shape[0], y.grad.shape[1]))
    grad_expand = grad_expand.at[:, :D].set(x.grad[:, :D])

    iii = D + supp[-P:].nonzero()[0]
    x.grad = grad_expand.at[:, iii].set(x.grad[:, iii])

    D = y.grad_precond.shape[1] - P
    grad_expand = jnp.zeros(shape=(x.grad_precond.shape[0], y.grad_precond.shape[1]))
    grad_expand = grad_expand.at[:, :D].set(x.grad_precond[:, :D])

    iii = D + supp[-P:].nonzero()[0]
    x.grad_precond = grad_expand.at[:, iii].set(x.grad_precond[:, iii])
    return x


def get_algo(model, seed):
    algo = SPGD(preconditioner.AdaGrad(), partial_fit=False)
    algo.init_mcmc(
        model,
        adaptative_sd=True,
        sd={"phi": 1.0, "psi": 1.0},
    )
    algo.max_iter = 15000 if not DEBUG_FLAG else 1000
    algo.estimate_average_length = 13000 if not DEBUG_FLAG else 700
    algo.pre_heating = 0
    algo._alpha = 1  #  2.8

    # Definition of gamma_0
    algo._preconditioner._scale = jnp.concatenate(
        [
            0.1 * jnp.ones(shape=(2,)),  # mu et eta
            0.5 * jnp.ones(shape=(2,)),  # gamma2 et Omega2
            1 * jnp.ones(shape=(1,)),  # sigma2
            1 * jnp.ones(shape=(5,)),  # alpha
            0.01 * jnp.ones(model.P),
        ]
    )

    algo.skip_initialization = False
    algo.save_all = seed == 0

    return algo


def _one_estim(algo: SPGD, theta0, prngkey, model, data, lbd, freeze=None):
    algo.set_seed(prngkey)
    algo.lbd = lbd
    out = algo.fit(model, data, theta0, freezed_components=freeze)
    return out


def selection_and_estim(prngkey, model, data, p_init, seed, lbd):
    prngkey_select, prngkey_estim = jrd.split(prngkey, 2)
    p = model.P
    theta0 = model.parametrization.params_to_reals1d(p_init)

    algo_select = get_algo(model, seed)
    out = _one_estim(algo_select, theta0, prngkey_select, model, data, lbd=lbd)
    if DEBUG_FLAG:
        return out

    supp = (out.last_theta != 0).at[: -model.P].set(True)
    dt = deepcopy(data)
    dt["cov"] = dt["cov"][:, supp[-model.P :]]

    model_shrink = deepcopy(model)
    model_shrink.P = supp[-model.P :].sum()
    model_shrink.init()

    t0 = jnp.concatenate([theta0[:-p], theta0[-p:][supp[-p:]]])
    algo_estim = get_algo(model_shrink, seed)
    algo_estim.save_all = seed == 0

    estimation = _one_estim(
        algo_estim,
        theta0=t0,
        prngkey=prngkey_estim,
        model=model_shrink,
        data=dt,
        lbd=None,
    )

    out += expand_mask(estimation, out, model.P)
    return out


def strong_selection_estim(selection_estim, ntry=1, ntry_exc=10, **kwargs):

    def _estim(prngkey, ntry_incr, ntry_exc_incr):
        out = None
        try:
            kwargs["prngkey"] = prngkey
            out = selection_estim(**kwargs)

        except Sdg4vsException as exc:
            print(exc)
            if ntry_exc_incr > 1:
                print(f"new try : {ntry_exc_incr}/{ntry_exc}")
                prngkey_retry, prngkey = jrd.split(prngkey)
                return _estim(
                    prngkey_retry, ntry_incr=ntry_incr, ntry_exc_incr=ntry_exc_incr - 1
                )

        if ntry_incr > 1:
            print(f"new try : {ntry_incr}/{ntry}")
            prngkey_retry, prngkey = jrd.split(prngkey)
            retry = _estim(
                prngkey_retry, ntry_incr=ntry_incr - 1, ntry_exc_incr=ntry_exc_incr
            )
            if out is None:
                out = retry
            elif retry is not None:
                if out.log_likelihood > retry.log_likelihood:
                    out = retry
            else:
                print("retry failed")
        return out

    return _estim(kwargs["prngkey"], ntry_incr=ntry, ntry_exc_incr=ntry_exc)
