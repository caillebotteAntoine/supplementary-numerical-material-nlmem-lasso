# pylint: skip-file
from copy import deepcopy

import jax.numpy as jnp
import jax.random as jrd


import sdg4varselect.plotting as sdgplt

sdgplt.FIGSIZE = 4


from sdg4varselect.algo import StochasticProximalGradientDescentPrecond as SPGD
import sdg4varselect.algo.preconditioner as preconditioner
from sdg4varselect.exceptions import Sdg4vsException


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


def algo_factory(model, max_iter=1000, length_history=500):
    AdaGrad = preconditioner.AdaGrad(scale=None, regularization=1e-8)

    algo = SPGD(AdaGrad, partial_fit=False)
    algo.init_mcmc(model, adaptative_sd=True)

    algo.max_iter = max_iter
    algo.estimate_average_length = length_history
    # algo.step_size.heating.step = algo.max_iter - 100
    algo.step_size.heating.step = None
    algo.step_size.max = 1

    return algo


def _one_estim(algo: SPGD, theta0, prngkey, model, data, lbd, freeze=None):
    algo.set_seed(prngkey)
    algo.lbd = lbd
    out = algo.fit(model, data, theta0, freezed_components=freeze)
    return out


def lm_algo_scale(model, max_iter):
    algo = algo_factory(
        model,
        max_iter=max_iter,
        length_history=max_iter * 3 // 4,
    )
    algo._preconditioner._scale = jnp.concatenate(
        [
            1 * jnp.ones(shape=(5,)),
            1 * jnp.ones(model.P),
        ]
    )
    algo.save_all = False
    return algo


def lm_selection_estim(prngkey, model, data, seed, lbd):

    prngkey_theta, prngkey_select, prngkey_estim = jrd.split(prngkey, 3)
    theta0 = jrd.normal(prngkey_theta, shape=(model.parametrization.size,))
    P = model.P
    # theta0 = theta0.at[(-P) : (-P + 10)].set(theta0[(-P) : (-P + 10)] + 10)
    # theta0 = theta0.at[(-P + 10) :].set(theta0[(-P + 10) :] + 1)

    algo_select = lm_algo_scale(model, 1000)
    algo_select.save_all = seed == 0
    out = _one_estim(
        algo_select,
        theta0,
        prngkey_select,
        model,
        data,
        lbd=lbd,
    )
    # return out

    supp = (out.last_theta != 0).at[: -model.P].set(True)
    dt = deepcopy(data)
    dt["cov"] = dt["cov"][:, supp[-model.P :]]

    model_shrink = deepcopy(model)
    model_shrink.P = supp[-model.P :].sum()
    model_shrink.init()

    t0 = jnp.concatenate([theta0[:-P], theta0[-P:][supp[-P:]]])
    algo_estim = lm_algo_scale(model_shrink, 10000)
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


def pkm_selection_estim(algo_select, algo_estim, prngkey, model, data, lbd):

    prngkey_theta, prngkey_select, prngkey_estim = jrd.split(prngkey, 3)

    theta0 = jrd.normal(prngkey_theta, shape=(model.parametrization.size,))
    # theta0 = theta0.at[2:5].set(jnp.array([3.11902645, 4.46064732, 0.0]))
    theta0 = theta0.at[2:4].set(jnp.array([3.11902645, 4.46064732]))
    # theta0 = theta0.at[4].set(jnp.array(0.5))

    theta0 = theta0.at[4].set(theta0[4] + 10)
    P = model.P
    theta0 = theta0.at[(-P) : (-P + 10)].set(1)
    theta0 = theta0.at[(-P + 10) : (-P // 2)].set(0)
    theta0 = theta0.at[(-P // 2) : (-P // 2 + 10)].set(1)
    theta0 = theta0.at[(-P // 2 + 10) :].set(0)

    freeze = jnp.zeros(shape=theta0.shape, dtype=jnp.bool)
    # freeze = freeze.at[4].set(True)
    # theta0 = theta0.at[2:5].set(jnp.array([3.11902645, 4.46064732, 0.0]))
    # algo_select.skip_initialization = False
    out = _one_estim(
        algo_select, theta0, prngkey_select, model, data, lbd=lbd, freeze=freeze
    )
    # algo_select.skip_initialization = True
    # algo_select._preconditioner.freezed_components = jnp.zeros(
    #     shape=theta0.shape, dtype=jnp.bool
    # )
    # out = _one_estim(algo_select, theta0, prngkey_select, model, data, lbd=lbd)

    # return out

    supp = (out.last_theta != 0).at[: -model.P].set(True)

    t0 = jnp.concatenate(
        [
            # out.last_theta_reals1d[:-P],
            theta0[:-P],
            # jnp.where(supp[-P:], out.last_theta_reals1d[-P:], 0),
            jnp.where(supp[-P:], theta0[-P:], 0),
        ]
    )

    estimation = _one_estim(
        algo_estim,
        t0,
        prngkey_estim,
        model,
        data,
        lbd=None,
        freeze=jnp.zeros(shape=t0.shape, dtype=jnp.bool).at[-P:].set(~supp[-P:]),
    )
    out += estimation
    return out


def nlm_algo_scale(model, max_iter):
    algo = algo_factory(
        model,
        max_iter=max_iter,
        length_history=max_iter * 3 // 4,
    )

    algo._preconditioner._scale = jnp.concatenate(
        [
            1 * jnp.ones(shape=(1,)),  # mu0
            2 * jnp.ones(shape=(1,)),  #  mu1
            4 * jnp.ones(shape=(1,)),  # tau
            4 * jnp.ones(shape=(1,)),  # omega 1
            1 * jnp.ones(shape=(1,)),  # omega 2
            1 * jnp.ones(shape=(1,)),  # sigma2
            20 * jnp.ones(model.P),  # beta
        ]
    )
    algo.save_all = False

    return algo


def nlm_selection_estim(prngkey, model, data, seed, lbd):
    prngkey_theta, prngkey_select, prngkey_estim = jrd.split(prngkey, 3)
    theta0 = jrd.normal(prngkey_theta, shape=(model.parametrization.size,))
    P = model.P
    theta0 = theta0.at[5].set(theta0[5] + 10)  # sigma2
    theta0 = theta0.at[2:4].set(theta0[2:4] + jnp.array([10, 0.8]))

    theta0 = theta0.at[(-P) : (-P + 10)].set(theta0[(-P) : (-P + 10)] + 10)
    theta0 = theta0.at[(-P + 10) :].set(theta0[(-P + 10) :] + 1)

    freeze = jnp.zeros(shape=theta0.shape, dtype=jnp.bool)
    # freeze = freeze.at[2:4].set(True)
    algo_select = nlm_algo_scale(model, 1000)
    algo_select.skip_initialization = False
    algo_select.save_all = seed == 0
    out = _one_estim(
        algo_select,
        theta0,
        prngkey_select,
        model,
        data,
        lbd=lbd,
        freeze=freeze,
    )
    # return out

    supp = (out.last_theta != 0).at[: -model.P].set(True)
    dt = deepcopy(data)
    dt["cov"] = dt["cov"][:, supp[-model.P :]]

    model_shrink = deepcopy(model)
    model_shrink.P = supp[-model.P :].sum()
    model_shrink.init()

    t0 = jnp.concatenate([theta0[:-P], theta0[-P:][supp[-P:]]])
    algo_estim = nlm_algo_scale(model_shrink, 10000)
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


def strong_selection_estim(selection_estim, ntry=1, **kwargs):

    def _estim(prngkey, ntry=ntry):
        out = None
        try:
            kwargs["prngkey"] = prngkey
            out = selection_estim(**kwargs)

        except Sdg4vsException as exc:
            print(exc)
            if ntry > 1:
                prngkey_retry, prngkey = jrd.split(prngkey)
                return _estim(prngkey_retry, ntry=ntry - 1)

        if ntry > 1:
            prngkey_retry, prngkey = jrd.split(prngkey)
            retry = _estim(prngkey_retry, ntry=ntry - 1)
            if out is None:
                out = retry
            elif retry is not None:
                if out.log_likelihood > retry.log_likelihood:
                    out = retry
            else:
                print("retry failed")
        return out

    return _estim(kwargs["prngkey"], ntry=ntry)
