from sdg4varselect.algo import StochasticProximalGradientDescentPrecond as SPGD
import sdg4varselect.algo.preconditioner as preconditioner


def _one_estim(algo: SPGD, theta0, prngkey, model, data, lbd, freeze=None):
    algo.set_seed(prngkey)
    algo.lbd = lbd
    out = algo.fit(model, data, theta0, freezed_components=freeze)
    return out
