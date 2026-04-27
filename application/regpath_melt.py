from sdg4varselect.outputs import GDResults, RegularizationPath


def melt(self, x):
    """Merge two MultiGDResults instances together by applying mean operation.

    Parameters
    ----------
    x : MultiGDResults
        Another MultiGDResults instance to merge.

    Returns
    -------
    MultiGDResults
        A new MultiGDResults instance with merged results.
    """
    assert len(self.results) == len(
        x.results
    ), "Both MultiGDResults must have the same number of GDResults instances to merge."

    new_results = []
    for self_res, x_res in zip(self.results, x.results):
        if self_res.ebic > x_res.ebic:
            best_theta = self_res.theta
            log_likelihood = self_res.log_likelihood
            ebic = self_res.ebic
            bic = self_res.bic
        else:
            best_theta = x_res.theta
            log_likelihood = x_res.log_likelihood
            ebic = x_res.ebic
            bic = x_res.bic

        merged_res = GDResults(
            theta=best_theta,
            theta_reals1d=None,
            fim=None,
            grad=None,
            grad_precond=None,
            log_likelihood=log_likelihood,
            bic=bic,
            ebic=ebic,
            chrono=self_res.chrono + x_res.chrono,
        )
        new_results.append(merged_res)

    return RegularizationPath(
        chrono=self.chrono + x.chrono,
        results=new_results,
        lbd_set=self.lbd_set,
    )


def melt_regpaths(*list_regpaths):
    """Merge multiple RegularizationPath instances together by applying mean operation.

    Parameters
    ----------
    list_regpaths : list of RegularizationPath
        List of RegularizationPath instances to merge.

    Returns
    -------
    RegularizationPath
        A new RegularizationPath instance with merged results.
    """
    assert len(list_regpaths) > 0, "The list of RegularizationPath instances is empty."

    merged_regpath = list_regpaths[0]
    for regpath in list_regpaths[1:]:
        merged_regpath = melt(merged_regpath, regpath)

    return merged_regpath
