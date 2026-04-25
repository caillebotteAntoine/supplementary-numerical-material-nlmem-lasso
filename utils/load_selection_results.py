import pandas as pd

import numpy as np

# true_theta = np.array(
#     [200.0, 1200.0, 49.0, 0, 0, 900.0, 300.0, 30.0, 120.0, 70.0, 40.0]
# )


def get_selection_results(filename, true_theta, beta_starting_index, skiprows=1):
    df = pd.read_csv(filename, skiprows=skiprows, sep=";")

    theta_ending_index = true_theta.shape[0]

    theta = df.iloc[:, :theta_ending_index].to_numpy()
    beta = df.iloc[:, beta_starting_index:].to_numpy()

    true_beta = np.concatenate(
        [
            true_theta[beta_starting_index:],
            np.zeros(beta.shape[1] - (theta_ending_index - beta_starting_index)),
        ]
    )

    # Support: indices where beta is nonzero
    estimated_support = beta != 0

    # Indices of true positives (p) and true negatives (n)
    p = np.where(true_beta != 0)[0]
    n = np.where(true_beta == 0)[0]
    # True Positives: number of correctly selected nonzero coefficients per row
    TP = (estimated_support[:, p]).sum(axis=1)
    # True Negatives: number of correctly identified zeros per row
    TN = (estimated_support[:, n] == 0).sum(axis=1)

    accuracy = (TP + TN) / (len(p) + len(n))
    sensitivity = TP / len(p)
    specificity = TN / len(n)
    support_exact = (TP == len(p)) & (TN == len(n))
    overselection = (TP == len(p)) & (TN < len(n))

    # Remove estimate when support is not exactly recovered
    theta_overselection = theta[overselection]
    theta_underselection = theta[~(overselection | support_exact)]
    theta = theta[support_exact]
    beta_overselection = beta[overselection]
    beta_underselection = beta[~(overselection | support_exact)]
    beta = beta[support_exact]

    # Take means over all rows
    accuracy = accuracy.mean()
    sensitivity = sensitivity.mean()
    specificity = specificity.mean()
    support_exact = support_exact.astype(float).mean()
    overselection = overselection.mean()

    # Calcul du MSE de beta (sur les colonnes p et n)
    mse_beta = (
        (beta[:, np.concatenate([p, n])] - true_beta[np.concatenate([p, n])]) ** 2
    ).mean(axis=1)
    mse_beta = mse_beta.mean()

    # Compute relative root mean squared error (RRMSE) for theta estimates
    # rrmse = sqrt(mean( (100 * (x - x_star) / x_star)^2 )) for each parameter (column)
    non_zero_true_theta = true_theta
    non_zero_true_theta[non_zero_true_theta == 0] = 1  # Avoid division by zero
    rrmse_theta = np.sqrt(
        np.mean(
            (
                (
                    100
                    * (theta - true_theta[None, : theta.shape[1]])
                    / true_theta[: theta.shape[1]]
                )
                ** 2
            ),
            axis=0,
        )
    )

    return {
        "theta": theta,
        "theta overselection": theta_overselection,
        "theta underselection": theta_underselection,
        "beta": beta,
        "beta overselection": beta_overselection,
        "beta underselection": beta_underselection,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "correct_support": support_exact,
        "overselection": overselection,
        "mse_beta": mse_beta,
        "rrmse_theta": rrmse_theta,
        "nrun": df.shape[0],
    }
