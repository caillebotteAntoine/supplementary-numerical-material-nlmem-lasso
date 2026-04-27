import pandas as pd
import jax.numpy as jnp


def get_chr(chr_name):
    chr_name = chr_name.lower()
    cov = pd.read_csv(
        f"application/data/chr{chr_name}.csv",
        sep=";",
        index_col=0,
        decimal=".",
        skiprows=0,
    )
    cov = cov.sort_values(by=["ID"])
    cov = cov.drop(columns=["ID", "GENOTYPE"])
    cov.head()

    return jnp.array(cov.to_numpy())


def load_data(chr_name):
    data_sen = pd.read_csv(
        "application/data/senescence.csv", sep=";", index_col=0, decimal=","
    )
    data_sen = data_sen.sort_values(by=["ID", "DATE"])
    data_sen.head()

    X = get_chr(chr_name)
    beta_init = X[0, 6:]

    shape = (data_sen.shape[0] // 18, 18)
    data = {
        "mem_obs_time": jnp.array(data_sen["DATE"]).reshape(shape),
        "Y": jnp.array(data_sen["SENESCENCE"]).reshape(shape),
        "Genotype": data_sen["GENOTYPE"].values.reshape(shape),
        "pca": X[1:, 1:6],
        "cov": X[1:, 6:],
    }

    return data, beta_init
