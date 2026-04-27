""" "Process results of senescence selection model."""

from jax import numpy as jnp
import pandas as pd

from sdg4varselect.outputs import RegularizationPath

from . import SenescenceModel, load_data, all_chromosome


def melt_csv_selected_snp():
    """Melt selected SNPs from multiple chromosomes into a single CSV file."""
    all_selected_snps = []

    for chr_name in all_chromosome:
        df = pd.read_csv(f"results/csv/selected_snp_chr{chr_name}.csv")
        all_selected_snps.append(df)

    melted_df = pd.concat(all_selected_snps, ignore_index=True)
    melted_df.to_csv("results/csv/selected_snp_all_chromosomes.csv", index=False)
    print(
        f"Wrote melted selected SNPs to results/csv/selected_snp_all_chromosomes.csv with {len(melted_df)} entries."
    )


def write_selected_snp(chr_name, folder="results/cluster_1", seed=None, ebic_shift=0):
    """Write selected SNPs to a CSV file

    Parameters
    ----------
    chr_name : str
        Name of the chromosome (e.g., "1a")
    folder : str
        Folder where the RegularizationPath results are stored
    ebic_shift : int
        Shift to apply to the EBIC index (default is 0)
    """
    data, _ = load_data(chr_name)

    n, p = data["cov"].shape
    _, j = data["Y"].shape  # 220, 18

    myModel = SenescenceModel(N=n, J=j, P=p)

    regpath = RegularizationPath.load(
        f"{folder}/results/senescence_chr{chr_name}_res"
        + (f"_{seed}" if seed is not None else "")
    )
    regpath.update_bic(myModel)
    regpath = regpath.standardize()

    ebic_argmin = jnp.argmin(regpath.ebic) - ebic_shift
    beta_estim = regpath[ebic_argmin].last_theta[-p:]

    cov = pd.read_csv(
        f"data/chr{chr_name}_pre_process_marion.csv",
        sep=";",
        index_col=0,
        decimal=".",
        skiprows=0,
    )
    cov = cov.sort_values(by=["ID"])
    cov = cov.drop(columns=["ID", "GENOTYPE"])
    cov.head()

    snp_names = cov.columns.to_list()[6:]

    selected_snp = [snp_names[i] for i in jnp.where(beta_estim != 0)[0]]

    out_path = f"{folder}/csv/selected_snp_chr{chr_name}.csv"
    pd.DataFrame({"selected_snp": selected_snp, "Chromosome": chr_name}).to_csv(
        out_path, index=False
    )

    print(f"Wrote {len(selected_snp)} selected SNPs to {out_path}.")
