import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

chr = "7b"

cov = pd.read_csv(
    f"../data/chr{chr}_pre_process.csv", sep=";", index_col=0, decimal=".", skiprows=2
)
cov = cov.sort_values(by=["ID"])
cov = cov.drop(columns=["ID", "GENOTYPE"])
cov.head()
X = cov.to_numpy()
print(X.shape)


# --- Données d'intérêt ---
heading_qtls = (
    pd.read_csv("../data/marker_HD_INRmon13LN.csv", sep=" ")["x"].str.lower().tolist()
)
major_flowering_df = pd.read_csv("../data/GenesMajeursFloraison.csv", sep=";")
major_flowering_df["Gene"] = major_flowering_df["Gene"].str.lower()
major_flowering_df["Chromosome"] = "chr" + major_flowering_df["Chromosome"].astype(str)
major_flowering_df = major_flowering_df.rename(
    columns={"Position": "V2", "Chromosome": "V1", "Gene": "name"}
)

major_genes = major_flowering_df["name"].tolist()
genes_names = cov.columns.str.lower().tolist()

np.where(np.isin(genes_names, major_genes))[0], np.where(np.isin(genes_names, heading_qtls))[0]
