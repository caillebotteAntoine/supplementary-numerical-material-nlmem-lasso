import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def do_plot_selected_snp(
    map_file,
    qtl_file,
    new_qtl_file,
    flowering_file,
    results_file,
) -> plt.Figure:
    """Plot selected SNPs, heading QTLs and major flowering genes on chromosomes.

    Args:
        map_file (str): Path to the SNP map file with chromosome and position information.
        qtl_file (str): Path to the file containing heading QTLs.
        flowering_file (str): Path to the file containing major flowering genes.
        results_file (str): Path to the file containing selected SNPs.
    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    # ============================================================ #
    # Loading selected SNPs
    # ============================================================ #
    df_selected_snp = pd.read_csv(results_file, sep=";")
    df_selected_snp = df_selected_snp["selected_snp"].str.lower().tolist()

    # ============================================================ #
    # Loading map : chromosome, position
    # ============================================================ #
    chr_map = pd.read_csv(map_file, sep=";")
    # V1 (chromosome), V2 (position), name (gène)
    chr_map = chr_map[chr_map["V1"] != "chrUn"]

    # ============================================================ #
    # Loading QTLs and Major Flowering Genes
    # ============================================================ #
    # heading_qtls and major_flowering_genes are list of gene names (lowercase)
    heading_qtls = pd.read_csv(qtl_file, sep=";")["x"].str.lower().tolist()
    new_heading_qtls = pd.read_csv(new_qtl_file, sep=";")["x"].str.lower().tolist()

    major_flowering = pd.read_csv(flowering_file, sep=";")
    # Adding major flowering genes to chr_map
    major_flowering["Gene"] = major_flowering["Gene"].str.lower()
    major_flowering["Chromosome"] = "chr" + major_flowering["Chromosome"].astype(str)
    major_flowering = major_flowering.rename(
        columns={"Position": "V2", "Chromosome": "V1", "Gene": "name"}
    )
    chr_map = pd.concat([chr_map, major_flowering], ignore_index=True)
    major_flowering_genes = major_flowering["name"].tolist()

    # ============================================================ #
    # Preparing DataFrame for Plotting
    # ============================================================ #
    chr_map["name"] = chr_map["name"].str.lower()
    # categorical variable for plotting order
    chr_map["Chromosome"] = pd.Categorical(
        chr_map["V1"],
        categories=sorted(chr_map["V1"].unique(), key=lambda x: (x[3], x[2])),
        ordered=True,
    )

    # sorting chromosomes
    chromosome_order = sum(
        [[f"chr{i}{LETTER}" for LETTER in ["A", "B", "D"]] for i in range(1, 8)],
        start=[],
    )
    order_map = {name: idx for idx, name in enumerate(chromosome_order)}
    chr_map["order"] = chr_map["Chromosome"].map(order_map).astype(int)
    chr_map = chr_map.sort_values(by="order")

    # masks for coloring each category
    is_heading = chr_map["name"].isin(heading_qtls)
    is_senescence = chr_map["name"].isin(new_heading_qtls)
    is_major = chr_map["name"].isin(major_flowering_genes)
    is_selected = chr_map["name"].isin(df_selected_snp)

    # colors and markers for each category
    colored_markers = {
        "Other": {"marker": "_", "c": "pink", "s": 60, "alpha": 0.9, "linewidths": 0.5},
        "Selected SNPs": {"marker": "x", "c": "black", "s": 100},
        "Heading QTLs": {"marker": "P", "c": "red", "edgecolor": "k", "s": 100},
        "Senescence SNPs": {"marker": "P", "c": "#90EE90", "edgecolor": "k", "s": 100},
        "Major flowering genes": {
            "marker": "*",
            "c": "green",
            "edgecolor": "k",
            "s": 100,
        },
    }
    # default _ scatter for other points (pink)

    # ============================================================ #
    # Plotting
    # ============================================================ #
    fig = plt.figure(figsize=(10, 6))

    # order of plotting matters for visibility
    plot_items = [
        # (~(is_heading | is_new_heading | is_major | is_selected), "Other"),
        (is_senescence, "Senescence SNPs"),
        (is_heading, "Heading QTLs"),
        # (is_major, "Major flowering genes"),
        (is_selected, "Selected SNPs"),
    ]

    for mask, label in plot_items:
        if mask.any():
            subset = chr_map[mask]
            plt.scatter(
                subset["Chromosome"],
                subset["V2"],
                label=label,
                **colored_markers[label],
            )

    # Vertical lines between chromosomes
    for chrom in chr_map["Chromosome"].cat.categories:
        plt.axvline(x=chrom, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Labels and legend
    plt.ylim(0, 8e8)
    plt.xlabel("Chromosome")
    plt.xticks(rotation=45)
    plt.ylabel("Position")
    plt.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    PACKAGE_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PACKAGE_ROOT.parent / "data"
    RESULTS_DIR = PACKAGE_ROOT.parent / "results"

    fig = do_plot_selected_snp(
        DATA_DIR / "carte_Axiom-TABW420k_WGAv1.csv",
        DATA_DIR / "marker_HD_INRmon13LN.csv",
        DATA_DIR / "markSignif_bonf.csv",
        DATA_DIR / "GenesMajeursFloraison.csv",
        RESULTS_DIR / "cluster_10" / "csv" / "selected_snp_all_chr.csv",
    )

    fig.savefig(RESULTS_DIR / "selected_snp_all_chr.png")
