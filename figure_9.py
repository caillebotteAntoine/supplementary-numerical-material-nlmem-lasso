from jax import numpy as jnp

from sdg4varselect.outputs import RegularizationPath
import sdg4varselect.plotting as sdgplt


from application import (
    SenescenceModel,
    load_data,
    write_selected_snp,
    do_plot_selected_snp,
    do_plot,
    plot_chr_regpath,
    all_chromosome,
    RESULTS_DIR,
)

from application.regpath_melt import melt_regpaths
import pandas as pd

seed = 1

for chr_name in all_chromosome:
    if seed is not None:

        try:
            list_regpaths = []
            for s in range(seed):
                filename = RESULTS_DIR / (f"senescence_chr{chr_name}_res_{s}")
                list_regpaths.append(RegularizationPath.load(filename.as_posix()))

            x = melt_regpaths(*list_regpaths)
            # filename = (
            #     RESULTS_DIR / folder / "results" / (f"senescence_chr{chr_name}_res")
            # )
            # x.save(filename.as_posix())

            # _ = plot_chr_regpath(chr_name, folder=folder, seed=None, save_fig=True)
        except Exception as e:
            print(f"melting {chr_name} failed:", e)

    try:
        write_selected_snp(chr_name, seed=None, ebic_shift=0)
    except Exception as e:
        print(f"Writing selected SNPs for chr {chr_name} failed:", e)


dfs = []
for chr_name in all_chromosome[1:]:
    filepath = RESULTS_DIR / f"selected_snp_chr{chr_name}.csv"
    df = pd.read_csv(filepath.as_posix())
    dfs.append(df)

# out_path = f"results/{folder}/csv/selected_snp_all_chr.csv"
# pd.concat(dfs, ignore_index=True).to_csv(out_path, index=False, sep=";")


out_path = RESULTS_DIR / "selected_snp_all_chr.csv"
pd.concat(dfs, ignore_index=True).to_csv(out_path.as_posix(), index=False, sep=";")

from application import APP_ROOT, RESULTS_DIR, DATA_DIR


fig = do_plot_selected_snp(
    DATA_DIR / "carte_Axiom-TABW420k_WGAv1.csv",
    DATA_DIR / "marker_HD_INRmon13LN.csv",
    DATA_DIR / "markSignif_bonf.csv",
    DATA_DIR / "GenesMajeursFloraison.csv",
    RESULTS_DIR / "selected_snp_all_chr.csv",
)
# fig.savefig(f"results/{folder}/fig/selected_snp_all_chr_{folder}.png")
