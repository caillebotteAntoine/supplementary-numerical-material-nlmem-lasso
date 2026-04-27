from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT.parent / "application" / "data"
RESULTS_DIR = PACKAGE_ROOT.parent / "results"

DEBUG_FLAG = False

all_chromosome = sum(
    [[f"{i}{LETTER}" for LETTER in ["a", "b", "d"]] for i in range(1, 8)],
    start=[],
)

from .load_data import load_data, get_chr
from .selection_function import selection_and_estim, strong_selection_estim
from .model import SenescenceModel, model_params_names
from .plot_results import do_plot, plot_chr_regpath

from .process_results import write_selected_snp

from .plot_chr import do_plot_selected_snp
