##############################

import sys

import sdg4varselect.plotting as sdgplt

from application import (
    SenescenceModel,
    load_data,
)


seed, chr = 0, "2b"
if len(sys.argv) > 2:
    seed = int(sys.argv[1]) - 1
    chr = str(sys.argv[2]).rstrip()

data, beta_init = load_data(chr)


N, J = data["Y"].shape  # 220, 18
myModel = SenescenceModel(N=N, J=J, P=data["cov"].shape[1])

selected_varieties = {
    "BOKARO": "#6a4c93",
    "ESPERIA": "#ff924c",
    "PRIMO": "#ffca3a",
    "QUALITY": "#c5ca30",
    "RIMBAUD": "#8ac926",
    "ROYSSAC": "#36949d",
    "SIRTAKI": "#1982c4",
    "SOBBEL": "#4267ac",
    "SOLDEAD": "#565aa0",
    "SOLLARIO": "#ff595e",
}

ax = sdgplt.ax(5, 8)
for i in range(data["Y"].shape[0]):
    if data["Genotype"][i, 0] in selected_varieties.keys():
        _ = ax.plot(
            data["mem_obs_time"][i],
            data["Y"][i],
            label=data["Genotype"][i, 0],
            marker="o",
            color=selected_varieties[data["Genotype"][i, 0]],
        )
_ = ax.legend(loc="center right")

Date = ["Jul 06", "Jul 13", "Jul 20", "Jul 27", "Aug 03"]
x_date = [2 + 7 * i for i in range(len(Date))]
_ = ax.set_xticks(x_date, Date, fontsize=15)
_ = ax.set_xlabel("Date", fontsize=15)

_ = ax.set_ylabel("Senescence (in %)", fontsize=15)
_ = ax.set_yticks(
    [0, 20, 40, 60, 80, 100],
    [0, 20, 40, 60, 80, 100],
    fontsize=15,
)

# ax.figure().savefig("results/senescence_curves.png")
