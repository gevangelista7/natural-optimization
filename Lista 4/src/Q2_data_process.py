import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from GeneticAlgorithm import GAEvolutionPlot

pathlist = Path('res_q2').glob('rodada*')
n_iter = []
for path in pathlist:
    data = pd.read_csv(path)
    n_iter.append(max(data.iter_n))

general_result_path = 'res_q2/Q2_150_rodadas_Algo=ES__Begin_2022525_11h50m44s.csv'
df = pd.read_csv(general_result_path)

if __name__ == "__main__":
    GAEvolutionPlot("res_q2/rodada103__Algo=ES__Begin_2022525_13h27m41s.csv").plot_evolution().savefig("img/exemplo_ES1")
    GAEvolutionPlot("res_q2/rodada79__Algo=ES__Begin_2022525_13h2m59s.csv").plot_evolution().savefig("img/exemplo_ES2")

    sr = df['success'].value_counts(normalize=True)[True]

    max_fit_ever_overall = df.max_fit
    mean_final_fit_overall = df.final_gen_mean_fit
    elapsed_time_overall = df['elapsed_time']

    success_df = df[df['success'] == True]
    max_fit_ever_success = success_df.max_fit
    mean_final_fit_success = success_df.final_gen_mean_fit
    elapsed_time_success = success_df['elapsed_time']

    f, axes = plt.subplots(2, 1)
    axes[0].hist(max_fit_ever_success, alpha=.75, label="Max fitness ever, success only")
    axes[0].legend()
    axes[1].hist(max_fit_ever_overall, alpha=.75, label="Max fitness ever")
    axes[1].legend()
    plt.show()
    f.savefig("img/hist_max_fit_Q2")

    f, axes = plt.subplots(2, 1)
    axes[0].hist(mean_final_fit_success, alpha=.75, label="Mean final fitness, success only")
    axes[0].legend()
    axes[1].hist(mean_final_fit_overall, alpha=.75, label="Mean final fitness")
    axes[1].legend()
    plt.show()
    f.savefig("img/hist_mean_final_fit_Q2")

    f, axes = plt.subplots(2, 1)
    axes[0].hist(elapsed_time_success, alpha=.75, label="Elapsed time, success only")
    axes[0].legend()
    axes[1].hist(elapsed_time_overall, alpha=.75, label="Elapsed time")
    axes[1].legend()
    plt.show()
    f.savefig("img/elapsed_time_Q2")
