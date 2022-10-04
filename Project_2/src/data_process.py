
import pandas as pd
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mount_conf(row):
    final_conf_name = "{}_m={}_l={}".format(
        row['algo'],
        row['mu'],
        row['lambda']
    )
    if row["epoch"] is not None:
        final_conf_name += "_e={}".format(row['epoch'])

    if row["n_island"] is not None:
        final_conf_name += "_n={}".format(row['n_island'])

    return final_conf_name


def mount_conf_runfile_folder(runfile_folder_name):
    params = re.findall('\d+', runfile_folder_name)

    if len(params) == 2:
        configuration = "{}_m={}_l={}".format(algo, params[1], params[0])
    elif len(params) == 3:
        configuration = "{}_m={}_l={}_e={}".format(algo, params[1], params[0], params[2])
    elif len(params) == 4:
        configuration = "{}_m={}_l={}_e={}_n=3".format(algo, params[1], params[0], params[3], params[2])
    else:
        configuration = 'unknown'

    return configuration, params[0]

if __name__ == "__main__":
    result_data_list = []
    runfile_data_list = []
    for path in Path("../res").glob("*_NC_*"):
        print(path.name)
        algo, _, NC = str(path.stem).split("_", 2)
        for resfile in Path(path).glob("*NC*"):
            df_result_i = pd.read_csv(resfile)
            df_result_i['NC'] = NC
            df_result_i['algo'] = algo

            if 'migration_period' in df_result_i:
                df_result_i.rename(columns={'migration_period': 'epoch'}, inplace=True)
            if 'epoch' not in df_result_i:
                df_result_i['epoch'] = None
            if 'n_island' not in df_result_i:
                df_result_i['n_island'] = None

            result_data_list.append(df_result_i)

    results_df = pd.concat(result_data_list)
    results_df['configuration'] = results_df.apply(lambda row: mount_conf(row), axis=1)
    results_df['SR'] = results_df.apply(lambda x: int(x['success']), axis=1)
    results_df['tgt_fit'] = results_df.apply(lambda x: -float(re.findall("\d+\.\d+", str(x['tgt_fit']))[0]), axis=1)
    results_df['best_fit_rel'] = results_df.apply(lambda x: - abs(x['best_fit'])/abs(x['tgt_fit']), axis=1)


    SR_table = results_df.pivot_table(values='SR',
                                      index=['configuration', 'NC', 'algo'],
                                      aggfunc=[np.mean, 'count'])\
        .sort_values(by=('mean', 'SR'), ascending=False)

    MBF_table = results_df.pivot_table(values='best_fit_rel',
                                       index=['configuration', 'NC', 'algo'],
                                       aggfunc=[np.mean])\
        .sort_values(by=('mean', 'best_fit_rel'), ascending=False)

    AES_table = results_df.pivot_table(values='eval_first_sol',
                                       index=['configuration', 'NC', 'algo'],
                                       aggfunc=[np.nanmean, 'count'])\
        .sort_values(by=('nanmean', 'eval_first_sol'), ascending=True)

    ElapsedTime_table = results_df.pivot_table(values='elapsed_time',
                                               index=['configuration', 'NC', 'algo'],
                                               aggfunc=np.mean)

    ordered_valid_index = SR_table.sort_index(level='configuration')[('count', 'SR')] > 75
    valid_results_sr = SR_table.sort_index(level='configuration')[ordered_valid_index]
    valid_results_mbf = MBF_table.sort_index(level='configuration')[ordered_valid_index]
    valid_results_aes = AES_table.sort_index(level='configuration')[ordered_valid_index]

    best_sr_grouped = valid_results_sr.groupby(by=['algo', 'NC']).max(('mean', 'SR'))['mean', 'SR']
    best_mbf_grouped = valid_results_mbf.groupby(by=['algo', 'NC']).max(('mean', 'best_fit_rel'))['mean', 'best_fit_rel']
    best_aes_grouped = valid_results_aes.groupby(by=['algo', 'NC']).min(('nanmean', 'eval_first_sol'))['nanmean', 'eval_first_sol']

    best_results = valid_results_sr.join(valid_results_mbf).join(valid_results_aes)
    best_results = best_results.loc[best_results.groupby(by=['algo', 'NC']).idxmax()[('mean', 'SR')]]
    best_results = best_results.reorder_levels(['NC', 'algo', 'configuration']).sort_values(('mean', 'SR'), ascending=False)

    best_results.to_csv(Path("../res/Resumo.csv"))

    NC_list = best_results.index.unique("NC").to_list()

    # sns.despine(left=True)

    for nc in NC_list:
        sns.set_theme(style="whitegrid")

        f, axes = plt.subplots(1, 3)
        # res_sr_nc = valid_results_sr[(valid_results_sr.index.get_level_values("NC") == nc)].droplevel("NC").droplevel("algo")
        # best_algo_list = [idx for idx in res_sr_nc.sort_values(('mean', 'SR'), ascending=False).index]

        res_sr_nc = best_results[(best_results.index.get_level_values('NC') == nc)].droplevel("NC").droplevel("algo")
        best_algo_list = [idx for idx in res_sr_nc.sort_values(('mean', 'SR'), ascending=False).index]

        g1 = sns.barplot(x=('mean', 'SR'),
                         y=res_sr_nc.index,
                         order=best_algo_list,
                         data=res_sr_nc,
                         ax=axes[0],
                         zorder=3)
        plt.tight_layout()
        g1.set(xlabel='Success Rate')
        # g1.grid(zorder=0)

        # MBF figure
        res_mbf_nc = results_df[(results_df['configuration'].isin(best_algo_list)) &
                                (results_df['NC'] == nc)]
        g2 = sns.boxplot(x='best_fit_rel',
                         y='configuration',
                         data=res_mbf_nc,
                         order=best_algo_list,
                         orient="h",
                         showfliers=False,
                         ax=axes[1])
        g2.set(yticklabels=[])
        g2.set(xlabel='Best Fit')
        g2.set(ylabel='')
        # g2.grid(zorder=0)

        # AES figure
        res_aes_nc = results_df[(results_df['configuration'].isin(best_algo_list)) &
                                (results_df['NC'] == nc) &
                                (~results_df['eval_first_sol'].isna())]
        g3 = sns.boxplot(x='eval_first_sol',
                         y='configuration',
                         data=res_aes_nc,
                         order=best_algo_list,
                         orient="h",
                         ax=axes[2])
        g3.set(yticklabels=[])
        g3.set(ylabel='')
        g3.set(xlabel='Evaluations to first Solution')
        # g3.grid(zorder=0)

        f.set_figwidth(16)
        f.suptitle('NC='+nc)
        f.tight_layout()
        # plt.show()
        # plt.savefig("../img/NC={}".format(nc))

    sns.set_theme(style="whitegrid")
    f, axes = plt.subplots(1, 3)
    SR_NC_plot = sns.lineplot(x=[int(i) for i in best_sr_grouped._get_label_or_level_values('NC')],
                              y=best_sr_grouped.values,
                              hue=best_sr_grouped._get_label_or_level_values('algo'),
                              ax=axes[0])
    SR_NC_plot.set(xlabel='NC')
    SR_NC_plot.set(ylabel='SR')
    SR_NC_plot.grid()

    MBF_NC_plot = sns.lineplot(x=[int(i) for i in best_mbf_grouped._get_label_or_level_values('NC')],
                               y=best_mbf_grouped.values,
                               hue=best_mbf_grouped._get_label_or_level_values('algo'),
                               ax=axes[1])

    MBF_NC_plot.set(xlabel='NC')
    MBF_NC_plot.set(ylabel='MBF')
    MBF_NC_plot.grid()

    AES_NC_plot = sns.lineplot(x=[int(i) for i in best_aes_grouped._get_label_or_level_values('NC')],
                               y=best_aes_grouped.values,
                               hue=best_aes_grouped._get_label_or_level_values('algo'),
                               ax=axes[2])

    AES_NC_plot.set(xlabel='NC')
    AES_NC_plot.set(ylabel='AES')
    AES_NC_plot.grid()

    f.set_figwidth(16)
    f.suptitle('Behavior with NC')
    f.tight_layout()

    for ax in axes:
        ax.grid()

    # plt.show()
    # plt.savefig("../img/criteria_behavior_with_nc")

