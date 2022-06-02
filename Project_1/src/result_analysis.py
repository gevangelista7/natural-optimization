import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

df = pd.read_csv('resultados_sensibilidade_NC16.csv')
df = df.round({"alpha": 3})
fig, ax = plt.subplots(2, 3)
table1 = pd.pivot_table(df, values='satisfactory_result', index='alpha', columns='T0')
ax[1, 0] = sns.heatmap(table1, cmap='viridis')
ax[0, 0].set_xlabel = 'alpha'
ax[0, 0].set_ylabel = 'T0'
fig.suptitle('Taxa de sucesso')
plt.show()

#
# table2 = pd.pivot_table(df, values='satisfactory_result', index='alpha', columns='eps')
# table3 = pd.pivot_table(df, values='satisfactory_result', index='alpha', columns='delta')
# table4 = pd.pivot_table(df, values='satisfactory_result', index='T0', columns='eps')
# table5 = pd.pivot_table(df, values='satisfactory_result', index='T0', columns='delta')
# table6 = pd.pivot_table(df, values='satisfactory_result', index='eps', columns='delta')
#
# table7 = pd.pivot_table(df, values='final_J', index='alpha', columns='T0')
# table8 = pd.pivot_table(df, values='final_J', index='alpha', columns='eps')
# table9 = pd.pivot_table(df, values='final_J', index='alpha', columns='delta')
# table10 = pd.pivot_table(df, values='final_J', index='T0', columns='eps')
# table11 = pd.pivot_table(df, values='final_J', index='T0', columns='delta')
# table12 = pd.pivot_table(df, values='final_J', index='eps', columns='delta')
#
