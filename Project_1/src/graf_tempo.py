import matplotlib.pyplot as plt

NC_list = range(10, 17)
np_tds_alpha = [17.50, 36.03, 42.84, 47.66, 58.57, 52.37, 210.46]
tc_tds_alpha = [30.41, 33.70, 25.10, 24.39, 25.08, 27.01, 29.3]
np_alp97 = [19.02, 36.03, 42.84, 47.66, 58.57, 67.69, 390.53]
tc_alp97 = [30.41, 33.70, 30.04, 31.76, 36.15, 38.48, 33.57]


plt.plot(NC_list, np_tds_alpha, label="CPU - todos alpha")
plt.plot(NC_list, np_alp97, label="CPU - alpha >= 0.97")
plt.plot(NC_list, tc_tds_alpha, label="GPU - todos alpha")
plt.plot(NC_list, tc_alp97, label="GPU - alpha >= 0.97")
plt.legend()
plt.xlabel("NC")
plt.ylabel("Tempo médio de execução [s]")
plt.grid()
plt.savefig("comparativo_tempo_exec.png")
plt.show()
