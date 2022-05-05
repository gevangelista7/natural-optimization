import matplotlib.pyplot as plt


def plot_curves_da(history_J, history_D, history_T, final_i, title):
    history_J = history_J[:final_i]
    history_D = history_D[:final_i]
    history_T = history_T[:final_i]

    plt.figure()
    plt.title(title)
    plt.plot(-history_J, 'r-', label='- Cost function')
    plt.plot(history_D, 'k-', label='Mean Distance')
    plt.plot(history_T, 'b-', label='Temperature')
    plt.legend()
    plt.grid()
    plt.show()

