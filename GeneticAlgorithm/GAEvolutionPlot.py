import pandas as pd
import matplotlib.pyplot as plt


class GAEvolutionPlot:
    def __init__(self, file_name, title=False):
        self.file_name = file_name
        self.title = title if title else file_name
        self.data = pd.read_csv(file_name, header=0)
        self.best_fit_curve = [None]+[max(self.data.gen_best_fit[:i]) for i in range(1, len(self.data.gen_best_fit))]

    def plot_evolution(self):
        f = plt.figure()
        plt.title(self.title)
        plt.fill_between(self.data.gen_n, self.data.gen_worst_fit, self.data.gen_best_fit, color='b', alpha=0.2)
        plt.plot(self.data.gen_n, self.data.gen_mean_fit,  'b-', label='Generation Mean Fitness')
        plt.plot(self.data.gen_n, self.best_fit_curve, 'r-', label="History Best fit")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid()
        plt.show()
        return f



