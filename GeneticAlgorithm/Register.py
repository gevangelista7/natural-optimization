import torch as t
import csv
import time
from datetime import datetime
import os
import numpy as np


class GARegister:
    def __init__(self, algo_name, data_header, filename=None):
        self.data_header = data_header
        begin = datetime.now()

        if filename is not None:
            self.file_name = filename + "_Algo={}__Begin_{}{}{}_{}h{}m{}s.csv" \
                .format(algo_name, begin.year, begin.month, begin.day, begin.hour, begin.minute, begin.second)
        else:
            self.file_name = "res/Algo_{}__{}{}{}_{}h{}m{}s.csv"\
                .format(algo_name, begin.year, begin.month, begin.day, begin.hour, begin.minute, begin.second)
        self.data_entry([])

    def data_entry(self, results):
        file_exists = os.path.isfile(self.file_name)
        try:
            with open(self.file_name, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.data_header)
                if not file_exists:
                    writer.writeheader()
                for data in results:
                    writer.writerow(data)
        except IOError:
            print("I/O error")


class DataPrep:
    def __init__(self, population, fitness):
        self.population = population
        self.fitness = fitness

        self.last_exec_end = time.time()

        self.header = ['iter_n', 'iter_time', 'gen_best_fit', 'gen_best_idv', 'gen_mean_fit', 'gen_worst_fit']
        self.entry_dic = {key: None for key in self.header}

    def processed_result(self, gen_n, eval_counter):
        iter_elapsed_time = time.time() - self.last_exec_end
        self.last_exec_end = time.time()
        gen_best_fit = t.max(self.fitness).item()
        gen_worst_fit = t.min(self.fitness).item()
        gen_mean_fit = t.mean(self.fitness).item()
        gen_best_idv = self.population[t.argmax(self.fitness)].tolist()
        result = {
            'gen_n': gen_n,
            'eval_counter': eval_counter,
            'iter_time': iter_elapsed_time,
            'gen_best_fit': gen_best_fit,
            'gen_mean_fit': gen_mean_fit,
            'gen_worst_fit': gen_worst_fit,
            'gen_best_idv': gen_best_idv
        }
        # if result['gen_best_fit'] != result['gen_best_fit']:
        #     # is nan
        #     raise Exception('ValueError')

        return result


class FinalResultProcessor:
    def __init__(self, offspring_fitness, tgt_fitness):
        self.final_result = {
            'max_fit': -np.inf,
            'best_idv':  None,
            'final_gen_mean_fit': None,
            'final_gen_best_fit': None,
            'final_gen_best_idv': None,
            'elapsed_time': 0,
            'success': None,
            'n_iter': 0
        }
        self.offspring_fitness = offspring_fitness
        self.tgt_fitness = tgt_fitness

    def process_iter(self, iter_result):
        self.final_result['elapsed_time'] += iter_result['iter_time']
        self.final_result['n_iter'] += 1
        if iter_result['gen_best_fit'] > self.final_result['max_fit']:
            self.final_result['max_fit'] = iter_result['gen_best_fit']
            self.final_result['best_idv'] = iter_result['gen_best_idv']

    def process_finnish(self, final_iter_result):
        self.final_result['final_gen_mean_fit'] = t.mean(self.offspring_fitness)
        self.final_result['final_gen_best_fit'] = t.max(self.offspring_fitness)
        self.final_result['final_gen_best_idv'] = final_iter_result['gen_best_idv']
        self.final_result['success'] = (self.final_result['final_gen_best_fit'] > self.tgt_fitness).item()

