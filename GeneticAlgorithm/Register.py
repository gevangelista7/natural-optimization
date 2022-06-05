import torch as t
import csv
import time
from datetime import datetime
import os
import numpy as np


class GARegister:
    def __init__(self, algo_name, data_header, filename=None, dir_name=None):
        self.data_header = data_header
        begin = datetime.now()
        if dir_name is None:
            dir_name = "./Teste_{}{}{}_{}h{}m{}s"\
                .format(begin.year, begin.month, begin.day, begin.hour, begin.minute, begin.second)

        self.dir_name = dir_name

        try:
            os.mkdir(self.dir_name)
        except OSError as e:
            print("Directory exists")

        self.dir_name = self.dir_name + "/"

        if filename is not None:
            self.file_name = filename + "_{}{}{}_{}h{}m{}s.csv" \
                .format(begin.year, begin.month, begin.day, begin.hour, begin.minute, begin.second)
        else:
            self.file_name = "Algo_{}__{}{}{}_{}h{}m{}s.csv"\
                .format(algo_name, begin.year, begin.month, begin.day, begin.hour, begin.minute, begin.second)

        self.complete_filename = self.dir_name + self.file_name
        self.data_entry([])

    def data_entry(self, results):
        file_exists = os.path.isfile(self.complete_filename)
        try:
            with open(self.complete_filename, 'a') as csvfile:
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

        self.header = ['gen_n',
                       'eval_counter',
                       'iter_time',
                       'gen_best_fit',
                       'gen_mean_fit',
                       'gen_worst_fit',
                       'gen_best_idv']

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

        return result


class FinalResultProcessor:
    def __init__(self, offspring_fitness, tgt_fitness, seed, _lambda, _mu):
        self.final_result = {
            'lambda': _lambda,
            'mu': _mu,

            'n_gen': 0,
            'success': False,
            'best_fit': -np.inf,
            'eval_first_sol': None,

            'tgt_fit': tgt_fitness.item(),
            'elapsed_time': 0,

            'final_eval_counter': 0,
            'final_gen_mean_fit': None,
            'final_gen_best_fit': None,
            'final_gen_worst_fit': None,

            'best_idv': None,
            'final_gen_best_idv': None,

            'seed': seed
        }
        self.offspring_fitness = offspring_fitness
        self.tgt_fitness = tgt_fitness

    def process_iter(self, iter_result):
        self.final_result['elapsed_time'] += iter_result['iter_time']
        self.final_result['n_gen'] += 1

        if iter_result['gen_best_fit'] > self.final_result['best_fit']:
            self.final_result['best_fit'] = iter_result['gen_best_fit']
            self.final_result['best_idv'] = iter_result['gen_best_idv']

        if not self.final_result['success'] and self.final_result['best_fit'] > self.tgt_fitness:
            self.final_result['success'] = True
            self.final_result['eval_first_sol'] = iter_result['eval_counter']

    def process_finnish(self, final_iter_result):
        self.final_result['final_gen_mean_fit'] = t.mean(self.offspring_fitness).item()
        self.final_result['final_gen_best_fit'] = t.max(self.offspring_fitness).item()
        self.final_result['final_gen_worst_fit'] = t.min(self.offspring_fitness).item()
        self.final_result['final_eval_counter'] = final_iter_result['eval_counter']

        self.final_result['final_gen_best_idv'] = final_iter_result['gen_best_idv']
