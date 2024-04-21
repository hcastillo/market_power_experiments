#!/usr/bin/env python
# coding: utf-8
"""
The results are generated with experiment3a, here we open the files csv to plot correctly the std() and errorbars
@author: hector@bith.net
"""
import random
import numpy as np
import util.utilities as utilities
from util.stats_array import PlotMethods
import statistics
from market_power.model import Model
import warnings
import pandas as pd
from progress.bar import Bar
import matplotlib.pyplot as plt
import os, sys
import math
import argparse
import scipy
import glob, shutil


class Experiment3:
    N = 100
    T = 1000
    MC = 10
    analyze_data = ['firms_Y', 'firms_A', 'bank_A', 'firms_r']
    OUTPUT_DIRECTORY = "experiment3a"
    number_of_tries_in_case_of_abort = 3
    beta = {
        # if instead of a list, you want to use a range, use np.arange(start,stop,step) to generate the list:
        # 'eta': np.arange(0.00001,0.9, 0.1) --> [0.0001, 0.1001, 0.2001... 0.9001]
        #
        'beta': [0.02, 0.03, 0.04, 0.05]
    }
    eta = {
        'eta': [0.0001, 0.1, 0.3, 0.5, 0.8]
    }

    @staticmethod
    def run_model(filename):
        data = pd.read_csv(filename, header=2)
        return data

    @staticmethod
    def plot(array_with_data, array_with_x_values, title, title_x, filename):
        for i in array_with_data:
            # each element in array will have a log_is_used bool, but we take the first for all:
            use_logarithm = array_with_data[i][0][2]
            mean = []
            standard_deviation = []
            for j in array_with_data[i]:
                # mean is 0, std is 1:
                mean.append(j[0])
                standard_deviation.append(j[1]/2)
            plt.clf()
            plt.title(title)
            plt.xlabel(title_x)
            plt.title(f"{title} log({i})" if use_logarithm else f"{title} {i}")
            try:
                plt.errorbar(array_with_x_values, mean, standard_deviation, linestyle='None', marker='^')
            except:
                pass
            else:
                plt.savefig(f"{filename}_{i}.png", dpi=300)
                with open(f"{filename}_{i}.txt", "w") as results:
                    results.write(f"{title} log({i})" if use_logarithm else f"{title} {i}\n")
                    for ii in range(len(array_with_x_values)):
                        results.write(f"eta={array_with_x_values[ii]} {i}={mean[ii]} std={standard_deviation[ii]}\n")


    @staticmethod
    def get_num_models(parameters):
        return len(list(Experiment3.get_models(parameters)))

    @staticmethod
    def get_models(parameters):
        return utilities.cartesian_product(parameters)


    @staticmethod
    def get_filename_for_iteration(parameters):
        result = str(parameters)
        for r in "{}',:. ":
            result = result.replace(r, "")
        return result.replace("00001", "00")


    @staticmethod
    def __verify_directories__():
        if not os.path.isdir(Experiment3.OUTPUT_DIRECTORY):
            os.mkdir(Experiment3.OUTPUT_DIRECTORY)


    @staticmethod
    def listnames():
        num = 0
        for parameters_not_eta in Experiment3.get_models(Experiment3.beta):
            for eta in Experiment3.get_models(Experiment3.eta):
                values = parameters_not_eta.copy()
                values.update(eta)
                model_name = Experiment3.get_filename_for_iteration(values)
                print(model_name)
                num += 1
        print("total: ", num)


    @staticmethod
    def do(model: Model):
        Experiment3.__verify_directories__()
        progress_bar = Bar('Executing models', max=Experiment3.get_num_models(Experiment3.beta) *
                                                   Experiment3.get_num_models(Experiment3.eta))
        progress_bar.update()
        for parameters_not_eta in Experiment3.get_models(Experiment3.beta):
            results_to_plot = {}
            log_results = {}
            results_x_axis = []
            for eta in Experiment3.get_models(Experiment3.eta):
                values = parameters_not_eta.copy()
                values.update(eta)
                filename_for_iteration = Experiment3.get_filename_for_iteration(values)

                result_iteration = pd.DataFrame()
                for i in range(Experiment3.MC):
                    result_mc = Experiment3.run_model(f"{Experiment3.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv")
                    result_iteration = pd.concat([result_iteration, result_mc])

                for k in result_iteration.keys():
                    k = k.strip()
                    if k == 't':
                        continue

                    if not k in results_to_plot:
                        log_results[k] = (result_iteration[k].max() - result_iteration[k].min()) > 1e6

                    if log_results[k]:
                        result_with_log = result_iteration[k].replace(0.0, 1e-10).apply(np.log)
                        mean_estimated = result_with_log.mean()
                        std_estimated = result_with_log.std()
                    else:
                        mean_estimated = result_iteration[k].mean()
                        std_estimated = result_iteration[k].std()
                    if k in results_to_plot:
                        results_to_plot[k].append([mean_estimated, std_estimated, log_results[k]])
                    else:
                        results_to_plot[k] = [[mean_estimated, std_estimated, log_results[k]]]
                results_x_axis.append(str(eta['eta']))
                progress_bar.next()
            Experiment3.plot(results_to_plot, results_x_axis, f"$\\beta={parameters_not_eta['beta']}$", "eta",
                             f"{Experiment3.OUTPUT_DIRECTORY}/beta{parameters_not_eta['beta']}")

        progress_bar.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ABM parameters")
    parser.add_argument('--do', default=False, action=argparse.BooleanOptionalAction,
                        help="Execute the experiment")
    parser.add_argument('--listnames', default=False, action=argparse.BooleanOptionalAction,
                        help="Print combinations to generate")
    args = parser.parse_args()
    if args.listnames:
        Experiment3.listnames()
    elif args.do:
        Experiment3.do(Model(export_datafile="exec", test=True))
