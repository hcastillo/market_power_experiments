#!/usr/bin/env python
# coding: utf-8
"""
ABM model calibrator, using the model
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
import nbformat as nbf
from IPython.display import Markdown


class Experiment3:
    N = 100
    T = 1000
    MC = 10
    analyze_data = ['firms_Y', 'firms_A', 'bank_A', 'firms_r']
    OUTPUT_DIRECTORY = "experiment3a"
    OUTPUT_TABLES = "tablesa"
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
    def manage_stats_options(model):
        model.statistics.add(what="bank", name="L", prepend="bank    ")
        model.statistics.add(what="bank", name="A", prepend=" | ", logarithm=True)
        model.statistics.add(what="bank", name="D", prepend="  ")
        model.statistics.add(what="bank", name="profits", symbol="π", prepend="  ", attr_name="profits")
        model.statistics.add(what="bank", name="bad debt", logarithm=True,
                             symbol="bd", prepend=" ", attr_name="bad_debt")
        model.statistics.add(what="firms", name="K", prepend="\n              firms   ", logarithm=True)
        model.statistics.add(what="firms", name="A", prepend=" |")
        model.statistics.add(what="firms", name="L", prepend=" ", logarithm=True)
        model.statistics.add(what="firms", name="profits", prepend=" ", symbol="π", attr_name="pi")
        model.statistics.add(what="firms", name="Y", prepend=" ", logarithm=True)
        model.statistics.add(what="firms", name="r", prepend=" ", function=statistics.mean)
        model.statistics.add(what="firms", name="I", prepend=" ")
        model.statistics.add(what="firms", name="gamma", prepend=" ", function=statistics.mean, symbol="γ")
        model.statistics.add(what="firms", name="u", function=statistics.mean, repr_function="¯")
        model.statistics.add(what="firms", name="desiredK", symbol="dK", show=False)
        model.statistics.add(what="firms", name="offeredL", symbol="oL", show=False, function=statistics.mean)
        model.statistics.add(what="firms", name="gap_of_L", show=False)
        model.statistics.add(what="firms", name="demandL", symbol="dL", show=False, function=statistics.mean)
        model.statistics.add(what="firms", name="failures", attr_name="failed", symbol="fail",
                             number_type=int, prepend=" ")

    @staticmethod
    def run_model(model: Model, description, values):
        values["T"] = Experiment3.T
        values["N"] = Experiment3.N
        model.statistics.interactive = False
        model.configure(**values)
        model.statistics.define_output_directory(Experiment3.OUTPUT_DIRECTORY)
        Experiment3.manage_stats_options(model)
        if description.endswith("_0"):
            # only the first model is plotted:
            model.statistics.enable_plotting(plot_format=PlotMethods.get_default(),
                                             plot_min=0, plot_max=Experiment3.T,
                                             plot_what='firms_Y,firms_failures,bank_bad_debt')
        number_of_tries_in_case_of_abort = Experiment3.number_of_tries_in_case_of_abort
        while number_of_tries_in_case_of_abort > 0:
            data, _ = model.run(export_datafile=description)
            if len(data) == Experiment3.T:
                # when exactly T iterations of data are returned = OK
                number_of_tries_in_case_of_abort -= 1
            else:
                # in case of aborted, we change the seed and run it again:
                model.t = 0
                model.config.default_seed += 1
                model.abort_execution = False
                model.config.T = Experiment3.T
        return data

    @staticmethod
    def plot(array_with_data, array_with_x_values, title, title_x, filename):
        for i in array_with_data:
            use_logarithm = abs(array_with_data[i][0][0] - array_with_data[i][1][0]) > 1e10
            mean = []
            standard_deviation = []
            for j in array_with_data[i]:
                # mean is 0, std is 1:
                mean.append(np.log(j[0]) if use_logarithm else j[0])
                standard_deviation.append(np.log(j[1] / 2) if use_logarithm else j[1] / 2)
            plt.clf()
            plt.title(title)
            plt.xlabel(title_x)
            plt.title(f"log({i})" if use_logarithm else f"{i}")
            try:
                plt.errorbar(array_with_x_values, mean, standard_deviation, linestyle='None', marker='^')
            except:
                pass
            else:
                plt.savefig(f"{filename}_{i}.png", dpi=300)

    @staticmethod
    def plot_ddf(array_with_data, title, filename):
        plt.clf()
        plt.title(title + " ddf")
        plt.xlabel("log firm_a")
        plt.ylabel("log rank")
        xx = []
        yy = []
        sorted = array_with_data['firms_A'].sort_values(ascending=False)
        j = 1
        for i in sorted:
            if not np.isnan(i) and i > 0:
                xx.append(math.log(i))
                yy.append(math.log(j))
                j += 1
        plt.plot(xx, yy,"ro")
        plt.savefig(f"{filename}_ddf.png", dpi=300)

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
    def generate_tables():
        try:
            os.mkdir(Experiment3.OUTPUT_TABLES)
        except:
            pass


        # to determine the keys (FirmsK, FirmsL... ) we open the first .csv:
        example_csv = pd.read_csv(glob.glob(f"{Experiment3.OUTPUT_DIRECTORY}\\*_0.csv")[0], header=2)
        keys = list(example_csv.keys()[1:])  # we remove the 't', which is the first

        results = {}
        results_std = {}
        results_log = {}
        results_log_std = {}

        for i in keys:
            results[i] = []
            results_std[i] = []
            results_log[i] = []
            results_log_std[i] = []

        rows_index = []
        for beta in Experiment3.get_models(Experiment3.beta):
            rows_index.append(beta['beta'])
            cols_index = []
            row_results = {}
            row_results_std = {}
            row_results_log = {}
            row_results_log_std = {}
            for i in keys:
                row_results[i] = []
                row_results_std[i] = []
                row_results_log[i] = []
                row_results_log_std[i] = []
            for eta in Experiment3.get_models(Experiment3.eta):
                cols_index.append(eta['eta'])
                values = beta.copy()
                values.update(eta)
                model_name = Experiment3.get_filename_for_iteration(values)
                model_data = pd.read_csv(f"experiment3a\\{model_name}_0.csv", header=2)
                for i in range(1, Experiment3.MC):
                    new_data = pd.read_csv(f"experiment3a\\{model_name}_{i}.csv", header=2)
                    model_data = pd.concat([model_data, new_data])

                for i in keys:
                    row_results[i].append(model_data[i].mean())
                    row_results_std[i].append(model_data[i].std())
                    with np.seterr(divide='ignore'):
                        temporal_dataframe = np.log(model_data[i])
                        temporal_dataframe[temporal_dataframe == -np.inf] = np.NaN
                        temporal_dataframe[temporal_dataframe == np.inf] = np.NaN
                        row_results_log[i].append(temporal_dataframe.mean())
                        row_results_log_std[i].append(temporal_dataframe.std())
            for i in keys:
                results[i].append(row_results[i])
                results_std[i].append(row_results_std[i])
                results_log[i].append(row_results_std[i])
                results_log_std[i].append(row_results_std[i])

        for i in keys:
            data_frame = pd.DataFrame(data=results[i], columns=cols_index, index=rows_index)
            data_frame.to_pickle(f'{Experiment3.OUTPUT_TABLES}\\{i}.pkl')
            data_frame_std = pd.DataFrame(data=results_std[i], columns=cols_index, index=rows_index)
            data_frame_std.to_pickle(f'{Experiment3.OUTPUT_TABLES}\\{i}_std.pkl')
            data_frame_log= pd.DataFrame(data=results_log[i], columns=cols_index, index=rows_index)
            data_frame_log.to_pickle(f'{Experiment3.OUTPUT_TABLES}\\{i}_log.pkl')
            data_frame_log_std = pd.DataFrame(data=results_log_std[i], columns=cols_index, index=rows_index)
            data_frame_log_std.to_pickle(f'{Experiment3.OUTPUT_TABLES}\\{i}_log_std.pkl')
        print(f"results are in directory {Experiment3.OUTPUT_TABLES}")


    @staticmethod
    def generate_notebook():
        header = """---
title: "Data of exploration over parameters in the ABM model ($2^{nd}$ part)"
author: "Hector Castillo"
format: pdf
toc: true
number-sections: false
jupyter: python3
ipynb-shell-interactivity: all

---"""
        nb = nbf.v4.new_notebook()

        # to determine the keys (FirmsK, FirmsL... ) we open the first .csv:
        example_csv = pd.read_csv(glob.glob(f"{Experiment3.OUTPUT_DIRECTORY}\\*_0.csv")[0], header=2)
        keys = list(example_csv.keys()[1:])  # we remove the 't', which is the first

        nb.cells = [ nbf.v4.new_markdown_cell(header)]
        for i in keys:
            nb.cells.append( nbf.v4.new_markdown_cell("{{< pagebreak >}} \n"+f"## {i}"))
            for j in ("","_std","_log","_log_std"):
                data = pd.read_pickle(f"{Experiment3.OUTPUT_TABLES}\\{i}{j}.pkl")
                nb.cells.append(
                    # nbf.v4.new_code_cell(
                    nbf.v4.new_markdown_cell(
                        f"### {j}\n" +
                        '$\\begin{matrix} & \\eta \\\\ \\beta & \\end{matrix}$' + data.to_markdown()[1:]))
        nbf.write(nb, 'test.ipynb')

    @staticmethod
    def do(model: Model):
        Experiment3.__verify_directories__()
        num_models_analyzed = 0
        log_experiment = open(f'{Experiment3.OUTPUT_DIRECTORY}/experiment3.txt', 'a')
        model.test = False
        progress_bar = Bar('Executing models', max=Experiment3.get_num_models(Experiment3.beta) *
                                                   Experiment3.get_num_models(Experiment3.eta))
        progress_bar.update()
        for parameters_not_eta in Experiment3.get_models(Experiment3.beta):
            results_to_plot = {}
            results_x_axis = []
            for eta in Experiment3.get_models(Experiment3.eta):
                values = parameters_not_eta.copy()
                values.update(eta)
                filename_for_iteration = Experiment3.get_filename_for_iteration(values)

                if os.path.exists(f"{Experiment3.OUTPUT_DIRECTORY}/{filename_for_iteration}_ddf.png") or \
                   os.path.exists(f"{Experiment3.OUTPUT_DIRECTORY}/bad/{filename_for_iteration}_ddf.png"):
                    progress_bar.next()
                    continue

                result_iteration = pd.DataFrame()
                aborted_models = 0
                for i in range(Experiment3.MC):
                    model_was_aborted = False
                    mc_iteration = random.randint(9999, 20000)
                    values['default_seed'] = mc_iteration
                    result_mc = Experiment3.run_model(model,
                                                      f"{Experiment3.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}",
                                                      values)
                    # rare, but still possible: we try 3 times with diff seed if aborted inside run_model
                    if len(result_mc) != Experiment3.T:
                        aborted_models += 1
                        model_was_aborted = True
                    result_iteration = pd.concat([result_iteration, result_mc])
                    coef_corr = scipy.stats.spearmanr(result_mc.firms_Y.to_numpy(), [i for i in range(len(result_mc))])
                    if coef_corr.statistic > 0.99 or model_was_aborted:
                        # plot of Y is a straight line, no shocks so this model is useless:
                        # also if the model was aborted
                        for file in glob.glob(rf'{Experiment3.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}*'):
                            try:
                                shutil.move(file, Experiment3.OUTPUT_DIRECTORY + "/bad")
                            except:
                                pass

                Experiment3.plot_ddf(result_iteration, f"{parameters_not_eta}{eta}",
                                     f"{Experiment3.OUTPUT_DIRECTORY}/{filename_for_iteration}")
                result_iteration_values = ""
                for k in result_iteration.keys():
                    mean_estimated = result_iteration[k].mean()
                    warnings.filterwarnings('ignore')  # it generates RuntimeWarning: overflow encountered in multiply
                    std_estimated = result_iteration[k].std()
                    if k in results_to_plot:
                        results_to_plot[k].append([mean_estimated, std_estimated])
                    else:
                        results_to_plot[k] = [[mean_estimated, std_estimated]]
                    result_iteration_values += f" {k}[avg:{mean_estimated},std:{std_estimated}]"
                del values['default_seed']
                del values['T']
                del values['N']
                results_x_axis.append(str(eta['eta']) + ("*" if aborted_models else ""))

                if aborted_models:
                    result_iteration_values = (f"aborted_models={aborted_models}/{Experiment3.MC} "
                                               + result_iteration_values)
                print(f"model #{num_models_analyzed} {filename_for_iteration}: {values}: {result_iteration_values}",
                                file=log_experiment)
                num_models_analyzed += 1
                progress_bar.next()
            Experiment3.plot(results_to_plot, results_x_axis, parameters_not_eta, "eta",
                             f"{Experiment3.OUTPUT_DIRECTORY}/beta{parameters_not_eta['beta']}")

        log_experiment.close()
        progress_bar.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ABM parameters")
    parser.add_argument('--do', default=False, action=argparse.BooleanOptionalAction,
                        help="Execute the experiment")
    parser.add_argument('--tables', default=False, action=argparse.BooleanOptionalAction,
                        help="Generate the csv tables (1st)")
    parser.add_argument('--notebook', default=False, action=argparse.BooleanOptionalAction,
                        help="Generate the csv tables (2nd, after 1st)")
    parser.add_argument('--listnames', default=False, action=argparse.BooleanOptionalAction,
                        help="Print combinations to generate")
    args = parser.parse_args()
    if args.listnames:
        Experiment3.listnames()
    elif args.tables:
        Experiment3.generate_tables()
    elif args.notebook:
        Experiment3.generate_notebook()
    elif args.do:
        Experiment3.do(Model(export_datafile="exec", test=True))
