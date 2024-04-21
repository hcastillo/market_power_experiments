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


class Experiment3:
    N = 100
    T = 1000
    MC = 10
    analyze_data = ['firms_Y', 'firms_A', 'bank_A', 'firms_r']
    OUTPUT_DIRECTORY = "experiment3b"
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
        model.statistics.add(what="bank", name="ploans", symbol="π1", prepend="  ", attr_name="profits_loans")
        model.statistics.add(what="bank", name="rdep", symbol="π2",
                             prepend="  ", attr_name="remunerations_of_deposits_and_networth")
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
    def __get_average_model__(filename, parameters_to_obtain, logarithm=False):
        # pass eta00beta002 as filename, and FirmsY as parameter_to_obtain, and it will generate the average of
        # the ten models we have with that characteristics
        if type(parameters_to_obtain) != type([]):
            parameters_to_obtain = [parameters_to_obtain]
        series = {}
        result = {}
        for param in parameters_to_obtain:
            series[param] = []
            result[param] = 0

        for file in glob.glob(rf'{Experiment3.OUTPUT_DIRECTORY}/{filename}_*.csv'):
            data = pd.read_csv(f"{file}", header=2)
            for param in parameters_to_obtain:
                series[param].append(data[param])
        for param in parameters_to_obtain:
            result[param] = sum(i for i in series[param])/len(series[param])
            if logarithm:
                result[param] = np.log(result[param])
        return result[0] if len(result)==1 else result


    @staticmethod
    def plot_average_y_for_beta():
        plt.clf()
        xx = [i for i in range(1000)]
        fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
        plt.xlabel("t")
        plt.ylabel('log ∑Y')
        for i in (2,3,4,5):
            serie = Experiment3.__get_average_model__(f"eta00beta00{i}","FirmsY", logarithm=True)
            axes[math.floor((i-2) / 2)][i % 2].plot(xx, serie)
            axes[math.floor((i-2) / 2)][i % 2].title.set_text(f'beta=0.00{i}')
            print()
            axes[math.floor((i-2) / 2)][i % 2].text(500, 40, f'$\\bar{{Y}}={round(serie.mean(),3)}$', fontsize=22)
            axes[math.floor((i - 2) / 2)][i % 2].text(500, 10, f'$Y_{{1000}}={round(serie[999], 3)}$', fontsize=22)
        plt.savefig(f'{Experiment3.OUTPUT_DIRECTORY}/beta_avgy.pdf')
        plt.show()

    @staticmethod
    def plot_average_profits_for_beta():
        plt.clf()
        xx = [i for i in range(1000)]
        fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
        plt.xlabel("t")
        plt.ylabel('log ∑π')
        for i in (2, 3, 4, 5):
            serie = Experiment3.__get_average_model__(f"eta00beta00{i}", "FirmsPROFITS", logarithm=True)
            axes[math.floor((i - 2) / 2)][i % 2].plot(xx, serie)
            axes[math.floor((i - 2) / 2)][i % 2].title.set_text(f'beta=0.00{i}')
            axes[math.floor((i - 2) / 2)][i % 2].text(500, 40, f'$\\bar{{\\pi}}={round(serie.mean(), 3)}$', fontsize=22)
            axes[math.floor((i - 2) / 2)][i % 2].text(500, 10, f'$\\pi_{{1000}}={round(serie[997], 3)}$', fontsize=22)
        plt.savefig(f'{Experiment3.OUTPUT_DIRECTORY}/beta_avgprof.pdf')
        plt.show()

    @staticmethod
    def generate_tablesa():
        try:
            os.mkdir("tables")
        except:
            pass

        keys = ['BankL', 'BankA', 'BankD', 'BankPROFITS', 'BankBD', 'FirmsK', 'FirmsA',
                'FirmsL', 'FirmsPROFITS', 'FirmsY', 'FirmsR', 'FirmsI', 'FirmsGAMMA',
                'FirmsU', 'FirmsDK', 'FirmsOL', 'FirmsGAP_OF_L', 'FirmsDL',
                'FirmsFAIL']

        keys_not_logarithm = [  'BankL', 'BankD', 'BankPROFITS',
                                'FirmsA', 'FirmsPROFITS',  'FirmsI', 'FirmsGAMMA',
                                'FirmsU', 'FirmsDK', 'FirmsOL', 'FirmsGAP_OF_L', 'FirmsDL', 'FirmsFAIL']

        results = {}
        std = {}
        for i in keys:
            results[i] = []
            std[i] = []

        rows_index = []
        for beta in Experiment3.get_models(Experiment3.beta):
            rows_index.append(beta['beta'])
            cols_index = []
            row_results = {}
            row_std = {}
            for i in keys:
                row_results[i] = []
                row_std[i] = []
            for eta in Experiment3.get_models(Experiment3.eta):
                cols_index.append(eta['eta'])
                values = beta.copy()
                values.update(eta)
                model_name = Experiment3.get_filename_for_iteration(values)
                model_data = pd.read_csv(f"experiment3a\\{model_name}_0.csv", header=2)
                for i in range(1, Experiment3.MC):
                    new_data = pd.read_csv(f"experiment3a\\{model_name}_{i}.csv", header=2)
                    model_data = model_data.append(new_data)

                for i in keys:
                    if i not in keys_not_logarithm:
                        row_results[i].append(np.log(model_data[i]).mean())
                        row_std[i].append(np.log(model_data[i]).std())
                    else:
                        row_results[i].append(model_data[i].mean())
                        row_std[i].append(model_data[i].std())
            for i in keys:
                results[i].append(row_results[i])
                std[i].append(row_std[i])

        for i in keys:
            data_frame = pd.DataFrame( data=results[i], columns=cols_index, index=rows_index)
            data_frame_std = pd.DataFrame( data=std[i], columns=cols_index, index=rows_index)
            data_frame.to_pickle(f'tablesa\\{i}.pickle')
            data_frame_std.to_pickle(f'tablesa\\{i}_std.pickle')
        print("results are in directory 'tablesA'")


    def generate_tablesb():
        try:
            os.mkdir("tablesb")
        except:
            pass

        keys = ['BankL', 'BankA', 'BankD', 'BankPROFITS', 'BankBD', 'FirmsK', 'FirmsA',
                'FirmsL', 'FirmsPROFITS', 'FirmsY', 'FirmsR', 'FirmsI', 'FirmsGAMMA',
                'FirmsU', 'FirmsDK', 'FirmsOL', 'FirmsGAP_OF_L', 'FirmsDL',
                'FirmsFAIL']

        keys_not_logarithm = [  'BankL', 'BankD', 'BankPROFITS',
                                # 'FirmsA',
                                #'FirmsPROFITS',
                                'FirmsI', 'FirmsGAMMA',
                                'FirmsU', 'FirmsDK', 'FirmsOL', 'FirmsGAP_OF_L', 'FirmsDL', 'FirmsFAIL']
        results = {}
        std = {}
        for i in keys:
            results[i] = []
            std[i] = []

        rows_index = []
        for eta in Experiment3.get_models(Experiment3.eta):
            rows_index.append(eta['eta'])
            cols_index = []
            row_results = {}
            row_std = {}
            for i in keys:
                row_results[i] = []
                row_std[i] = []
            for beta in Experiment3.get_models(Experiment3.beta):
                cols_index.append(beta['beta'])
                values = eta.copy()
                values.update(beta)
                model_name = Experiment3.get_filename_for_iteration(values)
                model_data = pd.read_csv(f"experiment3b\\{model_name}_0.csv", header=2)
                for i in range(1, Experiment3.MC):
                    new_data = pd.read_csv(f"experiment3b\\{model_name}_{i}.csv", header=2)
                    model_data = model_data.append(new_data)

                for i in keys:
                    if i not in keys_not_logarithm:
                        row_results[i].append(np.log(model_data[i]).mean())
                        row_std[i].append(np.log(model_data[i]).std())
                    else:
                        row_results[i].append(model_data[i].mean())
                        row_std[i].append(model_data[i].std())
            for i in keys:
                results[i].append(row_results[i])
                std[i].append(row_std[i])

        for i in keys:
            data_frame = pd.DataFrame( data=results[i], columns=cols_index, index=rows_index)
            data_frame_std = pd.DataFrame( data=std[i], columns=cols_index, index=rows_index)
            data_frame.to_pickle(f'tablesb\\{i}.pickle')
            data_frame_std.to_pickle(f'tablesb\\{i}_std.pickle')
        print("results are in directory 'tablesb'")

    @staticmethod
    def plot_average_bank_for_beta():
        plt.clf()
        xx = [i for i in range(1000)]
        fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
        plt.xlabel("t")
        for i in (2, 3, 4, 5):
            elements_to_plot = {"BankRDEP":['g','Deposits'],
                                "BankPLOANS":['r','Loans'],
                                "BankPROFITS":['bo','Profits'] }
            series = Experiment3.__get_average_model__(f"eta00beta00{i}",
                                                       list(elements_to_plot.keys()),
                                                       logarithm=True)
            for j in elements_to_plot:
                axes[math.floor((i - 2) / 2)][i % 2].plot(xx, series[j], elements_to_plot[j][0],
                                                          label=elements_to_plot[j][1])
            axes[math.floor((i - 2) / 2)][i % 2].title.set_text(f'beta=0.00{i}')
            axes[math.floor((i - 2) / 2)][i % 2].text(500, 40, f'$\\bar{{\\pi}}={round(series["BankPROFITS"].mean(), 3)}$', fontsize=22)
        plt.legend(loc='best')
        plt.savefig(f'{Experiment3.OUTPUT_DIRECTORY}/beta.pdf')
        #plt.show()

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
            os.mkdir(Experiment3.OUTPUT_DIRECTORY+"/bad")

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
        num_models_analyzed = 0
        log_experiment = open(f'{Experiment3.OUTPUT_DIRECTORY}/experiment3.txt', 'a')
        model.test = False
        progress_bar = Bar('Executing models', max=Experiment3.get_num_models(Experiment3.beta) *
                                                   Experiment3.get_num_models(Experiment3.eta))
        progress_bar.update()
        for eta in Experiment3.get_models(Experiment3.eta):
            results_to_plot = {}
            results_x_axis = []
            for beta in Experiment3.get_models(Experiment3.beta):
                values = eta.copy()
                values.update(beta)
                filename_for_iteration = Experiment3.get_filename_for_iteration(values)

                if os.path.exists(f"{Experiment3.OUTPUT_DIRECTORY}/{filename_for_iteration}_ddf.png"):
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

                Experiment3.plot_ddf(result_iteration, f"{beta}{eta}",
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
                results_x_axis.append(str(beta['beta']) + ("*" if aborted_models else ""))

                if aborted_models:
                    result_iteration_values = (f"aborted_models={aborted_models}/{Experiment3.MC} "
                                               + result_iteration_values)
                print(f"model #{num_models_analyzed} {filename_for_iteration}: {values}: {result_iteration_values}",
                                file=log_experiment)
                num_models_analyzed += 1
                progress_bar.next()
            Experiment3.plot(results_to_plot, results_x_axis, eta, "beta",
                             f"{Experiment3.OUTPUT_DIRECTORY}/eta{eta['eta']}")

        log_experiment.close()
        progress_bar.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ABM parameters")
    parser.add_argument('--do', default=False, action=argparse.BooleanOptionalAction,
                        help="Execute the experiment")
    parser.add_argument('--ploty', default=False, action=argparse.BooleanOptionalAction,
                        help="Obtains a plot with 4 subplots of Y for beta=0.02 etc")
    parser.add_argument('--profits', default=False, action=argparse.BooleanOptionalAction,
                        help="Obtains a plot with 4 subplots of profits for beta=0.02 etc")
    parser.add_argument('--bank', default=False, action=argparse.BooleanOptionalAction,
                        help="Obtains plots with 4 subplots of bank indicators for beta=0.02 etc")
    parser.add_argument('--listnames', default=False, action=argparse.BooleanOptionalAction,
                        help="Print combinations to generate")
    parser.add_argument('--tablesa', default=False, action=argparse.BooleanOptionalAction,
                        help="Generate the csv tables for Experiment3a")
    parser.add_argument('--tablesb', default=False, action=argparse.BooleanOptionalAction,
                        help="Generate the csv tables for Experiment3a")
    args = parser.parse_args()
    if args.listnames:
        Experiment3.listnames()
    elif args.ploty:
        Experiment3.plot_average_y_for_beta()
    elif args.profits:
        Experiment3.plot_average_profits_for_beta()
    elif args.bank:
        Experiment3.plot_average_bank_for_beta()
    elif args.tablesa:
        Experiment3.generate_tablesa()
    elif args.tablesb:
        Experiment3.generate_tablesb()
    elif args.do:
        Experiment3.do(Model(export_datafile="exec", test=True))
