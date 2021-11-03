'''
Created on 2020-09-15 11:09:06
Last modified on 2020-09-30 11:40:17

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# standard library
import os
import pickle
from collections import OrderedDict

# third-party
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# local library
from .utils import get_sims_info
from ..utils.file_handling import InfoReport
from ..utils.plot import BarPlot
from ..post_processing import collect_raw_data


# object definition

def analyze_times(example_name, data_filename='DoE.pkl',
                  sims_dir_name='analyses', raw_data_filename='raw_data.pkl',
                  print_info=True, report='', show_figures=False):
    '''
    Collects times and performs a simple analsis.

    Parameters
    ----------
    show_figures : bool
        It does not work if code runs from a notebook.
    '''
    # TODO: expand

    # initialization
    figs = []

    # running information
    info = get_sims_info(example_name, data_filename=data_filename,
                         sims_dir_name=sims_dir_name, print_info=False, report='')
    info.append(InfoReport(sections=[('time', 'Time-related information:')]))

    # collect times
    times_df, nested_times_df = collect_times(
        example_name, data_filename=data_filename,
        sims_dir_name=sims_dir_name, raw_data_filename=raw_data_filename)

    # total times
    time_info = info['time']
    column_names = times_df.columns
    totals, means, stds = times_df.sum().to_list(), times_df.mean().to_list(), times_df.std().to_list()
    total_col = np.argmax(totals)
    percs = np.array(totals) / totals[total_col] * 100
    for i, (column_name, total, mean, std, perc) in enumerate(zip(column_names, totals, means, stds, percs)):
        if i == total_col:
            time_info.add_info('{} time [s]: {:.2f}'.format(
                column_name.capitalize().replace('_', '-'), total))
        else:
            time_info.add_info('{} time [s]: {:.2f} ({:.2f}%)'.format(
                column_name.capitalize().replace('_', '-'), total, perc))
        time_info.add_info('{} time for each simulation (mean [s] and std [s]): {:.2f} and {:.2f}'.format(
            column_name.capitalize().replace('_', '-'), mean, std))

    # info for bar plots
    labels = [column_name.capitalize().replace('_', '-') for column_name in column_names]
    colors = len(labels) * [plt.rcParams['axes.prop_cycle'].by_key()['color'][0]]
    colors[total_col] = 'r'

    # create total time bar plot
    bar_plot = BarPlot().plot(totals, tick_label=labels, color=colors)
    text = [perc / 100 for perc in percs]
    text[total_col] = None
    bar_plot.add_text(totals, text=text, verticalalignment='bottom',
                      middle=False)
    bar_plot.ax.set_ylabel('Total time /$s$')
    figs.append(bar_plot)

    # create mean time bar plot
    bar_plot = BarPlot().plot(means, tick_label=labels, color=colors,
                              yerr=stds, capsize=10)
    bar_plot.ax.set_ylabel('Mean time /$s$')
    figs.append(bar_plot)

    # deal with models details
    column_names = nested_times_df.columns.levels[0].to_list()
    if len(column_names) > 1:
        model_names = nested_times_df.columns.levels[1].to_list()
        swapped_nested_times_df = nested_times_df.swaplevel(axis=1)
        totals, means, stds = [], [], []
        for model_name in model_names:
            totals.append(swapped_nested_times_df[model_name].sum().to_list())
            means.append(swapped_nested_times_df[model_name].mean().to_list())
            stds.append(swapped_nested_times_df[model_name].std().to_list())
        # info bar plots
        labels = [column_name for column_name in column_names]
        # total time bar plot
        bar_plot = BarPlot().plot(totals, tick_label=labels,)
        bar_plot.ax.set_ylabel('Total time /$s$')
        totals_ = np.sum(totals, axis=0)
        with np.errstate(all='ignore'):
            percs = np.array(totals) / totals_
        bar_plot.add_text(totals, text=percs, verticalalignment='bottom',
                          middle=False)
        figs.append(bar_plot)
        # mean bar plot
        bar_plot = BarPlot().plot(means, tick_label=labels, yerr=stds,
                                  capsize=10)
        bar_plot.ax.set_ylabel('Mean time /$s$')
        figs.append(bar_plot)

    # print information
    if print_info:
        info.print_info()

    # create report
    if report:
        with open(report, 'w') as file:
            info.write_report(file)

    # show figures
    if show_figures:
        plt.show()

    return info, figs


def collect_times(example_name, data_filename='DoE.pkl',
                  sims_dir_name='analyses', raw_data_filename='raw_data.pkl'):
    '''
    Parameters
    ----------
    raw_data_filename: str
        Name of the concatenated dict file. If empty, then times will be
        collected from the raw data pickle file.

    Notes
    -----
    1. Collects information only of successful simulations.
    '''

    # get successful simulations
    with open(os.path.join(example_name, data_filename), 'rb') as file:
        data = pickle.load(file)
    successful_sims = data['run_info']['successful_sims']

    # get raw_data
    raw_data = collect_raw_data(example_name, sims_dir_name=sims_dir_name,
                                sim_numbers=successful_sims, delete=False,
                                raw_data_filename=raw_data_filename)

    # collect times
    times_sim = pd.Series(
        [value['time'] for value in raw_data.loc[successful_sims].values],
        index=successful_sims)

    # reorganize times
    times_df, nested_times_df = _reorganize_times(times_sim)

    return times_df, nested_times_df


def _reorganize_times(times_sim):
    '''
    Parameters
    ----------
    times : pandas.Series (with dict or OrderedDict)
        Output of `collect_times`.

    Notes
    -----
    1. If there's only one type of simulation per point, then there's little
    difference between e.g. `run_times` and `run_times_mdl`: only the type of
    variable changes (from list to dict).
    '''

    # initialization
    first_sim = times_sim.iloc[0]
    time_names = list(first_sim.keys())
    nested_times_names = [key for key, value in first_sim.items() if type(value) in [dict, OrderedDict]]
    nonnested_times_names = [time_name for time_name in time_names if time_name not in nested_times_names]
    model_names = list(first_sim[nested_times_names[0]].keys())

    # store nested and non-nested data
    data_nested = []
    data = []
    for _, outer_value in times_sim.iteritems():
        data.append([])
        data_nested.append([])
        for inner_key, inner_value in outer_value.items():
            if inner_key in nested_times_names:
                data_nested[-1].extend([value for value in inner_value.values()])
            else:
                data[-1].append(inner_value)

    # create pandas frames
    nested_columns = pd.MultiIndex.from_product([nested_times_names, model_names])
    nonnested_times_df = pd.DataFrame(data, index=times_sim.index,
                                      columns=nonnested_times_names)
    nested_times_df = pd.DataFrame(data_nested, index=times_sim.index,
                                   columns=nested_columns)

    # add summed nested information
    times_df = nonnested_times_df.join(nested_times_df.sum(axis=1, level=0))

    return times_df, nested_times_df
