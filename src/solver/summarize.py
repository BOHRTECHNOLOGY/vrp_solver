#!/usr/bin/env python

import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd
import os
import ast
from graph_generator import lat_long_to_x_y
import itertools
import logging

LOGGER = logging.getLogger("Summarize")


DESCRIPTION = '''
Solution visualisation tool.
'''


def load_to_rgba(load, m, M):
    '''
    Create tuple of values coding color in RGBA using floats in [0, 1] range.
    If `load` varies between `m` and `M`, all RGA channels change linearly.
    `m` and `M` define loads corresponding to color limits.

    Args:
        load: given load
        m: minimal load 
        M: maximal load

    Returns:
        RGBA tuple
    '''
    if load < m:
        x = 0.5
    elif load > M:
        x = 0.95
    else:
        x = 0.5 + 0.45 * (load - m) / (M - m) 
    return (x, x, x, 0.9)


def draw_routes(ax, results, outposts):
    '''
    Draw routes from resulting routes.
    '''
    hues = np.linspace(0, 1, num=len(results))
    colors = [[1, hue, 1-hue] for hue in hues]
    for route_id, result in results.iterrows():
        route = result.route
        for i in range(len(route) - 1):
            id_A = route[i]
            id_B = route[i + 1]
            outpost_A = outposts[outposts['outpost_id']==id_A]
            outpost_B = outposts[outposts['outpost_id']==id_B]
            point_A = [float(outpost_A.latitude), float(outpost_A.longitude)]
            point_B = [float(outpost_B.latitude), float(outpost_B.longitude)]
            ax.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], '-.', c=colors[route_id], linewidth=1)
    return ax


def draw_outposts(ax, outposts):
    '''
    Draw outposts on the figure.

    Args:
        ax: matplotlib figure axes
        outposts: DataFrame with outposts data

    Returns:
        Axes
    '''
    for idx, current_outpost in outposts.iterrows():
        load = current_outpost.load
        marker_size = 6
        facecol = load_to_rgba(load, 0.05, 10)
        if idx is 0:
            ax.plot(current_outpost.latitude, current_outpost.longitude, 'rX', markersize=10)
        else:
            ax.plot(current_outpost.latitude, current_outpost.longitude, 'go', markersize=marker_size, markeredgewidth=0.5, markeredgecolor='black', markerfacecolor=facecol)
    return ax


def plot_routes(outposts, routes, savedir, name, title):
    '''
    Plot given routes.
    '''
    fig, ax = plt.subplots(1, figsize=(20,20))
    draw_outposts(ax, outposts)
    draw_routes(ax, routes, outposts)
    fig.suptitle(title, size=40)
    fig.tight_layout()
    savepath = os.path.join(savedir, name + '.png')
    fig.savefig(savepath)
    plt.close(fig)


def summarize_changes(df):
    '''
    Generate additional summary showing percentage changes.
    Args:
        df: DataFrame with detail results
    Returns:
        Tuple of DataFrames: (routes distance change, routes number change, overall change)
    '''
    bohr_routes_sums = df[df.source == 'BOHR'][['id','sum']].set_index('id')
    orig_routes_sums = df[df.source == 'ORIG'][['id','sum']].set_index('id')

    percent_sum_change = -100 * (orig_routes_sums - bohr_routes_sums) / orig_routes_sums
    percent_sum_change = percent_sum_change.rename(columns={'sum':'total distance change [%]'})

    bohr_routes_n = df[df.source == 'BOHR'][['id','vehicles_used']].set_index('id')
    orig_routes_n = df[df.source == 'ORIG'][['id','vehicles_used']].set_index('id')

    percent_routes_change = -100 * (orig_routes_n - bohr_routes_n) / orig_routes_n
    percent_routes_change = percent_routes_change.rename(columns={'vehicles_used':'routes number change [%]'})

    bohr_routes_sums = bohr_routes_sums.rename(columns={'sum': 'total distance [km] - BOHR'})
    orig_routes_sums = orig_routes_sums.rename(columns={'sum': 'total distance [km] - ORIG'})
    bohr_routes_n = bohr_routes_n.rename(columns={'vehicles_used': '#routes - BOHR'})
    orig_routes_n = orig_routes_n.rename(columns={'vehicles_used': '#routes - ORIG'})
    df_routes_change = pd.concat([orig_routes_n, bohr_routes_n, np.around(percent_routes_change, decimals=1)], sort=False, axis=1)
    df_sums_change = np.around(pd.concat([orig_routes_sums, bohr_routes_sums, percent_sum_change], sort=False, axis=1), decimals=1)

    results = {}
    for source in ['BOHR', 'ORIG']:
        data = {
            'total distance [km]': df[df.source == source]['sum'].sum(),
            'average total distance per day [km]': df[df.source == source]['sum'].mean(),
            'total number of routes': df[df.source == source]['vehicles_used'].sum(),
            'average total number of routes': df[df.source == source]['vehicles_used'].mean()
        }
        results[source] = data

    results_percent = {}
    for current_key in results['BOHR'].keys():
        diff = results['ORIG'][current_key] - results['BOHR'][current_key]
        relative_diff = diff / results['ORIG'][current_key]
        percent = -100 * relative_diff # Minus X% means that we reduced e.g. total distance
        results_percent[current_key] = percent
    results['change [%]'] = results_percent
    df_overall = np.around(pd.DataFrame(results), decimals=1)
    return df_sums_change, df_routes_change, df_overall


def summarize(outposts, data):
    '''
    Do summary of generated solutions.
    Args:
        outposts: DataFrame of outposts
        data: lits of ('<#day><d/n>', DataFrame with BOHR's solution, DataFrame with original solution
    Returns:
        DataFrame with summary
    '''
    def cost_stats(costs):
        '''
        Get global statistics of list of costs.
        '''
        return [np.sum(costs), np.min(costs), np.max(costs), np.std(costs), np.mean(costs)]

    columns = ['source', 'id', 'sum', 'min', 'max', 'std', 'mean', 'vehicles_used']
    results = []
    for suffix, routes, orig in data:
        # Change units to `km`
        stats_our = cost_stats(routes['cost'].values / 1000.)
        # This is a cost of Pocza Polska solution in `m` (calculated from point to point using WGS84 metrics)
        stats_orig = cost_stats(orig['cost'].values / 1000.)
        results.append(['BOHR', suffix] + stats_our + [len(routes)])
        results.append(['ORIG', suffix] + stats_orig + [len(orig)])

    df_detailed = pd.DataFrame(results, columns=columns) # Detailed data
    df_sums, df_routes, df_overall = summarize_changes(df_detailed)
    return df_detailed, df_sums, df_routes, df_overall

    
def plot_summary(df, save_dir):
    '''
    Plot nice bar charts from summary data
    Args:
        df: DataFrame with summary
        save_dir: save location for plots
    '''
    df_bohr_sum = df[df.source == 'BOHR'][['id', 'sum']].rename(columns={'sum': 'Bohr'}).set_index('id')
    df_orig_sum = df[df.source == 'ORIG'][['id', 'sum']].rename(columns={'sum': 'Original'}).set_index('id')
    df_sum = pd.concat((df_bohr_sum.T, df_orig_sum.T), axis='index').T

    ax = df_sum.plot.bar(figsize=(16,10), fontsize=16, legend=False, rot=0)
    ax.set_title('Total length of all routes', size=20)
    ax.set_xlabel('Week day and time shift', size=16)
    ax.set_ylabel('Total distance (km)', size=16)
    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, labels, fontsize=16)
    plt.savefig(os.path.join(save_dir, 'total_distance.png'))

    df_bohr_vu = df[df.source == 'BOHR'][['id', 'vehicles_used']].rename(columns={'vehicles_used': 'Bohr'}).set_index('id')
    df_orig_vu = df[df.source == 'ORIG'][['id', 'vehicles_used']].rename(columns={'vehicles_used': 'Original'}).set_index('id')
    df_vehicles_used = pd.concat((df_bohr_vu.T, df_orig_vu.T), axis='index').T

    ax = df_vehicles_used.plot.bar(figsize=(16,10), fontsize=16, legend=False, rot=0)
    ax.set_title('Vehicles used', size=20)
    ax.set_xlabel('Week day and time shift', size=16)
    ax.set_ylabel('Vehicles used', size=16)
    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, labels, fontsize=16)
    plt.savefig(os.path.join(save_dir, 'vehicles_used.png'))


def verify_solutions(outposts, solutions, vehicles, save_dir):
    '''
    Verify generated solutions and save results into `routes_verify.csv`.

    Args:
        outposts: outposts list DataFrame 
        solutions: list of (suffix, solution DataFrame, original solution DataFrame)
        vehicles: vehicles list DataFrame
        save_dir: where to save output files
    '''
    outposts_number = len(outposts)

    outposts_couter_global = np.zeros(outposts_number)
    original_couter_global = np.zeros(outposts_number)

    outposts_load_global = np.zeros(outposts_number)
    original_load_global = np.zeros(outposts_number)

    counters = {}
    loads = {}
    for suffix, solution, original in solutions:
        outposts_counter = np.zeros(outposts_number)
        original_counter = np.zeros(outposts_number)
        outposts_load = np.zeros(outposts_number)
        original_load = np.zeros(outposts_number)
        for _, row in solution.iterrows():
            vehicle_id = row.vehicle_id
            vehicle_capacity = float(vehicles[vehicles.vehicle_id == vehicle_id].capacity)
            if len(row.route) > 2:
                outpost_mean_load = vehicle_capacity / (len(row.route) - 2)
            else:
                outpost_mean_load = vehicle_capacity
            for outpost_id in row.route:
                outposts_counter[outpost_id] += 1
                outposts_couter_global[outpost_id] += 1
                outposts_load[outpost_id] += outpost_mean_load
                outposts_load_global[outpost_id] += outpost_mean_load

        for _, row in original.iterrows():
            if len(row.route) > 2:
                outpost_mean_load = row.vehicle_capacity / (len(row.route) - 2)
            else:
                outpost_mean_load = row.vehicle_capacity
            for outpost_id in row.route:
                original_counter[outpost_id] += 1
                original_couter_global[outpost_id] += 1
                original_load[outpost_id] += outpost_mean_load
                original_load_global[outpost_id] += outpost_mean_load

        for i in range(outposts_number):
            if outposts_counter[i] == 0 and original_counter[i] > 0:
                LOGGER.error("Suffix: %s - Outpost [id: %d, name: %s] not visited in our solution! \
Visited %d in original solution!" % (suffix, i, outposts.iloc[i].outpost_name, original_counter[i]))
        counters[suffix] = {'B': outposts_counter, 'O': original_counter}
        loads[suffix] = {'B': outposts_load, 'O': original_load}

    suffixes = [suffix for suffix, _, _ in solutions]
    result = []
    for _, row in outposts.iterrows():
        outpost_id = row.outpost_id
        outpost_name = row.outpost_name
        result += [
            dict([
                    ('outpost_id', outpost_id),
                    ('outpost_name', outpost_name),
                    ('B-all', int(outposts_couter_global[outpost_id])),
                    ('O-all', int(original_couter_global[outpost_id])),
                    ('B-all-l', outposts_load_global[outpost_id]),
                    ('O-all-l', original_load_global[outpost_id])
                ] + \
                [('B-' + suffix, int(counters[suffix]['B'][outpost_id])) for suffix in suffixes] + \
                [('O-' + suffix, int(counters[suffix]['O'][outpost_id])) for suffix in suffixes] + \
                [('B-%s-l' % suffix, loads[suffix]['B'][outpost_id]) for suffix in suffixes] + \
                [('O-%s-l' % suffix, loads[suffix]['O'][outpost_id]) for suffix in suffixes]
        )]
    count_columns_order = ['outpost_name'] \
        + list(itertools.chain.from_iterable([('B-' + suffix, 'O-' + suffix) for suffix in suffixes])) \
        + ['B-all', 'O-all']

    load_columns_order = ['outpost_name'] \
        + list(itertools.chain.from_iterable([('B-%s-l' % suffix, 'O-%s-l' % suffix) for suffix in suffixes])) \
        + ['B-all-l', 'O-all-l']

    results = pd.DataFrame(result)
    results_count = results.set_index(['outpost_id'])[count_columns_order]
    results_load = results.set_index(['outpost_id'])[load_columns_order]
    with open(os.path.join(save_dir, 'check_visits.txt'), 'w') as f:
        f.write(results_count.to_string())
    with open(os.path.join(save_dir, 'check_load.txt'), 'w') as f:
        f.write(results_load.to_string())


def generate_summary(outposts, summary_data, save_dir):
    '''
    Generate all summaries and plots.
    Args:
        outposts: outposts DataFrame
        summary_data: list of (suffix, our solution DataFrame, original solution DataFrame)
        save_dir: save directory
    '''
    for suffix, routes, orig in summary_data:
        plot_routes(outposts=outposts, routes=routes, savedir=save_dir, 
                name='bohr_'+suffix, title='BOHR SOLUTION - ' + suffix)
        plot_routes(outposts=outposts, routes=orig, savedir=save_dir, 
                name='original_'+suffix, title='ORIGINAL SOLUTION - ' + suffix)

    file_names = ['detail', 'routes_distance', 'routes_number', 'overall']
    summaries = summarize(outposts, summary_data)
    for fname, df in zip(file_names, summaries):
        path = os.path.join(save_dir, fname)
        df.to_csv(path + '.csv', sep=';')
        if fname == 'detail':
            plot_summary(df, save_dir) 
        with open(path + '.txt', 'w') as f:
            f.write(str(df))
            f.write('\n')


def run(outposts_dir, routes_dir, save_dir, verify=False):
    """
    Create routes plots with overall text summary.
    Verify solutions and exit if `verify` is True.
    """
    os.makedirs(save_dir, exist_ok=True)
    vehicles_path = os.path.join(routes_dir, 'vehicles.csv')
    outposts_base_path = os.path.join(outposts_dir, 'outposts.csv')
    suffixes = list(itertools.chain.from_iterable([['%dd' % i, '%dn' % i] for i in range(1, 8)]))

    outposts = pd.read_csv(outposts_base_path, sep=';')
    vehicles = pd.read_csv(vehicles_path, sep=';')
    summary_data = []
    for suffix in suffixes:
        routes_path = os.path.join(routes_dir, 'routes_%s.csv' % suffix)
        orig_path = os.path.join(outposts_dir, 'orig_routes_%s.csv' % suffix)
        
        routes = pd.read_csv(routes_path, sep=';', converters={'route': ast.literal_eval})
        orig = pd.read_csv(orig_path, sep=';', converters={'route': ast.literal_eval})
        summary_data.append((suffix, routes, orig))

    if verify:
        verify_solutions(outposts, summary_data, vehicles, save_dir)
    else:
        generate_summary(outposts, summary_data, save_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument('outposts_dir', help='Directory containing `outposts_*.csv` files')
    parser.add_argument('routes_dir', help='Directory containing solutions - `routes_*.csv` files')
    parser.add_argument('--verify', action='store_true', help='Verify solutions and exit.')
    parser.add_argument('--savedir', default='.', help='Save directory')
    parser.add_argument('-v', action='count', default=0, help='Increase level of verbosity')
    
    args = parser.parse_args()

    if args.v < 1:
        log_level = logging.ERROR
    elif args.v < 2:
        log_level = logging.WARNING
    elif args.v < 3:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    run(args.outposts_dir, args.routes_dir, args.savedir, verify=args.verify)

