import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import random
import pandas as pd
import seaborn as sns

def plot_solution(results, outposts, data_path=None, file_name="plot"):
    fig, ax = plt.subplots(1)
    ax = plot_outposts(ax, outposts)
    ax = plot_routes(ax, results, outposts)
    plt.title("Total cost: " + str(results.cost.sum()))
    if data_path:
        plt.savefig(os.path.join(data_path, file_name))
    else:
        plt.show()

def plot_outposts(ax, outposts):
    for _, current_outpost in outposts.iterrows():
        ax.plot(current_outpost.latitude, current_outpost.longitude, 'bo', markersize=15)
        ax.annotate(str(current_outpost.outpost_id), xy=(current_outpost.latitude, current_outpost.longitude), ha="center", va="center")
        ax.annotate(str(current_outpost.load), xy=(current_outpost.latitude + 0.3, current_outpost.longitude + 0.3), ha="center", va="center", fontsize=8)
    return ax

def plot_routes(ax, results, outposts):
    hues = np.linspace(0, 1, num=len(results))
    colors = [[1, hue, 1-hue] for hue in hues]
    for route_id, result in results.iterrows():
        route = result.route
        for i in range(len(route) - 1):
            id_A = route[i]
            id_B = route[i + 1]
            outpost_A = outposts[outposts.outpost_id==id_A]
            outpost_B = outposts[outposts.outpost_id==id_B]
            point_A = [float(outpost_A.latitude), float(outpost_A.longitude)]
            point_B = [float(outpost_B.latitude), float(outpost_B.longitude)]
            ax.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], '-', c=colors[route_id])
    return ax

def generate_feasible_tsp_matrix(num_outposts):
    """Generate given number of feasible solutions for given problem.

    .. note::
       This implicitly assumes that we have fully connected graph. Otherwise finding any feasible
       solution is equivalent to finding Hamilton cycle.
    """
    matrix = np.zeros((num_outposts-1, num_outposts-1), dtype=int)
    indices = list(range(num_outposts-1))
    random.shuffle(indices)
    for i, j in enumerate(indices):
        matrix[i, j] = 1
    return matrix

def add_uniqueness_violation(feasible_matrix, count):
    """Add violation of uniqueness constraint."""
    done = 0
    matrix = np.array(feasible_matrix)
    while done < count:
        i = random.randint(0, matrix.shape[0]-1)
        j = random.randint(0, matrix.shape[0]-1)
        if matrix[i, j] == 0:
            matrix[i, j] = 1
            done += 1
    return matrix

def add_completeness_violation(feasible_matrix, count):
    """Add violation of completeness constraint."""
    done = 0
    matrix = np.array(feasible_matrix)
    indices = random.sample(range(matrix.shape[0]), count)
    for index in indices:
        matrix[index:] = 0
    return matrix

def compute_energy(solution_matrix, qubo_dict):
    """Compute energy of sample defined by some solution matrix with respect to given QUBO."""
    sample = solution_matrix.flatten()
    energy = 0.0
    for (i, j), weight in qubo_dict.items():
        energy += sample[i] * sample[j] * weight
    return energy

def visualize_states(problem, cost_constant, constraint_constant, violation_count, sample_size):
    """Create visualization of some subset of given problem's "solutions"

    .. note ::
       The word "solution" is in quotes above because some of the solutions visualized are not
       feasible (and therefore are not solutions.

    :param problem: an instance of the problem to visualize.
    :type problem: :py:class:`problem.Problem`
    :param cost_constant: cost constant for QUBO
    :type cost_constant: float
    :param constraint_constant: constraint constant for QUBO
    :type constraint_constant: float
    :param violation_count: determines how many constraint violations should be generatd for
     each violating solution. For instance, if violation_count=2 every non-complete solution
     will miss precisely two nodes and every non-unique solution will have two repeatitions.
    :type violatoin_count: int
    :param sample_size: how many examples to generate for each category (feasible, non-complete,
     non-unique).
    :returns: seaborn stripplot with computed energies on y-axis and type of solution on x-axis.
    """
    if problem.use_capacity_constraints:
        raise NotImplementedError('Visualizations for CVRP are not implemented yet.')
    feasible = []
    non_complete = []
    non_unique = []
    for _ in range(sample_size):
        feasible_matrix = generate_feasible_tsp_matrix(len(problem.outposts))
        feasible.append(feasible_matrix)
        non_complete.append(add_completeness_violation(feasible_matrix, violation_count))
        non_unique.append(add_uniqueness_violation(feasible_matrix, violation_count))

    qubo_dict = problem.get_qubo_dict(cost_constant, constraint_constant, 0)
    df = make_energies_dataframe({
        'feasible': [compute_energy(matrix, qubo_dict) for matrix in feasible],
        'violating completeness': [compute_energy(matrix, qubo_dict) for matrix in non_complete],
        'violating uniqueness': [compute_energy(matrix, qubo_dict) for matrix in non_unique]})
    return sns.stripplot(x='type', y='energy', data=df)

def make_energies_dataframe(energies_dict):
    """Given energies dict construct DataFrame usable with seaborn.

    :param energies_dict: dictionary mapping type of solution for which energy was computed
     to an array of computed energies. This will usually have keys "feasible", "violating
     completeness" and "violating_uniqueness".
    :type energies_dict: mapping
    :returns: DataFrame resulting from "flattening" of energies_dict. More specifically,
     it contains entries of the form (type, energy) where type is the solution type
     and energy is energy of some solution. Number of rows with the same value in type
     column is equal to the number of energies passed in energies_dict for that particular
     value.
    :rtype: :py:class:`pandas.DataFrame`
    """
    rows = []
    for type_name, energies in energies_dict.items():
        for energy in energies:
            rows.append((type_name, energy))
    return pd.DataFrame(rows, columns=['type', 'energy'])
