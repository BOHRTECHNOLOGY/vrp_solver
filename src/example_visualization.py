"""Example usage of visualization API."""
import matplotlib.pylab as plt
from visualization import visualize_states
from problem import Problem
from utilities import generate_input

NUM_OUTPOSTS = 9
COST_CONSTANT = 1
CONSTRAINT_CONSTANT = 200
VIOLATION_COUNT = 5
SAMPLE_SIZE = 30

def main():
    """Entrypoint of this script.

    Synopsis:
      - generate random problem
      - visualize some sampled states.
    """
    outposts, vehicles, graph = generate_input(NUM_OUTPOSTS)
    # some sample dumb partition
    partition = [2 for _ in range(len(vehicles) - 1)] + [NUM_OUTPOSTS -1 - 2 * (len(vehicles)-1)]
    problem = Problem(vehicles, outposts, partition, graph, 0, False)
    visualize_states(problem, COST_CONSTANT, CONSTRAINT_CONSTANT, VIOLATION_COUNT, SAMPLE_SIZE)
    plt.show()

if __name__ == '__main__':
    main()
