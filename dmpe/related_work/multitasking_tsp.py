from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from pymoo.core.repair import Repair
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer

from mixed_GA import Permutation


class StartFromZeroRepair(Repair):

    def _do(self, problem, X, **kwargs):

        print(X)

        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])

        return X


def create_cities(n_cities, grid_width=100.0, grid_height=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    grid_height = grid_height if grid_height is not None else grid_width
    cities = np.random.random((n_cities, 2)) * [grid_width, grid_height]
    return cities


def visualize(problem, x, fig=None, ax=None, show=True, label=True):
    with plt.style.context("ggplot"):

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # plot cities using scatter plot
        ax.scatter(problem.cities[:, 0], problem.cities[:, 1], s=250)
        if label:
            # annotate cities
            for i, c in enumerate(problem.cities):
                ax.annotate(str(i), xy=c, fontsize=10, ha="center", va="center", color="white")

        # plot the line on the path
        for i in range(len(x)):
            current = x[i]
            next_ = x[(i + 1) % len(x)]
            ax.plot(problem.cities[[current, next_], 0], problem.cities[[current, next_], 1], "r--")

        fig.suptitle("Route length: %.4f" % problem.get_route_length(x))

        if show:
            fig.show()


class MultitaskingTravellingSalespersonProblem(ElementwiseProblem):

    def __init__(self, cities, **kwargs):
        """
        A two-dimensional traveling salesperson problem (TSP)

        Parameters
        ----------
        cities : numpy.array
            The cities with 2-dimensional coordinates provided by a matrix where where city is represented by a row.

        """
        n_cities, _ = cities.shape

        self.cities = cities
        self.D = cdist(cities, cities)

        variables = {number: Permutation(bounds=(0, n_cities)) for number in range(n_cities)}
        variables["x0"] = Integer(bounds=(-20, 20))
        variables["x1"] = Integer(bounds=(-20, 20))

        super(MultitaskingTravellingSalespersonProblem, self).__init__(vars=variables, n_obj=1)

        self.permutation_keys = []
        self.non_permutation_keys = []
        for key, value in variables.items():
            if isinstance(value, Permutation):
                self.permutation_keys.append(key)
            else:
                self.non_permutation_keys.append(key)

        self.permutation_keys = tuple(self.permutation_keys)
        self.non_permutation_keys = tuple(self.non_permutation_keys)

    def _evaluate(self, x, out, *args, **kwargs):

        x_perm = np.array(itemgetter(*self.permutation_keys)(x))
        x_nonperm = np.array(itemgetter(*self.non_permutation_keys)(x))

        out["F"] = self.get_route_length(x_perm) + self.rosenbrock_function(x_nonperm)

    def get_route_length(self, x):
        n_cities = len(x)
        dist = 0
        for k in range(n_cities - 1):
            i, j = x[k], x[k + 1]
            dist += self.D[i, j]

        last, first = x[-1], x[0]
        dist += self.D[last, first]  # back to the initial city
        return dist

    def rosenbrock_function(self, x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
