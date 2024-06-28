import math
from copy import deepcopy

import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Choice, Real, Integer, Binary, BoundedVariable
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.selection.rnd import RandomSelection


class Permutation(BoundedVariable):
    """
    Class for the representation of a bounded, permutation decision variable.
    """

    def _sample(
        self,
        n: int,
    ) -> np.ndarray:
        raise NotImplementedError()


class MixedVariableMating(InfillCriterion):

    def __init__(
        self,
        selection=RandomSelection(),
        crossover=None,
        mutation=None,
        repair=None,
        eliminate_duplicates=True,
        n_max_iterations=100,
        **kwargs,
    ):

        super().__init__(repair, eliminate_duplicates, n_max_iterations, **kwargs)

        if crossover is None:
            crossover = {
                Binary: UX(),
                Real: SBX(),
                Integer: SBX(vtype=float, repair=RoundingRepair()),
                Choice: UX(),
                Permutation: OrderCrossover(),
            }

        if mutation is None:
            mutation = {
                Binary: BFM(),
                Real: PM(),
                Integer: PM(vtype=float, repair=RoundingRepair()),
                Choice: ChoiceRandomMutation(),
                Permutation: InversionMutation(),
            }

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(self, problem, pop, n_offsprings, parents=False, **kwargs):

        # So far we assume all crossover need the same amount of parents and create the same number of offsprings
        XOVER_N_PARENTS = 2
        XOVER_N_OFFSPRINGS = 2

        # the variables with the concrete information
        vars = problem.vars

        # group all the variables by their types
        vars_by_type = {}
        for k, v in vars.items():
            clazz = type(v)

            if clazz not in vars_by_type:
                vars_by_type[clazz] = []
            vars_by_type[clazz].append(k)

        # # all different recombinations (the choices need to be split because of data types)
        recomb = []
        for clazz, list_of_vars in vars_by_type.items():
            if clazz == Choice:
                for e in list_of_vars:
                    recomb.append((clazz, [e]))
            else:
                recomb.append((clazz, list_of_vars))

        # create an empty population that will be set in each iteration
        off = Population.new(X=[{} for _ in range(n_offsprings)])

        if not parents:
            n_select = math.ceil(n_offsprings / XOVER_N_OFFSPRINGS)
            pop = self.selection(problem, pop, n_select, XOVER_N_PARENTS, **kwargs)

        for clazz, list_of_vars in recomb:

            crossover = self.crossover[clazz]
            assert crossover.n_parents == XOVER_N_PARENTS and crossover.n_offsprings == XOVER_N_OFFSPRINGS

            _parents = [
                [
                    Individual(
                        X=np.array([parent.X[var] for var in list_of_vars], dtype="O" if clazz is Choice else None)
                    )
                    for parent in parents
                ]
                for parents in pop
            ]

            _vars = {e: vars[e] for e in list_of_vars}
            _xl = np.array([vars[e].lb if hasattr(vars[e], "lb") else None for e in list_of_vars])
            _xu = np.array([vars[e].ub if hasattr(vars[e], "ub") else None for e in list_of_vars])
            _problem = Problem(vars=_vars, xl=_xl, xu=_xu)

            _off = crossover(_problem, _parents, **kwargs)

            mutation = self.mutation[clazz]
            _off = mutation(_problem, _off, **kwargs)

            for k in range(n_offsprings):
                for i, name in enumerate(list_of_vars):
                    off[k].X[name] = _off[k].X[i]

        return off


class MixedVariableSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        nonperm_vars = {name: problem.vars[name].sample(n_samples) for name in problem.non_permutation_keys}

        helper = np.full((n_samples, len(problem.permutation_keys)), 0, dtype=int)
        for i in range(n_samples):
            helper[i, :] = np.random.permutation(len(problem.permutation_keys))
        helper = helper.T
        perm_vars = {name: value for name, value in zip(problem.permutation_keys, helper)}

        all_vars = dict(perm_vars, **nonperm_vars)

        X = []
        for k in range(n_samples):
            X.append({name: all_vars[name][k] for name in all_vars.keys()})

        return X
