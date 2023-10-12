"""
Genetic Programming in Python, with a scikit-learn inspired API

The :mod:`gplearn.genetic` module implements Genetic Programming. These
are supervised learning methods based on applying evolutionary operations on
computer programs.
"""

import os
import itertools
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn
from copy import deepcopy
from threading import Thread

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from ._program import _Program
from .functions import function_space
from .fields import field_space
from .utils import check_random_state, _partition_estimators

from FactorExplore.FactorExplore_GpLearn import execute_unit, aggregate_threads

__all__ = ['SymbolicTransformer']

MAX_INT = np.iinfo(np.int32).max


def _parallel_evolve(n_programs, parents, seeds, params):
    """Private function used to build a batch of programs within a job."""

    # Unpack parameters
    tournament_size = params['tournament_size']
    function_space = params['function_space']
    field_space = params['field_space']
    function_field_prob = params['function_field_prob']
    trade_when_y_n_prob = params['trade_when_y_n_prob']
    init_depth = params['init_depth']
    init_method = params['init_method']
    p_point_replace = params['p_point_replace']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    

    def _tournament():
        # Find the fittest individual from a sub-population
        # Note: randint is likely to choose duplicated elements
        contender_ind = list(set(random_state.randint(0, len(parents), tournament_size)))
        fitness = [parents[p].fitness_ for p in contender_ind]
        # greater_is_better:
        parent_ind = contender_ind[np.argmax(fitness)]
        return parents[parent_ind], parent_ind

    # Build programs
    programs = []
    program_name_list = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            # which allows us to build a new program
            genome = None
        else:
            parent, parent_index = _tournament()
            method = random_state.choice(method_probs["label"], p=method_probs["probs"])
            if method == "crossover":
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program, random_state)
                genome = {'method': method,
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method == "subtree_mutation":
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': method,
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method == "hoist_mutation":
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': method,
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method == "point_mutation":
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': method,
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': method,
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_space=function_space,
                           field_space=field_space, 
                           function_field_prob=function_field_prob,
                           trade_when_y_n_prob=trade_when_y_n_prob,
                           init_depth=init_depth,
                           init_method=init_method,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           random_state=random_state,
                           program=program)

        program.parents = genome
        programs.append(program)

    return programs



class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    """Base class for symbolic regression estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    parameters
    ----------
    population_size : integer, optional (default=50)
        The number of programs in each generation.

    n_components : integer, or None, optional (default=None)
        The number of best programs to return after searching the population
        in every generation.
        If `None`, the entire population will be used.

    generations : integer, optional (default=7)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation (not fixed).

    stopping_criteria : float, optional (default=10.0)
        The required fitness value required in order to stop evolution early.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and terminals, 
          allowing for smaller trees than `init_depth` allows. Tends to grow 
          asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    function_space : class, see in '.functions.py' file

    field_space : class, see in '.fields.py' file

    function_field_prob : list of float numbers (default=[0.5, 0.5])

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    low_memory : bool, optional (default=True)
        When set to ``False``, all of the program infomation will be stored.
        Otherwise, only parent information will be retained. Other program 
        information will be discarded. For very large populations or runs with 
        many generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=True)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicRegressor

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.


    """

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=50,
                 n_components=None,
                 generations=7,
                 tournament_size=20,
                 stopping_criteria=10.0,
                 init_depth=(1, 4),
                 init_method='half and half',
                 parsimony_coefficient=0.001,
                 function_space=function_space,
                 field_space=field_space,
                 function_field_prob=[0.5, 0.5],
                 trade_when_y_n_prob=[0.5, 0.5],
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 low_memory=True,
                 n_jobs=1,
                 verbose=True,
                 random_state=None,
                 settings=None,
                 engines=None,
                 log_dir='D:/_Files/F.Projects/Worldquant/GpLearn/',
                 project_name='1',
                 output_dir='0'):
        self.population_size = population_size
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.init_depth = init_depth
        self.init_method = init_method
        self.parsimony_coefficient = parsimony_coefficient
        self.function_space = function_space
        self.field_space=field_space
        self.function_field_prob=function_field_prob
        self.trade_when_y_n_prob=trade_when_y_n_prob
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.settings = settings
        self.engines = engines
        self.log_dir = log_dir
        self.project_name = project_name
        self.output_dir = output_dir


    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ')
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness'))

        else:
            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16}'
            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     oob_fitness))


    def fit(self):
        random_state = check_random_state(self.random_state)

        if type(self.engines) != dict:
            raise TypeError("The type of 'self.engines' must be dict !")

        if self.n_components is None:
            self.n_components = self.population_size
        if self.n_components > self.population_size or self.n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to population_size (%d) ' 
                             'and larger than 0.' % (self.n_components, self.population_size))

        _method_probs = [self.p_crossover, self.p_subtree_mutation, self.p_hoist_mutation, self.p_point_mutation]
        _method_probs.append(1 - np.array(_method_probs).sum())
        self._method_probs = {
            "probs": _method_probs,
            "label": ["crossover", "subtree_mutation", "hoist_mutation", "point_mutation", "reproduction"]
        }

        if self._method_probs["probs"][-1] < 0:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        params = self.get_params()
        # A method of sklearn.base.BaseEstimator
        params['method_probs'] = self._method_probs

        if not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations <= 0:
            raise ValueError('generations=%d must be larger len(_programs)=%d'
                             % (self.generations, len(self._programs)))

        ##########################################################
        # Here is the first of the only two places interacting with
        # FactorExplore Module.

        # Create working directory
        self.factor_dir = self.log_dir + "project_" + self.project_name + "/"
        self.engine_dir = self.factor_dir + self.output_dir + "/"
        if not os.path.exists(self.engine_dir):
            os.makedirs(self.engine_dir)

        # Generating the setting of current project
        with open(self.factor_dir + "factor_library_settings.txt", "a") as f:
            region, universe, delay, neutralization, decay, truncation = self.settings
            f.write("Setting: Region: " + region + "; Delay: " + str(delay) + "; Universe: " + str(universe) + "; Neutralization: " 
                    + neutralization + "; Decay: " + str(decay) + "; Truncation: " + str(truncation) + "\n")

        # Generate a Dataframe to store the fitness data fetched from platform        
        self.raw_fitness_dfs = {thread : pd.DataFrame([],  
                                                      index = range(1, self.n_components * (self.generations + 1)), 
                                                      columns = ["Score", 
                                                                 "Level", 
                                                                 "Fitness", 
                                                                 "Sharpe", 
                                                                 "Returns", 
                                                                 "Turnover", 
                                                                 "Drawdown", 
                                                                 "Fail_num", 
                                                                 "Exec_time", 
                                                                 "Corr"
                                                    ]) for thread in self.engines.keys()}

        # Generate a Fitness dict to store the data which
        # can be share among different Processes (Threads) 
        try:
            self.factor_lib = np.load(self.factor_dir + 'factor_library.npy', allow_pickle=True).item()
        except:
            self.factor_lib = {}
        # Block Ends
        ##########################################################

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Pseudo-Parallel loop Begins
            n_jobs, n_programs, starts = _partition_estimators(self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs, verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))
            # Pseudo-Parallel loop Ends

            ##########################################################
            # Note: we have moved the api of parallel computation here.
            # And here is the second of the only two places interacting 
            # with FactorExplore Module.
            
            """
            # Try every program and their reverse

            temp_population = deepcopy(population)
            for program in population:
                new_program = deepcopy(program)
                new_program.print_reverse_mode = True
                temp_population.append(new_program)
            population = temp_population


            # Get the dict from the directory
            try:
                factor_lib = np.load(self.factor_dir + 'factor_library.npy', allow_pickle=True).item()
            except:
                factor_lib = {}
            """          
            def parallel_calculation(self, sub_population, thread_num, log_info=False):

                for program in sub_population:

                    if log_info:
                         print("Thread %d - " % (thread_num) + str(os.getpid()) + " is working on " + str(program))
                    try:
                        fitness_score = self.factor_lib[str(program)]
                    except:
                        self.factor_lib[str(program)] = -5
                        fitness_score = None

                    if type(fitness_score) == type(None):
                        index = (self.raw_fitness_dfs[thread_num].any(axis=1)).sum() + 1
                        if index == 1:
                            program.raw_fitness_ = execute_unit(self.engines[thread_num], 
                                                                index, 
                                                                self.settings, 
                                                                str(program), 
                                                                self.raw_fitness_dfs[thread_num],
                                                                self.engine_dir, 
                                                                str(thread_num),
                                                                gen + 1,
                                                                setting_remain=False,
                                                                log_hide_setting=False)
                        else:
                            program.raw_fitness_ = execute_unit(self.engines[thread_num], 
                                                                index, 
                                                                self.settings, 
                                                                str(program),
                                                                self.raw_fitness_dfs[thread_num],
                                                                self.engine_dir, 
                                                                str(thread_num),
                                                                gen + 1,
                                                                setting_remain=True,
                                                                log_hide_setting=True)

                        self.factor_lib[str(program)] = program.raw_fitness_
                        np.save(self.factor_dir + 'factor_library.npy', self.factor_lib)
                    else:
                        try:
                            # there is a fuck damn bug I cannot figure out when I use 'factor_lib[str(program)]'
                            program.raw_fitness_ = fitness_score
                        except:
                            # should never go here
                            raise Exception("Parallel Computation Block Design Error ! Please Check !")


            process_pool = []
            for thread_index, sub_population in enumerate(np.array_split(population, len(self.engines))):
                process_pool.append(Thread(target=parallel_calculation, args=(self, sub_population, thread_index + 1,)))
            for p in process_pool:
                p.start()
            for p in process_pool:
                p.join()
            # parallel computation block ends
            ##########################################################


            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] / np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None

            # Record run details
            # greater_is_better
            best_program = population[np.argmax(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            # greater_is_better:
            best_fitness = fitness[np.argmax(fitness)]
            if best_fitness >= self.stopping_criteria:
                break


        # Find the best individuals in the final generation      
        program_list = []
        fitness_list = []
        for program in self._programs[-1]:
            if program:
                program_list.append(program)
                fitness_list.append(program.raw_fitness_)
        program_list = np.array(program_list)        
        fitness_list = np.array(fitness_list)
        # greater_is_better:
        sort_index = fitness_list.argsort()[::-1][:]
        program_list = program_list[sort_index]
        fitness_list = fitness_list[sort_index]

        while len(program_list) > self.n_components:
            program_list = program_list[:-1]
            fitness_list = fitness_list[:-1]
        self._best_programs = program_list
        aggregate_threads(len(self.engines), self.engine_dir)

        return self



class SymbolicTransformer(BaseSymbolic, TransformerMixin):

    """A Genetic Programming symbolic transformer.

    A symbolic transformer is a supervised transformer that begins by building
    a population of naive random formulas to represent a relationship. The
    formulas are represented as tree-like structures with mathematical
    functions being recursively applied to variables and constants. Each
    successive generation of programs is then evolved from the one that came
    before it by selecting the fittest individuals from the population to
    undergo genetic operations such as crossover, mutation or reproduction.
    The final population is searched for the fittest individuals with the least
    correlation to one another.
    """
    def __init__(self,
                 *,
                 population_size=50,
                 n_components=None,
                 generations=7,
                 tournament_size=20,
                 stopping_criteria=10.0,
                 init_depth=(2, 6),
                 init_method='half and half',
                 parsimony_coefficient=0.001,
                 function_space=function_space,
                 field_space=field_space,
                 function_field_prob=[0.5, 0.5],
                 trade_when_y_n_prob=[0.5, 0.5],
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 low_memory=True,
                 n_jobs=1,
                 verbose=True,
                 random_state=None,
                 settings=None,
                 engines=None,
                 project_name='1',
                 log_dir='D:/_Files/F.Projects/Worldquant/GpLearn/',
                 output_dir='0'):

        super(SymbolicTransformer, self).__init__(
            population_size=population_size,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            init_depth=init_depth,
            init_method=init_method,
            parsimony_coefficient=parsimony_coefficient,
            function_space=function_space,
            field_space=field_space,
            function_field_prob=function_field_prob,
            trade_when_y_n_prob=trade_when_y_n_prob,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            settings=settings,
            engines=engines,
            project_name=project_name,
            log_dir=log_dir,
            output_dir=output_dir)

    def __len__(self):
        """Overloads `len` output to be the number of fitted components."""
        if not hasattr(self, '_best_programs'):
            return 0
        return self.n_components

    def __getitem__(self, item):
        """Return the ith item of the fitted components."""
        if item >= len(self):
            raise IndexError
        return self._best_programs[item]

    def __str__(self):
        """Overloads `print` output of the object to resemble LISP trees."""
        if not hasattr(self, '_best_programs'):
            return self.__repr__()
        output = str([gp.__str__() for gp in self])
        return output.replace("',", ",\n").replace("'", "")