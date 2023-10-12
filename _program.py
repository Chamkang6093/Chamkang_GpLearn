"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

import numpy as np
from copy import deepcopy
from sympy import sympify
from .functions import _Function


class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ---------------------------------------------------------------------------
    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.


    Attributes
    ---------------------------------------------------------------------------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.


    # And use property() to define:

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_space,
                 field_space,
                 function_field_prob,
                 trade_when_y_n_prob,
                 init_depth,
                 init_method,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 print_reverse_mode=False,
                 program=None):
    
        self.function_space = function_space
        self.field_space = field_space
        self.function_field_prob = function_field_prob
        self.trade_when_y_n_prob = trade_when_y_n_prob
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state
        self.print_reverse_mode = print_reverse_mode
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.converter = self.function_space.f_converter
        # The following attributes will be used later in BaseSymbolic.fit()
        self.parents = None
        self.raw_fitness_ = None
        self.fitness_ = None
        
        

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = random_state.choice(['full', 'grow'], p=[0.5, 0.5])
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = self.function_space.choose_function(random_state)
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = random_state.choice(['function', 'field'], p=self.function_field_prob)

            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or choice == 'function'):
                function = self.function_space.choose_function(random_state)
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                # But constant seems to be useless, thus we do not use in this code !
                terminal = self.field_space.choose_field(random_state)
                """
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with const_range=None.')
                """
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None


    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]


    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1


    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)


    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        standard_bool = [True]
        params = []
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                if node.op_type == "standard":
                    terminals.append(node.arity)
                    standard_bool.append(True)
                    output += node.name + '('
                else:
                    terminals.append(node.arity + 1)
                    standard_bool.append(False)
                    split_idx = node.name.rfind("_")
                    temp_f = node.name[:split_idx]
                    params.append(node.name[split_idx+1:])
                    output += temp_f + '('
            else:
                output += str(node)
                # since we do not consider constant in this code, 
                # and we do not use some parameters like 'self.feature_names',
                # it is different from origial code
                """        
                    if isinstance(node, int):
                        if self.feature_names is None:
                            output += 'X%s' % node
                        else:
                            output += self.feature_names[node]
                    else:
                        output += '%.3f' % node
                """

                terminals[-1] -= 1
                while (terminals[-1] == 0 and standard_bool[-1] == True) or (terminals[-1] == 1 and standard_bool[-1] == False):
                    if standard_bool[-1] == True:
                        terminals.pop()
                        standard_bool.pop()
                        if terminals:
                            terminals[-1] -= 1
                            output += ')'
                    else:
                        terminals.pop()
                        standard_bool.pop()
                        terminals[-1] -= 1
                        output += ', ' + params[-1] + ')'
                        params.pop()
                if i != len(self.program) - 1:
                    output += ', '

        output = str(sympify(output, locals=self.converter))
        if self.random_state.choice([True, False], p=self.trade_when_y_n_prob):
            output = str(sympify(output, locals=self.converter))
            output = "trade_when(ts_mean(returns, 20) - 0.01, " + output + ", -1)"
        else:
            output = str(sympify(output, locals=self.converter))

        if not self.print_reverse_mode:
            return output
        else:
            return "- (" + output + ")" 


    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                            % (i, node.name, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None


    def raw_fitness(self):
        """Evaluate the raw fitness of the program.

        We directlt implement this part in 'genetic.py'.
        """
        pass


    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program)
        return self.raw_fitness_ - penalty


    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1 for node in program])
        probs = probs / probs.sum()
        inds = np.array(range(len(program)))
        start = random_state.choice(inds, p=probs)

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end


    def reproduce(self):
        """Return a copy of the embedded program."""
        return deepcopy(self.program)


    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, program=donor)
        donor_removed = list(set(range(len(donor))) - set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] + donor[donor_start:donor_end] + self.program[end:]), removed, donor_removed


    def subtree_mutation(self, random_state):

        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)


    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) - set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed


    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = deepcopy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) < self.p_point_replace)[0]

        for node in mutate:                
            if isinstance(program[node], _Function):
                # Find a valid replacement with same arity
                program[node] = self.function_space.choose_function(random_state, arity=program[node].arity)
            else:
                # We've got a terminal, add a const or variable
                program[node] = self.field_space.choose_field(random_state)
                """
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with const_range=None.')
                """
        return program, list(mutate)



    # The following lines are executed in class
    depth_ = property(_depth)
    length_ = property(_length)