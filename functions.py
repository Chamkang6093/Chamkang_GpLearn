"""The functions used to create and support programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""


import sympy
import numpy as np
from copy import deepcopy


class _Function(object):
    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """
    def __init__(self, name, arity, freq, op_type="standard"):
        self.name = name
        self.arity = arity
        self.freq = freq
        self.op_type = op_type        
        if self.op_type == "ts":
        	self.param = None
        	if self.name in ["ts_delay", "ts_delta"]:
        		self.param_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 21, 30, 45, 60, 120, 180, 250]
        		self.prob_list = [1 / len(self.param_list)] * len(self.param_list)   # eqaul likely
        	else:
        		self.param_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 21, 30, 45, 60, 120, 180, 250]
        		self.prob_list = [1 / len(self.param_list)] * len(self.param_list)   # eqaul likely
        if self.op_type == "special":
        	self.param = None
        	if self.name == "group_neutralize":
        		self.param_list = ["market", "sector", "industry", "subindustry"]
        		self.prob_list = [1 / len(self.param_list)] * len(self.param_list)   # eqaul likely
        	elif self.name == "power":
        		self.param_list = [2, 3, 4, 5]
        		self.prob_list = [1 / len(self.param_list)] * len(self.param_list)   # eqaul likely       		
        	else:
        		raise AttributeError("Please reset attribute 'name' as a special type operator")


    def __call__(self, *args):
        if self.op_type == "standard":
            return self.function(*args)
        elif self.op_type == "ts":
            if self.p not in self.param_list:
                raise AttributeError("Please reset attribute 'p'")
            else:
                return self.function(*args, self.p)
        elif self.op_type == "special":
        	if self.p not in self.param_list:
        		raise AttributeError("Please reset attribute 'p'")
        	else:
        		return self.function(*args, self.p)
        else:
        	raise AttributeError("Please reset attribute 'op_type'")


    def set_param(self, p):
        self.param = p
        self.name += '_%s' % str(self.param)


    def function_instance(self, random_state):
    	if self.op_type == "standard":
    		raise AttributeError("Cannot make an instance on a standard function")
    	elif self.op_type == "special" or self.op_type == "ts":
    		pass
    	else:
    		raise AttributeError("Please reset attribute 'op_type'")
    	
    	_function_ins = deepcopy(self)
    	# randomly choose a parameter in param_list according to our probability setting
    	p = random_state.choice(self.param_list, p=self.prob_list)
    	_function_ins.set_param(p)
    	return _function_ins


class _Function_Space(object):

	def __init__(self):

		# Define functions by categories in function space
		self.operator_dict = {
		    # *Operators*
		    # The following are operators that all arities can support:
		    # const, variable(field)
		    # operator, restricted operator
		    # restricted ts_operator (ALL of the ts_operators are restricted !)
		    'add' : _Function(name='add', arity=2, freq=5),
		    'sub' : _Function(name='sub', arity=2, freq=5),
		    'mul' : _Function(name='mul', arity=2, freq=5),
		    'div' : _Function(name='div', arity=2, freq=3),
		    'reverse' : _Function(name='reverse', arity=1, freq=3),
		    # the operators above will be subsitute by symbols(+,-,*,/ and etc.)
		    'log' : _Function(name='log', arity=1, freq=1),
		    'inverse' : _Function(name='inverse', arity=1, freq=1),
		    'abs' : _Function(name='abs', arity=1, freq=1),
		    'sign' : _Function(name='sign', arity=1, freq=2),
		    # Note: the operators are well protected by Worldquant platform
		    #       so we do not need to go into depth about this 
		    #       these operators are:
		    #       div, power, log

		    # *Restricted Operators*
		    # The following are operators that all arities cannot support: const 
		    # But constant seems to be useless, thus we do not use in this code !
		    # The others remain the same as *Operators*
			'rank' : _Function(name='rank', arity=1, freq=7),
			'max' : _Function(name='max', arity=2, freq=1),
			'min' : _Function(name='min', arity=2, freq=1)

			# *Do-Not-Use Operators*
			# And we do not use the following operators
			# 1. 'signed_power' = _Function(name='signed_power', arity=2)
			# Reason: Low Frequency
		}


		self.ts_operator_dict = {
			# *Restricted Operators*
		    # The following are operators that all arities can support:
		    # variable(field), operator, restricted operator
		    # restricted ts_operator (ALL of the ts_operators are restricted !)
		    # And cannot support: const
			'ts_delay' : _Function(name='ts_delay', arity=1, freq=4, op_type="ts"),
			'ts_delta' : _Function(name='ts_delta', arity=1, freq=5, op_type="ts"),
			# The following operators have additional constraints ( d cannot be 1 ),
			# which are reflected in _Function.__init__() !
			'ts_decay_linear' : _Function(name='ts_decay_linear', arity=1, freq=3, op_type="ts"),
			'ts_rank' : _Function(name='ts_rank', arity=1, freq=5, op_type="ts"),
			'ts_sum' : _Function(name='ts_sum', arity=1, freq=2, op_type="ts"),
			'ts_product' : _Function(name='ts_product', arity=1, freq=1, op_type="ts"),
			'ts_mean' : _Function(name='ts_mean', arity=1, freq=5, op_type="ts"),
			'ts_max' : _Function(name='ts_max', arity=1, freq=2, op_type="ts"),
			'ts_min' : _Function(name='ts_min', arity=1, freq=2, op_type="ts"),
			'ts_std_dev' : _Function(name='ts_std_dev', arity=1, freq=2, op_type="ts"),        
			'ts_corr' : _Function(name='ts_corr', arity=2, freq=3, op_type="ts"),             
			'ts_covariance' : _Function(name='ts_covariance', arity=2, freq=1, op_type="ts"),
			'ts_corr_rank' : _Function(name='ts_corr_rank', arity=2, freq=4, op_type="ts"),
			# 'ts_corr_ts_rank' : _Function(name='ts_corr_ts_rank', arity=2, freq=2, op_type="ts"),
			'ts_covariance_rank' : _Function(name='ts_covariance_rank', arity=2, freq=3, op_type="ts")
		}


		self.special_operator_dict = {
			'power' : _Function(name='power', arity=1, freq=1, op_type="special"),
			'group_neutralize' : _Function(name='group_neutralize', arity=1, freq=2, op_type="special")
		}


        # For building trees and randomly choosing fields
		self.arities = {}
		self.function_set = []
		self.prob_set = []
		for function in list(self.operator_dict.values()) + list(self.ts_operator_dict.values()) + list(self.special_operator_dict.values()):
			self.function_set.append(function)
			self.prob_set.append(function.freq)
			arity = function.arity
			self.arities[arity] = self.arities.get(arity, [])
			self.arities[arity].append(function)
		# Normalize probabilities
		self.prob_set = list(np.array(self.prob_set) / np.array(self.prob_set).sum())


		# For converting alpha factors to formatted versions
		self.f_converter = {
		    'add': lambda x, y : x + y,
		    'sub': lambda x, y : x - y,
		    'mul': lambda x, y : x * y,
		    'div': lambda x, y : x / y,
		    # not a big problem - worldquant platform supports ** as power function 
		    'power': lambda x, y : x ** y, 
		    'reverse': lambda x : - x,
		    'ts_corr_rank': lambda x, y, d : sympy.Function('ts_corr')(sympy.Function('rank')(x), sympy.Function('rank')(y), d),
		    # lack of the parameter of ts_rank, too much parameters, thus not use 
		    # 'ts_corr_ts_rank': lambda x, y, d : sympy.Function('ts_corr')(sympy.Function('ts_rank')(x), sympy.Function('ts_rank')(y), d),
		    'ts_covariance_rank': lambda x, y, d : sympy.Function('ts_covariance')(sympy.Function('rank')(x), sympy.Function('rank')(y), d),
		}
		for op in list(self.operator_dict.keys()) + list(self.ts_operator_dict.keys()) + list(self.special_operator_dict.keys()):
			if op not in list(self.f_converter.keys()):
				self.f_converter[op] = sympy.Function(op)


	def choose_function(self, random_state, arity = None):
		if not arity:
			f = random_state.choice(self.function_set, p=self.prob_set)

		else:
			if type(arity) != int:
				raise ValueError("'arity' must be a int type value !")
			# we should never get here, if we correctly implement the functions
			if self.arities.get(arity, []) == []:
				raise ValueError("'arity' wrong input !")
			function_set = []
			prob_set = []
			for function in self.arities[arity]:
				function_set.append(function)
				prob_set.append(function.freq)
			prob_set = list(np.array(prob_set) / np.array(prob_set).sum())
			f = random_state.choice(function_set, p=prob_set)
			
		if f.op_type == "standard":
			return f
		elif f.op_type == "ts" or f.op_type == "special":
			return f.function_instance(random_state)
		else:
			raise AttributeError("Please reset attribute 'op_type'")



# The following is main program
function_space = _Function_Space()
