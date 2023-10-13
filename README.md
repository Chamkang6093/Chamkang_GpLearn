# Genetic Programming (Windows Version)

## Description
* Applied genetic programming algorithm to find alphas automatically in multiple threads using Python3, and then deployed it on Amazon Web Server (in Ubuntu).

## Notes
1.  Before running the code, some files are supposed to be briefly modified in order to meet some expectations.
    * Add your own operators to 'self.operator_dict' or 'self.ts_operator_dict' in _Function_Space of functions.py with specifying the frequencies and arities.
    * Change 'self.param_list' and 'self.prob_list' in '_Function' of functions.py to modify the ways that the random numbers occur in the operators.
    * Add your own dataset to 'self.fields_set' in '_Field_Space' of fields.py with specifying the frequencies and units.
    * Add your own usernames and corresponding passwords to 'account_dict' of Control.py, while 'threads_dict' can be modified too to change the way multiple threads work.
    * Change 'BaseSymbolic.fit()' if necessary to use Chamkang_GpLearn to generate alphas at your will, i.e., not restricted in Worldquant Brain platform any more.
2.  The functionalities of each file are as follows:
    * main.py - Overall structure of generating alphas automatically, testing and recording the performance.
    * Control.py - It interacts with Worldquant Brain platform (Input the alphas and then send back the performance in multiple threads).
    * genetic.py - It implements genetic programming with a scikit-learn inspired API, which are supervised learning methods based on applying evolutionary operations.
    * _program.py - Underlying data-structure used by the public classes in Genetic.py, which is used for creating and evolving programs.
    * fields.py - All of the fields (also known as dataset) and their attributes used by gplearn programs.
    * functions.py - All of the functions (also known as operators) and their attributes used by gplearn programs, which connect different fields in certain rules under the designated probabilities.
    * utils.py - Some lightweight but helpful tools, including parallel computation (abandoned in my lib) and random states designation.
    * __init__.py - (Unimportant) Some configuration infomaiton.

## Main Reference
* <a href='https://github.com/trevorstephens/gplearn' target='_blank'>https://github.com/trevorstephens/gplearn</a>
