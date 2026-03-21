'''
Problem definition.
'''
import os
import platform
import time

import numpy as np
import numexpr as ne
from scipy.spatial.distance import cdist

from typing import Tuple, List

from AeroOpt.core.settings import SettingsData, SettingsProblem


class Problem(object):
    '''
    Problem for optimization.
    
    Parameters:
    -----------
    data_settings: SettingsData
        Settings of the data.
    problem_settings: SettingsProblem
        Settings of the problem.
        
    Attributes:
    -----------
    input_fname: str
        Name of the input file.
    output_fname: str
        Name of the output file.
    calculation_folder: str
        Name of the calculation folder.
    '''
    def __init__(self, data_settings: SettingsData, problem_settings: SettingsProblem):

        self.data_settings = data_settings
        self.problem_settings = problem_settings

        self.input_fname : str = 'input.txt'
        self.output_fname : str = 'output.txt'
        
        self.calculation_folder : str = 'Calculation'

    def __eq__(self, other):
        '''
        User defined comparison operator [=].
        '''
        if not isinstance(other, Problem):
            return NotImplemented
        
        if self.problem_settings.name != other.problem_settings.name:
            return False
        
        return True
    
    @property
    def name(self) -> str:
        '''
        Name of the problem in the settings.
        '''
        return self.problem_settings.name
    
    @property
    def n_input(self) -> int:
        '''
        Number of input variables.
        '''
        return self.data_settings.n_input
    
    @property
    def n_output(self) -> int:
        '''
        Number of output variables.
        '''
        return self.data_settings.n_output
    
    @property
    def n_constraint(self) -> int:
        '''
        Number of constraints.
        '''
        return self.problem_settings.n_constraint
    
    @property
    def n_objective(self) -> int:
        '''
        Number of objective variables.
        '''
        return self.problem_settings.n_objective
    
    @property
    def output_type(self) -> List[int]:
        '''
        Per-output role from settings (e.g. minimize / maximize).
        '''
        return self.problem_settings.output_type
    
    @property
    def mask_for_deactivated_inputs(self) -> np.ndarray:
        '''
        Mask for deactivated input variables, i.e.,
        the range of the variable is less than the precision.
        '''
        span = self.data_settings.input_upp - self.data_settings.input_low
        return span < self.data_settings.input_precision
        
    @property
    def mask_for_deactivated_outputs(self) -> np.ndarray:
        '''
        Mask for deactivated output variables, i.e.,
        the range of the variable is less than the precision.
        '''
        span = self.data_settings.output_upp - self.data_settings.output_low
        return span < self.data_settings.output_precision
    
    @property
    def critical_scaled_distance(self) -> float:
        '''
        Critical scaled distance for checking duplication of individuals.
        '''
        return self.data_settings.critical_scaled_distance
    
    #* External evaluation of the output variable by calling run.bat/.sh.

    def external_run(self, folder_name: str, x: np.ndarray,
                information: bool = True, bash_name: str = 'run', 
                timeout: float | None = None) -> Tuple[bool, np.ndarray]:
        '''
        External calculation by calling run.bat/.sh.
        
        Parameters
        -----------------
        folder_name: str
            name of the current running folder, the working folder is ./Calculation/folder_name.
        x: ndarray [dim_input]
            function input
        information: bool
            whether print information on screen
        bash_name: str
            name of the external running script, the default is 'run'.
        timeout: float, or None
            if `timeout` is None, waits for the application to end.
            If `timeout` is a float, wait for `timeout` seconds.

        Returns
        ----------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_output]
            function output

        I/O files
        ----------------
        input_fname: str
            name of the file that contains information of `x`.
            Each line contains the name and value of one variable, e.g. 'x1   1.0'.
        input_fname: str
            name of the file that contains information of `y`.
            Each line contains the name and value of one variable, e.g. 'y1   1.0'.
        
        '''
        
        folder = os.path.join(self.calculation_folder, folder_name)
        out_name = os.path.join(folder, self.output_fname)
        in_name = os.path.join(folder, self.input_fname)
        
        os.makedirs(folder, exist_ok=True)

        if platform.system() in 'Windows':
            
            if not os.path.exists(in_name):
                os.system('xcopy /s /y  .\\Runfiles  '+folder+'\\  > nul')

                self.write_input(in_name, x)
                
                if isinstance(timeout, int) or isinstance(timeout, float):
                    os.system('start /min /d   '+folder+'  %s.bat'%(bash_name))
                    time.sleep(float(timeout))
                    
                else:
                    os.system('start /wait /min /d   '+folder+'  %s.bat'%(bash_name))
                    os.system('del   '+folder+'\\%s.bat'%(bash_name))

        else:

            if not os.path.exists(in_name):
                #* Note: check input.txt because if the folder exists,
                #* command 'cp' will copy the Runfiles folder inside the targeted folder
                #* instead of overwrite it.
                os.system('cp -rf  ./Runfiles  '+folder+'/ ')

                self.write_input(in_name, x)

                os.system('cd '+folder+' &&  sh ./%s.sh >/dev/null'%(bash_name))
                os.system('rm -f '+folder+'/%s.sh'%(bash_name))

        #* Process results 
        succeed, y = self.read_output(out_name)

        if information and not succeed:
            print('    warning: [external_run] failed: %s'%(folder_name))

        return succeed, y

    def write_input(self, fname: str, x: np.ndarray) -> None:
        '''
        Write x into fname. (each line: var_name, value)
        '''
        f = open(fname, 'w', encoding='utf-8')
        for i in range(x.shape[0]):
            f.write('  %20s  %20.9f \n'%(self.data_settings.name_input[i], x[i]))
        f.close()
        
    def read_input(self, fname: str) -> Tuple[bool, np.ndarray]:
        '''
        Read input file [fname], (each line: var_name, value)

        Returns
        -------------
        succeed: bool
            whether the evaluation succeed or not
        x: ndarray [dim_input]
            function input
        '''
        
        succeed = True
        x = np.ones(self.n_input)

        if not os.path.exists(fname):
            return False, x

        f = open(fname, 'r+', encoding='utf-8')
        lines = f.readlines()
        
        if len(lines) == 0:
            return False, x
        
        dict_out = dict()
        for line in lines:
            line = line.split()
            dict_out[line[0]] = float(line[1])

        for i in range(self.n_input):
            name_var = self.data_settings.name_input[i]
            if not name_var in dict_out.keys():
                print('  Error: input [%s] is not in %s'%(name_var, fname))
                succeed = False
                continue
            x[i] = dict_out[name_var]

        return succeed, x

    def read_output(self, fname: str) -> Tuple[bool, np.ndarray]:
        '''
        Read output file [fname], (each line: var_name, value)

        Returns
        -------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_input]
            function output
        '''
        
        succeed = True
        y = np.ones(self.n_output)

        if not os.path.exists(fname):
            return False, y

        f = open(fname, 'r+', encoding='utf-8')
        lines = f.readlines()
        
        if len(lines) == 0:
            return False, y
        
        dict_out = dict()
        for line in lines:
            line = line.split()
            if len(line)==0 and len(dict_out)>0:
                break
            dict_out[line[0]] = float(line[1])

        for i in range(self.n_output):
            name_out = self.data_settings.name_output[i]
            if not name_out in dict_out.keys():
                print('  Error: output [%s] is not in %s'%(name_out, fname))
                succeed = False
                continue
            y[i] = dict_out[name_out]

        return succeed, y

    #* Evaluation of the constraint function.
    
    def eval_constraints(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        '''
        Evaluate all the constraint functions.
        
        Parameters
        -------------
        x: ndarray [dim_input]
            function input
        y: ndarray [dim_output]
            function output

        Returns
        -------------
        sum_violation: float
            sum of the constraint violations, only the violated constraints are counted.
        violations: ndarray [n_constraint]
            constraint violations, the constraint is violated if the violation is greater than 0.
            All the original constraint values are returned.
        '''
        violations = np.zeros(self.n_constraint)
        
        i_constraint = 0
        for constraint_str in self.problem_settings.constraint_strings:
            violation = self.eval_constraint_string(constraint_str, x, y)
            violations[i_constraint] = violation
            i_constraint += 1
            
        for constraint_func in self.problem_settings.constraint_functions:
            violation = constraint_func(x, y)
            violations[i_constraint] = violation
            i_constraint += 1
            
        sum_violation = np.sum(np.maximum(0.0, violations))
            
        return sum_violation, violations
    
    def eval_constraint_string(self, formula: str, x: np.ndarray, y: np.ndarray) -> float:
        '''
        Evaluate the constraint string.
        
        Parameters
        -------------
        formula: str
            constraint string
        x: ndarray [dim_input]
            function input
        y: ndarray [dim_output]
            function output
        
        Returns
        -------------
        violation: float
            constraint violation, the constraint is violated if the violation is greater than 0.
        '''
        new_formula = ''
        items = formula.split(' ')
        
        for item in items:
            
            if item in self.data_settings.name_input:
                i = self.data_settings.name_input.index(item)
                # Wrap numeric substitution with parentheses so negative values
                # keep expected precedence, e.g. (-1.0)**2 instead of -1.0**2.
                new_formula = new_formula + f'({x[i]})'
                
            elif item in self.data_settings.name_output:
                i = self.data_settings.name_output.index(item)
                new_formula = new_formula + f'({y[i]})'
                
            else:
                new_formula = new_formula + item

        result = ne.evaluate(new_formula)
  
        return result

    #* Pareto dominance.

    def check_pareto_dominance(self, y1: np.ndarray, y2: np.ndarray) -> int:
        '''
        Check the dominance relationship between self and other
        
        Parameters
        -------------
        y1, y2: ndarray [n_output]
            function outputs
        
        Returns
        -------------
        i_dominance: int
            dominance relationship between y1 and y2
            - `0`: equal
            - `1`: y1 dominates y2
            - `-1`: y1 is dominated by y2
            - `9`: y1 and y2 are non-dominated
        '''
        dominance_list = []
        for i in range(self.n_output):

            ii = 0
            if self.problem_settings.output_type[i] == 1:
                if y1[i] > y2[i]:
                    ii = 1
                elif y1[i] < y2[i]:
                    ii = -1
                
            elif self.problem_settings.output_type[i] == -1:
                if y1[i] > y2[i]:
                    ii = -1
                elif y1[i] < y2[i]:
                    ii = 1

            dominance_list.append(ii)
        
        i_dominance = 0

        if 1 in dominance_list and -1 in dominance_list:
            i_dominance = 9

        if 1 in dominance_list and not -1 in dominance_list:
            i_dominance = 1

        if not 1 in dominance_list and -1 in dominance_list:
            i_dominance = -1

        return i_dominance

    #* Perturbation of the input vector.
    
    def perturb_scaled_x(self, scaled_x: np.ndarray, 
                            n_perturb: int = 1, dx: float = 0.01) -> np.ndarray:
        '''
        Perturb the scaled input vector.
        
        Parameters
        -------------
        scaled_x: ndarray [n_input]
            scaled input vector
        n_perturb: int
            number of perturbations
        dx: float
            relative perturbation scale (0~1)
        
        Returns
        -------------
        perturbed_scaled_x: ndarray [n_perturb, n_input]
            perturbed input vectors
        '''
        perturbed_scaled_x = np.zeros([n_perturb, self.n_input])
        dxs = np.random.rand(n_perturb, self.n_input) # [0, 1]
        dxs = (2*dxs-1.0)*dx
        perturbed_scaled_x = scaled_x + dxs
        
        # apply bounds of [0,1]
        mask_upper = perturbed_scaled_x > 1.0
        mask_lower = perturbed_scaled_x < 0.0
        perturbed_scaled_x[mask_upper] = 1.0
        perturbed_scaled_x[mask_lower] = 0.0
        
        return perturbed_scaled_x

    def perturb_x(self, x: np.ndarray, n_perturb: int = 1, dx: float = 0.01) -> np.ndarray:
        '''
        Perturb the input vector.
        '''
        scaled_x = self.scale_x(x)
        perturbed_scaled_x = self.perturb_scaled_x(scaled_x, n_perturb, dx)
        perturbed_x = self.scale_x(perturbed_scaled_x, reverse=True)
        self.apply_bounds_x(perturbed_x)
        return perturbed_x

    #* Support functions
    
    def check_bounds_x(self, x: np.ndarray) -> bool:
        '''
        Check if the input vector is within the bounds.
        '''
        return np.all(x >= self.data_settings.input_low) and np.all(x <= self.data_settings.input_upp)
    
    def check_bounds_y(self, y: np.ndarray) -> bool:
        '''
        Check if the output vector is within the bounds.
        '''
        return np.all(y >= self.data_settings.output_low) and np.all(y <= self.data_settings.output_upp)
    
    def apply_bounds_x(self, x: np.ndarray) -> bool:
        '''
        Apply the bounds to the input vector.
        
        Parameters
        -------------
        x: ndarray [n_input] or [:, n_input]
            input vector
        
        Returns
        -------------
        within_bounds: bool
            whether the input vector is within the bounds
        '''
        mask_upper = x > self.data_settings.input_upp
        mask_lower = x < self.data_settings.input_low
        upp = np.broadcast_to(self.data_settings.input_upp, x.shape)
        low = np.broadcast_to(self.data_settings.input_low, x.shape)
        x[mask_upper] = upp[mask_upper]
        x[mask_lower] = low[mask_lower]
        
        within_bounds = not (np.any(mask_upper) or np.any(mask_lower))
        return within_bounds
    
    def apply_bounds_y(self, y: np.ndarray) -> bool:
        '''
        Apply the bounds to the output vector.
        
        Parameters
        -------------
        y: ndarray [n_output] or [:, n_output]
            output vector
        
        Returns
        -------------
        within_bounds: bool
            whether the output vector is within the bounds
        '''
        mask_upper = y > self.data_settings.output_upp
        mask_lower = y < self.data_settings.output_low
        upp = np.broadcast_to(self.data_settings.output_upp, y.shape)
        low = np.broadcast_to(self.data_settings.output_low, y.shape)
        y[mask_upper] = upp[mask_upper]
        y[mask_lower] = low[mask_lower]
        
        within_bounds = not (np.any(mask_upper) or np.any(mask_lower))
        return within_bounds
    
    def scale_x(self, x: np.ndarray, reverse: bool = False) -> np.ndarray:
        '''
        Scale the input vector to [0, 1] or from [0, 1] to the original range.
        
        Parameters
        -------------
        x: ndarray [n_input]
            input vector
        reverse: bool
            if True, scale [0, 1] to the original range
            if False, scale the original range to [0, 1]
        
        Returns
        -------------
        x: ndarray [n_input]
            scaled input vector, precision applied.
        '''
        span = self.data_settings.input_upp - self.data_settings.input_low
        
        if reverse:
            span[self.mask_for_deactivated_inputs] = 0.0
            x = x * span + self.data_settings.input_low
            SettingsData.apply_precision(x, self.data_settings.input_precision)
            return x
        else:
            SettingsData.apply_precision(x, self.data_settings.input_precision)
            span[self.mask_for_deactivated_inputs] = 1.0
            x = (x - self.data_settings.input_low) / span
            if x.ndim == 1:
                x[self.mask_for_deactivated_inputs] = 0.0
            else:
                x[:, self.mask_for_deactivated_inputs] = 0.0
            return x
    
    def scale_y(self, y: np.ndarray, reverse: bool = False) -> np.ndarray:
        '''
        Scale the output vector to [0, 1] or from [0, 1] to the original range.
        '''
        span = self.data_settings.output_upp - self.data_settings.output_low
        if reverse:
            span[self.mask_for_deactivated_outputs] = 0.0
            y = y * span + self.data_settings.output_low
            SettingsData.apply_precision(y, self.data_settings.output_precision)
            return y
        else:
            SettingsData.apply_precision(y, self.data_settings.output_precision)
            span[self.mask_for_deactivated_outputs] = 1.0
            y = (y - self.data_settings.output_low) / span
            if y.ndim == 1:
                y[self.mask_for_deactivated_outputs] = 0.0
            else:
                y[:, self.mask_for_deactivated_outputs] = 0.0
            return y
    
    def get_output_by_type(self, y: np.ndarray, type_list: List[int]) -> np.ndarray:
        '''
        Get the output by the type list.
        
        Parameters
        -------------
        y: ndarray [n_output]
            output vector
        type_list: List[int]
            type list
        
        Returns
        -------------
        y: ndarray [n]
            output vector by the type list
        '''
        return y[np.isin(self.problem_settings.output_type, type_list)]
    
    def calculate_scaled_distance(self, x1: np.ndarray, x2: np.ndarray,
                            is_scaled_x: bool = False,
                            metric: str = 'euclidean') -> np.ndarray:
        '''
        Calculate the scaled distance between two input vectors.
        
        Parameters
        -------------
        x1, x2: ndarray [n, n_input] or [n_input]
            (scaled) input vectors
        is_scaled_x: bool
            if True, the input vectors are already scaled.
        metric: str
            metric for distance calculation, refer to scipy.spatial.distance.cdist.
        
        Returns
        -------------
        distance: ndarray [n1, n2]
            distance between scaled x1 and scaled x2.
        '''
        if x1.ndim == 1:
            x1 = x1[np.newaxis, :]
        if x2.ndim == 1:
            x2 = x2[np.newaxis, :]
        
        if not is_scaled_x:
            x1 = self.scale_x(x1)
            x2 = self.scale_x(x2)
            
        distance_matrix = cdist(x1, x2, metric=metric)

        return distance_matrix
    
    def is_subset_of(self, other: 'Problem') -> bool:
        '''
        Check if the problem is a subset of another problem.
        '''
        flag_1 = set(self.data_settings.name_input).issubset(other.data_settings.name_input)
        flag_2 = set(self.data_settings.name_output).issubset(other.data_settings.name_output)
        return flag_1 and flag_2
    