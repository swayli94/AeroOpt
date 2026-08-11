'''
Problem definition.
'''

from __future__ import annotations

import os
import platform
import shutil
import subprocess

import numpy as np
import numexpr as ne
from scipy.spatial.distance import cdist

from typing import Any, Tuple, List, cast

from aeroopt.core.settings import SettingsData, SettingsProblem
from aeroopt.sampling import latin_hypercube_sampling


class StaleCaseFolderError(RuntimeError):
    '''
    An external case folder holds an input file for a different design.

    Raised by :meth:`Problem.external_run`. It means the working folder belongs
    to an earlier study (case numbering restarts at 1 for every fresh study, and
    a re-parameterization changes what the same variable names mean), so reading
    its output would score the previous design's results against the current
    `x`. Clear or move the calculation folder, resume the previous study, or set
    :attr:`Problem.rerun_stale_cases`.
    '''


class Problem:
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
        Name of the folder holding one working folder per external evaluation.
    runfiles_folder: str
        Name of the folder whose contents (solver, run script, templates) are
        copied into each working folder before the external run.
    rerun_stale_cases: bool
        What to do when a case folder already holds an input file for a
        *different* design. False (the default) raises
        :class:`StaleCaseFolderError`, so leftovers from a previous study are
        reported instead of quietly overwritten. True re-prepares and re-runs
        the folder, discarding the old result.
    '''
    def __init__(self, data_settings: SettingsData, problem_settings: SettingsProblem):

        self.data_settings = data_settings
        self.problem_settings = problem_settings

        self.input_fname : str = 'input.txt'
        self.output_fname : str = 'output.txt'

        self.calculation_folder : str = 'Calculation'
        self.runfiles_folder : str = 'Runfiles'

        self.rerun_stale_cases : bool = False

    def __eq__(self, other) -> bool:
        '''
        Two problems are equal when their problem settings share the same name.
        '''
        if not isinstance(other, Problem):
            return NotImplemented

        return self.problem_settings.name == other.problem_settings.name

    def __hash__(self) -> int:
        '''
        Hash consistent with `__eq__`, so problems can be used in sets and dict keys.
        Defining `__eq__` alone would set `__hash__` to None and make `Problem` unhashable.
        '''
        return hash(self.problem_settings.name)


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
    def name_input(self) -> List[str]:
        '''
        Name of the input variables.
        '''
        return self.data_settings.name_input

    @property
    def name_output(self) -> List[str]:
        '''
        Name of the output variables.
        '''
        return self.data_settings.name_output

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
        Evaluate `x` by running an external solver in its own working folder.

        The working folder is `<calculation_folder>/<folder_name>`. When it does
        not already contain an input file, the contents of `runfiles_folder` are
        copied in, `x` is written to the input file, and the run script is
        executed with the working folder as its current directory.

        An existing input file **holding this same `x`** means the case was
        already prepared (and possibly already run), so it is left alone and
        only the output file is read. That makes an interrupted study
        restartable.

        An existing input file holding a *different* `x` is a **stale** folder,
        left by an earlier study that used this name --- a fresh study numbers
        its cases from 1 again, and a re-parameterization changes what the same
        variable names mean. Reading that folder's output would record the old
        design's `y` against the new `x`, a mismatch no downstream check can
        detect, so :class:`StaleCaseFolderError` is raised instead. Clear (or
        move) `calculation_folder` before starting a new study, or set
        `rerun_stale_cases` to re-prepare such folders automatically.

        Parameters
        -----------------
        folder_name: str
            name of the case folder inside `calculation_folder`.
        x: ndarray [dim_input]
            function input
        information: bool
            whether print information on screen
        bash_name: str
            base name of the external run script, without extension:
            `run.bat` on Windows, `run.sh` elsewhere.
        timeout: float, or None
            seconds to wait for the solver. If None, wait indefinitely.
            A run that exceeds the timeout is reported as failed.

        Returns
        ----------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_output]
            function output

        Raises
        ----------------
        StaleCaseFolderError
            The case folder holds an input file for a different design and
            `rerun_stale_cases` is False.

        I/O files
        ----------------
        input_fname: str
            file written with the values of `x`, one `name value` pair per line.
        output_fname: str
            file the solver is expected to write, one `name value` pair per line.
        '''

        folder = os.path.join(self.calculation_folder, folder_name)
        out_name = os.path.join(folder, self.output_fname)
        in_name = os.path.join(folder, self.input_fname)

        os.makedirs(folder, exist_ok=True)

        is_prepared = os.path.exists(in_name)

        if is_prepared:

            matches, x_recorded = self._input_file_matches(in_name, x)

            if not matches:

                if not self.rerun_stale_cases:
                    recorded = ('unreadable' if x_recorded is None
                                else np.array2string(x_recorded, precision=6))
                    raise StaleCaseFolderError(
                        'Case folder [%s] already holds an input file for a different '
                        'design, so its results belong to another study. Reading them '
                        'would attach the wrong output to this candidate.\n'
                        '  Requested x: %s\n'
                        '  Recorded  x: %s\n'
                        'Clear or move [%s] before starting a new study, resume the '
                        'previous one (`resume` in the optimization settings), or set '
                        '`problem.rerun_stale_cases = True` to re-run such folders.'
                        % (folder, np.array2string(x, precision=6), recorded,
                           self.calculation_folder))

                if information:
                    print('    warning: [external_run] stale case folder re-run: %s'
                          % (folder_name))

                # The stale output belongs to the previous design. Deleting it
                # means a failed re-run is reported as a failure instead of
                # silently returning the old `y`.
                if os.path.exists(out_name):
                    os.remove(out_name)

                is_prepared = False

        if not is_prepared:

            # `dirs_exist_ok` copies the *contents* of the run-files folder.
            # A plain `cp -r Runfiles folder/` would nest it as a subdirectory,
            # because the case folder was just created above.
            if os.path.isdir(self.runfiles_folder):
                shutil.copytree(self.runfiles_folder, folder, dirs_exist_ok=True)
            elif information:
                print('    warning: [external_run] run-files folder not found: %s'
                      % (self.runfiles_folder))

            self.write_input(in_name, x)

            if platform.system() == 'Windows':
                command = [os.path.join('.', bash_name + '.bat')]
            else:
                command = ['sh', os.path.join('.', bash_name + '.sh')]

            try:
                subprocess.run(command, cwd=folder, timeout=timeout, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            except subprocess.TimeoutExpired:
                if information:
                    print('    warning: [external_run] timeout after %.1f s: %s'
                          % (float(timeout or 0.0), folder_name))

            except OSError as e:
                if information:
                    print('    warning: [external_run] failed to start %s: %s'
                          % (command, e))

        #* Process results
        succeed, y = self.read_output(out_name)

        if information and not succeed:
            print('    warning: [external_run] failed: %s'%(folder_name))

        return succeed, y

    def _input_file_matches(self, fname: str,
                            x: np.ndarray) -> Tuple[bool, np.ndarray | None]:
        '''
        Whether the input file `fname` holds the input vector `x`.

        A file that cannot be read, or that misses one of the input variables
        (as after a re-parameterization), counts as *not* matching, with `None`
        for the recorded input.

        The tolerance follows the `%20.9f` format of :meth:`write_input`, which
        keeps nine decimals, plus a relative term for large values whose decimal
        representation cannot resolve that many digits.

        Returns
        -------------
        matches: bool
            whether the file holds this same input vector.
        x_recorded: ndarray [dim_input], or None
            the input vector read back, None when the file could not be read.
        '''
        try:
            succeed, x_recorded = self.read_input(fname)
        except OSError:
            return False, None

        if not succeed:
            return False, None

        return bool(np.allclose(x_recorded, x, rtol=1e-8, atol=1e-9)), x_recorded

    def write_input(self, fname: str, x: np.ndarray) -> None:
        '''
        Write `x` to `fname`, one `name value` pair per line.
        '''
        with open(fname, 'w', encoding='utf-8') as f:
            for i in range(x.shape[0]):
                f.write('  %20s  %20.9f \n' % (self.data_settings.name_input[i], x[i]))

    @staticmethod
    def _read_name_value_file(fname: str) -> dict[str, float]:
        '''
        Parse a `name value` text file into a dictionary.

        Blank lines and lines that do not hold a parsable numeric value are
        skipped, so a partially written file from a crashed external solver
        yields a partial dictionary instead of raising.
        '''
        values: dict[str, float] = {}

        with open(fname, encoding='utf-8') as f:
            for line in f:
                items = line.split()
                if len(items) < 2:
                    continue
                try:
                    values[items[0]] = float(items[1])
                except ValueError:
                    continue

        return values

    def _read_variables(self, fname: str, names: List[str]) -> Tuple[bool, np.ndarray]:
        '''
        Read the named variables from a `name value` text file.

        Returns
        -------------
        succeed: bool
            True when the file exists and holds every requested name.
        values: ndarray [len(names)]
            Values of the requested variables; entries stay at 1.0 when missing.
        '''
        values = np.ones(len(names))

        if not os.path.exists(fname):
            return False, values

        parsed = self._read_name_value_file(fname)
        if not parsed:
            return False, values

        succeed = True
        for i, name in enumerate(names):
            if name not in parsed:
                print('  Error: variable [%s] is not in %s' % (name, fname))
                succeed = False
                continue
            values[i] = parsed[name]

        return succeed, values

    def read_input(self, fname: str) -> Tuple[bool, np.ndarray]:
        '''
        Read input file `fname` (each line: `var_name value`).

        Returns
        -------------
        succeed: bool
            whether every input variable was found
        x: ndarray [dim_input]
            function input
        '''
        return self._read_variables(fname, self.data_settings.name_input)

    def read_output(self, fname: str) -> Tuple[bool, np.ndarray]:
        '''
        Read output file `fname` (each line: `var_name value`).

        Returns
        -------------
        succeed: bool
            whether every output variable was found
        y: ndarray [dim_output]
            function output
        '''
        return self._read_variables(fname, self.data_settings.name_output)

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
        y = np.asarray(y, dtype=float)
        if y.size == 0 and self.n_output > 0:
            y = np.zeros(self.n_output, dtype=float)

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

        return float(result)

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

        if 1 in dominance_list and -1 not in dominance_list:
            i_dominance = 1

        if 1 not in dominance_list and -1 in dominance_list:
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
        dxs = np.random.rand(n_perturb, self.n_input) # [0, 1]
        dxs = (2*dxs-1.0)*dx
        perturbed_scaled_x = scaled_x + dxs

        # apply bounds of [0,1]
        np.clip(perturbed_scaled_x, 0.0, 1.0, out=perturbed_scaled_x)

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

    #* Sampling of input/output vectors.

    def latin_hypercube_sampling(self, n: int,
                scaled_values: bool = False,
                sample_variables: List[str]|None = None,
                seed: int|None = None) -> np.ndarray:
        '''
        Latin Hypercube Sampling for the input/output vectors.

        Parameters
        -------------
        n: int
            number of samples
        scaled_values: bool
            if True, return the scaled values.
            if False, return the original values.
        sample_variables: List[str]
            list of variables (n_variables) to be sampled.
            If None, sample all the input variables.
        seed: int, or None
            seed for the random number generator.
            If None, use the default random number generator.

        Returns
        -------------
        samples: ndarray [n, n_variables]
            sampled input/output vectors
        '''

        if sample_variables is None:
            n_variables = self.n_input
        elif isinstance(sample_variables, list):
            n_variables = len(sample_variables)
        else:
            raise ValueError('Invalid sample_variables.')

        v_samples = latin_hypercube_sampling(n_variables, n, criterion='m', seed=seed)

        if scaled_values:
            return v_samples

        if sample_variables is None:
            v_samples = self.scale_x(v_samples, reverse=True)
            return v_samples

        for i_variable in range(n_variables):

            name = sample_variables[i_variable]

            if name in self.data_settings.name_input:
                i = self.data_settings.name_input.index(name)
                low = self.data_settings.input_low[i]
                upp = self.data_settings.input_upp[i]
                precision = self.data_settings.input_precision[i]
            elif name in self.data_settings.name_output:
                i = self.data_settings.name_output.index(name)
                low = self.data_settings.output_low[i]
                upp = self.data_settings.output_upp[i]
                precision = self.data_settings.output_precision[i]
            else:
                raise ValueError('Invalid name of variable %s.'%(name))

            v_samples[:, i_variable] = low + v_samples[:, i_variable] * (upp - low)

            # The whole-vector path scales through `scale_x`, which snaps to the
            # precision grid; sampling a subset must not skip that.
            if precision != 0.0:
                v_samples[:, i_variable] = (
                    np.round(v_samples[:, i_variable] / precision) * precision)

        return v_samples

    #* Support functions

    def check_bounds_x(self, x: np.ndarray) -> bool:
        '''
        Check if the input vector is within the bounds.
        '''
        return bool(
            np.all(x >= self.data_settings.input_low) and np.all(x <= self.data_settings.input_upp)
        )

    def check_bounds_y(self, y: np.ndarray) -> bool:
        '''
        Check if the output vector is within the bounds.
        '''
        return bool(
            np.all(y >= self.data_settings.output_low) and np.all(y <= self.data_settings.output_upp)
        )

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

    def apply_precision_x(self, x: np.ndarray) -> None:
        '''
        Snap the input vector to the precision grid of each input variable (in place).

        Every operator that builds a new `x` must call this after
        :meth:`apply_bounds_x`, because `input_precision` is a *hard* property of
        the design variable --- an integer count of ribs, a ply number that must
        be even, a thickness the solver only accepts on a 0.1 mm grid. A value
        off that grid is not a slightly worse design, it is one the external
        evaluation rejects outright.

        The bounds are themselves snapped to the grid when the settings are
        checked, so snapping after clipping cannot leave `x` out of bounds.

        Parameters
        -------------
        x: ndarray [n_input] or [:, n_input]
            input vector, modified in place.

        Returns
        -------------
        None
        '''
        SettingsData.apply_precision(x, self.data_settings.input_precision)

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

    @staticmethod
    def _scale(values: np.ndarray, low: np.ndarray, upp: np.ndarray,
               precision: np.ndarray, mask_deactivated: np.ndarray,
               reverse: bool) -> np.ndarray:
        '''
        Scale a variable vector between its original range and `[0, 1]`.

        A new array is always returned; the caller's array is never modified,
        even though `SettingsData.apply_precision` works in place.

        Deactivated variables (span smaller than the precision) map to `0.0`
        in scaled space and to their lower bound in original space.
        '''
        span = upp - low

        if reverse:
            span[mask_deactivated] = 0.0
            scaled = np.asarray(values, dtype=float) * span + low
            SettingsData.apply_precision(scaled, precision)
            return scaled

        scaled = np.array(values, dtype=float, copy=True)
        SettingsData.apply_precision(scaled, precision)
        span[mask_deactivated] = 1.0
        scaled = (scaled - low) / span
        if scaled.ndim == 1:
            scaled[mask_deactivated] = 0.0
        else:
            scaled[:, mask_deactivated] = 0.0
        return scaled

    def scale_x(self, x: np.ndarray, reverse: bool = False) -> np.ndarray:
        '''
        Scale the input vector to [0, 1] or from [0, 1] to the original range.

        Parameters
        -------------
        x: ndarray [n_input] or [n, n_input]
            input vector; not modified
        reverse: bool
            if True, scale [0, 1] to the original range
            if False, scale the original range to [0, 1]

        Returns
        -------------
        x: ndarray, same shape as the input
            scaled input vector, precision applied.
        '''
        return self._scale(
            x,
            self.data_settings.input_low,
            self.data_settings.input_upp,
            self.data_settings.input_precision,
            self.mask_for_deactivated_inputs,
            reverse,
        )

    def scale_y(self, y: np.ndarray, reverse: bool = False) -> np.ndarray:
        '''
        Scale the output vector to [0, 1] or from [0, 1] to the original range.

        The caller's array is not modified; see :meth:`scale_x`.
        '''
        return self._scale(
            y,
            self.data_settings.output_low,
            self.data_settings.output_upp,
            self.data_settings.output_precision,
            self.mask_for_deactivated_outputs,
            reverse,
        )

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
        mask = np.isin(np.asarray(self.problem_settings.output_type), type_list)
        if y.ndim == 1:
            return y[mask]
        return y[:, mask]

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

        distance_matrix = cdist(x1, x2, metric=cast(Any, metric))

        return distance_matrix

    def is_subset_of(self, other: Problem) -> bool:
        '''
        Check if the problem is a subset of another problem.
        '''
        flag_1 = set(self.data_settings.name_input).issubset(other.data_settings.name_input)
        flag_2 = set(self.data_settings.name_output).issubset(other.data_settings.name_output)
        return flag_1 and flag_2
