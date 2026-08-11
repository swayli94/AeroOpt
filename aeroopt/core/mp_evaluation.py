'''
Multi-process evaluation of a user-defined function

Source:

    https://github.com/swayli94/mpEvaluation

ProcessPoolExecutor:

    https://docs.python.org/3/library/concurrent.futures.html

'''
from __future__ import annotations

import numpy as np
import time
from concurrent.futures import (
    Future, ProcessPoolExecutor, as_completed,
    TimeoutError as FuturesTimeoutError,
)
from aeroopt.core.problem import Problem, StaleCaseFolderError
from typing import Callable, Dict, Iterator, List, Tuple


class MultiProcessEvaluation:
    '''
    Multi-process evaluation of a user-defined function `y=func(x, **kwargs)`.

    >>> mpRun = MultiProcessEvaluation(dim_input, dim_output, func=None,
    >>>                 n_process=None, information=True, timeout=None)

    Parameters
    --------------
    dim_input: int
        dimension of the function input `x`
    dim_output: int
        dimension of the function input `y`
    func: callable or None
        the user-defined function.
        If `func` is None, it uses an external evaluation script to get the result.
        The details are explained in function `external_run`.
    n_process: int or None
        maximum number of processors. If `n_process` is None, use serial computation.
    information: bool
        whether print information on screen
    timeout: float or None
        seconds one *external* evaluation may take before its process tree is
        killed and the design is recorded as a failure, see
        :meth:`~aeroopt.core.problem.Problem.external_run`. None means no limit.
        It does not bound a user-defined `func`, which runs in-process.
    batch_timeout: float or None
        seconds the *whole batch* may take. None (the default) means no limit,
        which is what a per-evaluation `timeout` already bounds: with more
        designs than processes the batch legitimately takes several times
        longer than one evaluation. When it does fire, the designs still
        running are recorded as failures and their solvers are left to finish.

    Notes
    ----------------
    The `if __name__ == '__main__'` is necessary for python multiprocessing.

    https://docs.python.org/3/library/multiprocessing.html

    For an explanation of why the `if __name__ == '__main__'` part is necessary, see Programming guidelines.

    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming


    User-defined function:

    >>> succeed, y = func(x, **kwargs)
    >>> # x: ndarray [dim_input]
    >>> # y: ndarray [dim_output]
    >>> # succeed: bool

    Evaluation of `n` inputs:

    >>> list_succeed, ys = Multiprocessing.evaluate(xs, **kwargs)
    >>> # xs: ndarray [n, dim_input]
    >>> # ys: ndarray [n, dim_output]
    >>> # list_succeed: list [bool], length is n
    '''
    def __init__(self, dim_input: int, dim_output: int, func: Callable|None = None,
                    n_process: int|None = None, information: bool = True,
                    timeout: float|None = None, batch_timeout: float|None = None):
        '''
        Configure the evaluator.

        Parallel evaluation is performed with
        :class:`concurrent.futures.ProcessPoolExecutor`, whose ``submit``
        schedules a callable and returns a ``Future`` representing its
        execution::

            executor = ProcessPoolExecutor(max_workers=None, mp_context=None,
                                           initializer=None, initargs=())
            future = executor.submit(fn, *args, **kwargs)

        Only ``max_workers`` is configurable here, through ``n_process``: it is
        the maximum number of worker processes, defaulting to the number of
        processors when None.
        '''
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.func = func
        self.n_process = n_process
        self.information = information
        self.timeout = timeout
        self.batch_timeout = batch_timeout

    def external_run(self, name: str, x: np.ndarray, prob: Problem):
        '''
        External calculation by calling run.bat/.sh.

        >>> succeed, y = external_run(self, name, x, prob)

        Parameters
        -----------------
        name: str
            name of the current running folder, the working folder is ./Calculation/name.
        x: ndarray [dim_input]
            function input
        prob: Problem
            the problem for external runs

        Returns
        ----------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_output]
            function output
        '''
        return prob.external_run(name, x, information=self.information, timeout=self.timeout)

    def func_mp(self, x: np.ndarray, i: int, **kwargs):
        '''
        Callable function for the ProcessPoolExecutor

        >>> succeed, y, i = func_mp(self, x, i, **kwargs)

        An evaluation that raises is reported as a failed design, not as an
        error: a solver bridge that rejects its input, or a mesher that throws,
        must not take the whole study down with it. Two things are re-raised
        instead, because they are setup mistakes that would fail identically for
        every design and that no amount of retrying fixes: a missing folder name
        or problem object, and
        :class:`~aeroopt.core.problem.StaleCaseFolderError`.

        Parameters
        -----------------
        x: ndarray [dim_input]
            function input
        i: int
            index of this `x` in xs[n, dim_input]
        name: str
            name of the current running folder, the working folder is ./Calculation/name
        prob: Problem
            the problem for external runs

        Returns
        ----------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_output]
            function output
        i: int
            index of this `x` in xs[n, dim_input]
        '''
        if self.func is None:

            if 'name' not in kwargs.keys():
                raise Exception('Must define name as the the working folder')

            if 'prob' not in kwargs.keys():
                raise Exception('Must provide Problem object `prob` for external running')

        try:
            if self.func is None:
                succeed, y = self.external_run(kwargs['name'], x, kwargs['prob'])
            else:
                succeed, y = self.func(x, **kwargs)

        except StaleCaseFolderError:
            raise

        except Exception as e:
            if self.information:
                print('    warning: [evaluation] failed for #%d: %s: %s'
                      % (i, type(e).__name__, e))
            return False, np.zeros(self.dim_output), i

        return succeed, y, i

    def evaluate(self, xs: np.ndarray, list_name: List[str]|None = None, **kwargs):
        '''
        Evaluation of the multiple inputs `xs`.

        >>> list_succeed, ys = evaluate(xs, list_name)

        Parameters
        -----------------
        xs: ndarray [n, dim_input]
            function input
        list_name: list or None
            list of working folder names for external runs
        prob: Problem
            the problem for external runs
        n_show: int
            print number of succeed runs each n_show succeed runs

        Returns
        -----------------
        list_succeed: list [bool]
            list of succeed for each input
        ys: ndarray [n, dim_output]
            function output

        Notes
        -----------------
        Schedule the callable functions to be executed

        >>> future = executor.submit(fn, *args, **kwargs)

        returns a Future object representing the execution of the callable

        Yield futures as they complete (finished or cancelled)

        >>> for f in as_completed(futures, timeout=None):
        >>>     f.result()

        Any futures that completed before as_completed() is called will be yielded first.
        The returned iterator raises a concurrent.futures.TimeoutError
        if __next__() is called and the result isn't available after timeout seconds
        from the original call to as_completed(). timeout can be an int or float.
        If timeout is not specified or None, there is no limit to the wait time.
        #! This timeout will raise an Error
        '''
        n = xs.shape[0]
        ys = np.zeros([n, self.dim_output])
        list_succeed = [False for _ in range(n)]

        n_show = int(kwargs.get('n_show', 100))

        if 'prob' in kwargs:
            prob = kwargs['prob']
        elif self.func is None:
            raise Exception('Must provide Problem object `prob` for external running')
        else:
            prob = None

        #* Serial calculation
        if self.n_process is None:

            if self.func is None:

                if list_name is None:
                    raise Exception('Must provide a list of working folder names')

                if not isinstance(prob, Problem):
                    raise Exception('Must provide Problem object `prob` for external running')

                for i in range(n):
                    list_succeed[i], ys[i,:], _ = self.func_mp(
                        xs[i,:], i, name=list_name[i], prob=prob)

            else:
                for i in range(n):
                    list_succeed[i], ys[i,:], _ = self.func_mp(
                        xs[i,:], i, **kwargs)

        #* Multiprocessing calculation
        else:

            with ProcessPoolExecutor(max_workers=self.n_process) as executor:

                index_of_future = {}

                for i in range(n):

                    if self.func is None:

                        if list_name is None:
                            raise Exception('Must provide a list of working folder names')

                        future = executor.submit(self.func_mp, xs[i,:], i, name=list_name[i], **kwargs)

                    else:
                        future = executor.submit(self.func_mp, xs[i,:], i, **kwargs)

                    index_of_future[future] = i

                num = 0
                t0 = time.perf_counter()

                for f in self._completed_futures(index_of_future):

                    succeed, y, i = f

                    ys[i,:] = y
                    list_succeed[i] = succeed

                    if succeed:
                        num += 1
                        if num%n_show==0:
                            t1 = time.perf_counter()
                            print('  > parallel calculation done: n = %d, t = %.2f min'%(num, (t1-t0)/60.0))

        return list_succeed, ys

    def _completed_futures(self, index_of_future: Dict[Future, int]
                           ) -> Iterator[Tuple[bool, np.ndarray, int]]:
        '''
        Yield `(succeed, y, i)` for every submitted design, in completion order.

        A worker that dies rather than returns --- a killed process, a result
        that cannot be pickled, a `BrokenProcessPool` --- yields a failure for
        its design instead of propagating, so one broken evaluation cannot end
        the study. When `batch_timeout` expires, everything still running is
        yielded as a failure; those solvers are left to finish, since the pool
        has no way to interrupt them.

        Parameters:
        -----------
        index_of_future: Dict[Future, int]
            Submitted futures, mapped to the row of `xs` each one evaluates.

        Yields:
        --------
        (succeed, y, i): Tuple[bool, np.ndarray, int]
        '''
        pending = set(index_of_future)

        try:
            for future in as_completed(index_of_future, timeout=self.batch_timeout):

                pending.discard(future)
                i = index_of_future[future]

                try:
                    yield future.result()

                except StaleCaseFolderError:
                    raise

                except Exception as e:
                    if self.information:
                        print('    warning: [evaluation] worker for #%d died: %s: %s'
                              % (i, type(e).__name__, e))
                    yield False, np.zeros(self.dim_output), i

        except FuturesTimeoutError:
            if self.information:
                print('    warning: [evaluation] batch timeout after %.1f s, '
                      '%d of %d designs unfinished'
                      % (float(self.batch_timeout or 0.0), len(pending),
                         len(index_of_future)))

            for future in pending:
                future.cancel()
                yield False, np.zeros(self.dim_output), index_of_future[future]


def template_user_func(x: np.ndarray, **kwargs) -> tuple[bool, np.ndarray]:
    '''
    Reference implementation of a user evaluation function: `succeed, y = func(x)`.
    '''
    return True, np.array([np.sum(x**2)])

