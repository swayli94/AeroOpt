'''
Multi-process evaluation of a user-defined function

Source:

    https://github.com/swayli94/mpEvaluation

ProcessPoolExecutor:

    https://docs.python.org/3/library/concurrent.futures.html

'''
import numpy as np
import time
import concurrent
import concurrent.futures
from concurrent.futures import as_completed
from aeroopt.core.problem import Problem
from typing import List, Callable


class MultiProcessEvaluation():
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
        limit to the wait time. If `timeout` is None, no limit on wait time.
        
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
                    n_process: int|None = None, information: bool = True, timeout: float|None = None):
        '''
        Using concurrent.futures.ProcessPoolExecutor as executor

        Using submit to schedule the callable, fn, to be executed as 
        fn(*args **kwargs) and returns a Future object 
        representing the execution of the callable.

        >>> executor =  ProcessPoolExecutor(max_workers=None,
                    mp_context=None, initializer=None, initargs=())

        >>> future = executor.submit(fn, *args, **kwargs)

        Args:
        ---
        max_workers:    The maximum number of processes that can be used to execute the given calls. 
                        If None or not given then as many worker processes will be created as the machine has processors.
        mp_context:     A multiprocessing context to launch the workers. 
                        This object should provide SimpleQueue, Queue and Process.
        initializer:    A callable used to initialize worker processes.
        initargs:       A tuple of arguments to pass to the initializer.
        '''
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.func = func
        self.n_process = n_process
        self.information = information
        self.timeout = timeout

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
            
            if 'name' in kwargs.keys():
                name : str = kwargs['name']
            else:
                raise Exception('Must define name as the the working folder')

            if 'prob' in kwargs.keys():
                prob : Problem = kwargs['prob']
            else:
                raise Exception('Must provide Problem object `prob` for external running')

            succeed, y = self.external_run(name, x, prob)

        else:

            succeed, y = self.func(x, **kwargs)

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

        n_show = 100
        if 'n_show' in kwargs.keys():
            n_show : int = kwargs['n_show']

        if 'prob' in kwargs.keys():
            prob = kwargs['prob']
        elif self.func is None:
            raise Exception('Must provide Problem object `prob` for external running')
        else:
            prob = None

        #* Serial calculation
        if self.n_process==None:

            if self.func is None:
                
                if list_name is None:
                    raise Exception('Must provide a list of working folder names')
                
                if not isinstance(prob, Problem):
                    raise Exception('Must provide Problem object `prob` for external running')
                
                for i in range(n):
                    list_succeed[i], ys[i,:] = self.external_run(
                        list_name[i], xs[i,:], prob)
            
            else:
                for i in range(n):
                    list_succeed[i], ys[i,:] = self.func(xs[i,:], **kwargs)

        #* Multiprocessing calculation
        else:

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_process) as executor:

                futures = []
                
                for i in range(n):

                    if self.func is None:
                        
                        if list_name is None:
                            raise Exception('Must provide a list of working folder names')
                        
                        futures.append(executor.submit(self.func_mp, xs[i,:], i, name=list_name[i], **kwargs))
                        
                    else:
                        futures.append(executor.submit(self.func_mp, xs[i,:], i, **kwargs))

                num = 0
                t0 = time.perf_counter()
                        
                for f in as_completed(futures, timeout=self.timeout):

                    succeed, y, i = f.result()

                    ys[i,:] = y
                    list_succeed[i] = succeed
                    
                    if succeed:
                        num += 1
                        if num%n_show==0:
                            t1 = time.perf_counter()
                            print('  > parallel calculation done: n = %d, t = %.2f min'%(num, (t1-t0)/60.0))
                            
        return list_succeed, ys


def template_usr_func(x, **kwargs):
    return True, np.array([np.sum(x**2)])

