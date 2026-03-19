'''
Utility functions.
'''

import datetime
import numpy as np
import platform
import os

from typing import List, Tuple


def init_log(folder_result, fname='logging.log') -> None:
    '''
    Initialize logging
    '''
    f0 = open(fname, 'w')
    f0.write('\n')
    now_time = datetime.datetime.now()
    f0.write('Time:        '+now_time.strftime('%Y-%m-%d %H:%M:%S \n'))
    f0.write('Result path: ' + str(folder_result) + '\n')

    f0.write('\n')
    f0.write('============================== \n')
    f0.write('\n')
    f0.close()
    
def log(text: str, prefix='>>> ', show_time=True, fname='logging.log') -> None:
    '''
    Log time and text
    '''
    print(prefix+text)
    
    if fname is None:
        return

    _time = ''
    if show_time:
        now_time = datetime.datetime.now()
        _time = now_time.strftime('%Y-%m-%d %H:%M:%S | ')
    
    with open(fname, 'a') as f:
        f.write(_time+prefix+text+'\n')


def check_folder(name: str):
    '''
    Check if folder ./Calculation/name exists
    '''
    if platform.system() in 'Windows':
        folder = '.\\Calculation\\'+name
        exist = os.path.exists(folder)

    else:
        folder = './Calculation/'+name
        exist = os.path.exists(folder)

    return exist

def compare_ndarray(x1: np.ndarray, x2: np.ndarray) -> int:
    '''
    Compare ndarray x1, x2 lexicographically.

    Parameters:
    -----------
    x1, x2: np.ndarray
        Arrays to compare.
    
    Returns:
    --------
    value: int
        - `0`: x1 = x2
        - `1`: x1 > x2
        - `-1`: x1 < x2
    '''
    n = x1.shape[0]
    if n != x2.shape[0]:
        raise ValueError(f"Must compare 2 arrays of same shape, got {x1.shape} and {x2.shape}")

    for i in range(n):
        if x1[i] < x2[i]:
            return -1
        if x1[i] > x2[i]:
            return 1
    return 0
    


def output_database(fname: str, xs: np.ndarray,
                    ids: List[int] = None, ys: np.ndarray = None,
                    list_succeed: List[bool] = None,
                    name_inputs: List[str] = None,
                    name_outputs: List[str] = None):
    '''
    Output database, including the inputs and the outputs.
    
    Parameters
    ---------------
    fname: str
        database file name
        
    xs: ndarray [n, dim_input]
        function inputs
    
    ids: None, or list [n]
        list of sample ID
    
    ys: None, or ndarray [n, dim_output]
        function outputs
        
    list_succeed: None, or list [bool]
        list of succeed for each input.
        If it is provided, only output samples that succeeded its evaluation.
    '''
    if ids is None:
        ids = [i+1 for i in range(xs.shape[0])]
        
    with open(fname, 'w') as f:
        
        f.write('Variables= ID')
        
        if ys is not None:
            for j in range(ys.shape[1]):
                f.write(' %14s'%(name_outputs[j]))
            
        for j in range(xs.shape[1]):
            f.write(' %14s'%(name_inputs[j]))
                
        f.write('\n')
        
        for i in range(xs.shape[0]):
            
            if list_succeed is not None:
                if not list_succeed[i]:
                    continue
                
            f.write('   %10d'%(ids[i]))
            
            if ys is not None:
                for j in range(ys.shape[1]):
                    f.write(' %14.6E'%(ys[i,j]))
                
            for j in range(xs.shape[1]):
                f.write(' %14.6E'%(xs[i,j]))
            f.write('\n')

def read_database(fname: str, have_output: bool = True,
                    n_input: int = None, n_output: int = None):
    '''
    Read database from file.
    
    Parameters
    ----------------
    fname: str
        database file name
        
    have_output: bool
        whether the file contains sample output.
    
    n_input: None, or int
        user specified input dimension.
        This is needed when load database from another problem.
    
    n_output: None, or int
        user specified output dimension.
        This is needed when load database from another problem.
        
    Returns
    ----------------
    ids: list [n]
        list of sample ID
    
    xs: ndarray [n, dim_input]
        function inputs
    
    ys: None, or ndarray [n, dim_output]
        function outputs
    '''
    with open(fname, 'r') as f:
        lines = f.readlines()
        
    ids = []
    xs = []
    ys = [] if have_output else None
    
    for line in lines:
        
        line = line.split()
        
        if line[0]=='Variables=' or line[0]=='#':
            continue
        
        ids.append(int(line[0]))
        
        if have_output:
            
            ys.append([float(line[i+1]) for i in range(n_output)])
        
            xs.append([float(line[i+1+n_output]) for i in range(n_input)])
            
        else:
            
            xs.append([float(line[i+1]) for i in range(n_input)])
    
    if have_output:
        ys = np.array(ys)
    
    return ids, np.array(xs), ys

def remove_duplicate_samples(xs: np.ndarray, ys: np.ndarray = None, ids: List[int] = None, 
                                indexes_parameter: List[int] | None = None) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    '''
    Remove duplicate samples from the dataset.
    
    Parameters
    ----------------
    xs: ndarray [n, dim_input]
        function inputs
    
    ys: ndarray [n, dim_output] | None
        function outputs
    
    ids: list [n] | None
        list of sample ID
    
    indexes_parameter: list [n_parameter] or None
        list of indexes of the parameters that are used to determine the uniqueness of the samples.
        If `indexes_parameter` is None, all parameters are used.
    
    Returns
    ----------------
    xs_new: ndarray [n_new, dim_input]
        function inputs
        
    ys_new: ndarray [n_new, dim_output] | None
        function outputs
        
    ids_new: list [n_new]
        list of sample ID
    
    indexes: list [n]
        list of indexes of the unique samples in the original dataset.
    '''
    xs_for_check = xs if indexes_parameter is None else xs[:,indexes_parameter]
    
    _, indexes = np.unique(xs_for_check, axis=0, return_index=True)
                    
    xs_new = xs[indexes,:]
    
    if ys is None:
        ys_new = None
    else:
        ys_new = ys[indexes,:]
    
    if ids is None:
        ids_new = [i+1 for i in range(xs_new.shape[0])]
    else:
        ids_new = [ids[i] for i in indexes]
    
    return xs_new, ys_new, ids_new, indexes

def remove_duplicate_samples_in_old_database(xs: np.ndarray, xs_old: np.ndarray, 
                ys: np.ndarray = None, ids: List[int] = None, 
                indexes_parameter: List[int] | None = None) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    '''
    Remove duplicate samples in the new dataset that are already in the old dataset.
    
    Parameters
    ----------------
    xs: ndarray [n, dim_input]
        function inputs
    
    xs_old: ndarray [n_old, dim_input]
        function inputs in the old dataset
    
    ys: ndarray [n, dim_output] | None
        function outputs
    
    ids: list [n] | None
        list of sample ID
    
    indexes_parameter: list [n_parameter] or None
        list of indexes of the parameters that are used to determine the uniqueness of the samples.
        If `indexes_parameter` is None, all parameters are used.
    
    Returns
    ----------------
    xs_new: ndarray [n_new, dim_input]
        function inputs
        
    ys_new: ndarray [n_new, dim_output] | None
        function outputs
        
    ids_new: list [n_new]
        list of sample ID
    
    indexes: list [n]
        list of indexes of the unique samples in the original dataset.
    '''
    indexes = []
    
    xs_for_check = xs if indexes_parameter is None else xs[:,indexes_parameter]
    xs_old_for_check = xs_old if indexes_parameter is None else xs_old[:,indexes_parameter]
    
    for i in range(xs.shape[0]):
        
        is_duplicate = False
        
        for j in range(xs_old.shape[0]):
            if np.all(xs_old_for_check[j,:] == xs_for_check[i,:]):
                is_duplicate = True
                break
            
        if not is_duplicate:
            indexes.append(i)
            
    xs_new = xs[indexes,:]
    
    if ys is None:
        ys_new = None
    else:
        ys_new = ys[indexes,:]
        
    if ids is None:
        ids_new = [i+1 for i in range(xs_new.shape[0])]
    else:
        ids_new = [ids[i] for i in indexes]
        
    return xs_new, ys_new, ids_new, indexes      


    
