'''
Utility functions.
'''

import datetime
import numpy as np
import platform
import os


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
    

