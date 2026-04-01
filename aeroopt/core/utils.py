'''
Utility functions.
'''

import datetime
import numpy as np
import platform
import os

from typing import List, Tuple, Optional


def check_folder(folder: str) -> str:
    '''
    Ensure a folder exists and return its path.
    '''
    if folder is None or len(str(folder).strip()) == 0:
        raise ValueError('Folder path must be a non-empty string.')
    os.makedirs(folder, exist_ok=True)
    return folder


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
    
def log(text: str, prefix='>>> ', fname: Optional[str] = 'logging.log', print_on_screen: bool = True) -> None:
    '''
    Log time and text. If ``fname`` is None, only print to screen when ``print_on_screen`` is True.
    '''
    if print_on_screen:
        print(prefix+text)
        
    if fname is None:
        return

    now_time = datetime.datetime.now()
    _time = now_time.strftime('%Y-%m-%d %H:%M:%S | ')
    
    with open(fname, 'a', encoding='utf-8') as f:
        f.write(_time+prefix+text+'\n')

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
