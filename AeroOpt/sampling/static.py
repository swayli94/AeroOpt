'''
Static sampling methods.
'''

import numpy as np
import pyDOE
from typing import List, Set

from AeroOpt.core.problem import Problem
from AeroOpt.core.database import Database
from AeroOpt.core.individual import Individual
from AeroOpt.analysis.analyze_database import AnalyzeDatabase


def sample_x_LHS(n: int, problem: Problem) -> np.ndarray:
    '''
    Latin Hypercube Sampling for the design variables.
    
    Parameters:
    -----------
    n: int
        Number of samples.
    problem: Problem
        Problem.
        
    Returns:
    --------
    x_LHS: np.ndarray
        Latin Hypercube Sampling points.
    '''
    x_LHS = pyDOE.lhs(problem.n_var, samples=n, criterion='m')
    x_LHS = problem.scale_x(x_LHS, reverse=True)
    return x_LHS

def sample_individual_from_database(
        database: Database, n: int) -> List[Individual]:
    '''
    Sample individuals from the database.
    
    Parameters:
    -----------
    database: Database
        Database.
    n: int
        Number of samples.
        
    Returns:
    --------
    individuals: List[Individual]
        List of sampled individuals.
    '''
    if n > database.size:
        raise ValueError('Number of samples > size of the database.')    
    index_list = np.random.choice(database.size, size=n, replace=False)
    individuals = [database.individuals[i] for i in index_list]
    return individuals

def sample_individual_from_groups(
        analyze_database: AnalyzeDatabase, n: int) -> List[Individual]:
    '''
    Sample individuals from the groups of the database,
    try to evenly sample from each group.
    
    Parameters:
    -----------
    analyze_database: AnalyzeDatabase
        AnalyzeDatabase.
    n: int
        Number of samples.
    
    Returns:
    --------
    individuals: List[Individual]
        List of sampled individuals.
    '''
    if n > analyze_database.size:
        raise ValueError('Number of samples > size of the database.')
    
    if analyze_database.n_group <= 0:
        raise ValueError('Grouping has not been calculated.')
    
    analyze_database.database.update_id_list()
    
    ID_set : Set[int] = set()
    individuals : List[Individual] = []
    j_group = 0
    
    while len(ID_set) < n:
        
        candidate_ID_set = set(analyze_database.ID_in_group[j_group])
        intersect_set = candidate_ID_set & ID_set
        candidate_ID_set = candidate_ID_set - intersect_set
        
        if len(candidate_ID_set) > 0:
            k = np.random.choice(list(candidate_ID_set), size=1)
            ID_set.add(k)
            individuals.append(analyze_database.database.individuals[k])
            
        j_group += 1
        if j_group >= analyze_database.n_group:
            j_group = 0
    
    return individuals
