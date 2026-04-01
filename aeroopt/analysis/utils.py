'''
Utility functions for analysis.
'''

import numpy as np
from typing import overload
from sklearn.cluster import KMeans

'''
Crowding metrics:
- `d_typical`: Average minimum distance to adjacent points.
- `potentials`: Potential of each point.
'''

@overload
def func_potential(r: np.ndarray, c: float) -> np.ndarray:
    ...

@overload
def func_potential(r: float, c: float) -> float:
    ...

def func_potential(r: np.ndarray | float, c: float) -> np.ndarray | float:
    '''
    Calculate the potential function of force `f=c*r*exp(-c*r)`.
    
    The potential equals 1 when `r` is 0,
    and equals 0 when `r` is infinity.
    A larger `c` makes the potential decrease faster.
    
    Parameters:
    -----------
    r: np.ndarray|float
        distance
    c: float
        coefficient to scale the distance `r`.
    '''
    return (c*r+1)*np.exp(-c*r)

def calculate_potential_coefficient(
        critical_distance: float, critical_potential: float) -> float:
    '''
    Calculate the coefficient `c` of potential.

    Parameters:
    -----------
    critical_distance: float
        Critical distance, means potential drop to `critical_potential` at `critical_distance`.
    critical_potential: float
        Desired critical potential.
    
    Returns:
    --------
    c: float
        Coefficient `c` of potential.
    '''
    
    if critical_distance <= 0.0:
        raise ValueError("critical_distance must be positive.")

    # Keep previous behavior for invalid input, but make bounds explicit.
    if critical_potential <= 0.0 or critical_potential >= 1.0:
        critical_potential = 0.2

    def objective(c: float) -> float:
        return float(func_potential(critical_distance, c) - critical_potential)

    # objective(0) = 1 - critical_potential > 0
    lo, hi = 0.0, 1.0
    while objective(hi) > 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket potential coefficient.")

    # Binary search for robust and accurate solution.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if objective(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def idw_interpolation(d: np.ndarray, ys: np.ndarray) -> np.ndarray:
    '''
    Inverse distance weighted interpolation.
    
    Parameters:
    -----------
    d: np.ndarray [n]
        Distances to the reference points.
    ys: np.ndarray [n, ny]
        Output values of the reference points.
    
    Returns:
    --------
    y: np.ndarray [ny]
        Interpolated output values.
    '''
    zero_mask = d == 0.0
    if np.any(zero_mask):
        return np.mean(ys[zero_mask], axis=0)

    weights = 1.0 / d
    weights /= np.sum(weights)
    return np.sum(ys * weights[:, None], axis=0)


def clustering_kmeans(vs: np.ndarray, n_clusters: int) -> np.ndarray:
    '''
    Cluster the scaled variables using k-means algorithm.
    
    Parameters:
    -----------
    vs: np.ndarray [n, n_variable]
        Scaled variables.
    n_clusters: int
        Number of clusters.
    
    Returns:
    --------
    cluster_labels: np.ndarray [n]
        Labels of the clusters.
    '''
    # Standardize variables
    vs_std = (vs - np.mean(vs, axis=0)) / (np.std(vs, axis=0) + 1e-8)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vs_std)
    
    return np.array(kmeans.labels_)
