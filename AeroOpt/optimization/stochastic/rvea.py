'''
RVEA: Reference Vector Guided Evolutionary Algorithm

In RVEA, a scalarization approach, termed angle penalized distance (APD), is adopted to 
balance the convergence and diversity of the solutions in the high-dimensional objective space.
Furthermore, an adaptation strategy is proposed to dynamically adjust the 
reference vectors' distribution according to the objective functions' scales. 

Note that the APD is adapted based on the progress the algorithm has made.
Thus, termination criteria such as n_gen or n_evals should be used.

References:
    
    Ran Cheng, Yaochu Jin, Markus Olhofer, and Bernhard Sendhoff.
    A reference vector guided evolutionary algorithm for many-objective optimization.
    IEEE Transactions on Evolutionary Computation, 20(5):773-791, 2016. doi:10.1109/TEVC.2016.2519378.
    
    https://pymoo.org/algorithms/moo/rvea.html#nb-rvea
    
    https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/rvea.py
    
'''
