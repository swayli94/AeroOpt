'''
NRBO: Newton-Raphson-based Optimizer

NRBO is a population-based metaheuristic algorithm for single-objective continuous optimization problems. 
NRBO is inspired by the Newton-Raphson method used in numerical analysis for finding roots of real-valued functions. 
The algorithm combines the exploration capabilities of population-based methods with the exploitation power of gradient-based approaches.

Key Features:

- Population-based approach: Maintains a population of solutions to explore the search space
- Newton-Raphson Search Rule (NRSR): Utilizes a modified Newton-Raphson update rule for generating new solutions
- Trap Avoidance Operator (TAO): Includes a mechanism to escape local optima
- Dynamic adaptation: The algorithm adapts its search behavior based on the current iteration

References:

    R. Sowmya, M. Premkumar, and P. Jangir. Newton-raphson-based optimizer:
    a new population-based metaheuristic algorithm for continuous optimization problems.
    Engineering Applications of Artificial Intelligence, 128:107532, 2024. doi:10.1016/j.engappai.2023.107532.
    
    https://pymoo.org/algorithms/soo/nrbo.html#nb-nrbo
    
    https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/soo/nonconvex/nrbo.py

'''
