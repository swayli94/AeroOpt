'''
NSGA-II implementation.
'''

import random
from typing import List, Tuple

import numpy as np

from AeroOpt.core import Database, Problem, SettingsOptimization
from AeroOpt.core.individual import Individual
from AeroOpt.core.settings import SettingsNSGAII
from AeroOpt.optimization.base import OptBaseFramework, PostProcess, PreProcess


class NSGAII(OptBaseFramework):
    """
    NSGA-II optimization algorithm.
    """

    def __init__(self,
        problem: Problem,
        optimization_settings: SettingsOptimization,
        nsgaii_settings: SettingsNSGAII,
        pre_process: PreProcess = None,
        post_process: PostProcess = None,
        ):
        self.nsgaii_settings = nsgaii_settings
        super().__init__(problem, optimization_settings, pre_process, post_process)

    def initialize_population(self) -> None:
        '''
        Initialize population if no resumed database exists.
        '''
        self.iteration = 1

        if self.db_total.size <= 0:
            xs = self._generate_random_xs(self.population_size)
            self.db_candidate = Database(self.problem, database_type="population")
            for x in xs:
                indi = Individual(self.problem, x=np.array(x, dtype=float))
                indi.source = "random"
                indi.generation = self.iteration
                self.db_candidate.add_individual(
                    indi,
                    check_duplication=False,
                    check_bounds=True,
                    deepcopy=False,
                )

            if self.pre_process is not None:
                self.pre_process.apply()
            self.evaluate_db_candidate()
            if self.post_process is not None:
                self.post_process.apply()

        self.select_valid_elite_from_total()
        self.log(f"Initial population prepared: valid={self.db_valid.size}", level=1)

    def generate_candidate_individuals(self) -> None:
        '''
        Generate offspring from current valid population.
        '''
        if self.db_valid.size <= 0:
            raise RuntimeError("No valid individuals available for NSGA-II evolution.")

        mating_population = self.binary_tournament_selection(self.db_valid, self.population_size)
        self.db_candidate = Database(self.problem, database_type="population")

        mute_rate = self.nsgaii_settings.mut_rate / max(self.problem.n_input, 1)
        pow_poly = self.nsgaii_settings.pow_poly

        n_pairs = int(np.ceil(self.population_size / 2))
        for i in range(n_pairs):
            i1 = 2 * i
            i2 = min(2 * i + 1, self.population_size - 1)
            p1 = mating_population[i1]
            p2 = mating_population[i2]

            x1, x2 = self.sbx_crossover(
                p1.x,
                p2.x,
                self.problem,
                cross_rate=self.nsgaii_settings.cross_rate,
                pow_sbx=self.nsgaii_settings.pow_sbx,
            )

            x1 = self.polynomial_mutation(
                x1,
                self.problem,
                mut_rate=mute_rate,
                pow_poly=pow_poly,
            )
            x2 = self.polynomial_mutation(
                x2,
                self.problem,
                mut_rate=mute_rate,
                pow_poly=pow_poly,
            )

            children = [x1, x2]
            for k, x_child in enumerate(children):
                if self.db_candidate.size >= self.population_size:
                    break
                indi = Individual(self.problem, x=np.array(x_child, dtype=float))
                indi.source = "GA"
                indi.generation = self.iteration
                self.db_candidate.add_individual(
                    indi,
                    check_duplication=False,
                    check_bounds=True,
                    deepcopy=False,
                )

    def select_valid_elite_from_total(self) -> None:
        '''
        Select valid population and elite set from total database.
        '''
        valid = Database(self.problem, database_type="valid")
        valid.merge_with_database(self.db_total, deepcopy=True)
        valid.eliminate_invalid_individuals()

        if valid.size <= 0:
            self.db_valid = valid
            self.db_elite = Database(self.problem, database_type="elite")
            return

        fronts = self.fast_non_dominated_sort(valid)
        self._assign_crowding_distance(valid, fronts)

        selected_indices = self._select_population_indices(valid, fronts, self.population_size)
        self.db_valid = valid.get_sub_database(index_list=selected_indices, deepcopy=True)
        self.db_valid.sort_database(sort_type=0)

        if len(fronts) > 0:
            self.db_elite = valid.get_sub_database(index_list=fronts[0], deepcopy=True)
            self.db_elite.sort_database(sort_type=0)
        else:
            self.db_elite = Database(self.problem, database_type="elite")

    def binary_tournament_selection(self,
                pool: Database, n_select: int) -> List[Individual]:
        '''
        Binary tournament selection from a sorted population.
        '''
        if pool.size <= 0:
            raise ValueError("Selection pool is empty.")

        selected: List[Individual] = []
        for _ in range(n_select):
            i, j = random.sample(range(pool.size), 2) if pool.size > 1 else (0, 0)
            a = pool.individuals[i]
            b = pool.individuals[j]

            if a < b:
                selected.append(a)
            elif b < a:
                selected.append(b)
            else:
                selected.append(a if random.random() < 0.5 else b)
        return selected

    @staticmethod
    def fast_non_dominated_sort(db: Database) -> List[List[int]]:
        '''
        Return fronts as index lists, and update indi.pareto_rank (1-based).
        '''
        n = db.size
        if n == 0:
            return []

        dominating_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for p in range(n - 1):
            for q in range(p + 1, n):
                flag = db.individuals[p].check_dominance(db.individuals[q])
                if flag == 1:
                    dominated_set[p].append(q)
                    dominating_count[q] += 1
                elif flag == -1:
                    dominated_set[q].append(p)
                    dominating_count[p] += 1

        for i in range(n):
            if dominating_count[i] == 0:
                db.individuals[i].pareto_rank = 1
                fronts[0].append(i)

        i_front = 0
        while i_front < len(fronts) and len(fronts[i_front]) > 0:
            next_front: List[int] = []
            for p in fronts[i_front]:
                for q in dominated_set[p]:
                    dominating_count[q] -= 1
                    if dominating_count[q] == 0:
                        db.individuals[q].pareto_rank = i_front + 2
                        next_front.append(q)
            if len(next_front) > 0:
                fronts.append(next_front)
            i_front += 1

        return fronts

    def _assign_crowding_distance(self, db: Database,
                        fronts: List[List[int]]) -> None:
        '''
        Assign crowding distance on each front.
        '''
        for front in fronts:
            if len(front) == 0:
                continue

            for idx in front:
                db.individuals[idx].crowding_distance = 0.0

            if len(front) <= 2:
                for idx in front:
                    db.individuals[idx].crowding_distance = float("inf")
                continue

            n_obj = self.problem.n_objective
            obj_matrix = np.zeros((len(front), n_obj))
            for i_local, idx in enumerate(front):
                obj_matrix[i_local, :] = db.individuals[idx].objectives

            # Convert maximization objectives to minimization consistently.
            sign = []
            for out_type in self.problem.problem_settings.output_type:
                if abs(out_type) == 1:
                    sign.append(-1.0 if out_type == 1 else 1.0)
            sign = np.array(sign, dtype=float)
            obj_matrix = obj_matrix * sign

            for m in range(n_obj):
                order = np.argsort(obj_matrix[:, m])
                sorted_values = obj_matrix[order, m]
                f_min = sorted_values[0]
                f_max = sorted_values[-1]
                span = max(f_max - f_min, 1.0e-12)

                db.individuals[front[order[0]]].crowding_distance = float("inf")
                db.individuals[front[order[-1]]].crowding_distance = float("inf")

                for i in range(1, len(front) - 1):
                    idx_mid = front[order[i]]
                    if np.isinf(db.individuals[idx_mid].crowding_distance):
                        continue
                    db.individuals[idx_mid].crowding_distance += (
                        sorted_values[i + 1] - sorted_values[i - 1]
                    ) / span

    @staticmethod
    def _select_population_indices(
        db: Database, fronts: List[List[int]], population_size: int
        ) -> List[int]:
        '''
        Select top individuals by rank then crowding distance.
        '''
        selected: List[int] = []
        for front in fronts:
            if len(selected) + len(front) <= population_size:
                selected.extend(front)
                continue

            remaining = population_size - len(selected)
            if remaining > 0:
                sorted_front = sorted(
                    front,
                    key=lambda idx: db.individuals[idx].crowding_distance,
                    reverse=True,
                )
                selected.extend(sorted_front[:remaining])
            break

        return selected

    def _generate_random_xs(self, n_samples: int) -> np.ndarray:
        '''
        Generate random points uniformly in design space.
        '''
        xs = np.random.rand(n_samples, self.problem.n_input)
        xs = self.problem.scale_x(xs, reverse=True)
        return xs

    @staticmethod
    def sbx_crossover(x1: np.ndarray, x2: np.ndarray, problem: Problem,
        cross_rate: float = 1.0, pow_sbx: float = 20.0,
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Simulated Binary Crossover (SBX) Operation.

        Deb, Kalyanmoy, and Ram Bhushan Agrawal. 
        "Simulated binary crossover for continuous search space."
        Complex systems 9.2 (1995): 115-148.
        '''
        child1 = x1.copy()
        child2 = x2.copy()
        if random.random() > cross_rate:
            return child1, child2

        low = problem.data_settings.input_low
        upp = problem.data_settings.input_upp
        precision = problem.data_settings.input_precision

        for i in range(problem.n_input):
            if random.random() > 0.5:
                continue
            if abs(x1[i] - x2[i]) <= precision[i]:
                continue

            y1, y2 = (x1[i], x2[i]) if x1[i] < x2[i] else (x2[i], x1[i])
            rand = random.random()

            beta = 1.0 + (2.0 * (y1 - low[i]) / (y2 - y1))
            alpha = 2.0 - beta ** (-(pow_sbx + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (pow_sbx + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (pow_sbx + 1.0))
            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

            beta = 1.0 + (2.0 * (upp[i] - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(pow_sbx + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (pow_sbx + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (pow_sbx + 1.0))
            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            c1 = min(max(c1, low[i]), upp[i])
            c2 = min(max(c2, low[i]), upp[i])
            if random.random() <= 0.5:
                child1[i], child2[i] = c2, c1
            else:
                child1[i], child2[i] = c1, c2

        problem.apply_bounds_x(child1)
        problem.apply_bounds_x(child2)
        
        return child1, child2

    @staticmethod
    def polynomial_mutation(x: np.ndarray, problem: Problem,
        mut_rate: float = 1.0, pow_poly: float = 20.0,
        ) -> np.ndarray:
        '''
        Polynomial mutation.
        '''
        out = x.copy()
        low = problem.data_settings.input_low
        upp = problem.data_settings.input_upp

        for i in range(problem.n_input):
            if random.random() > mut_rate:
                continue

            span = upp[i] - low[i]
            if span <= 0.0:
                continue

            delta1 = (out[i] - low[i]) / span
            delta2 = (upp[i] - out[i]) / span
            rnd = random.random()
            mut_pow = 1.0 / (pow_poly + 1.0)

            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (pow_poly + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (pow_poly + 1.0))
                deltaq = 1.0 - val ** mut_pow

            out[i] += deltaq * span

        problem.apply_bounds_x(out)
        
        return out




    