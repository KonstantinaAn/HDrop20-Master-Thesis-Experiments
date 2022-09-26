"""
This class implements DeterministicUSM
(1/3) approximation
"""
import logging
from copy import deepcopy


class CSG:
    """
    CSG algorithm implementation
    """

    def __init__(self, submodular_func,cost_func, set_of_workers, k=None,_logger=None):
        """
        Constructor
        :param submodular_func:
        :param cost_func:
        :param set_of_workers:
        :param k:
        """
        if _logger is not None:
            self.logger = _logger
        else:
            self.logger = logging.getLogger(__name__)
        self.submodular_func = submodular_func
        self.cost_func = cost_func
        self.workers = set_of_workers

        if k is None:
            self.k = len(self.workers)
        else:
            self.k = k

    def find_greedy_element(self, workers, temporary_solution):
        """
        Finds the greedy element e to add to the current solution sol
        :param workers: Set of workers
        :param temporary_solution: set. Subset of workers
        :return e:
        """
        # self.logger.debug("Temp solution: {}, workers: {}".format(temporary_solution, workers))

        greedy_element = max(workers, key=lambda x: self.calc_marginal_gain(temporary_solution, x))
        self.logger.debug("Greedy element: {}".format(greedy_element))

        return greedy_element

    def calc_marginal_gain(self, temporary_solution, e):
        """
        Calculates the marginal gain for adding element e to the current solution sol
        :param temporary_solution:
        :param e:
        :return marginal_gain:
        """

        prev_val = self.submodular_func(temporary_solution) - 2*self.cost_func(temporary_solution)
        new_temporary_solution = temporary_solution.union({e})
        new_val = self.submodular_func(new_temporary_solution)-2* self.cost_func(new_temporary_solution)

        marginal_gain = new_val - prev_val

        return marginal_gain,prev_val

    def run(self):
        """
        Execute algorithm
        :param:none.
        :return best_sol:
        """
        curr_solution = set()
        temporary_workers = deepcopy(self.workers)
        for i in range(self.k):
            self.logger.info("Item CSG: {},k: {}".format(i, self.k))
            self.logger.debug("Finding Greedy Element")
            # find greedy element
            e_i = self.find_greedy_element(temporary_workers, curr_solution)
            self.logger.debug("Calculating Modified Objective Function value")
            # calculate modified objective value
            Q_i,val = self.calc_marginal_gain(curr_solution, e_i)
            self.logger.debug("Modified Objective Function value is {}  and current val is {}".format(Q_i,val))
            # if the best element can't increase the modified objective break loop
            if Q_i <= 0:
                self.logger.debug("Breaking loop")
                break
            # self.logger.debug("Curr solution before: {}".format(curr_solution))
            curr_solution = curr_solution.union({e_i})
            # self.logger.debug("Curr solution after: {}".format(curr_solution))
            # self.logger.debug("Temp workers before: {}".format(temporary_workers))
            temporary_workers = temporary_workers.difference({e_i})
            # self.logger.debug("Temp workers after: {}".format(temporary_workers))


        solution = curr_solution
        value = self.submodular_func(curr_solution)-self.cost_func(curr_solution)
        self.logger.info("Best solution: {}\nBest value: {}\nNumber of elements in solution: {}\n".format(solution, value,
                                                                                             len(solution)))

        return solution
