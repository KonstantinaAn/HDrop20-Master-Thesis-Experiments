"""
This class implements distorted greedy algorithm
(1 - 1/e) approximation in expectation
"""
import logging


class DistortedGreedy:
    """
    Distored Greedy algorithm implementation
    """

    def __init__(self, submodular_func, cost_func, set_of_workers, k=None,_logger=None):
        """
        Constructor
        :param submodular_func:
        :param cost_func:
        :param set_of_workers: -- a python set
        :param k:
        :return:
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

    def calc_marginal_gain(self, temporary_solution, e):
        """
        Calculates the marginal gain for adding element e to the current solution sol
        :param temporary_solution:
        :param e:
        :return marginal_gain:
        """

        prev_val = self.submodular_func(temporary_solution)
        new_temporary_solution = temporary_solution.union({e})
        new_val = self.submodular_func(new_temporary_solution)

        marginal_gain = new_val - prev_val

        return marginal_gain

    def distorted_greedy_criterion(self, temporary_solution, e, k, i, gamma=1):
        """
        Calculates the contribution of element e to greedy solution
        :param temporary_solution: set. Subset of workers
        :param e: string. Element to be added
        :param k: integer. k-value. default equals len(workers)
        :param i: integer. Used for reverse counting of the exponent
        :param gamma:float in (0,1). Parameter of week submodularity. default=1
        :return greedy_contrib:
        """
        # Weight scaling is distorted
        rho = (k - gamma) / k
        marginal_gain = self.calc_marginal_gain(temporary_solution, e)
        cost = self.cost_func({e})
        weighted_gain = rho ** (k - i - 1) * marginal_gain
        greedy_contrib = weighted_gain - cost

        return greedy_contrib

    def find_greedy_element(self, workers, temporary_solution, k, i):
        """
        Finds the greedy element e to add to the current solution sol
        :param workers: Set of workers
        :param temporary_solution: set. Subset of workers
        :param k: integer. k-value. default equals len(workers)
        :param i: integer. Used for reverse counting of the exponent
        :return e:
        """

        greedy_element = max(workers, key=lambda x: self.distorted_greedy_criterion(temporary_solution, x, k,
                                                                                    i))  # takes the maximum contribution element

        return greedy_element

    def run(self):
        """
        Execute algorithm
        :param:
        :return best_sol:
        """
        # Keep track of current solution for a given value of k
        curr_sol = set()
        # Keep track of the submodular value
        curr_val = 0
        self.temporary_solution = set()

        for i in range(0, self.k):
            self.logger.info("Item distorted:{} {}".format( i, self.k))

            # Greedy element decided wrt distorted objective
            greedy_element = self.find_greedy_element(self.workers, self.temporary_solution, self.k, i)
            # Element is added to the solution
            self.logger.info("Item:{}, {}".format( greedy_element,
                             self.distorted_greedy_criterion(self.temporary_solution, greedy_element, self.k, i)))

            if self.distorted_greedy_criterion(self.temporary_solution, greedy_element, self.k, i) > 0:
                curr_sol.add(greedy_element)
                self.temporary_solution = self.temporary_solution.union({greedy_element})
                submodular_gain = self.submodular_func(self.temporary_solution)
                curr_val += submodular_gain

        # Computing the original objective value for current solution
        curr_val = curr_val - self.cost_func(curr_sol)
        self.logger.info(
            "Best solution: {}\nBest value: {}\nNumber of elements in solution: {}\n".format(curr_sol, curr_val,
                                                                                             len(curr_sol)))

        return curr_sol
