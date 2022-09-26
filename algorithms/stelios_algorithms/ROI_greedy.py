"""
#TODO add documentation
"""
import logging


class ROIGreedy:
    """
    ROI Greedy algorithm implementation
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

    def roi_greedy_criterion(self, temporary_solution, e):
        """
        Calculates the contribution of element e to greedy solution
        :param temporary_solution: set. Subset of workers
        :param e: string. Element to be added
        :return greedy_contrib:
        """
        # Weight scaling is distorted
        marginal_gain = self.calc_marginal_gain(temporary_solution, e)
        cost = self.cost_func({e})
        greedy_contrib = marginal_gain / cost

        return greedy_contrib

    def find_greedy_element(self, workers, temporary_solution):
        """
        Finds the greedy element e to add to the current solution sol
        :param workers: Set of workers
        :param temporary_solution: set. Subset of workers
        :param k: integer. k-value. default equals len(workers)
        :param i: integer. Used for reverse counting of the exponent
        :return greedy_element:
        """

        greedy_element = max(workers, key=lambda x: self.roi_greedy_criterion(temporary_solution,
                                                                              x))  # takes the maximum contribution element

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

        for i in range(0, self.k):
            self.logger.info("Item roi:{}, {}".format(i, self.k))

            greedy_element = self.find_greedy_element(self.workers, curr_sol)

            # Element is added to the solution
            self.logger.info("Item: {} {}".format(greedy_element, self.roi_greedy_criterion(curr_sol, greedy_element)))

            if self.calc_marginal_gain(curr_sol, greedy_element) - self.cost_func({greedy_element}) > 0:
                curr_sol.add(greedy_element)
                submodular_gain = self.submodular_func(curr_sol)
                curr_val += submodular_gain
            else:
                break

        # Computing the original objective value for current solution
        curr_val = curr_val - self.cost_func(curr_sol)
        self.logger.info(
            "Best solution: {}\nBest value: {}\nNumber of elements in solution: {}\n".format(curr_sol, curr_val,
                                                                                             len(curr_sol)))

        return curr_sol
