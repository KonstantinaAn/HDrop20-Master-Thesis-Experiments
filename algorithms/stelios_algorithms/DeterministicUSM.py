"""
This class implements DeterministicUSM
(1/3) approximation
"""
import logging
from copy import deepcopy


class DeterministicUSM:
    """
    DeterministicUSM algorithm implementation
    """

    def __init__(self, submodular_func, set_of_workers, k=None,_logger=None):
        """
        Constructor
        :param submodular_func:
        :param set_of_workers -- a python set:
        :param k:
        :return:
        """
        if _logger is not None:
            self.logger = _logger
        else:
            self.logger = logging.getLogger(__name__)
        self.submodular_func = submodular_func
        self.workers = set_of_workers
        self.set_X = set()
        self.set_Y = deepcopy(self.workers)

        if k is None:
            self.k = len(self.workers)
            self.elements = deepcopy(self.workers)
        else:
            self.k = k
            # TODO add sampling from workers

    def calc_marginal_gain(self, set_XorY, e, state):
        """
        Calculates the marginal gain for adding element e to the current solution sol
        :param set_X
        :param e:
        :return marginal_gain:
        """

        prev_val = self.submodular_func(set_XorY)
        # self.logger.debug("State: {},SetXorY: {},element:{},Previous val:{}".format(state, set_XorY, e, prev_val))
        if state == 'add':
            new_set_XorY = set_XorY.union({e})
        elif state == 'remove':
            new_set_XorY = set_XorY.difference({e})
        else:
            raise ValueError
        new_val = self.submodular_func(new_set_XorY)

        marginal_gain = new_val - prev_val
        self.logger.debug("New val:{}, marginal gain: {}".format(new_val, marginal_gain))

        return marginal_gain

    def run(self):
        """
        Execute algorithm
        :param:none.
        :return best_sol:
        """

        for element in self.elements:
            self.logger.info("Item DeterministicUSM: {}, {}".format(element, self.k))

            self.logger.debug("Calculating marginal gain for a_i ")
            a_i = self.calc_marginal_gain(self.set_X, element, 'add')
            self.logger.debug("Calculating marginal gain for b_i ")

            b_i = self.calc_marginal_gain(self.set_Y, element, 'remove')
            self.logger.debug("a_i: {}, b_i: {}, element: {}, ".format(a_i,b_i,element))
            # self.logger.debug("set_X before: {}, set_Y before: {}".format(self.set_X, self.set_Y))
            if a_i >= b_i:
                self.set_X = self.set_X.union({element})
            else:
                self.set_Y = self.set_Y.difference({element})
            # self.logger.debug("set_X after: {}, set_Y after: {}".format(self.set_X, self.set_Y))


        solution = self.set_X
        value = self.submodular_func(self.set_X)
        self.logger.info("Best solution: {}\nBest value: {}\nNumber of elements in solution: {}\n".format(solution, value,
                                                                                             len(solution)))

        return solution
