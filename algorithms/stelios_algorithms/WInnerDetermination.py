"""
This class implements DeterministicUSM
(1/3) approximation
"""
import logging
from copy import deepcopy


class WinnerDeetermination:
    """
    Winner Determination algorithm implementation
    """

    def __init__(self, submodular_func, cost_func, cost_dict, set_of_workers, WD_LP_solver, worker_df, task_df, k=None,_logger=None):
        """
        Constructor
        :param cost_dict: dict
        :param submodular_func:
        :param cost_func:
        :param set_of_workers:
        :param winner determination linear programming solver:
        :param k:
        """
        if _logger is not None:
            self.logger = _logger
        else:
            self.logger = logging.getLogger(__name__)
        self.submodular_func = submodular_func
        self.cost_dict = cost_dict
        self.cost_func = cost_func
        self.workers = set_of_workers
        self.wd_lp_solver = WD_LP_solver
        self.worker_df = worker_df
        self.task_df = task_df

        if k is None:
            self.k = len(self.workers)
        else:
            self.k = k

    def solve_WD_LP(self, worker_dataframe, task_dataframe):
        """
        :param worker_dataframe:
        :param task_dataframe:
        :return:
        """
        opt, solution_worker_dataframe, solution_task_dataframe = self.wd_lp_solver(
            task_dataframe.shape[0], worker_dataframe, task_dataframe)
        self.logger.debug("Opt: {}, Solution_worker_dataframe: \n{}, Solution_task_dataframe: \n{} ".format(opt,
                                                                                                        solution_worker_dataframe,
                                                                                                        solution_task_dataframe))
        sorted_workers = self.helper_get_sorted(solution_worker_dataframe)
        self.logger.debug("Sorted workers: \n{}".format(sorted_workers))

        return sorted_workers

    @staticmethod
    def helper_get_sorted(solution_dataframe):
        """

        :param solution_dataframe:
        :return:
        """
        df = deepcopy(solution_dataframe)
        df = df.sort_values('y_star_index', ascending=False)
        sorted_set = df['user_name']
        return sorted_set

    def calc_marginal_gain(self, temporary_solution, e):
        """
        Calculates the marginal gain for adding element e to the current solution sol
        :param temporary_solution:
        :param e:
        :return marginal_gain:
        """

        prev_val = self.submodular_func(temporary_solution)
        self.logger.debug("Previous value : {}".format(prev_val))
        new_temporary_solution = temporary_solution.union({e})
        # self.logger.debug("Temporary solution : {}".format(new_temporary_solution))
        new_val = self.submodular_func(new_temporary_solution)
        self.logger.debug("New value : {}".format(new_val))
        self.logger.debug("Cost of worker e : {}".format(self.cost_dict[e]))
        marginal_gain = new_val - prev_val - self.cost_dict[e]
        self.logger.debug("Marginal gain : {}".format(marginal_gain))
        return marginal_gain

    def run(self):
        """
        Execute algorithm
        :param:none.
        :return best_sol:
        """
        curr_solution = set()
        self.logger.info('Starting WD LP')
        y_star = self.solve_WD_LP(self.worker_df, self.task_df)  # return sorted set of workers
        self.logger.info('Finished WD LP')
        for selected_worker in y_star:
            self.logger.debug(
                "Item Winner Determination selected workers: {}, Current solution: {}".format(selected_worker,
                                                                                              curr_solution))
            marginal_gain = self.calc_marginal_gain(curr_solution, selected_worker)
            if marginal_gain > 0:
                self.logger.debug("Marginal gain (>0) : {}".format(marginal_gain))
                curr_solution = curr_solution.union({selected_worker})
                self.logger.debug(
                    " selected workers: {}, with marginal gain : {} added, New Current solution: {}".format(
                        selected_worker, marginal_gain, curr_solution))

        solution = curr_solution
        value = self.submodular_func(curr_solution) - self.cost_func(curr_solution)
        self.logger.info(
            "Best solution: {}\nBest value: {}\nNumber of elements in solution: {}\n".format(solution, value,
                                                                                                 len(solution)))

        return solution
