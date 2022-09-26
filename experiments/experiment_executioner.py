from algorithms.distorted_greedy import DistortedGreedy
from algorithms.DeterministicUSM import DeterministicUSM
from algorithms.CSG import CSG
from algorithms.WInnerDetermination import WinnerDeetermination
from algorithms.ROI_greedy import ROIGreedy
from algorithms.optimization import Optimization

import math

from datetime import datetime
import logging


class ExperimentExecutioner:

    def __init__(self, objective_function, workers, algorithm, exec_uid, logger, workers_df=None, tasks_df=None):
        """

        :param objective_function:
        :param workers:
        :param algorithm:
        :param exec_uid:
        :param logger:
        """

        self.uid = exec_uid + "_exec"
        self.obj_func = objective_function
        self.workers = workers
        self.algorithm = algorithm
        self.logger = logger
        self.solution = {}
        self.results = {}
        self.alg_obj = None
        self.workers_df = workers_df
        self.tasks_df = tasks_df
        self.opt = Optimization(workers_df, tasks_df)

    def execute_algorithm(self):
        """

        :return:
        """
        self.logger.debug("Prepare to run {} ".format(self.algorithm))
        if self.algorithm == 'dgreedy':
            self.alg_obj = DistortedGreedy(self.obj_func.submodular_function_eval_fast, self.obj_func.linear_function_eval,
                                           self.workers, k=math.ceil(0.1*len(self.workers)), _logger=self.logger)
            self.solution = self.alg_obj.run()
        elif self.algorithm == 'dUSM':
            self.alg_obj = DeterministicUSM(self.obj_func.objective_function_eval, self.workers, k=math.ceil(0.1*len(self.workers)), _logger=self.logger)
            self.solution = self.alg_obj.run()
        elif self.algorithm == 'CSG':
            self.alg_obj = CSG(self.obj_func.submodular_function_eval_fast, self.obj_func.linear_function_eval,
                               self.workers, k=math.ceil(0.1*len(self.workers)), _logger=self.logger)
            self.solution = self.alg_obj.run()
        elif self.algorithm == 'Winner':
            self.opt.set_k_constrained(math.ceil(0.1*len(self.workers)))
            print('k_constrained', self.opt.k_constrained)
            wp_lp_solver = self.opt.wd_lp_solver
            self.alg_obj = WinnerDeetermination(self.obj_func.submodular_function_eval_fast,
                                                self.obj_func.linear_function_eval,
                                                self.obj_func.costs, self.workers, wp_lp_solver, self.workers_df,
                                                self.tasks_df, k=math.ceil(0.1*len(self.workers)), _logger=self.logger)
            self.solution = self.alg_obj.run()
        elif self.algorithm == 'ROI':
            self.alg_obj = ROIGreedy(self.obj_func.submodular_function_eval_fast, self.obj_func.linear_function_eval,
                                     self.workers, k=math.ceil(0.1*len(self.workers)), _logger=self.logger)
            self.solution = self.alg_obj.run()
        self.logger.debug("Finished running of {} ".format(self.algorithm))
        obj_eval = self.obj_func.objective_function_eval(self.solution)
        lin_eval = self.obj_func.linear_function_eval(self.solution)
        sub_eval = self.obj_func.submodular_function_eval(self.solution)
        total_valuation_of_tasks = self.tasks_df["valuation"].sum()
        self.logger.info("Solution: {} ".format(self.solution))
        self.logger.info("Objective function: {}".format(obj_eval))
        self.logger.info("Linear function: {}".format(lin_eval))
        self.logger.info("Submodular function: {}".format(sub_eval))
        self.results.update({self.uid: {
            "solution": self.solution,
            "obj_val": obj_eval,
            "lin_val": lin_eval,
            "sub_val": sub_eval,
            "total_valuation_of_tasks": total_valuation_of_tasks
        }})
        return self.results, self.uid

    def get_optimal_solution(self, worker_df, task_df):
        """
        Get the optimal solution
        :param worker_df:
        :param task_df:
        :return:
        """
        optimal_uid = self.uid

        self.opt.set_k_constrained(math.ceil(0.1*len(self.workers)))
        print('k_constrained', self.opt.k_constrained)

        problem_obj, xvals, yvals, optimal_workers_df, optimal_tasks_df = self.opt.wd_ip_solver(task_df.shape[0],
                                                                                                worker_df, task_df)
        optimal_solution = set(optimal_workers_df.query("y_star_index >0.9")[
                                   "user_name"])  # if it is approxiametly 1 then it belongs to the soln
        opt_value = problem_obj.value
        optimal_results = {
            "optimal_solution": optimal_solution,
            "optimal_value": opt_value,
            "problem_object": problem_obj,
            "optimal_workers_df": optimal_workers_df,
            "optimal_tasks_df": optimal_tasks_df

        }

        return optimal_results, optimal_uid
