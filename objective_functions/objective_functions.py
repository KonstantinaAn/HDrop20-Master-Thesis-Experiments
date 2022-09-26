from copy import deepcopy
import pandas as pd
import numpy as np

class ObjectiveFunction:
    """
    #TODO: add documentation
    """

    def __init__(self, tasks, worker_tasks, valuations, costs):
        """

        :param tasks: set. Set of tasks
        :param worker_tasks: Dict. Dict[worker:list[tasks]]
        :param valuations: Dict. Dict[task:valuation]
        :param costs:Dict. Dict[worker:cost]
        """
        self.set_of_tasks = tasks
        self.worker_tasks = worker_tasks
        self.valuations = valuations
        self.costs = costs
        self.tasks_assigned = {}
        self.objective_last_evaluation = None
        self.cache = {}



    def submodular_function_eval_fast(self,set_S):
        covered_tasks = []
        [covered_tasks.extend(self.worker_tasks[worker]) for worker in set_S if worker]
        covered_tasks = list(dict.fromkeys(covered_tasks))
        val = sum([self.valuations[task] for task in covered_tasks])
        return val

    def submodular_function_eval(self, set_S, verbose=False):
        """
            Crates the evaluation of the submodular function given a set S (subset of workers)
            returns a dict of workers with their submodulars function evaluation
            f(s) = sum of valuations
        """
        sum = 0
        set_of_tasks = deepcopy(self.set_of_tasks)
        for worker in set_S:
            worker_tasks_dict = {}
            for task in self.worker_tasks[worker]:
                if task in set_of_tasks:
                    val = self.valuations[task]
                    sum += val
                    worker_tasks_dict.update({task: val})
                    set_of_tasks.remove(task)
            self.tasks_assigned.update({worker: worker_tasks_dict})
        if verbose:
            print(self.tasks_assigned)
        return sum


    def linear_function_eval(self, set_S,verbose=False):
        #TODO add verbose
        linear_sum = 0
        for worker in set_S:
            linear_sum += self.costs[worker]
        return linear_sum


    def objective_function_eval(self,set_S,verbose=False):
        """
            Crates the evaluation of the f(s)-c(s) given a set S (subset of workers)
            returns a dict of workers with their objective function evaluation
        """
        submodular_val = self.submodular_function_eval(set_S,verbose)
        submodular_val_fast = self.submodular_function_eval_fast(set_S)
        linear = self.linear_function_eval(set_S,verbose)
        self.objective_last_evaluation = submodular_val-linear
        return submodular_val - linear
