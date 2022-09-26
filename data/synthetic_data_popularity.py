import copy
import random
import json
import numpy as np
import pandas as pd
import os
import scipy.stats as ss
from data.samplers import NormDiscrete
from data.sliding_window import *

import math


class SyntheticDataPop:
    """
    This class creates synthetic data for the execution of the experiments
    #TODO: create documenation for this classs
    """

    valuations_distribution = None
    worker_number_tasks_distribution = None
    worker_tasks_distribution = None
    worker_costs_distribution = None

    @classmethod
    def read_config(cls, config_json_path):
        """
        Reads config file
        :param config_json_path:
        :return config:
        """
        with open(config_json_path) as f:
            config = json.load(f)

        return config

    def __init__(self, n_length_tasks, m_length_workers, popular_ratio=0.2, common_ratio=0.6, json_path='', seed=2021):
        self.n = n_length_tasks
        self.m = m_length_workers
        self.set_of_tasks = sorted(self.set_tasks())  # create the set of tasks in the form of ti
        self.workers = sorted(self.set_workers())  # create the set of workers in the form of wi
        self.valuations = {}  # task valuations
        self.worker_number_of_tasks = {}
        self.worker_tasks = {}
        self.worker_costs = {}
        if json_path:
            self.config = self.read_config(json_path)
        else:
            self.config = ''  # TODO self.read_config('default_path')
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.popular_ratio = popular_ratio
        self.common_ratio = common_ratio
        self.popular_tasks, self.common_tasks, self.non_popular_tasks = self.generate_popular_tasks(self.popular_ratio,
                                                                                                    self.common_ratio)
        self.sliding_window_popular = SlidingWindow(self.popular_tasks, seed=self.seed)
        self.sliding_window_common = SlidingWindowOverlap(self.common_tasks, seed=self.seed)
        #self.sliding_window_common = SlidingWindow(self.common_tasks, seed=self.seed)
        self.sliding_window_non_popular = SlidingWindowOverlap(self.non_popular_tasks, seed=self.seed)

    def generate_popular_tasks(self, popular_ratio, common_ratio):
        popular_tasks = set()
        common_tasks = set()
        non_popular_tasks = set()
        for index, task in enumerate(self.set_of_tasks):
            if index < math.ceil(popular_ratio * self.n):
                popular_tasks.add(task)
            elif math.ceil(popular_ratio * self.n) <= index < math.ceil((popular_ratio + common_ratio) * self.n):
                common_tasks.add(task)
            else:
                non_popular_tasks.add(task)
        print('popular tasks', popular_tasks)
        print('common tasks', common_tasks)
        print('non popular tasks', non_popular_tasks)
        return popular_tasks, common_tasks, non_popular_tasks

    def set_tasks(self):
        number_of_leading_zeros = len(str(self.n))
        return {"t" + "{:0{}d}".format(s, number_of_leading_zeros) for s in range(self.n)}

    def set_workers(self):
        number_of_leading_zeros = len(str(self.m))
        return {"w" + "{:0{}d}".format(s, number_of_leading_zeros) for s in range(self.m)}

    def create_valuations(self, value_range, distribution=None):
        """
            Create a valuation for every task
        """
        lower_bound = value_range[0]
        upper_bound = value_range[1]
        for task in self.set_of_tasks:
            self.valuations.update(
                {task: random.uniform(lower_bound, upper_bound)})  # uniformly create valuation values

    def create_worker_number_of_tasks(self, range, ceil_ratio=3 / 2, distribution=None):
        """
            Create for every worker the number of tasks to be assigned. Default Uniformly
        """
        lower_bound = range[0]
        upper_bound = range[1]
        if distribution is None:
            if lower_bound < 1:
                lower_bound = 1
            if upper_bound > self.n:
                upper_bound = self.n
            for worker in self.workers:
                self.worker_number_of_tasks.update(
                    {worker: random.randrange(lower_bound,
                                              upper_bound)})  # uniformly create the number of tasks per worker
            print('The numbers of tasks', self.worker_number_of_tasks)
        elif distribution == 'Discrete_Normal':
            print('Distribution is Discrete Normal')
            norm_discrete = NormDiscrete(self.config, lower_bound, upper_bound, seed=self.seed)
            for worker in self.workers:
                self.worker_number_of_tasks.update(
                    {worker: norm_discrete.discrete_norm()[0]})
            print('The numbers of tasks', self.worker_number_of_tasks)
        elif distribution == 'Augmented_Discrete_Normal':
            print('Distribution is Augmented Discrete Normal')
            norm_discrete_95_percent = NormDiscrete(self.config, lower_bound, upper_bound, seed=self.seed)
            norm_discrete_5_percent = NormDiscrete(self.config, lower_bound, math.ceil(ceil_ratio * upper_bound),
                                                   seed=self.seed)
            for index, worker in enumerate(self.workers):
                if index <= math.ceil(ceil_ratio * upper_bound):
                    self.worker_number_of_tasks.update(
                        {worker: norm_discrete_95_percent.discrete_norm()[0]})
                else:
                    self.worker_number_of_tasks.update(
                        {worker: norm_discrete_5_percent.discrete_norm()[0]})
            print('The numbers of tasks', self.worker_number_of_tasks)

    def create_worker_tasks(self, popular_ratio=None, common_ratio=None, distribution=None):
        """
            Create for every worker the tasks to be assigned. Default Uniformly

        """
        if popular_ratio is None:
            popular_ratio = self.popular_ratio

        if common_ratio is None:
            common_ratio = self.common_ratio

        if distribution is None:
            for worker in self.workers:
                # uniformly get the tasks for every worker based on the number of tasks to be assigned for every worker
                self.worker_tasks.update(
                    {worker: random.sample(self.set_of_tasks, self.worker_number_of_tasks[worker])})
                print('worker', worker, 'has the tasks', self.worker_tasks)
        elif distribution == 'sliding_window':

            for worker in self.workers:
                num_tasks = int(self.worker_number_of_tasks[worker])
                print('worker', worker, 'has number of tasks', int(self.worker_number_of_tasks[worker]))
                num_popular_tasks = math.ceil(popular_ratio * num_tasks)  # 10 * 0.2 = 2
                print('num_popular_tasks', num_popular_tasks)
                num_common_tasks = math.ceil(common_ratio * num_tasks)  # 10 * 0.6 = 6
                if num_common_tasks + num_popular_tasks > num_tasks:
                    num_common_tasks = num_tasks - num_popular_tasks

                print('num_common_tasks', num_common_tasks)
                num_non_popular_tasks = num_tasks - (num_popular_tasks + num_common_tasks) # 10 - 8 + 2 = 2
                print('num_non_popular_tasks', num_non_popular_tasks)

                popular_tasks = self.sliding_window_popular.sliding_window(num_popular_tasks)
                print('popular_tasks', popular_tasks)

                common_tasks = []
                print('The window for common tasks:')
                if num_common_tasks > 0:
                    if num_common_tasks == 1:
                        common_tasks = self.sliding_window_common._sliding_window(num_common_tasks)
                    else:
                        common_tasks = self.sliding_window_common.sliding_window_with_overlaps(
                            num_common_tasks, overlaps=1)

                # common_tasks = self.sliding_window_common.sliding_window(num_common_tasks)
                print('common_tasks', common_tasks)

                non_popular_tasks = []
                print('The window for non popular tasks:')
                if num_non_popular_tasks > 0:
                    if num_non_popular_tasks == 1:
                        non_popular_tasks = self.sliding_window_non_popular._sliding_window(num_non_popular_tasks)
                    else:
                        non_popular_tasks = self.sliding_window_non_popular.sliding_window_with_overlaps(
                            num_non_popular_tasks, overlaps=1)

                print('non_popular_tasks', non_popular_tasks)

                tasks = popular_tasks + common_tasks + non_popular_tasks
                #tasks = popular_tasks + non_popular_tasks

                # FIXME fix popular non popular ratio
                self.worker_tasks.update({worker: tasks})
                print('worker', worker, 'has the tasks', self.worker_tasks)

    def create_worker_costs(self, value_range=None, distribution=None):
        """
            Create for every worker the cost of the tasks he wants to complete this must be such that the f(s)-c(s) is not always positive. Default Uniformly
        """

        if not self.worker_number_of_tasks:
            raise Exception("Number of workers are not defined")
        if distribution is None:
            lower_bound = value_range[0]
            upper_bound = value_range[1]
            for worker in self.workers:
                self.worker_costs.update(
                    {worker: random.uniform(lower_bound, upper_bound)})  # uniformly create costs for the tasks
        else:
            # randomnly break workers to 0-50,50-85,85-100 percent
            set_of_workers = copy.deepcopy(self.workers)
            set_of_workers = list(set_of_workers)
            first_part = set(np.random.choice(set_of_workers, size=int(0.5 * self.m), replace=False))
            for worker in first_part:
                set_of_workers.remove(worker)
                num_of_tasks = self.worker_number_of_tasks[worker]
                self.worker_costs.update(
                    {worker: random.uniform(num_of_tasks, 4 * num_of_tasks)})
            second_part = set(np.random.choice(set_of_workers, size=int(0.7 * len(set_of_workers)), replace=False))
            for worker in second_part:
                set_of_workers.remove(worker)
                num_of_tasks = self.worker_number_of_tasks[worker]
                self.worker_costs.update(
                    {worker: random.uniform(4 * num_of_tasks, 8 * num_of_tasks)})
            third_part = set_of_workers
            for worker in third_part:
                set_of_workers.remove(worker)
                num_of_tasks = self.worker_number_of_tasks[worker]
                self.worker_costs.update(
                    {worker: random.uniform(8 * num_of_tasks, 12 * num_of_tasks)})
                # print("worker: {} , num of tasks : {} , worker_cost: {} ".format(worker, num_of_tasks, self.worker_costs[worker]))

            # check everything worked fine
            for worker in self.workers:
                try:
                    self.worker_costs[worker]
                except KeyError as e:
                    print("Worker: {} was not set error occured : {}".format(worker, e))
                    num_of_tasks = self.worker_number_of_tasks[worker]
                    self.worker_costs.update(
                        {worker: random.uniform(8 * num_of_tasks, 12 * num_of_tasks)})

    def create_synthetic_data(self, valuations_range, number_of_tasks_range, costs_range=None):
        self.create_worker_number_of_tasks(number_of_tasks_range, distribution=self.worker_number_tasks_distribution)
        self.create_worker_tasks(popular_ratio=0.2, common_ratio=0.6, distribution=self.worker_tasks_distribution)
        self.create_worker_costs(value_range=costs_range, distribution=self.worker_costs_distribution)
        self.create_valuations(valuations_range, distribution=self.valuations_distribution)

    def tasks_to_df(self):
        task_list = list(self.set_of_tasks)
        task_list.sort()
        ids = [id for id in range(self.n)]
        valuations_list = [self.valuations[task] for task in task_list]
        prepare_data = np.array([task_list, ids, valuations_list]).transpose()
        return pd.DataFrame(prepare_data, columns=['skill', 'skill_id', 'valuation'])

    def workers_task_cost_to_df(self):
        workers_list = list(self.workers)
        workers_list.sort()
        workers_costs_list = [self.worker_costs[worker] for worker in self.worker_costs]
        workers_tasks_list = [self.worker_tasks[worker] for worker in self.worker_tasks]
        ids = [id for id in range(self.m)]
        prepare_data = np.array([workers_costs_list, workers_tasks_list, ids, workers_list], dtype=object).transpose()
        return pd.DataFrame(prepare_data, columns=['cost', 'skills', 'user_id', 'user_name'])

    def dataframe_to_csv(self, dataframe, filename, path='', suffix='.csv'):
        if path:
            filepath = path
        else:
            filename = self.config
        filepath = os.path.join(path, filename)
        dataframe.to_csv(filepath + suffix, sep=',', header=True, index=False)
        return True


if __name__ == "__main__":
    synth_data = SyntheticDataPop(20, 20, popular_ratio=0.20, json_path='', seed=2022)
    costs_range = (1, 25)
    valuations_range = (1, 20)
    number_of_tasks_range = (1, 6)

    synth_data.create_synthetic_data(valuations_range, number_of_tasks_range, costs_range=costs_range)
    for item in synth_data.__dict__:
        print(f'{item}:  {synth_data.__dict__[item]}')
    tasks_df = synth_data.tasks_to_df()
    worker_df = synth_data.workers_task_cost_to_df()
    print(synth_data.tasks_to_df())
    print(synth_data.workers_task_cost_to_df())
    synth_data.dataframe_to_csv(tasks_df, 'tasks',
                                path=r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\synthetic_data_konna\\')
    synth_data.dataframe_to_csv(worker_df, 'workers',
                                path=r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\synthetic_data_konna\\')

    synth_data_2 = SyntheticDataPop(20, 20, popular_ratio=0.20, json_path='', seed=2021)
    synth_data_2.worker_number_tasks_distribution = "Discrete_Normal"
    synth_data_2.worker_tasks_distribution = "sliding_window"
    synth_data_2.worker_costs_distribution = "default"

    costs_range_2 = None
    valuations_range_2 = (1, 20)
    number_of_tasks_range_2 = (1, math.ceil(20 / 3))
    synth_data_2.create_synthetic_data(valuations_range_2, number_of_tasks_range_2, costs_range=costs_range_2)
    for item in synth_data_2.__dict__:
        print(f'{item}:  {synth_data_2.__dict__[item]}')
    tasks_df_2 = synth_data_2.tasks_to_df()
    worker_df_2 = synth_data_2.workers_task_cost_to_df()
    print(synth_data_2.tasks_to_df())
    print(synth_data_2.workers_task_cost_to_df())
    synth_data_2.dataframe_to_csv(tasks_df_2, 'tasks_2',
                                path=r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\synthetic_data_konna\\')
    synth_data_2.dataframe_to_csv(worker_df_2, 'workers_2',
                                path=r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\synthetic_data_konna\\')
    synth_data_3 = SyntheticDataPop(20, 20, popular_ratio=0.20, json_path='', seed=2019)

    synth_data_3.worker_number_tasks_distribution = "Augmented_Discrete_Normal"
    synth_data_3.worker_tasks_distribution = "sliding_window"
    synth_data_3.worker_costs_distribution = "default"
    costs_range_3 = None
    valuations_range_3 = (1, 20)
    number_of_tasks_range_3 = (1, math.ceil(20 / 3))
    synth_data_3.create_synthetic_data(valuations_range_3, number_of_tasks_range_3, costs_range=costs_range_3)
    for item in synth_data_3.__dict__:
        print(f'{item}:  {synth_data_3.__dict__[item]}')
    tasks_df_3 = synth_data_3.tasks_to_df()
    worker_df_3 = synth_data_3.workers_task_cost_to_df()
    print(synth_data_3.tasks_to_df())
    print(synth_data_3.workers_task_cost_to_df())
    synth_data_3.dataframe_to_csv(tasks_df_3, 'tasks_3',
                                  path=r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\synthetic_data_konna\\')
    synth_data_3.dataframe_to_csv(worker_df_3, 'workers_3',
                                  path=r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\synthetic_data_konna\\')
    print(synth_data_3.popular_tasks)
    print(synth_data_3.non_popular_tasks)

    synth_data_4 = SyntheticDataPop(20, 20, popular_ratio=0.20, json_path='', seed=2020)
    costs_range = (1, 25)
    valuations_range = (1, 20)
    number_of_tasks_range = (1, 6)
    synth_data_4.create_synthetic_data(valuations_range, number_of_tasks_range, costs_range=costs_range)
    for item in synth_data_4.__dict__:
        print(f'{item}:  {synth_data_4.__dict__[item]}')
    tasks_df = synth_data_4.tasks_to_df()
    worker_df = synth_data_4.workers_task_cost_to_df()
    print(synth_data_4.tasks_to_df())
    print(synth_data_4.workers_task_cost_to_df())
    synth_data_4.dataframe_to_csv(tasks_df, 'tasks_4',
                                  path=r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\synthetic_data_konna\\')
    synth_data_4.dataframe_to_csv(worker_df, 'workers_4',
                                  path=r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\synthetic_data_konna\\')

