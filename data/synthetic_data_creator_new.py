from data.utils import config_filepath
import json
from data.synthetic_data_popularity import SyntheticDataPop
import math
from datetime import datetime
from pathlib import Path

with open(config_filepath) as f:
    config = json.load(f)

import logging

time_now = datetime.now()
time_date = '{}_{}_{}_{}'.format(time_now.hour, time_now.minute, time_now.second, time_now.microsecond)
logger = logging.getLogger(__name__)
filename = r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\config\logs\log{}.log'.format(time_date)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=filename,
                    filemode='w')


def create_synth_data(n, m, filepath, num_task_distribution, tasks_filename, workers_filename, valuation_upper=20,
                      seed=2021):
    """TODO: add cocumentation in this module"""
    synth_data = SyntheticDataPop(n, m, seed=seed)
    valuations_range = (1, valuation_upper)
    number_of_tasks_range = (1, math.ceil(n / 3))

    # #for the first 4 synthetic dataset
    # number_of_tasks_range = (1, math.ceil(n / 2))
    #
    # print('number of tasks range', number_of_tasks_range)


    costs_range = None
    synth_data.worker_tasks_distribution = "sliding_window"
    if num_task_distribution == "Discrete_Normal":
        synth_data.worker_costs_distribution = "default"
        synth_data.worker_number_tasks_distribution = "Discrete_Normal"
    elif num_task_distribution == "Augmented_Discrete_Normal":
        synth_data.worker_costs_distribution = "default"
        synth_data.worker_number_tasks_distribution = "Augmented_Discrete_Normal"
    elif num_task_distribution == "Uniform":
        costs_range = (1, 25)
    synth_data.create_synthetic_data(valuations_range, number_of_tasks_range, costs_range=costs_range)
    tasks_df = synth_data.tasks_to_df()
    worker_df = synth_data.workers_task_cost_to_df()
    logger.info("\nCreating file: {} with n:{} and m: {} ".format(str(Path(filepath)) + tasks_filename + ".csv", n, m))
    logger.info("\n" + str(synth_data.tasks_to_df()))
    synth_data.dataframe_to_csv(tasks_df, tasks_filename, path=str(Path(filepath)))
    logger.info("\nCreating file: {} with n:{} and m: {} ".format(str(Path(filepath)) + workers_filename + ".csv", n, m))
    logger.info("\n" + str(synth_data.workers_task_cost_to_df()))
    synth_data.dataframe_to_csv(worker_df, workers_filename, path=str(Path(filepath)))
    logger.info("\nPopular tasks: " + str(synth_data.popular_tasks))
    logger.info("\nNon-Popular tasks: " + str(synth_data.non_popular_tasks))
    logger.info("\nTasks: " + str(synth_data.set_of_tasks))
    logger.info("\nValuations: " + str(synth_data.valuations))
    logger.info("\nCosts: " + str(synth_data.worker_costs))
    logger.info("\nWorker tasks: " + str(synth_data.worker_tasks))
    logger.info("\nWorker num of tasks: " + str(synth_data.worker_number_of_tasks))
    return True


def create_instances(n, m, k_instances, dataset_number):

    for i in range(k_instances):

        if i < 9:
            #num_task_distribution = "Uniform"
            num_task_distribution = "Discrete_Normal"
            filepath = Path(
                r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\test\%s_dataset_%d_workers_%d_tasks\instance_0%d'
                % (dataset_number, m, n, i + 1))
            filepath.parent.mkdir(parents=True, exist_ok=True)
        elif i == 9:
            # num_task_distribution = "Uniform"
            num_task_distribution = "Discrete_Normal"
            filepath = Path(
                r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\test\%s_dataset_%d_workers_%d_tasks\instance_%d'
                % (dataset_number, m, n, i + 1))
            filepath.parent.mkdir(parents=True, exist_ok=True)
        elif 9 < i <= 19:
            num_task_distribution = "Augmented_Discrete_Normal"
            filepath = Path(
                r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\test\%s_dataset_%d_workers_%d_tasks\instance_%d'
                % (dataset_number, m, n, i + 1))
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            #num_task_distribution = "Discrete_Normal"
            num_task_distribution = "Uniform"
            filepath = Path(
                r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\test\%s_dataset_%d_workers_%d_tasks\instance_%d'
                % (dataset_number, m, n, i + 1))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            print(i, 'Uniform')

        # filepath = Path(r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\%s_dataset_%d_workers_%d_tasks\instance_%d'
        #                         % (dataset_number, m, n, i+1))
        # filepath.parent.mkdir(parents=True, exist_ok=True)

        print('For instance:',i+1)
        create_synth_data(n, m, filepath, num_task_distribution, "tasks", "workers", valuation_upper=20, seed=2000+i)

        # #TODO
        # break


if __name__ == '__main__':

    create_instances(n=10, m=5, k_instances=30, dataset_number='01')
    # create_instances(n=5, m=10, k_instances=30, dataset_number='02')
    # create_instances(n=10, m=10, k_instances=30, dataset_number='03')
    # create_instances(n=20, m=10, k_instances=30, dataset_number='04')
    # create_instances(n=10, m=20, k_instances=30, dataset_number='05')
    # create_instances(n=50, m=20, k_instances=30, dataset_number='06')
    # create_instances(n=100, m=20, k_instances=30, dataset_number='07')
    # create_instances(n=25, m=50, k_instances=30, dataset_number='08')
    # create_instances(n=50, m=50, k_instances=30, dataset_number='09')
    # create_instances(n=100, m=50, k_instances=30, dataset_number='10')

