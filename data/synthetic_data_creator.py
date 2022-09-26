from data.utils import config_filepath
import json
from data.synthetic_data_popularity import SyntheticDataPop
from data.synthetic_data import SyntheticData
import math
from datetime import datetime

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
    synth_data = SyntheticData(n, m, seed=seed)
    valuations_range = (1, valuation_upper)
    number_of_tasks_range = (1, math.ceil(n / 3))
    costs_range = None
    synth_data.worker_tasks_distribution = "sliding_window"
    if num_task_distribution == "Discrete_Normal":
        synth_data.worker_costs_distribution = "default"
        synth_data.worker_number_tasks_distribution = "Discrete_Normal"
    elif num_task_distribution == "Augmented_Discrete_Normal":
        synth_data.worker_costs_distribution = "default"
        synth_data.worker_number_tasks_distribution = "Augmented_Discrete_Normal"
    else:
        costs_range = (1, 25)
    synth_data.create_synthetic_data(valuations_range, number_of_tasks_range, costs_range=costs_range)
    tasks_df = synth_data.tasks_to_df()
    worker_df = synth_data.workers_task_cost_to_df()
    logger.info("\nCreating file: {} with n:{} and m: {} ".format(filepath + tasks_filename + ".csv", n, m))
    logger.info("\n" + str(synth_data.tasks_to_df()))
    synth_data.dataframe_to_csv(tasks_df, tasks_filename, path=filepath)
    logger.info("\nCreating file: {} with n:{} and m: {} ".format(filepath + workers_filename + ".csv", n, m))
    logger.info("\n" + str(synth_data.workers_task_cost_to_df()))
    synth_data.dataframe_to_csv(worker_df, workers_filename, path=filepath)
    logger.info("\nPopular tasks: " + str(synth_data.popular_tasks))
    logger.info("\nNon-Popular tasks: " + str(synth_data.non_popular_tasks))
    logger.info("\nTasks: " + str(synth_data.set_of_tasks))
    logger.info("\nValuations: " + str(synth_data.valuations))
    logger.info("\nCosts: " + str(synth_data.worker_costs))
    logger.info("\nWorker tasks: " + str(synth_data.worker_tasks))
    logger.info("\nWorker num of tasks: " + str(synth_data.worker_number_of_tasks))
    return True


def main(suffix='', set_upper=False):
    synthetic_project = config['project']['synthetic']
    tasks_filename = synthetic_project['tasks_filename'] + suffix
    workers_filename = synthetic_project['workers_filename'] + suffix
    datasets = synthetic_project['datasets_paths']
    datasets_keys = datasets.keys()

    seed = 2021
    for key in datasets_keys:
        n = datasets[key]['n_m'][0]
        m = datasets[key]['n_m'][1]
        if set_upper:
            valuation_upper = n
        else:
            valuation_upper = 20
        for index, instance in enumerate(datasets[key]['instances']):
            instance_path = datasets[key]['instances'][instance]
            logger.info("\n" + "-" * 200)
            logger.info(instance)
            logger.info("seed: {}".format(seed))
            if index <= 10:
                logger.info(
                    "\nAccessing dataset: {} with settings: n = {}, m = {}, num_tasks_distribution = {}, cost_distribution = Default cost , valuations = Default Valuations".format(
                        instance_path, n, m, "Discrete_Normal"))
                create_synth_data(n, m, instance_path, "Discrete_Normal", tasks_filename, workers_filename,
                                  valuation_upper=valuation_upper, seed=seed)
            elif index <= 20:
                logger.info(
                    "\nAccessing dataset: {} with settings: n = {}, m = {}, num_tasks_distribution = {}, cost_distribution = Default cost , valuations = Default Valuations".format(
                        instance_path, n, m, "Augmented_Discrete_Normal"))
                create_synth_data(n, m, instance_path, "Augmented_Discrete_Normal", tasks_filename, workers_filename,
                                  valuation_upper=valuation_upper, seed=seed)
            else:
                logger.info(
                    "\nAccessing dataset: {} with settings: n = {}, m = {}, num_tasks_distribution = {}, cost_distribution = Default cost , valuations = Default Valuations".format(
                        instance_path, n, m, "Uniform"))
                create_synth_data(n, m, instance_path, "Uniform", tasks_filename, workers_filename, seed=seed)
            seed += 1


if __name__ == '__main__':
    main(suffix='', set_upper=False)
    create_synth_data(10,10,r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\test_dir','Discrete_Normal', "tasks_discrete","workers_discrete")
    create_synth_data(10,10,r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\test_dir','Augmentened_Discrete_Normal', "tasks_discrete_aug","workers_discrete_aug")
    create_synth_data(10,10,r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\test_dir','Uniform', "tasks_uniform","workers_uniform")
