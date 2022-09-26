from data.utils import config_filepath, get_config

import os
import glob


# parsers for synthetic project only
def main_parser(dataset, instance, suffix=''):
    config = get_config()
    synthetic_project = config['project']['synthetic']
    tasks_filename = synthetic_project['tasks_filename'] + suffix
    workers_filename = synthetic_project['workers_filename'] + suffix
    datasets = synthetic_project['datasets_paths']
    instance_tasks_paths = []
    instance_workers_paths = []
    path = datasets[dataset]['instances'][instance]
    instance_tasks_paths.append(os.path.join(path, tasks_filename + ".csv"))
    instance_workers_paths.append(os.path.join(path, workers_filename + ".csv"))
    return instance_tasks_paths, instance_workers_paths


def file_path_parser_from_config(dataset_key, instance_key,_project='synthetic'):
    print('in file_path_parser_from_config')
    config = get_config()
    project = config['project'][_project]
    print('project', project)
    datasets = project['datasets_paths']
    print('datasets', datasets)
    print('dataset key', dataset_key)
    file_path = datasets[dataset_key]['instances'][instance_key]
    print('file path', file_path)
    return file_path


def pattern_parser(dataset, instance, pattern=''):
    config = get_config()
    synthetic_project = config['project']['synthetic']
    tasks_filename = synthetic_project['tasks_filename']
    workers_filename = synthetic_project['workers_filename']
    datasets = synthetic_project['datasets_paths']
    instance_tasks_paths = []
    instance_workers_paths = []
    path = datasets[dataset]['instances'][instance]
    instance_tasks_paths.append(glob.glob(path + tasks_filename + pattern + ".csv"))
    instance_workers_paths.append(glob.glob(path + workers_filename + pattern + ".csv"))
    return instance_tasks_paths, instance_workers_paths


def file_pattern_parser(file_path, file_name, pattern=''):
    files = glob.glob(file_path + file_name + pattern+".csv")
    print('In file_pattern_parser', files)
    if len(files)==1:
        return files[0]
    else:
        return files


if __name__ == "__main__":
    print(main_parser("01_dataset", "instance_01", suffix='_2'))
    print(pattern_parser("01_dataset", "instance_01", pattern='20*'))
