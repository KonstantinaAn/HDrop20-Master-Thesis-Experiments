from data.utils import config_filepath
from data.data_file_parser import file_path_parser_from_config, file_pattern_parser
import json


def get_config():
    with open(config_filepath) as f:
        config = json.load(f)
    return config


def get_data_file_paths(dataset_keys=None, instance_keys=None, file_suffix='',_project = 'synthetic' ):
    """
    Get a list of all the file paths give dataset_keys, instance keys and file suffix
    :param dataset_keys:
    :param instance_keys:
    :param file_suffix:
    :return: task_paths, worker_paths
    """
    config = get_config()
    project = config['project'][_project]
    print('project', project)
    if dataset_keys is None:
        print('Dataset keys none')
        dataset_keys = project['dataset_keys']
        print('dataset keys', dataset_keys)
    file_paths = []
    tasks_filename = project["tasks_filename"]
    print('tasks_filename', tasks_filename)
    workers_filename = project["workers_filename"]
    print('workers_filename', workers_filename)
    for dataset_key in dataset_keys:
        if instance_keys is None:
            instance_keys = project[dataset_key]
            print('None', instance_keys)
        for instance_key in instance_keys:
            file_path = file_path_parser_from_config(dataset_key, instance_key,_project=_project)
            print(file_path)
            print('tasks filename', tasks_filename)
            tasks_file = file_pattern_parser(file_path, tasks_filename, pattern=file_suffix)
            print('tasks_file', tasks_file)
            print('workers filename', workers_filename)
            workers_file = file_pattern_parser(file_path, workers_filename, pattern=file_suffix)
            print('workers_file', workers_file)
            file_paths.append((tasks_file, workers_file, (dataset_key,instance_key)))
        return  file_paths

def get_guru_files(dataset_keys=None, instance_keys=None, file_suffix=''):
    config = get_config()
    guru_project = config['project']['guru']
    tasks_file = guru_project["guru_skill_filepath"]
    workers_file = guru_project["guru_user_filepath"]
    return [tasks_file, workers_file]


def get_freelancer_files(dataset_keys=None, instance_keys=None, file_suffix=''):
    config = get_config()
    freelancer_project = config['project']['freelancer']
    tasks_file = freelancer_project["freelancer_skill_filepath"]
    workers_file = freelancer_project["freelancer_user_filepath"]
    return [tasks_file, workers_file]
