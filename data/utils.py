from itertools import chain, combinations
import json
import os

config_filepath = r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\config\config.json'


def get_config():
    with open(config_filepath) as f:
        config = json.load(f)
    return config


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    only for small instances i.e. max 10
    :param iterable:
    :return:
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_worker_power_set(worker_df):
    powset = powerset(worker_df['user_name'])
    return powset


def dataframe_to_csv( dataframe, filename, path, suffix='.csv'):
    filepath = os.path.join(path, filename)
    dataframe.to_csv(filepath + suffix, sep=',', header=True, index=False)
    return True