import numpy as np
import statistics as st

import pandas as pd

from data import data_provider
from data.utils import config_filepath, dataframe_to_csv
from scipy import stats
from algorithms.optimization import Optimization

reader = data_provider.DataProvider(config_filepath)
worker_df = reader.read_data_csv2df(file_path=
                                    r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\related_work\data_x\guru\guru_user_df.csv")
task_df = reader.read_data_csv2df(file_path=
                                  r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\related_work\data_x\guru\guru_skill_df.csv")
task_list = task_df['skill'].to_list()


def worker_df_clean(worker_df, task_list):
    # lists are stored as strings so apply(eval) will fix this
    column = worker_df['skills'].apply(eval).to_list()
    new_skills = []
    for row in column:
        new_row = []
        for skill in row:
            if skill not in task_list:
                continue
            else:
                new_row.append(skill)
        new_skills.append(new_row)
    worker_df['skills'] = new_skills
    # remove empty rows
    mask = worker_df.skills.apply(lambda x: x == [])
    removed = worker_df.loc[mask]
    worker_df = worker_df.loc[~mask]
    # reset indeces
    worker_df = worker_df.reset_index(drop=True)
    return worker_df, removed


cleaned_df, removed = worker_df_clean(worker_df, task_list)


def remove_outlier_from_df(worker_df, col_name):
    mask = (np.abs(stats.zscore(worker_df[col_name])) < 3)
    removed = worker_df.loc[~mask]
    worker_df = worker_df.loc[mask]
    worker_df = worker_df.reset_index(drop=True)
    return worker_df, removed


clean_outlier_df, removed_outliers = remove_outlier_from_df(cleaned_df, 'cost')


def sample_workers(worker_df, seed=1, fraction=1.0):
    """
    Samples users instead of using all users
    :param worker_df:
    :param seed:
    :param fraction:
    :return:
    """
    if seed != 1:
        np.random.seed(seed)
    if fraction < 1.0:
        n = len(worker_df)
        num_sampled_workers = int(fraction * n)
        sampled_workers = np.random.choice(worker_df['user_id'], size=num_sampled_workers, replace=False)
        mask = worker_df['user_id'].isin(sampled_workers)
        worker_df = worker_df.loc[mask]
        worker_df = worker_df.reset_index(drop=True)
        return worker_df
    else:
        return worker_df


new_df = sample_workers(clean_outlier_df, seed=200002, fraction=0.2)
print(new_df)


def get_statistics_worker_cost(worker_df):
    mean_cost = st.mean(worker_df['cost'].to_list())
    median_cost = st.median(worker_df['cost'].to_list())
    std_cost = st.stdev(worker_df['cost'].to_list())
    return mean_cost, median_cost, std_cost


mean_cost ,median_cost,std_cost = get_statistics_worker_cost(clean_outlier_df)
print("mean_cost: {}, median_cost: {}, std_cost: {}".format(mean_cost, median_cost, std_cost))

def add_user_name(worker_df):
    number_of_leading_zeros = len(str(len(worker_df)))  # get number of digits could use log but saves the import
    user_name = ["w" + "{:0{}d}".format(s, number_of_leading_zeros) for s in range(len(worker_df))]
    worker_df['user_name'] = user_name
    return worker_df


print(add_user_name(clean_outlier_df))


def get_popularities(worker_df, task_list, fraction=1.0, split=(0.2, 0.6)):
    # lists are stored as strings so apply(eval) will fix this
    column = worker_df['skills'].to_list()
    frequiencies = dict.fromkeys(task_list, 0)
    for row in column:
        for skill in row:
            frequiencies[skill] += 1
    df_freq = pd.DataFrame(frequiencies.items(), columns=['skill', 'freq'])
    df_freq = df_freq.sort_values('freq')
    covered_skills = df_freq['skill']
    df_freq = np.random.choice(df_freq['skill'].to_list(), size=int(len(covered_skills) * fraction), replace=False)
    rare, common, popular = np.split(df_freq, [int(split[0] * len(df_freq)), int(split[1] * len(df_freq))])
    return rare, common, popular, covered_skills.to_list()


rare, common, popular, covered = get_popularities(clean_outlier_df, task_list, fraction=1)


def get_dataframe_from_categories(rare, common, popular, task_df):
    combined = [*rare,*common, *popular]
    mask = task_df['skill'].isin(combined)
    task_df = task_df.loc[mask]
    task_df = task_df.reset_index(drop=True)

    return task_df


sample_df = get_dataframe_from_categories(rare, common, popular, task_df)

print(sample_df)


#
# def sample_task(task_df, rare, common, popular,number_of_tasks,  quantiles_array=(0.15, 0.75), ratio=1, split=(0.33, 0.33, 0.33)):
#     if not (isinstance(quantiles_array, (tuple, list)) and len(quantiles_array) == 2) and (1 - sum(quantiles_array)) > 0:
#         raise TypeError(
#             'quantiles array must be tuple or list and must have len == 2 and'
#             'must have sum less than 1 instead of type: {}, len: {}, sum: {}'.format(
#                 type(quantiles_array), len(quantiles_array) if '__len__' in dir(quantiles_array) else None,
#                 sum(quantiles_array)))
#     sample_number_of_tasks = number_of_tasks * ratio
#     rare_perc = min(int(len(rare) * ratio), int(sample_number_of_tasks * quantiles_array[0]))
#     common_perc = min(int(len(common) * ratio), int(sample_number_of_tasks * quantiles_array[1]))
#     popular_perc = min(int(len(popular) * ratio), int(sample_number_of_tasks * (1- sum(quantiles_array))))
#     sample_rare = np.random.choice(rare, size=rare_perc, replace=False)
#     sample_common = np.random.choice(common, size=common_perc, replace=False)
#     sample_popular = np.random.choice(popular, size=popular_perc, replace=False)
#
#     mask = task_df['skill'].isin(sample_rare) or task_df['skill'].isin(sample_common) or task_df['skill'].isin(sample_popular)
#     task_df = task_df.loc[mask]
#     task_df = task_df.reset_index(drop=True)
#
#     return task_df
#
# print(sample_task(task_df, rare, common, popular, len(task_df), ratio=1))

def add_valuation(task_df, upper_bound, std, rare, common, seed=1, lower_bound=1):
    """
    Adding Valutiations to the dataset.
    :param task_df:
    :param upper_bound:
    :param std:
    :param rare:
    :param common:
    :param seed:
    :param lower_bound:
    :return:
    """
    if seed != 1:
        np.random.seed(seed)

    task_list = task_df['skill'].to_list()
    val_list = []
    for task in task_list:
        if task in rare:
            val_list.append(np.random.uniform(lower_bound, upper_bound + 1.2*std))
        elif task in common:
            val_list.append(np.random.uniform(lower_bound, upper_bound))
        else:
            val_list.append(np.random.uniform(lower_bound, max(0, upper_bound - std / 4)))

    task_df["valuation"] = val_list
    return task_df


final_task_df = add_valuation(sample_df, mean_cost, std_cost, rare, common , seed=2022)



# dataframe_to_csv(final_task_df,'task_df_guru_complete','/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/data/data_files/real_data/test_real_dir/instance_2')
# dataframe_to_csv(clean_outlier_df,'worker_df_guru_complete','/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/data/data_files/real_data/test_real_dir/instance_2')
#
# opt = Optimization(clean_outlier_df, final_task_df)
# problem_obj, xvals, yvals, optimal_workers_df, optimal_tasks_df = opt.wd_ip_solver(final_task_df.shape[0],
#                                                                                       clean_outlier_df, final_task_df,verbose=1)
# print(problem_obj,xvals,yvals,optimal_workers_df,optimal_tasks_df)
