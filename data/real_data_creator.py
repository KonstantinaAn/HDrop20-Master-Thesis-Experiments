import pandas as pd
import numpy as np
import ast
import scipy.stats as ss
from pathlib import Path

# Freelancer

#pd.option_context('display.max_rows', None, 'display.max_columns', None)  # more options can be specified also
desired_width=5000
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)


def get_popularity(worker_df, task_df, split=(0.2, 0.8)):

    task_list = task_df['skill'].to_list()
    column = worker_df['skills'].to_list()
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    frequencies = dict.fromkeys(task_list, 0)

    for row in column:
        # Converting string to list
        row = ast.literal_eval(row)
        for skill in row:
            frequencies[skill] += 1

    df_freq = pd.DataFrame(frequencies.items(), columns=['skill', 'freq'])
    df_freq = df_freq.sort_values('freq')

    df_freq = pd.merge(df_freq, task_df, on='skill', how='inner')
    #print('Joined',df_freq)

    rare, common, popular = np.split(df_freq, [int(split[0] * len(df_freq)), int(split[1] * len(df_freq))])

    # print('popular', popular, len(popular))
    # print('common', common, len(common))
    # print('rare', rare, len(rare))

    return rare, common, popular


def get_random_tasks(n, rare, common, popular):

    random_rare_tasks = rare.sample(n=int(n*0.2))
    random_common_tasks = common.sample(n=int(n * 0.6))
    random_popular_tasks = popular.sample(n=int(n * 0.2))
    # print(random_rare_tasks)
    # print(random_common_tasks)
    # print(random_popular_tasks)

    frames = [random_rare_tasks, random_common_tasks, random_popular_tasks]
    result = pd.concat(frames)

    #print('result',result)
    return result


# selecting the workers that are included when the n_random_tasks are picked
# then, m of them will be selected (in the same for loop the outcome will be 10 instances)
def select_workers(n_random_tasks, worker_df):

    sum_of_workers = len(worker_df)
    sum_cost_of_all = 0

    for index, row in worker_df.iterrows():
        sum_cost_of_all += row['cost']

    skills_list = n_random_tasks['skill'].to_list()
    sum_skills_of_all = 0

    for index, row in worker_df.iterrows():
        skills_count = len([i for i in skills_list if i in row['skills']])
        sum_skills_of_all += skills_count

    # v_average * #tasks_average > cost_average
    v_average = (sum_cost_of_all / sum_of_workers) / (sum_skills_of_all / sum_of_workers)

    # Discrete Norm-------------------------------------
    supremum = 2 * v_average
    x = np.arange(1, supremum)
    xU, xL = x + 0.5, x - 0.5
    loc = (supremum + 1) / 2
    prob = ss.norm.cdf(xU, loc, scale=(supremum + 1 - loc) / 3) - ss.norm.cdf(xL, loc,
                                                                              scale=(supremum + 1 - loc) / 3)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1

    # Value to each task with Discrete Normal Distribution
    n_random_tasks['new_value'] = 0

    n_random_tasks = n_random_tasks.reset_index()

    for index, row in n_random_tasks.iterrows():
        nums = np.random.choice(x, size=1, p=prob)
        if nums[0] == 0:
            value = 1
        else:
            value = nums[0]

        n_random_tasks.at[index, 'new_value'] = value
        #print('task:', row['skill'],'value', n_random_tasks.at[index, 'new_value'])

    sum_of_included_workers = 0
    sum_of_workers_that_have_the_task = 0

    included_workers = pd.DataFrame()
    workers_df = worker_df.reset_index()
    for index, row in workers_df.iterrows():
        # the number of the needed skills that this worker has
        skills_count = len([i for i in skills_list if i in row['skills']])
        # # random value
        # nums = np.random.choice(x, size=1, p=prob)
        # if nums[0] == 0:
        #     value = 1
        # else:
        #     value = nums[0]
        skills = [i for i in skills_list if i in row['skills']]
        sum_value = 0
        for i in skills:
            for j, r in n_random_tasks.iterrows():
                if r["skill"] == i:
                    sum_value += r["new_value"]

        #if skills_count * value > row['cost']:
        if sum_value > row['cost']:
            sum_of_included_workers += 1
            included_workers = included_workers.append(row, ignore_index=True)

        if skills_count >= 1:
            sum_of_workers_that_have_the_task += 1

    included_workers.drop(columns=['index'], inplace=True)
    n_random_tasks.drop(columns=['index'], inplace=True)
    n_random_tasks.drop(columns=['freq'], inplace=True)
    n_random_tasks.drop(columns=['valuation'], inplace=True)
    n_random_tasks.rename(columns={'new_value': 'valuation'}, inplace=True)

    print('Sum of workers that have at least one of these tasks', len(n_random_tasks), 'tasks:', sum_of_workers_that_have_the_task, 'out of',
          len(workers_df), 'workers')
    print('Sum of workers that got included:', sum_of_included_workers, 'That is',
          sum_of_included_workers / sum_of_workers_that_have_the_task * 100, '%')

    return included_workers, n_random_tasks


def create_instances(rare, common, popular, n_tasks, m_workers, k_instances, dataset_number, dataset_name):

    for i in range(k_instances):

        n_random_tasks = get_random_tasks(n_tasks, rare, common, popular)
        included_workers, n_random_tasks = select_workers(n_random_tasks, worker_df)

        df = included_workers.sample(n=m_workers, replace=False)
        #print('df number:', i)
        #print(df)

        if i < 9:
            filepath_workers = Path(r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_0%d\workers.csv' %(dataset_name, dataset_number, m_workers, len(n_random_tasks), i+1))
            filepath_workers.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath_workers, index=False)
            filepath_tasks = Path(r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_0%d\tasks.csv' % (
                dataset_name, dataset_number, m_workers, len(n_random_tasks), i + 1))
            n_random_tasks.to_csv(filepath_tasks, index=False)
        else:
            filepath_workers = Path(r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_%d\workers.csv' %(dataset_name, dataset_number, m_workers, len(n_random_tasks), i+1))
            filepath_workers.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath_workers, index=False)
            filepath_tasks = Path(r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_%d\tasks.csv' % (
                dataset_name, dataset_number, m_workers, len(n_random_tasks), i + 1))
            n_random_tasks.to_csv(filepath_tasks, index=False)


# TODO
# main

if __name__ == '__main__':

    #Freelancer

    # task_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/tasks_df_freelancer_complete_steliou.csv')
    # worker_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/workers_df_freelancer_complete_steliou.csv')
    #
    # split = (0.2, 0.8)  # arr[0:0.2] arr[0.2:0.8] arr[0.8:1]
    # rare, common, popular = get_popularity(worker_df, task_df)
    # print('rare', rare)
    # print('common', common)
    # print('popular', popular)
    #
    # create_instances(rare, common, popular, n_tasks=10, m_workers=5, k_instances=10, dataset_number='01', dataset_name='freelancer1')
    #
    # create_instances(rare, common, popular, 5, 10, 10, '02', 'freelancer1')
    #
    # create_instances(rare, common, popular, 10, 10, 10, '03', 'freelancer1')
    #
    # create_instances(rare, common, popular, 20, 10, 10, '04', 'freelancer1')
    #
    # create_instances(rare, common, popular, 10, 20, 10, '05', 'freelancer1')
    #
    # create_instances(rare, common, popular, 50, 20, 10, '06', 'freelancer1')
    #
    # create_instances(rare, common, popular, 100, 20, 10, '07', 'freelancer1')
    #
    # create_instances(rare, common, popular, 25, 50, 10, '08', 'freelancer1')
    #
    # create_instances(rare, common, popular, 50, 50, 10, '09', 'freelancer1')
    #
    # create_instances(rare, common, popular, 100, 50, 10, '10', 'freelancer1')
    #
    # create_instances(rare, common, popular, 100, 20, 10, '11', 'freelancer1')
    #
    # create_instances(rare, common, popular, 150, 40, 10, '12', 'freelancer1')

    #Guru

    task_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_2/tasks_df_guru_complete_steliou.csv')
    worker_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_2/workers_df_guru_complete_steliou.csv')

    split = (0.2, 0.8)  # arr[0:0.2] arr[0.2:0.8] arr[0.8:1]
    rare, common, popular = get_popularity(worker_df, task_df)
    print('rare', rare, len(rare))
    print('common', common, len(common))
    print('popular', popular, len(popular))

    # create_instances(rare, common, popular, 50, 20, 10, '01', 'guru1')
    #
    # create_instances(rare, common, popular, 50, 50, 10, '02', 'guru1')
    #
    # create_instances(rare, common, popular, 100, 20, 10, '03', 'guru1')
    #
    # create_instances(rare, common, popular, 100, 50, 10, '04', 'guru1')
    #
    # create_instances(rare, common, popular, 200, 20, 10, '05', 'guru1')
    #
    # create_instances(rare, common, popular, 200, 50, 10, '06', 'guru1')

    create_instances(rare, common, popular, 400, 50, 10, '07', 'guru1')

    create_instances(rare, common, popular, 400, 100, 10, '08', 'guru1')

