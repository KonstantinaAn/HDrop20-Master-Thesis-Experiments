import pandas as pd
import numpy as np
import ast
import scipy.stats as ss
from pathlib import Path

#task_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/tasks_df_freelancer_complete.csv')
#worker_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/workers_df_freelancer_complete.csv')

#print('sum of values of the tasks.csv', task_df['valuation'].sum())


# a function to add the tasks that do not matter (valuation=0), but are covered by the workers, so they have to be included
# in order for the algorithms to run

def add_tasks_of_workers(dataset_name, dataset_number, m_workers, n_tasks, k_instances):

    for i in range(k_instances):

        if i < 9:
            worker_df = pd.read_csv(
                r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_0%d\workers.csv' % (
                    dataset_name, dataset_number, m_workers, n_tasks, i + 1))

            task_df = pd.read_csv(
                r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_0%d\tasks.csv' % (
                    dataset_name, dataset_number, m_workers, n_tasks, i + 1))

        else:
            worker_df = pd.read_csv(
                r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_%d\workers.csv' % (
                    dataset_name, dataset_number, m_workers, n_tasks, i + 1))

            task_df = pd.read_csv(
                r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_%d\tasks.csv' % (
                    dataset_name, dataset_number, m_workers, n_tasks, i + 1))

        print(worker_df)
        print(task_df)

        column_worker_tasks = worker_df['skills'].to_list()
        worker_tasks = []
        print('Sum of skills of workers')
        for row in column_worker_tasks:
            # Converting string to list
            row = ast.literal_eval(row)
            for skill in row:
                worker_tasks.append(skill)
                print(skill)

        worker_tasks = list(dict.fromkeys(worker_tasks))

        print('Skills in tasks.csv')
        column_tasks = task_df['skill'].to_list()
        tasks = []
        for skill in column_tasks:
            tasks.append(skill)
            print(skill)

        print('Check tasks')
        for skill in worker_tasks:
            if skill not in tasks:
                print(skill, 'not in tasks.csv')
                df = {'skill': skill, 'skill_id': 0, 'valuation': 0}
                task_df = task_df.append(df, ignore_index=True)

        print(task_df)

        if i < 9:
            filepath_tasks = Path(r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_0%d\tasks.csv' % (
                    dataset_name, dataset_number, m_workers, n_tasks, i + 1))

        else:
            filepath_tasks = Path(r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\%s\%s_dataset_%d_workers_%d_tasks\instance_%d\tasks.csv' % (
                    dataset_name, dataset_number, m_workers, n_tasks, i + 1))

        task_df.to_csv(filepath_tasks, index=False)




if __name__ == '__main__':
    # add_tasks_of_workers('freelancer1', '01', 5, 10, 10)
    # add_tasks_of_workers('freelancer1', '02', m_workers=10, n_tasks=5, k_instances=10)
    # add_tasks_of_workers('freelancer1', '03', m_workers=10, n_tasks=10, k_instances=10)
    # add_tasks_of_workers('freelancer1', '04', m_workers=10, n_tasks=20, k_instances=10)
    # add_tasks_of_workers('freelancer1', '05', m_workers=20, n_tasks=10, k_instances=10)
    # add_tasks_of_workers('freelancer1', '06', m_workers=20, n_tasks=50, k_instances=10)
    # add_tasks_of_workers('freelancer1', '07', m_workers=20, n_tasks=100, k_instances=10)
    # add_tasks_of_workers('freelancer1', '08', m_workers=50, n_tasks=25, k_instances=10)
    # add_tasks_of_workers('freelancer1', '09', m_workers=50, n_tasks=50, k_instances=10)
    # add_tasks_of_workers('freelancer1', '10', m_workers=50, n_tasks=100, k_instances=10)
    # add_tasks_of_workers('freelancer1', '11', m_workers=20, n_tasks=100, k_instances=10)
    # add_tasks_of_workers('freelancer1', '12', m_workers=40, n_tasks=150, k_instances=10)
    # add_tasks_of_workers('guru1', '01', m_workers=20, n_tasks=50, k_instances=10)
    # add_tasks_of_workers('guru1', '02', m_workers=50, n_tasks=50, k_instances=10)
    # add_tasks_of_workers('guru1', '03', m_workers=20, n_tasks=100, k_instances=10)
    # add_tasks_of_workers('guru1', '04', m_workers=50, n_tasks=100, k_instances=10)
    # add_tasks_of_workers('guru1', '05', m_workers=20, n_tasks=200, k_instances=10)
    # add_tasks_of_workers('guru1', '06', m_workers=50, n_tasks=200, k_instances=10)
    add_tasks_of_workers('guru1', '07', m_workers=50, n_tasks=400, k_instances=10)
    add_tasks_of_workers('guru1', '08', m_workers=100, n_tasks=400, k_instances=10)


