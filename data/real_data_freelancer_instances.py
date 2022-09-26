import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from pathlib import Path

# #Guru
# freelancer

print('Freelancer dataset')
tasks_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/tasks_df_freelancer_complete.csv')
workers_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/workers_df_freelancer_complete.csv')

sum_cost_of_all = 0
sum_of_workers = 0
sum_cost = 0
for index, row in workers_df.iterrows():
    i = 0
    while i <= 140:
        if row['cost'] > i and row['cost'] <= i+10:
            #print('My cost is from', i,'to', i+10,'',row['cost'])
            sum_of_workers += 1
            sum_cost_of_all += row['cost']
        i += 10
    sum_cost += row['cost']

print('Number of workers is:', sum_of_workers, len(workers_df))
print('Sum cost of all', sum_cost_of_all, sum_cost)
print('The average cost of this dataset is:', sum_cost_of_all/sum_of_workers)

n = 10
while n <= len(tasks_df):

    print('-----')
    print('For the n first tasks, where n =', n)
    tasks_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/tasks_df_freelancer_complete.csv')
    workers_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/workers_df_freelancer_complete.csv')

    #n = len(tasks_df)
    tasks_df = tasks_df.head(n)
    skills_list = tasks_df['skill'].to_list()
    print(skills_list)

    costs = workers_df['cost']
    max_cost = costs.max()
    min_cost = costs.min()
    #print('Freelancer dataset')
    #print('Max cost:', max_cost, 'Min cost:', min_cost, 'Number of tasks:', len(tasks_df), 'Number of workers:', len(workers_df))
    workers_df = workers_df.reset_index()
    sum_cost = []
    for i in range(0, 15):
        sum_cost.append(0)

    sum_cost_of_all = 0
    sum_of_workers = 0
    for index, row in workers_df.iterrows():
        i = 0
        while i <= 140:
            if row['cost'] > i and row['cost'] <= i+10:
                #print('My cost is from', i,'to', i+10,'',row['cost'])
                sum_cost[i//10] += 1
                sum_of_workers += 1
                sum_cost_of_all += row['cost']
            i += 10

    #print('Workers are:', sum_of_workers)

    #print('The average cost of the dataset is:',sum_cost_of_all/sum_of_workers)


    #-----------------Skills---------------------
    max_skills_count = 0
    for index, row in workers_df.iterrows():

        skills_count = len([i for i in skills_list if i in row['skills']])

        if max_skills_count < skills_count:
            max_skills_count = skills_count
    #print('Max skills count: ', max_skills_count)

    sum_skills = []
    for i in range(0, max_skills_count+1):
        sum_skills.append(0)

    sum_skills_of_all = 0
    for index, row in workers_df.iterrows():
        i = 0
        skills_count = len([i for i in skills_list if i in row['skills']])

        sum_skills_of_all += skills_count

        while i <= max_skills_count:
            if skills_count == i:
                #print('My number of skills, when the', n, 'first tasks are chosen, is ', skills_count)
                sum_skills[i]+=1

            i+=1

    #for i in range(0,len(sum_skills)):
        #print(i,':', sum_skills[i])
    #print('Sum skills of all:', sum_skills_of_all)
    print('The average count of skills of this instance of the dataset is:', sum_skills_of_all/sum_of_workers)

    v_average = (sum_cost_of_all/sum_of_workers)/(sum_skills_of_all/sum_of_workers)
    print('The average value must be greater than', v_average)
    # fig = plt.figure()
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(range(len(sum_cost)), sum_cost)
    # plt.xticks(np.arange(0, len(sum_cost), 10.0))
    # plt.yticks(np.arange(0, max(sum_cost), 100.0))
    # plt.xlabel('Ranges of cost, e.g. 50 means cost 500-510')
    #
    # plt.subplot(2, 1, 2)
    #
    # plt.plot(range(len(sum_skills)), sum_skills)
    # plt.xticks(np.arange(0, len(sum_skills), 1.0))
    # plt.yticks(np.arange(0, max(sum_skills), 100.0))
    # plt.xlabel('Ranges of number of skills')
    #
    # plt.show()

    tasks_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/tasks_df_freelancer_complete.csv')
    workers_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/workers_df_freelancer_complete.csv')

    # Discrete Norm-------------------------------------

    workers_df = workers_df.reset_index()
    # for index, row in workers_df.iterrows():
    # print('Initial cost:', row['cost']);

    # the first n tasks

    df = tasks_df.head(n)
    skills_list = df['skill'].to_list()
    #print(skills_list)

    sum_of_workers = 0
    sum_of_workers_that_have_the_task = 0
    # workers_df = workers_df.reset_index()

    supremum = 2*v_average
    x = np.arange(1, supremum)
    xU, xL = x + 0.5, x - 0.5
    loc = (supremum + 1) / 2
    prob = ss.norm.cdf(xU, loc, scale=(supremum + 1 - loc) / 3) - ss.norm.cdf(xL, loc, scale=(supremum + 1 - loc) / 3)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1


    # Value to each task with Discrete Normal Distribution

    tasks_df['new_value'] = 0
    for index, row in tasks_df.iterrows():
        nums = np.random.choice(x, size=1, p=prob)
        if nums[0] == 0:
            value = 1
        else:
            value = nums[0]

        tasks_df.at[index, 'new_value'] = value

    #print('The new tasks_df with the new_values for each task')
    #print(tasks_df)

    filepath_tasks = Path(r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\real_data\test_real_dir\instance_1\tasks.csv')
    filepath_tasks.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.to_csv(filepath_tasks, index=False)


    for index, row in workers_df.iterrows():

        # the number of the needed skills that this worker has
        skills_count = len([i for i in skills_list if i in row['skills']])
        skills = [i for i in skills_list if i in row['skills']]
        sum_value = 0
        for i in skills:
            for j, r in tasks_df.iterrows():
                if r["skill"] == i:
                    sum_value += r["new_value"]
        #print('For worker:',row['user_id'],'Sum value:', sum_value)
        # # random value
        # nums = np.random.choice(x, size=1, p=prob)
        # if nums[0] == 0:
        #     value = 1
        # else:
        #     value = nums[0]

        #print('Worker with id:', row['user_id'], 'has skills_count:',skills_count)
        #print('and cost:',row['cost'])
        #print('and value:', value)
        if sum_value > row['cost']:
            # print(row['skills'])
            # print('Worker with id:', row['user_id'], 'Old Cost:', row['cost'], 'New cost:', skills_count*value - row['cost'], 'Has number of skills:', skills_count)
            #print('Worker got included')
            #print('----')
            sum_of_workers += 1
            # if skill in row['skills']:
            # print('Worker with id:', row['user_id'], 'Cost', row['cost'],'Has skill:',skill)
        if skills_count >= 1:
            sum_of_workers_that_have_the_task += 1

    #print('For the first', n, 'tasks')
    print('Sum of workers that have at least on of these', n, 'tasks:',sum_of_workers_that_have_the_task,'out of',len(workers_df),'workers')
    print('Sum of workers that got included:', sum_of_workers, 'That is', sum_of_workers / sum_of_workers_that_have_the_task * 100, '%')

    if n == 170:
        n = len(tasks_df)
    else:
        n += 10

