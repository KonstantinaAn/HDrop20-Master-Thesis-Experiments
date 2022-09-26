import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

task_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/tasks.csv')
worker_df = pd.read_csv('./data_files/real_data/test_real_dir/instance_1/workers.csv')

k_constrained = 4
k_array = np.array([k_constrained])
print(k_array)


b = np.zeros((task_df.shape[0], 1))
print(b)

b = np.append(b, [k_array], axis=0)
print('new_b')
print(b)

b = np.squeeze(b)
print('squeezed_b', b)
#
# mlb = MultiLabelBinarizer(classes=task_df.skill.values)
# worker_skills = worker_df['skills'].tolist()
# tasks_worker_array = mlb.fit_transform(worker_skills)
# print('tasks_worker_array', tasks_worker_array)
# eye_n = np.eye(len(task_df.skill.values))
# print('eye_n', eye_n)
# tasks_worker_array = tasks_worker_array.transpose()
# print('transposed_tasks_worker_array', tasks_worker_array)
#
# # n x n+m size
# # A represents the inequality xn< Sum(ym) for m: rn in Qm
# # identity matrix is the first part because x1 < sum..., x2<sum.. etc
# # so for the first row 1 followed by 24 zeros and then -1 for all the y's that can perform the task
# # multiply this row with the column vector X hwre 25 first elements are xn and 50 last elements are ym
# # and you get the inequality
# A = np.concatenate((eye_n, tasks_worker_array * -1), axis=1)
# print(A)
#
# arr = []
# for i in range(len(task_df)):
#     arr.append(0)
# for i in range(len(worker_df)):
#     arr.append(1)
#
# print(arr)
#
# A = np.append(A, [arr], axis=0)
# print('new_A')
# print(A)

# x = np.array([[0],[1],[2],[3],[4]])
# print(x.shape)
# print(x)
# #(1, 3, 1)
# print(np.squeeze(x).shape)
# print(np.squeeze(x))
# #(3,)
# print(np.squeeze(x, axis=(1,)).shape)
# print(np.squeeze(x, axis=(1,)))
# #(1, 3)
