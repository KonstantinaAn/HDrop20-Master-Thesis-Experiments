from data.synthetic_data import *
# algorithms
from algorithms.distorted_greedy import DistortedGreedy
from algorithms.DeterministicUSM import DeterministicUSM
from algorithms.CSG import CSG
from algorithms.WInnerDetermination import WinnerDeetermination
from algorithms.ROI_greedy import ROIGreedy

from objective_functions.objective_functions import ObjectiveFunction

synth_data = SyntheticData(50, 90)
costs_range = (0, 3.39033)
valuations_range = (1, 4)
number_of_tasks_range = (1, 7)

synth_data.create_synthetic_data(costs_range,
                                 valuations_range,
                                 number_of_tasks_range)

obj_func = ObjectiveFunction(synth_data.set_of_tasks,
                             synth_data.worker_tasks,
                             synth_data.valuations,
                             synth_data.worker_costs)

dgreedy = DistortedGreedy(obj_func.submodular_function_eval,
                          obj_func.linear_function_eval,
                          synth_data.workers)
dUSM = DeterministicUSM(obj_func.submodular_function_eval,
                        synth_data.workers)

CSG = CSG(obj_func.submodular_function_eval,
           obj_func.linear_function_eval,
           synth_data.workers)

Winner = WinnerDeetermination(obj_func.submodular_function_eval, obj_func.linear_function_eval, obj_func.costs,
                               synth_data.workers,)

ROI = ROIGreedy(obj_func.submodular_function_eval,
                          obj_func.linear_function_eval,
                          synth_data.workers)

# solution = dgreedy.run()
# solution = dUSM.run()
# solution = CSG.run()
# solution = Winner.run()
solution = ROI.run()
print('solution: ', solution)
print('val:', obj_func.submodular_function_eval(solution))
print('len solution: ', len(solution))
for item in synth_data.__dict__:
    print(f'{item}:  {synth_data.__dict__[item]}')
print(obj_func.tasks_assigned)
