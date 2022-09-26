from experiments.First_Generation.experiment import Experiment

first_experiment = Experiment('Official_Experiment')
first_experiment.run_experiments(result_suffix='_ofc_2_low_valuation_fix', file_suffix='_2')
print("Finished")
print("!"*100)