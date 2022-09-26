from experiments.First_Generation.experiment import Experiment

test_experiment = Experiment('guru_complete')
# algorithms =[
#     "dgreedy",
#     "dUSM",
#     "CSG",
#     "Winner",
#     "ROI"
# ]
algorithms = ["dUSM", "CSG", "Winner", "ROI", "dgreedy"]
dataset_keys = ["test_real_dir"]
dataset_paths = {
    "test_real_dir": {
        "path": "/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/data/data_files/real_data/test_real_dir/",
        "instances": {
            "instance_2": "/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/data/data_files/real_data/test_real_dir/instance_2/"
        },
        "n_m": [1639, 4477],
        "instance_keys": ["instance_2"]
    }
}
file_name_universl = "guru_complete_test"
excel_filename = file_name_universl + "_excel"
csv_filename = file_name_universl + "_csv"
pickle_filename = file_name_universl + "_pickle"
results_paths = {
    "dgreedy": "/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/results/test_real_data/dgreedy/",
    "dUSM": "/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/results/test_real_data/dUSM/",
    "CSG": "/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/results/test_real_data/CSG/",
    "Winner": "/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/results/test_real_data/Winner/",
    "ROI": "/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/results/test_real_data/ROI/"
}
override_params = (
    algorithms, dataset_keys, dataset_paths, excel_filename, csv_filename, pickle_filename, results_paths)
test_experiment.run_experiments(override_params=override_params, project='Guru', result_suffix='_test_run',
                                file_suffix='_df_guru_complete')