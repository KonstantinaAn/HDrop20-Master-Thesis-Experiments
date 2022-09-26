from experiments.First_Generation.experiment import Experiment

test_experiment = Experiment('freelancer_complete')
algorithms = ["dUSM", "CSG", "Winner", "ROI", "dgreedy"]
dataset_keys = ["test_real_dir"]
dataset_paths = {
    "test_real_dir": {
        "path": r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\real_data\test_real_dir\\',
        "instances": {
            "instance_1": r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\data_files\real_data\test_real_dir\instance_1\\'
        },
        "n_m": [
            175,
            1747],
        "instance_keys": ["instance_1"]
    }
}
file_name_universl = "freelancer_complete_test"
excel_filename = file_name_universl + "_excel"
csv_filename = file_name_universl + "_csv"
pickle_filename = file_name_universl + "_pickle"
results_paths = {
    "dgreedy": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\dgreedy\\",
    "dUSM": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\dUSM\\",
    "CSG": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\CSG\\",
    "Winner": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\Winner\\",
    "ROI": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\ROI\\"
}
override_params = (
    algorithms, dataset_keys, dataset_paths, excel_filename, csv_filename, pickle_filename, results_paths)
test_experiment.run_experiments(override_params=override_params, project='Freelancer', result_suffix='_test_run',
                                file_suffix='_df_freelancer_complete')
