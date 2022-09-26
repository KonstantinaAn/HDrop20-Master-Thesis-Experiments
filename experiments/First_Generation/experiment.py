import csv
import random

from data.utils import config_filepath
from data.data_provider import DataProvider
from experiments.utils import get_data_file_paths, get_guru_files, get_freelancer_files, get_config
from objective_functions.objective_functions import ObjectiveFunction
from experiments.experiment_executioner import ExperimentExecutioner
from datetime import datetime
import logging
import math
import pandas as pd
import pickle


class Experiment:
    @classmethod
    def get_logger(cls, uid, time_date):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%a, %d %b %Y %H:%M:%S')
        time_date = '{}_{}_{}_{}'.format(time_date.hour, time_date.minute, time_date.second, time_date.microsecond)
        log_file_name = 'experiment{}_log{}.log'.format(uid, time_date)

        # log_path = '/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/results/experiment_logs/'
        log_path = r'C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\experiment_logs\\'
        file_handler = logging.FileHandler(log_path + log_file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def __init__(self, experiment_uid):
        """

        :param experiment_uid:
        """
        self.config = get_config()
        self.uid = experiment_uid
        self.time_date = datetime.now()
        self.logger = self.get_logger(self.uid, self.time_date)
        self.logger.info("Initializing experiment: {}".format(self.uid))
        self.algorithm = ""
        self.uid_keys = []
        self.execution_data = {}
        self.results = {}
        self.optimal_results = {}
        self.file_paths = []
        self.cache = {}

    def get_file_paths(self, dataset_keys, instance_keys, file_suffix='', project='synthetic'):
        """

        :param dataset_keys:
        :param instance_keys:
        :param file_suffix:
        :param project:
        :return:
        """
        file_paths = []
        if project == 'synthetic':
            file_paths = get_data_file_paths(dataset_keys=dataset_keys, instance_keys=instance_keys,
                                             file_suffix=file_suffix)
        elif project == 'Guru':
            file_paths = get_data_file_paths(dataset_keys=dataset_keys, instance_keys=instance_keys,
                                             file_suffix=file_suffix, _project='guru')
        elif project == 'Freelancer':
            file_paths = get_data_file_paths(dataset_keys=dataset_keys, instance_keys=instance_keys,
                                             file_suffix=file_suffix, _project='freelancer')
        elif project == 'real_data_freelancer':
            file_paths = get_data_file_paths(dataset_keys=dataset_keys, instance_keys=instance_keys,
                                             file_suffix=file_suffix, _project='real_data_freelancer')

        file_paths.sort()
        print(file_paths)
        if not file_paths:
            self.logger.debug("Empty file paths")
        else:
            self.logger.debug("File paths: {}".format(file_paths))
        return file_paths

    def run_instances(self, algorithm, file_paths, exec_uid):
        """

        :param algorithm:
        :param file_paths:
        :param exec_uid:
        :return:
        """
        dataset_results = {}
        dataset_results_data = {}
        dataset_optimal_results = {}
        workers = 0
        for index, file_tuple in enumerate(file_paths):
            tasks_file_path = file_tuple[0]
            workers_file_path = file_tuple[1]
            instance = file_tuple[2][1]
            instance_uid = "{}_{}_{}".format(exec_uid, instance, index)
            self.logger.info("Running instance: {} for algorithm: {} with instance_uid: {}".format(instance, algorithm,
                                                                                                   instance_uid))
            data = DataProvider(config_filepath)
            workers = data.read_csv_to_set(file_path=workers_file_path, element_column_number=3)
            print('len of workers:',len(workers))
            workers_df = data.read_data_csv2df(file_path=workers_file_path, verbose=False, worker=True)
            print('Workers:', workers_df)
            tasks_df = data.read_data_csv2df(file_path=tasks_file_path, verbose=False, worker=False)
            print('Tasks_file_path', tasks_file_path)
            print('Tasks:', tasks_df)
            task_set, worker_task_dict, task_val_dict, worker_cost = data.df_2_objective_inputs(workers_df, tasks_df)
            obj_func = ObjectiveFunction(task_set,
                                         worker_task_dict,
                                         task_val_dict,
                                         worker_cost)

            exp_exec = ExperimentExecutioner(obj_func, workers, algorithm, instance_uid, self.logger,
                                             workers_df=workers_df, tasks_df=tasks_df)
            self.logger.info("Running experiment with the following parameters")
            self.logger.info("Task file: {}".format(tasks_file_path))
            self.logger.info("Worker file: {}".format(workers_file_path))
            self.logger.info("Workers set : {}".format(workers))
            self.logger.info("Task set : {}".format(task_set))
            self.logger.info("Worker task dict  : {}".format(worker_task_dict))
            self.logger.info("Task val dict: {}".format(task_val_dict))
            self.logger.info("Worker cost : {}".format(worker_cost))
            self.logger.info("n,m(#task,#worker) : {},{}".format(len(task_set), len(workers)))
            results, results_uid = exp_exec.execute_algorithm()
            optimal_results, optimal_results_uid = exp_exec.get_optimal_solution(workers_df, tasks_df)
            # hash_name = "n:{}, m: {}".format(len(workers_df), len(tasks_df))
            # if not self.cache.get("dataframes") or not (workers_df, tasks_df) in self.cache.get("dataframes", []):
            #     self.cache["dataframes"] = self.cache.get("dataframes", [])
            #     self.cache["dataframes"].append((workers_df, tasks_df))
            #     optimal_results, optimal_results_uid = exp_exec.get_optimal_solution(workers_df, tasks_df)
            #     self.cache[hash_name] = {'optimal_results': optimal_results, 'optimal_results_uid': optimal_results_uid}
            # else:
            #     optimal_results, optimal_results_uid = self.cache[hash_name]['optimal_results'], self.cache[hash_name][
            #         'optimal_results_uid']
            dataset_results.update({results_uid: results})
            dataset_results_data.update(
                {results_uid: {"task_file": tasks_file_path,
                               "worker_file": workers_file_path,
                               "workers": workers,
                               "workers_df": workers_df,
                               "tasks_df": tasks_df,
                               "task_set": task_set,
                               "worker_task_dict": worker_task_dict,
                               "task_val_dict": task_val_dict,
                               "worke_cost": worker_cost,
                               "n": len(task_set),
                               "m": len(workers),
                               "solution": results[results_uid]["solution"],
                               "value": results[results_uid]["obj_val"],
                               }
                 })
            print('dataset_results_data')
            print(dataset_results_data[results_uid]["tasks_df"])
            print(dataset_results_data[results_uid]["tasks_df"]["valuation"].sum())
            optimal_solution = optimal_results["optimal_solution"]
            optimal_linear_eval = obj_func.linear_function_eval(optimal_solution)
            optimal_sub_eval = obj_func.submodular_function_eval(optimal_solution)
            optimal_results.update({"optimal_linear_val": optimal_linear_eval, "optimal_sub_val": optimal_sub_eval})
            dataset_optimal_results.update({optimal_results_uid: optimal_results})
            opt_res = dataset_optimal_results[optimal_results_uid]
            self.logger.info("Optimal Solution: {} ".format(opt_res["optimal_solution"]))
            self.logger.info("Optimal Objective function: {}".format(opt_res["optimal_value"]))
            self.logger.info("Optimal Linear function: {}".format(opt_res["optimal_linear_val"]))
            self.logger.info("Optimal Submodular function: {}".format(opt_res["optimal_sub_val"]))
        return dataset_results, dataset_results_data, dataset_optimal_results, len(workers), results_uid

    def run_experiments(self, file_suffix='', project='synthetic', result_suffix='', override_params=(),algorithm_param=()):
        """

        :param override_params:
        :param file_suffix:
        :param project:
        :param result_suffix:
        :return:
        """
        if algorithm_param and not override_params:
            algorithms = algorithm_param
        else:
            algorithms = self.config["algorithms"]
        if override_params:
            print('In override params')
            algorithms = override_params[0]
            print('algorithms', algorithms)
            dataset_keys = override_params[1]
            dataset_paths = override_params[2]
            print(dataset_paths)
            excel_filename = override_params[3] + result_suffix
            csv_filename = override_params[4] + result_suffix
            pickle_filename = override_params[5] + result_suffix
            results_paths = override_params[6]
        else:
            print('Not override params')
            dataset_keys = self.config["project"][project]["dataset_keys"]
            dataset_paths = self.config["project"][project]["datasets_paths"]
            excel_filename = self.config["results"]["base_file_names"]["excel"] + result_suffix
            csv_filename = self.config["results"]["base_file_names"]["csv"] + result_suffix
            pickle_filename = self.config["results"]["base_file_names"]["pickle"] + result_suffix
            results_paths = self.config["results"]["algorithm_results_path"]
        for algorithm in algorithms:
            print('algorithm', algorithm)
            for dataset_key in dataset_keys:
                instance_keys = dataset_paths[dataset_key]["instance_keys"]
                self.file_paths = self.get_file_paths(dataset_keys=[dataset_key], instance_keys=instance_keys,
                                                      file_suffix=file_suffix, project=project)
                exec_id = '{}_{}_{}'.format(self.uid, algorithm, dataset_key)
                print(self.file_paths)
                results, data_results, optimal_results, n_worker_len, results_uid = self.run_instances(algorithm, self.file_paths,
                                                                                          exec_uid=exec_id)
                self.results.update({exec_id: results})
                self.execution_data.update({exec_id: data_results})
                self.optimal_results.update({exec_id: optimal_results})
                self.logger.debug("Preparing data")
                excel_data = self.prepare_excel_data(results, data_results[results_uid], optimal_results, n_worker_len, dataset_key)
                csv_data = self.prepare_csv_data(data_results)
                pickle_data = self.prepare_pickle_data(data_results)
                current_path = results_paths[algorithm] + dataset_key + "/"
                self.logger.debug("Finished data preperation")
                self.logger.debug("Writing Results")
                self.write_excel_results(current_path, excel_data, excel_filename)
                self.write_csv_results(current_path, csv_data, csv_filename)
                self.write_pickle_results(current_path, pickle_data, pickle_filename)
                self.logger.debug("Finish writing results")
        return True

    def prepare_excel_data(self, results, data_results, optimal_results, n_worker_len, dataset_key):
        self.logger.debug("Starting preparing excel data")
        results_keys = list(results.keys())
        results_keys.sort()
        lines = []
        labels = ("dataset",
                  "instance",
                  "results_key",
                  "n",
                  "|S*|",
                  "V(S*)",
                  "C(S*)",
                  "SW(S*)",
                  "|S*|/n",
                  "SW(OPT)",
                  "|OPT|",
                  "|OPT|/n",
                  "SW(S*)/SW(OPT)",
                  "V(OPT)",
                  "C(OPT)",
                  "ln(V(OPT)/C(OPT))",
                  "V(OPT)-C(OPT)-ln(V(OPT)/C(OPT))*C(OPT)",
                  "V(OPT)/C(OPT)",
                  "V(S*)/C(S*)",
                  "ln(V(S*)/C(S*))",
                  "V(S*)-C(S*)-ln(V(S*)/C(S*))*C(S*)",
                  "SC(S*)=-SW(S*)+V(N)",
                  "V(N)",
                  "SC(OPT)=-SW(OPT)+V(N)",
                  "SC(S*)/SC(OPT)"
                  )
        for index, result_key in enumerate(results_keys):
            results_dict = results[result_key][result_key]
            optimal_results_dict = optimal_results[result_key]
            solution = results_dict["solution"]
            sub_val = results_dict["sub_val"]
            lin_val = results_dict["lin_val"]
            obj_val = results_dict["obj_val"]
            total_valuation_of_tasks = results_dict["total_valuation_of_tasks"]
            optimal_solution = optimal_results_dict["optimal_solution"]
            optimal_value = optimal_results_dict["optimal_value"]
            optimal_sub_val = optimal_results_dict["optimal_sub_val"]
            optimal_lin_val = optimal_results_dict["optimal_linear_val"]
            lines.append((
                dataset_key,
                index + 1,
                result_key,
                n_worker_len,
                len(solution),
                sub_val,
                lin_val,
                obj_val,
                len(solution) / n_worker_len,
                optimal_value,
                len(optimal_solution),
                len(optimal_solution) / n_worker_len,
                obj_val / optimal_value,
                optimal_sub_val,
                optimal_lin_val,
                math.log(
                    optimal_sub_val / optimal_lin_val) if optimal_lin_val > 0 and optimal_lin_val > 0 else "undefined",
                (optimal_sub_val - optimal_lin_val - (math.log(
                    optimal_sub_val / optimal_lin_val)) * optimal_lin_val) if optimal_lin_val > 0 and optimal_sub_val > 0 else "undefined",
                optimal_sub_val / optimal_lin_val if optimal_lin_val > 0 else "undefined",
                sub_val / lin_val if lin_val > 0 else "undefined",
                math.log(sub_val / lin_val) if lin_val > 0 and sub_val > 0 else "undefined",
                (sub_val - lin_val - (math.log(
                    sub_val / lin_val)) * lin_val) if lin_val > 0 and sub_val > 0 else "undefined",
                -obj_val+total_valuation_of_tasks,
                total_valuation_of_tasks,
                -optimal_value+total_valuation_of_tasks,
                (-obj_val+total_valuation_of_tasks)/(-optimal_value+total_valuation_of_tasks)
            ))
        print('In preparing excel:')
        print('dataset_results_data')
        print(data_results["tasks_df"])
        print(data_results["tasks_df"]["valuation"].sum())
        df = pd.DataFrame(lines, columns=labels)
        self.logger.debug("Finishing preparation of excel data")
        return df

    def prepare_csv_data(self, results):
        self.logger.debug("Starting preparation of csv data")
        self.logger.debug("Finishing preparation of csv data")
        return results

    def prepare_pickle_data(self, results):
        self.logger.debug("Starting preparation of csv data")
        self.logger.debug("Finishing preparation of csv data")
        return results

    def write_excel_results(self, file_path, data, filename, suffix=".xlsx"):
        try:
            full_path = file_path + filename + suffix
            self.logger.debug("Writing excel data to file : {}".format(full_path))
            data.to_excel(full_path)
        except Exception as E:
            self.logger.debug("Exception while writing excel data")
            raise E
        return True

    def write_csv_results(self, file_path, data, filename, suffix=".csv"):
        try:
            full_path = file_path + filename + suffix
            self.logger.debug("Writing csv data to file : {}".format(full_path))
            with open(full_path, "w") as csvfile:
                writer = csv.writer(csvfile)
                for key, value in data.items():
                    writer.writerow([key, value])
        except Exception as E:
            self.logger.debug("Exception while writing excel data")
            raise E
        return True

    def write_pickle_results(self, file_path, data, filename, suffix=".pickle"):
        try:
            full_path = file_path + filename + suffix
            self.logger.debug("Writing csv data to file : {}".format(full_path))
            with open(full_path, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as E:
            self.logger.debug("Exception while writing pickle data")
            raise E
        return True


if __name__ == "__main__":
    # test_experiment = Experiment('freelancer_complete')
    # # algorithms =[
    # #     "dgreedy",
    # #     "dUSM",
    # #     "CSG",
    # #     "Winner",
    # #     "ROI"
    # # ]
    # algorithms = ["Winner"]
    # dataset_keys = [
    #   "dataset_01",
    #   "dataset_02",
    #   "dataset_03",
    #   "dataset_04",
    #   "dataset_05",
    #   "dataset_06",
    #   "dataset_07",
    #   "dataset_08",
    #   "dataset_09",
    #   "dataset_10",
    #   "dataset_11",
    #   "dataset_12"
    # ]
    # # dataset_paths = {
    # #   "dataset_01": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\01_dataset_5_workers_10_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       10,
    # #       5
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_02": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\02_dataset_10_workers_5_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       5,
    # #       10
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_03": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\03_dataset_10_workers_10_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       10,
    # #       10
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_04": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\04_dataset_10_workers_20_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       20,
    # #       10
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_05": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\05_dataset_20_workers_10_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       10,
    # #       20
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_06": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\06_dataset_20_workers_50_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       50,
    # #       20
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_07": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\07_dataset_20_workers_100_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       100,
    # #       20
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_08": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\08_dataset_50_workers_25_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       25,
    # #       50
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_09": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\09_dataset_50_workers_50_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       50,
    # #       50
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_10": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\10_dataset_50_workers_100_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       100,
    # #       50
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_11": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\11_dataset_20_workers_100_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       100,
    # #       20
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   },
    # #   "dataset_12": {
    # #     "path": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\",
    # #     "instances": {
    # #       "instance_01": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_1\\",
    # #       "instance_02": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_2\\",
    # #       "instance_03": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_3\\",
    # #       "instance_04": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_4\\",
    # #       "instance_05": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_5\\",
    # #       "instance_06": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_6\\",
    # #       "instance_07": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_7\\",
    # #       "instance_08": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_8\\",
    # #       "instance_09": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_9\\",
    # #       "instance_10": r"C:\\Users\\Konstantina\\Desktop\\HDrop20-master-thesis-experiments\\data\\real_data_konna\\freelancer\\12_dataset_40_workers_150_tasks\\instance_10\\"
    # #     },
    # #     "n_m": [
    # #       150,
    # #       40
    # #     ],
    # #     "instance_keys": [
    # #       "instance_1",
    # #       "instance_2",
    # #       "instance_3",
    # #       "instance_4",
    # #       "instance_5",
    # #       "instance_6",
    # #       "instance_7",
    # #       "instance_8",
    # #       "instance_9",
    # #       "instance_10"
    # #     ]
    # #   }
    # # }
    # file_name_universl = "freelancer_1_test"
    # excel_filename = file_name_universl + "_excel"
    # csv_filename = file_name_universl + "_csv"
    # pickle_filename = file_name_universl + "_pickle"
    # results_paths = {
    #     # "dgreedy": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\dgreedy\\",
    #     #"dUSM": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\dUSM\\",
    #     #"CSG": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\CSG\\",
    #     "Winner": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\Winner\\",
    #     #"ROI": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\ROI\\"
    # }
    # # override_params = (
    # #     algorithms, dataset_keys, dataset_paths, excel_filename, csv_filename, pickle_filename, results_paths)
    # test_experiment.run_experiments(project='real_data_freelancer', result_suffix='_test_run',
    #                                 file_suffix='_df_freelancer_complete')
    # # test_experiment = Experiment('fixed_CSG_final')
    # # algorithms = ['CSG']
    # # test_experiment.run_experiments( project='synthetic', result_suffix='_low_valuation_fixed_CSG_Final',
    # #                                 file_suffix='_2',algorithm_param=algorithms)

    # test_experiment = Experiment('guru_complete')
    # # algorithms =[
    # #     "dgreedy",
    # #     "dUSM",
    # #     "CSG",
    # #     "Winner",
    # #     "ROI"
    # # ]
    # algorithms = ["dgreedy"]
    # dataset_keys = ["test_real_dir"]
    # dataset_paths = {
    #     "test_real_dir": {
    #         "path": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\guru1\01_dataset_20_workers_50_tasks\\",
    #         "instances": {
    #             "instance_2": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\guru1\01_dataset_20_workers_50_tasks\instance_01\\"
    #         },
    #         "n_m": [50, 50],
    #         "instance_keys": ["instance_2"]
    #     }
    # }
    # file_name_universl = "synthetic_22_8_22_test"
    # excel_filename = file_name_universl + "_excel"
    # csv_filename = file_name_universl + "_csv"
    # pickle_filename = file_name_universl + "_pickle"
    # results_paths = {
    #     "dgreedy": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\dgreedy\\",
    #     # "dUSM": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\dUSM\\",
    #     # "CSG": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\CSG\\",
    #     # "Winner": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\Winner\\",
    #     # "ROI": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\ROI\\"
    # }
    # override_params = (
    #     algorithms, dataset_keys, dataset_paths, excel_filename, csv_filename, pickle_filename, results_paths)
    # test_experiment.run_experiments(override_params=override_params, project='Guru', result_suffix='_test_run',
    #                                 file_suffix='')
    # test_experiment = Experiment('fixed_CSG_final')
    # algorithms = ['CSG']
    # test_experiment.run_experiments( project='synthetic', result_suffix='_low_valuation_fixed_CSG_Final',
    #                                 file_suffix='_2',algorithm_param=algorithms)

    test_experiment = Experiment('freelancer_complete')
    # algorithms =[
    #     "dgreedy",
    #     "dUSM",
    #     "CSG",
    #     "Winner",
    #     "ROI"
    # ]
    algorithms = ["Winner"]
    dataset_keys = ["test_real_dir"]
    dataset_paths = {
        "test_real_dir": {
            "path": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\freelancer1\02_dataset_10_workers_15_tasks\\",
            "instances": {
                "instance_1": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\data\real_data_konna\freelancer1\02_dataset_10_workers_5_tasks\instance_01\\"
            },
            "n_m": [175, 1747],
            "instance_keys": ["instance_1"]
        }
    }
    file_name_universl = "freelancer_constrained_13_9_22_test"
    excel_filename = file_name_universl + "_excel"
    csv_filename = file_name_universl + "_csv"
    pickle_filename = file_name_universl + "_pickle"
    results_paths = {
        # "dgreedy": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\dgreedy\\",
        # "dUSM": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\dUSM\\",
        # "CSG": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\CSG\\",
        "Winner": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\Winner\\",
        # "ROI": r"C:\Users\Konstantina\Desktop\HDrop20-master-thesis-experiments\results\test_real_data\ROI\\"
    }
    override_params = (
        algorithms, dataset_keys, dataset_paths, excel_filename, csv_filename, pickle_filename, results_paths)
    test_experiment.run_experiments(override_params=override_params, project='Freelancer', result_suffix='_test_run',
                                    file_suffix='')
    # test_experiment = Experiment('fixed_CSG_final')
    # algorithms = ['CSG']
    # test_experiment.run_experiments( project='synthetic', result_suffix='_low_valuation_fixed_CSG_Final',
    #                                 file_suffix='_2',algorithm_param=algorithms)

