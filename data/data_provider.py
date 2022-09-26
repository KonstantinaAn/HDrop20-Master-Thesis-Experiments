import json
import pandas as pd
import csv
import ast
#FIXME fix configuration usage or remove

class DataProvider:

    def __init__(self, config_path):
        self.config = self.read_config(config_path)

    @classmethod
    def read_config(cls, config_json_path):
        """
        Reads config file
        :param config_json_path: string. Path to json config
        :return config: return config file.
        """
        with open(config_json_path) as f:
            config = json.load(f)

        return config

    def read_data_csv2df(self, file_name=None,file_path=None, dataset_name='synthetic', header=0, index_col=False, sep=",",
                         verbose=False, worker=False):
        """
        Reads user info from 'dataset_name'dataset
        :param file_name: string.   <dataset_name>_job_filepath
                                    <dataset_name>_skill_filepath
                                    <dataset_name>_user_filepath
        :param dataset_name: string. options: 'guru', 'freelancer', 'synthetic
        :param header: boolean. Include header
        :param index_col: boolean. Add index column
        :param sep: string. Separate on delimiter.
        :param verbose: boolean. Print dataframe. default:False
        For more see pandas documentation
        :return: pandas dataframe.
        """
        if file_path is None:
            file_path = self.config['project'][dataset_name][file_name]
        df = pd.read_csv(
            file_path,
            header=header,
            index_col=index_col,
            sep=sep)

        if worker:
            worker_skills = [ast.literal_eval(el) for el in df['skills']]
            df['skills'] = worker_skills
        if verbose:
            print(df)
        return df

    def read_csv_to_dict(self, file_name=None,file_path=None, dataset_name='synthetic', first_element_column_number=0,
                         second_element_column_number=1, header=0,
                         verbose=False, limit=True, worker=False):
        """
        Read from a csv and return dictionary
        :param file_name: string.   <dataset_name>_job_filepath
                                    <dataset_name>_skill_filepath
                                    <dataset_name>_user_filepath
        :param dataset_name: string. Dataset name
        :param first_element_column_number:integer. column of the csv to take the key
        :param second_element_column_number:integer. column of the csv to take the value
        :param header: boolean. True to include header, default:False
        :param verbose: boolean. True to print dictionary created
        :param limit: boolean. True to limit print output
        :return: dictionary. Dictionary from csv file
        """
        first = first_element_column_number
        second = second_element_column_number
        if file_path is None:
            file_path = self.config['project'][dataset_name][file_name]
        with open(file_path, mode='r') as inp:
            reader = csv.reader(inp)
            if worker:
                if header:
                    dict_from_csv = {rows[first]: ast.literal_eval(rows[second]) for rows in reader}
                else:
                    dict_from_csv = {rows[first]: ast.literal_eval(rows[second]) for index, rows in enumerate(reader) if index > 0}
            else:
                if header:
                    dict_from_csv = {rows[first]: rows[second] for rows in reader}
                else:
                    dict_from_csv = {rows[first]: rows[second] for index, rows in enumerate(reader) if index > 0}
        if verbose:
            if limit and len(dict_from_csv) > 20:
                for index, key in enumerate(dict_from_csv.keys()):
                    if index < 5 or index > len(dict_from_csv) - 5:
                        print(key, dict_from_csv[key])
                    elif index == 5:
                        print('...  ' * 2)
            else:
                for key in dict_from_csv.keys():
                    print(key, dict_from_csv[key])

        return dict_from_csv

    def read_csv_to_set(self, file_name=None, file_path=None, dataset_name='synthetic', element_column_number=0, header=0,
                        verbose=False, limit=True):
        """
        Read from csv and return a set
        :param file_name: string.   <dataset_name>_job_filepath
                                    <dataset_name>_skill_filepath
                                    <dataset_name>_user_filepath
        :param dataset_name: string. Dataset name
        :param element_column_number:integer. column of the csv to take the element
        :param header: boolean. True to include header, default:False
        :param verbose: boolean. True to print dictionary created
        :param limit: boolean. True to limit print output
        :return: set. Set from csv file TODO:this returns a set which is not sorted maybe it will cause trouble use sorted() to sort
        """
        first = element_column_number
        if file_path is None:
            file_path = self.config['project'][dataset_name][file_name]
        with open(file_path, mode='r') as inp:
            reader = csv.reader(inp)
            if header:
                set_from_csv = {rows[first] for rows in reader}
            else:
                set_from_csv = {rows[first] for index, rows in enumerate(reader) if index > 0}
        if verbose:
            if limit and len(set_from_csv) > 20:
                for index, element in enumerate(sorted(set_from_csv)):
                    if index < 5 or index > len(set_from_csv) - 5:
                        print(element)
                    elif index == 5:
                        print('...  ' * 2)
            else:
                for element in set_from_csv:
                    print(element)
        return set_from_csv

    def df_2_objective_inputs(self, worker_df, task_df):
        """
        #TODO docum.
        :param worker_df:
        :param task_df:
        :return:
        """
        task_set = set(task_df['skill'])
        worker_task_dict = dict(zip(worker_df['user_name'], worker_df['skills']))
        task_val_dict = dict(zip(task_df['skill'], task_df['valuation']))
        worker_cost = dict(zip(worker_df['user_name'], worker_df['cost']))

        return task_set, worker_task_dict, task_val_dict, worker_cost


if __name__ == '__main__':
    config_path = '/Users/hdrop/Documents/Personal FIles/master-thesis-experiments/config/config.json'
    data = DataProvider(config_path)
    worker_set = data.read_csv_to_set(file_name='synthetic_user_filepath', element_column_number=3, verbose=True)
    worker_tasks = data.read_csv_to_dict(file_name='synthetic_user_filepath', first_element_column_number=3,
                                         second_element_column_number=1, verbose=True,worker=True)
    user_df = data.read_data_csv2df(file_name='guru_job_filepath', dataset_name='guru', verbose=True,worker=False)
    print(user_df)
