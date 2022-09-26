from data_file_parser import pattern_parser
from utils import config_filepath
import json
import os

with open(config_filepath) as f:
    config = json.load(f)


def main(suffix=''):
    synthetic_project = config['project']['synthetic']
    datasets = synthetic_project['datasets_paths']
    datasets_keys = datasets.keys()
    instances_path = []
    for key in datasets_keys:
        for instance in datasets[key]['instances']:
            tasks_files, workers_files = pattern_parser(key, instance, pattern=suffix)
            files = tasks_files + workers_files
            [instances_path.append(el) for el in files]
    while True:
        print("Files: \n")
        print(*instances_path,sep='\n')
        print("You are deleteting {} files.\n".format(len(instances_path)))
        prompt = input("Are you sure you want to delete those files\n (yes/y=confirm, no/n=cancel):  ")
        if prompt.lower() == 'y' or prompt.lower() == 'yes':
            for file_path in instances_path:
                if file_path:
                    print('File {} removed. '.format(file_path))
                    os.remove(file_path[0])
            break
        elif prompt.lower() == 'n' or prompt.lower() == 'no':
            print("Cancelled.")
            break

if __name__ == '__main__':
    pattern = input("Give pattern: ")
    main(pattern)