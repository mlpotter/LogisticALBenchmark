import os
import os.path as osp
import re

import numpy as np
from mlflow.tracking import MlflowClient
import mlflow
import argparse
from subprocess import Popen
import pdb
from time import sleep


def check_experiment_completed(file_args,result_path):
    # Regex patterns
    dataset_pattern = r"--dataset=(\w+)"
    lam_pattern = r"--lam=([\d\.]+)"
    query_method_pattern = r"--query_method_name=([\w_]+)"

    # Extracting the values
    dataset_name = re.search(dataset_pattern, file_args)
    lam_value = re.search(lam_pattern, file_args)
    query_method_name = re.search(query_method_pattern, file_args)

    # Getting the matched values
    dataset_name = dataset_name.group(1) if dataset_name else None
    lam_value = str(float(lam_value.group(1))) if lam_value else None
    query_method_name = query_method_name.group(1) if query_method_name else None

    folder_path = osp.join(result_path,dataset_name,query_method_name,f"lam_{lam_value}")

    return folder_path


def parse_args():
    parser = argparse.ArgumentParser(
        prog='ActiveLearning',add_help=True)

    group = parser.add_argument_group('Dataset', 'Dataset description such as the dataset, validation size,test size, initial labeled size')

    group.add_argument('--experiment_name',type=str,default="AL_HYPERPARAM",help="experimennt name for MLFLOW. All runs will be logged to this experiment.")
    group.add_argument('--result_path',type=str,default=osp.join("results","hyperparam"),help="Directory to save the results of the hyperparam search")
    group.add_argument('--check_if_completed', action=argparse.BooleanOptionalAction,default=False,help='Do you not want to repeat already ran experiments in the folder? --check_if_completed for yes --no-check_if_completed for no')

    args = parser.parse_args()

    return args




if __name__ == "__main__":
    args = parse_args()

    os.makedirs("logs",exist_ok=True)
    #  set to true if you do not want to run the experiments again when results exist
    check_if_completed = args.check_if_completed

    query_methods = ['entropy','max_model_change','fivr','error_reduction','max_error_reduction']
    datasets =  ["ionosphere","user_knowledge","heart_disease","haberman","breast_cancer","parkinsons","acute","vehicle","pima","planning","sonar","diabetes"]
    lam = 0.01
    query_perc = 0.7


    experiments = []

    for query_method in query_methods:
        for dataset in datasets:
            experiments.append(f"--dataset={dataset} "
                           f"--query_method_name={query_method} "
                           f"--query_perc={query_perc} "
                           f"--num_trials=20 "
                           f"--n_initial=2 "
                           f"--test_size=0.2 "
                           f"--lam={lam} "
                           f"--mlflow_track "
                           f"--result_path={args.result_path} "
                           f"--experiment_name={args.experiment_name} ")

    for exp_args in experiments:
        #  write  code to check if the expeeriment  has run based on the hyperparam foldere structure
        #  if the experiment has been run, skip
        if check_if_completed:
            folder_path = check_experiment_completed(exp_args,args.result_path)
            if osp.exists(folder_path) and len(os.listdir(folder_path)) > 0:
                # print(f"Experiment {exp_args} already exists. Skipping...")
                continue

        file_full = f"python main_hyperparam.py {exp_args}"
        print(f"sbatch execute.bash '{file_full}'")
        Popen(f"sbatch execute.bash '{file_full}'",shell=True)
        sleep(0.1)