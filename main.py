# from sklearnex import patch_sklearn
# patch_sklearn()
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 5) # not sufficient

import warnings
warnings.filterwarnings('ignore')
# import warnings
# warnings.filterwarnings('ignore')

# import logging as lg


# lg.getLogger("sklearnex").setLevel("INFO")

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
import os.path as osp
import contextlib
from functools import partial

import mlflow
import pprint

from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# from sklearn.linear_model import LogisticRegression
import numpy as np
import random
from tqdm import tqdm
import time

import torch
torch.use_deterministic_algorithms(True)

from src.acquisition_fns import *
from src.datasets import load_data,create_datasets
from src.plotting import plot_scores,plot_times
from src.utils import al_experiment,baseline_performance


import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog='ActiveLearning',add_help=True)


    group1 = parser.add_argument_group('Dataset', 'Dataset description such as the dataset, validation size,test size, initial labeled size')
    group1.add_argument('--dataset', default="iris", type=str,help='Dataset choice')
    group1.add_argument('--n_initial', default=25,type=int, help='Initial training dataset size')
    group1.add_argument('--test_size', default=0.15, type=float, help='testing percentage size (whole data)')

    group2 = parser.add_argument_group('AL', 'Active Learning information')
    group2.add_argument('--query_perc', default=0.5,type=float, help='The percentage of the poool to use')
    group2.add_argument('--num_trials', default=25,type=int, help='Number of monte carlo trials to evaluate AL method')

    group3 = parser.add_argument_group('Model', 'Hyperparameters of Neural Network')
    group3.add_argument('--lam', default=1e-3, type=float,help='Regularization for the prior on Logistic Regression weights')

    group3 = parser.add_argument_group('Experiment', 'Experiment design and tracking settings')
    group3.add_argument('--experiment_name',type=str,default="AL_BENCHMARK",help="experimennt name for MLFLOW. All runs will be logged to this experiment.")
    group3.add_argument('--mlflow_track', action=argparse.BooleanOptionalAction,default=False,help='Do you wish to track experiments with mlflow? --mlflow_track for yes --no-mlflow_track for no')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    seed = int(0)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    args = parse_args()

    tmp_save_folder = osp.join("results","temp",args.dataset)
    os.makedirs(tmp_save_folder,exist_ok=True)

    # set the mlflow experiment to save the runs to
    if args.mlflow_track:
        mlflow.set_experiment(args.experiment_name)

    # context manager for mlflow or Null context manager if not tracking
    ctx = mlflow.start_run(run_name=f"{args.dataset}") if args.mlflow_track else contextlib.suppress()

    args.model_template = LogisticRegression

    with (ctx):
        # random seeds to use for each monte carlo realization of AL experiment
        seeds = np.random.randint(0,1000,args.num_trials)

        # load the dataset as numpy arrays
        X,y = load_data(args.dataset)

        args.n_input = X.shape[-1]
        args.n_class = len(np.unique(y))


        # split the data into a data subsets as a dictionary
        # {"init": (X_init,y_init), "test": (X_test,y_test), "pool": (X_pool,y_pool)}
        data_dict = create_datasets(X, y, test_size=args.test_size, n_initial=args.n_initial,
                                    seed=123)

        # get the sizes of the datasets that will be used for the AL experiment
        print("Dataset: ",args.dataset)
        print("Initial Size: ",data_dict['init'][0].shape)
        print("Test Size: ",data_dict['test'][0].shape)
        print("Pool Size: ",data_dict['pool'][0].shape)

        args.n_queries  = int(np.floor(args.query_perc * data_dict['pool'][0].shape[0]))
        args.query_size = 1

        pprint.pprint(vars(args), width=1)

        # ------------------- BASELINE MODEL ------------------------- #

        args.query_method_name = None
        baseline_accuracy = 0
        for seed in seeds:
            data_dict = create_datasets(X, y,  test_size=args.test_size, n_initial=args.n_initial,
                                        seed=seed)
            classifier, baseline_accuracy_i = baseline_performance(LogisticRegression(C=1/args.lam,n_jobs=None,random_state=123),
                                                                 data_dict,args)
            baseline_accuracy += baseline_accuracy_i
            del classifier,baseline_accuracy_i

        baseline_accuracy /= args.num_trials
        print("Baseline LR Accuracy fit on pool and init data together: ",baseline_accuracy)


        # a dictionary of the acquisition functions to be used for selecting the new data points for the oracle to label
        uncertainty_dict = {
        # 'two_error_reduction': two_error_reduction,
        'random': random_partition,
        'entropy': entropy,
        'error_reduction': error_reduction,
        'max_error_reduction': max_error_reduction,
        'fivr': fisher_information_variance_reduction,
        'max_model_change': max_model_change,
        }

        # a score matrix of dim ->  # acq methods x # mc trials x # al rounds + 1)
        # score is currently accuracy on the test data
        score_matrix_test = np.zeros((len(uncertainty_dict),args.num_trials,args.n_queries+1))
        alc_matrix_test = np.zeros((args.num_trials,))

        # a time matrix of dim ->  # acq methods x # mc trials x # al rounds + 1)
        # time is currently how long a AL query round takes
        time_matrix =  np.zeros((len(uncertainty_dict),args.num_trials,args.n_queries))

        # iterate through each AL acquisition strategy
        for query_idx,query_strategy in enumerate(uncertainty_dict.keys()):

            print(f"\n Query Method: {query_strategy}")
            args.query_method_name = query_strategy

            # Monte carlo trials for each AL acquisition strategy
            # pbar = tqdm(total=args.num_trials*(args.n_queries+1))
            pbar = tqdm(enumerate(seeds))

            for idx,seed in pbar: #enumerate(seeds):
                if idx != 0:
                    pbar.set_description("Baseline Accuracy {:.2f} , Model {} Final Accuracy: {:.2f}, ALC {:.2f}".format(baseline_accuracy,idx-1,scores_test.ravel()[-1],alc_matrix_test[idx-1]))
                else:
                    pbar.set_description("Baseline Accuracy: {:.2f}".format(baseline_accuracy))

                time.sleep(0.01)
                seed = int(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)

                # create the data dictionary as above
                data_dict = create_datasets(X, y,test_size=args.test_size, n_initial=args.n_initial, seed=seed)

                # create the classifier in SKORCH (sklearn like wrapper which has .fit(), .predict() , ... so on)
                classifier = LogisticRegression(C=1/args.lam,n_jobs=None,random_state=123)

                # perform the AL experiment (do multiple AL rounds, use acquistion function to acquire new data pts, refit model, and so on)
                learner,scores_test,times = al_experiment(classifier,data_dict,
                                                         query_strategy=uncertainty_dict[query_strategy],
                                                         seed=seed,**vars(args))

                # update the time matrix for the monte carlo trial of the AL acquisition strategy
                time_matrix[query_idx,idx] = times

                # update the score matrix for the monte carlo trial of the AL acquisition strategy
                score_matrix_test[query_idx,idx] = scores_test
                alc_matrix_test[idx] = np.trapz(scores_test, dx=1 / args.n_queries)
                # # empty memory of GPU for useless tensors on GPU

            print("Mean ALC: {:.3f} , Std ALC: {:.3f}".format(alc_matrix_test.mean(),alc_matrix_test.std()))

        # plot the accuracy curves (accuracy versus number of data points, for each AL acquisition methods)
        fig_scores,ax_scores = plot_scores(score_matrix_test,baseline_accuracy,uncertainty_dict,args)
        ax_scores.set_title(f"Test {args.dataset} scores")
        fig_scores.savefig(osp.join(tmp_save_folder,"scores_test.png"))

        # plot the timing curves (time in seconds versus number of data points, for each AL acquisition methods)
        fig_times,ax_times = plot_times(time_matrix,uncertainty_dict,args)
        fig_times.savefig(osp.join(tmp_save_folder,"times.png"))

        if args.mlflow_track:
            # ---------------- MLFLOW LOGGING --------------------- #
            args.model_template = None

            # log all the argparse arguments (because why not?)
            mlflow.log_params(vars(args))
            mlflow.log_metrics({"accuracy_baseline": baseline_accuracy,
                                "accuracy_test":score_matrix_test[:,-1].mean(),
                                "alc_test":alc_matrix_test.mean(),
                                })

            np.save(osp.join(tmp_save_folder,"score_matrix_test.npy"),score_matrix_test)
            np.save(osp.join(tmp_save_folder,"alc_matrix_test.npy"),alc_matrix_test)
            np.save(osp.join(tmp_save_folder,"matrix_time.npy"),time_matrix)

            mlflow.log_text(np.array2string(seeds, separator=","), "seeds.txt")


            mlflow.log_artifacts(tmp_save_folder)

            # mlflow.log_param("random_seeds", args.random_seed)


