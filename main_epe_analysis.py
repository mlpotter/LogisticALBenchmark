
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
import os.path as osp
import contextlib
from functools import partial

import mlflow
import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import numpy as np
import random
from tqdm import tqdm
import time
from copy import deepcopy

import torch
torch.use_deterministic_algorithms(True)

from src.acquisition_fns import *
from src.datasets import load_data,create_datasets
from src.plotting import plot_confidence_interval
from src.utils import baseline_performance


import argparse

def error_reduction(model,X_L,y_L,X_U,y_U,n_instances=1,**kwargs):
    """
    Error reduction acquisition function
    """
    model_template = kwargs['model_template']

    N,F = X_U.shape

    scores = np.zeros((N,))

    pY_given_L = model.predict_proba(X_U)

    model_L_plus = model_template(C=1/kwargs['lam'],random_state=123,n_jobs=None)

    for i in range(N):
        x = X_U[i,:]
        x = x.reshape(1,F)
        py_given_L = pY_given_L[i,:]

        score_temp = 0

        for c in range(2):
            y = np.array([[c]])

            model_L_plus.fit(np.vstack((X_L,x)),np.vstack((y_L,y)))

            score_temp = score_temp +  py_given_L[c] * entr(model_L_plus.predict_proba(X_U),axis=-1).sum()

        scores[i] = score_temp

    # take the minimum of the expected error
    scores = -scores

    idx = np.argpartition(scores, -n_instances)[-n_instances:]

    return idx, -scores.mean()

def al_experiment(classifier, data,  query_strategy, n_queries=10, query_size=100,**kwargs):
    # print(f"\n Query Method: {kwargs.get('query_method_name')}")
    X_train,y_train = data["init"]
    X_test,y_test = data['test']
    X_pool,y_pool = data['pool']

    # pbar = kwargs.get("pbar",None)


    scores_test = np.zeros(n_queries + 1)
    epe_pool = np.zeros(n_queries)
    times = np.zeros(n_queries)


    classifier.fit(X_train,y_train)
    scores_test[0] = classifier.score(X_test, y_test)


    # the active learning loop
    for idx in range(n_queries):
        # print(f"AL Round {idx+1}")
        cycle_start  = time.time()

        query_idx,epe_score = query_strategy(classifier,X_train,y_train,X_pool,y_pool,n_instances=query_size,**kwargs)

        X_label = X_pool[query_idx]
        y_label = y_pool[query_idx]

        X_train = np.vstack((X_train,X_label))
        y_train = np.vstack((y_train,y_label))

        classifier.fit(
            X=X_train, y=y_train
        )

        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        score_i =  classifier.score(X_test, y_test)
        scores_test[idx + 1] = float(score_i)
        epe_pool[idx] = epe_score

        # if pbar is not None:
        #     pbar.set_description("Model {} Final Accuracy: {:.2f}".format(idx,score_i))
        #     pbar.update(1)

        cycle_end  = time.time()

        times[idx] = cycle_end-cycle_start
    return classifier,scores_test, times,epe_pool

def parse_args():
    parser = argparse.ArgumentParser(
        prog='ActiveLearning',add_help=True)


    group1 = parser.add_argument_group('Dataset', 'Dataset description such as the dataset, validation size,test size, initial labeled size')
    group1.add_argument('--dataset', default="haberman", type=str,help='Dataset choice')
    group1.add_argument('--n_initial', default=25,type=int, help='Initial training dataset size')
    group1.add_argument('--test_size', default=0.15, type=float, help='testing percentage size (whole data)')

    group2 = parser.add_argument_group('AL', 'Active Learning information')
    group2.add_argument('--query_perc', default=0.5,type=float, help='The percentage of the poool to use')
    group2.add_argument('--num_trials', default=25,type=int, help='Number of monte carlo trials to evaluate AL method')

    group3 = parser.add_argument_group('Model', 'Hyperparameters of Neural Network')
    group3.add_argument('--lam', default=1e-3, type=float,help='Regularization for the prior on Logistic Regression weights')

    group3 = parser.add_argument_group('Experiment', 'Experiment design and tracking settings')
    group3.add_argument('--experiment_name',type=str,default="AL_HYPERPARAM",help="experimennt name for MLFLOW. All runs will be logged to this experiment.")
    group3.add_argument('--result_path',type=str,default=osp.join("results","hyperparam"),help="Directory to save the results of the hyperparam search")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    seed = int(0)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    args = parse_args()
    args.query_method_name = "epe"

    save_folder = osp.join(args.result_path,args.dataset,args.query_method_name,f"lam_{args.lam}")
    os.makedirs(save_folder,exist_ok=True)


    args.model_template = LogisticRegression

    # random seeds to use for each monte carlo realization of AL experiment
    seeds = np.random.randint(0,1000,args.num_trials)

    # load the dataset as numpy arrays
    X,y = load_data(args.dataset)

    args.n_input = X.shape[-1]
    args.n_class = len(np.unique(y))


    # split the data into a data subsets as a dictionary
    # {"init": (X_init,y_init), "val": (X_val,y_val), "test": (X_test,y_test), "pool": (X_pool,y_pool)}
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
    query_method_name = deepcopy(args.query_method_name)
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

    args.query_method_name = query_method_name

    # a dictionary of the acquisition functions to be used for selecting the new data points for the oracle to label
    query_method = error_reduction

    # a score matrix of dim ->  # acq methods x # mc trials x # al rounds + 1)
    # score is currently accuracy on the test data
    score_matrix_test = np.zeros((args.num_trials,args.n_queries+1))
    alc_matrix_test = np.zeros((args.num_trials,))
    epe_matrix_test = np.zeros((args.num_trials,args.n_queries))

    # a time matrix of dim ->  # acq methods x # mc trials x # al rounds + 1)
    # time is currently how long a AL query round takes
    time_matrix =  np.zeros((args.num_trials,args.n_queries))


    print(f"\n Query Method: {args.query_method_name}")


    # Monte carlo trials for each AL acquisition strategy
    pbar = tqdm(enumerate(seeds))
    for idx,seed in pbar:
        if idx != 0:
            pbar.set_description("Baseline Accuracy {:.2f} , Model {} Final Accuracy: {:.2f}, ALC {:.2f}".format(baseline_accuracy,idx - 1,scores_test.ravel()[ -1],
                                                                                                alc_matrix_test[ idx - 1]))

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
        classifier = LogisticRegression(C=1 / args.lam, n_jobs=None, random_state=123)

        # perform the AL experiment (do multiple AL rounds, use acquistion function to acquire new data pts, refit model, and so on)
        learner,scores_test,times,epes_test = al_experiment(classifier,data_dict,
                                                 query_strategy=query_method,
                                                 seed=seed,**vars(args))

        # update the time matrix for the monte carlo trial of the AL acquisition strategy
        time_matrix[idx] = times

        # update the score matrix for the monte carlo trial of the AL acquisition strategy
        score_matrix_test[idx] = scores_test
        alc_matrix_test[idx] = np.trapz(scores_test, dx=1 / args.n_queries)
        epe_matrix_test[idx] = epes_test

    print("Mean ALC: {:.3f} , Std ALC: {:.3f}".format(alc_matrix_test.mean(), alc_matrix_test.std()))

    # plot the accuracy curves (accuracy versus number of data points, for each AL acquisition methods)
    fig, ax = plt.subplots()
    data_sizes = (np.arange(args.n_queries + 1)) * args.query_size + args.n_initial
    ax.plot(data_sizes, [baseline_accuracy] * len(data_sizes), label="baseline", color="black", linestyle="--")
    plot_confidence_interval(data_sizes, score_matrix_test, ax, color='b',
                             label=f"{args.query_method_name} Test")

    ax.set_xlabel("# Datapoints")
    ax.set_ylabel("Accuracy")
    fig.legend(loc='lower right')
    fig.tight_layout()
    ax.set_title(f"{args.dataset} scores")
    fig.savefig(osp.join(save_folder,"scores_test.png"))


    # plot the epe curves (epe versus number of data points, for each AL acquisition methods)
    fig, ax = plt.subplots()
    data_sizes = (np.arange(args.n_queries)) * args.query_size + args.n_initial
    plot_confidence_interval(data_sizes, epe_matrix_test, ax, color='b',
                             label=f"{args.query_method_name} Test")

    ax.set_xlabel("# Datapoints")
    ax.set_ylabel("EPE")
    fig.legend(loc='lower right')
    fig.tight_layout()
    ax.set_title(f"{args.dataset} scores")
    fig.savefig(osp.join(save_folder,"epe_test.png"))


    # plot the epe curves (epe versus number of data points, for each AL acquisition methods)
    from scipy.stats import pearsonr

    fig, ax = plt.subplots()
    data_sizes = (np.arange(args.n_queries)) * args.query_size + args.n_initial

    pearson_corr = np.array([pearsonr(epe_matrix_test[:,i],score_matrix_test[:,i])[0] for i in range(args.n_queries)])

    ax.plot(data_sizes,pearson_corr,label="Pearson Correlation",color="black",linestyle="--")

    ax.set_xlabel("# Datapoints")
    ax.set_ylabel("Pearson Correlation")
    fig.legend(loc='lower right')
    fig.tight_layout()
    ax.set_title(f"{args.dataset} EPE and Accuracy Correlation")
    fig.savefig(osp.join(save_folder,"epe_acc_correlation.png"))


    fig, ax = plt.subplots()


    ax.scatter(epe_matrix_test.ravel(),score_matrix_test[:,:-1].ravel(),label="scatter",c="blue",alpha=0.5)

    ax.set_xlabel("EPE")
    ax.set_ylabel("Accuracy")
    fig.legend(loc='lower right')
    fig.tight_layout()
    ax.set_title(f"{args.dataset} EPE and Accuracy Correlation")
    fig.savefig(osp.join(save_folder,"epe_acc_correlation_scatter.png"))


    np.save(osp.join(save_folder,"score_matrix_test.npy"),score_matrix_test)
    np.save(osp.join(save_folder,"alc_matrix_test.npy"),alc_matrix_test)
    np.save(osp.join(save_folder,"matrix_time.npy"),time_matrix)
    np.save(osp.join(save_folder,"data_sizes.npy"),data_sizes)

