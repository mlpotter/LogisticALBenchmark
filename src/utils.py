# from sklearnex import patch_sklearn
# patch_sklearn()
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 5) # not sufficient

# import warnings
# warnings.filterwarnings('ignore')
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from copy import deepcopy




def baseline_performance(classifier,data_dict,args):


    base_X_train = np.vstack((data_dict['init'][0],data_dict['pool'][0]))
    base_y_train = np.vstack((data_dict['init'][1],data_dict['pool'][1]))

    classifier.fit(base_X_train,base_y_train)
    base_y_pred = classifier.predict(data_dict['test'][0])
    accuracy = accuracy_score(data_dict['test'][1],base_y_pred)

    del base_X_train,base_y_train,base_y_pred

    return classifier,accuracy

def al_experiment(classifier, data,  query_strategy, n_queries=10, query_size=100,**kwargs):
    # print(f"\n Query Method: {kwargs.get('query_method_name')}")
    X_train,y_train = data["init"]
    X_test,y_test = data['test']
    X_pool,y_pool = data['pool']

    # pbar = kwargs.get("pbar",None)


    scores_test = np.zeros(n_queries + 1)
    times = np.zeros(n_queries)


    classifier.fit(X_train,y_train)
    scores_test[0] = classifier.score(X_test, y_test)


    # the active learning loop
    for idx in range(n_queries):
        # print(f"AL Round {idx+1}")
        cycle_start  = time.time()

        query_idx = query_strategy(classifier,X_train,y_train,X_pool,y_pool,n_instances=query_size,**kwargs)

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

        # if pbar is not None:
        #     pbar.set_description("Model {} Final Accuracy: {:.2f}".format(idx,score_i))
        #     pbar.update(1)

        cycle_end  = time.time()

        times[idx] = cycle_end-cycle_start
    return classifier,scores_test, times
