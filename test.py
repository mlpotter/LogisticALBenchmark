import warnings

warnings.filterwarnings('ignore')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from src.datasets import load_data
from src.models import BayesianLogisticRegression
from tqdm import tqdm
from time import time

import numpy as np
import random

def test_real(dataset,lam,n_init,n_points,seed=123):
    np.random.seed(seed)
    random.seed(seed)
    X,y = load_data(dataset)
    print(f"Dataset {dataset} Size: ",X.shape)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=seed,stratify=y)


    model = BayesianLogisticRegression(input_dim=X_train.shape[1],lam=lam,seed=seed)

    print("="*20,f"Bayesian LR Classifier Random Prediction {dataset}","="*20)

    y_pred = model.predict(X_test,mode='posterior')
    print(classification_report(y_test,y_pred))


    start_time = time()
    model.fit(X_train,y_train)
    end_time = time()

    model.jac_unit_debug(X,y)
    model.hess_unit_debug(X,y)

    print("Model Fit Time: ",end_time-start_time)
    print("="*20,f"Bayesian LR Classifier Prediction {dataset}","="*20)

    y_pred = model.predict(X_test,mode='classifier')
    print(classification_report(y_test,y_pred))

    print("="*20,f"Bayesian LR Posterior Prediction {dataset}","="*20)

    y_pred = model.predict(X_test,mode="posterior")
    print(classification_report(y_test,y_pred))

    # print(model.w.ravel())
    plt.imshow(model.w_cov,origin="lower")
    plt.colorbar()
    plt.title(f"{dataset} Covariance Matrix")
    plt.show(block=False)

    print("="*20,f"Logistic Regression {dataset}","="*20)

    lr = LogisticRegression().fit(X_train,y_train)

    y_pred = lr.predict(X_test)
    print(classification_report(y_test,y_pred))
    # print(lr.coef_)
    # print(lr.intercept_)

    print("="*20,f"Online Learning {dataset}","="*20)
    model = BayesianLogisticRegression(input_dim=X_train.shape[1],lam=lam,seed=seed)

    X_rest,X_init,y_rest,y_init = train_test_split(X_train,y_train,test_size=n_init,random_state=seed,stratify=y_train.ravel())

    model.fit(X_init,y_init)
    y_pred = model.predict(X_test,mode="posterior")
    print(f"Initial Model on n={n_init} samples {dataset}")
    print(classification_report(y_test,y_pred))
    accuracy_scores = [accuracy_score(y_test,y_pred)]
    data_sizes = [n_init]
    for i in tqdm(range(n_points)): #X_rest.shape[0])):
        model.fit(X_rest[[i],...], y_rest[[i],...])
        # print(model.w)
        accuracy_scores.append(model.score(X_test,y_test))
        data_sizes.append(n_init+i+1)

    print(f"Model After recursion {dataset}")
    y_pred = model.predict(X_test,mode="posterior")
    print(classification_report(y_test,y_pred))

    plt.figure()
    plt.plot(data_sizes,accuracy_scores)
    plt.title(f"{dataset} UPDATE POSTERIOR")
    plt.show(block=False)

    # lam = 1e-3
    print("="*20,f"Fit Model Repeatedly {dataset}","="*20)

    lr = LogisticRegression()
    lr_bayesian = BayesianLogisticRegression(input_dim=X_train.shape[1],lam=lam,seed=seed)

    X_rest,X_init,y_rest,y_init = train_test_split(X_train,y_train,test_size=n_init,random_state=seed,stratify=y_train.ravel())

    lr.fit(X_init,y_init)
    lr_bayesian.fit(X_init,y_init)

    accuracy_scores_lr = [lr.score(X_test,y_test)]
    accuracy_scores_bayesian = [lr_bayesian.score(X_test,y_test)]

    data_sizes = [n_init]
    for i in tqdm(range(n_points)): #X_rest.shape[0])):
        X_init,y_init = jnp.vstack((X_init,X_rest[[i],...])),jnp.vstack((y_init,y_rest[[i],...]))

        lr_bayesian.refit(X_init,y_init)
        lr.fit(X_init, y_init)
        # print(model.w)
        accuracy_scores_lr.append(lr.score(X_test,y_test))
        accuracy_scores_bayesian.append(lr_bayesian.score(X_test,y_test))

        data_sizes.append(n_init+i+1)

    print(f"Model After recursion {dataset}")
    y_pred = lr_bayesian.predict(X_test,mode="posterior")
    print(classification_report(y_test,y_pred))

    plt.figure()
    plt.plot(data_sizes,accuracy_scores_lr,label="LR")
    plt.plot(data_sizes,accuracy_scores_bayesian,'--',label="Bayesian LR")
    plt.title(f"{dataset} REFIT EVERYTIME")
    plt.legend()
    plt.show()

def test_synth(dataset,lam,n_init,n_points,seed=123):
    np.random.seed(seed)
    random.seed(seed)

    w_true,X,y = load_data(dataset)
    print(f"Dataset {dataset} Size: ",X.shape)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=seed,stratify=y)


    num_classes = len(np.unique(y))


    model = BayesianLogisticRegression(input_dim=X_train.shape[1],lam=lam,seed=seed)

    print("="*20,f"Bayesian LR Classifier Random Prediction {dataset}","="*20)

    y_pred = model.predict(X_test,mode='posterior')
    print(classification_report(y_test,y_pred))


    start_time = time()
    model.fit(X_train,y_train)
    end_time = time()

    model.jac_unit_debug(X,y)
    model.hess_unit_debug(X,y)

    print("Model Fit Time: ",end_time-start_time)
    print("="*20,f"Bayesian LR Classifier Prediction {dataset}","="*20)

    y_pred = model.predict(X_test,mode='classifier')
    print(classification_report(y_test,y_pred))

    print("="*20,f"Bayesian LR Posterior Prediction {dataset}","="*20)

    y_pred = model.predict(X_test,mode="posterior")
    print(classification_report(y_test,y_pred))

    # print(model.w.ravel())
    plt.imshow(model.w_cov,origin="lower")
    plt.colorbar()
    plt.title(f"{dataset} Covariance Matrix")
    plt.show(block=False)

    print("="*20,f"Logistic Regression {dataset}","="*20)

    lr = LogisticRegression().fit(X_train,y_train)

    y_pred = lr.predict(X_test)
    print(classification_report(y_test,y_pred))

    # write code in matplotlib to have side by side barplots for three sets of arrays
    plt.figure()
    plt.bar(np.arange(len(w_true.ravel())),w_true.ravel(),width=0.2,label="True")
    plt.bar(np.arange(len(model.w.ravel()))+0.2,model.w.ravel(),width=0.2,label="Bayesian LR")
    lr_weights = np.hstack((lr.intercept_.ravel(),lr.coef_.ravel()))
    plt.bar(np.arange(len(lr_weights.ravel()))+0.4,lr_weights.ravel(),width=0.2,label="LR")
    plt.legend()
    plt.title(f"{dataset} Weights")
    plt.xlabel("Weights")
    plt.show()


if __name__ == "__main__":
    seed = 555

    if True:
        lam = 5e-1
        n_init = 5
        n_points = 50
        test_real("ionosphere",lam,n_init,n_points,seed)

    if False:
        lam = 10
        n_init = 5
        n_points = 50
        test_synth("synth_1",lam,n_init,n_points,seed)