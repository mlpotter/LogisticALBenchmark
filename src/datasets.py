
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml, load_wine, load_breast_cancer,load_iris,make_moons,make_circles
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import StratifiedShuffleSplit

import random
import numpy as np
import pandas as pd
import torch


def load_data(dataset="mnist"):
    std_scaler = StandardScaler()

    if dataset == "user_knowledge":
        X, y = fetch_openml("user-knowledge", return_X_y=True)
        y = y.map({"1": "high", "3": "high", "2": "low", "5": "low", "4": "low"})
        y = pd.factorize(y)[0]
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset == "diabetes":
        X, y = fetch_openml("diabetes", return_X_y=True)
        y = pd.factorize(y)[0]
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset =="acute":
        X, y = fetch_openml("acute-inflammations", return_X_y=True)
        y = pd.factorize(y)[0]
        X.V2 = pd.factorize(X.V2)[0]
        X.V3 = pd.factorize(X.V3)[0]
        X.V4 = pd.factorize(X.V4)[0]
        X.V5 = pd.factorize(X.V5)[0]
        X.V6 = pd.factorize(X.V6)[0]

        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset =="parkinsons":
        X, y = fetch_openml("parkinsons", return_X_y=True)
        y = pd.factorize(y)[0]
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset == "vehicle":
        X, y = fetch_openml("vehicle", return_X_y=True)
        y = pd.factorize(y)[0]
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset == "ionosphere":
        X, y = fetch_openml("ionosphere", return_X_y=True)
        y = pd.factorize(y)[0]
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))


    elif dataset == "haberman":
        X, y = fetch_openml("haberman",return_X_y=True)
        y = pd.factorize(y)[0]
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset == "heart_disease":
        X,_ = fetch_openml("Heart-Disease-Prediction",return_X_y=True)
        y = X.pop("Heart_Disease")
        y = pd.factorize(y)[0]
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset == "pima":
        X, y = fetch_openml("Diabetes_Dataset", return_X_y=True)
        y = X.pop('Outcome').values
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset == "planning":
        X, y = fetch_openml("planning-relax", return_X_y=True)
        y = pd.factorize(y)[0]
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))

    elif dataset == "sonar":
        X, y = fetch_openml("sonar", return_X_y=True)
        y = pd.factorize(y)[0]
        X = X.values
        X = std_scaler.fit_transform(X)
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))


    elif dataset == "synth1":
        N = 200
        mu1 = np.array([1,1])
        mu2 = np.array([-1,-1])
        cov1 = np.eye(2)
        cov2 = np.eye(2)
        X1 = np.random.multivariate_normal(mu1,cov1,N)
        X2 = np.random.multivariate_normal(mu2,cov2,N)
        y1 = np.zeros((N,1))
        y2 = np.ones((N,1))

        X = np.vstack((X1,X2))
        y = np.vstack((y1,y2))

        # X = std_scaler.fit_transform(X)


        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y,marker='.')
        plt.show()


        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))
    
    elif dataset == "synth2":
        N_middle = 150
        mu1 = np.array([0,4])
        mu2 = np.array([0,-4])

        cov1 = np.array([[2,0],[0,1]])
        cov2 = np.array([[2,0],[0,1]])

        X2_middle = np.random.multivariate_normal(mu1,cov1,N_middle)
        X1_middle = np.random.multivariate_normal(mu2,cov2,N_middle)

        N_other=25
        mu_br = np.array([8.5,-7.5])
        mu_tr = np.array([8.5,7.5])

        X1_br = np.random.multivariate_normal(mu_br,cov1,N_other)
        X1_tr = np.random.multivariate_normal(mu_tr,cov1,N_other)

        mu_bl = np.array([-8.5,-7.5])
        mu_tl = np.array([-8.5,7.5])

        X2_bl = np.random.multivariate_normal(mu_bl,cov2,N_other)
        X2_tl = np.random.multivariate_normal(mu_tl,cov2,N_other)


        y1 = np.zeros((N_middle+2*N_other,1))
        y2 = np.ones((N_middle+2*N_other,1))

        X = np.vstack((X1_middle,X1_tr,X1_br,X2_middle,X2_tl,X2_bl))
        y = np.vstack((y1,y2))

        X = std_scaler.fit_transform(X)

        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(X[:, 0], X[:, 1], c=y,marker='.')
        # plt.show()
        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))
    
    elif dataset == "synth3":
        N1 = 134
        N2 = 133
        N3 = 133

        mu1 = np.array([1.5,1.5])
        mu2 = np.array([1,-1])
        mu3 = np.array([-1.75,-1])

        cov1 = np.array([[1,-0.8],[-.8,1]])
        cov2 = np.array([[1,0.8],[.8,1]])
        cov3 = np.array([[1,-0.8],[-.8,1]])

        X1 = np.random.multivariate_normal(mu1,cov1,N1)
        X2 = np.random.multivariate_normal(mu2,cov2,N2)
        X3 = np.random.multivariate_normal(mu3,cov3,N3)

        y1 = np.zeros((N1+N3,1))
        y2 = np.ones((N2,1))

        X = np.vstack((X1,X3,X2))
        y = np.vstack((y1,y2))

        X = std_scaler.fit_transform(X)


        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y,marker='.')
        plt.show()

        return np.asarray(X.astype(np.float64)), np.asarray(y.astype(np.int64).reshape(-1, 1))


    else:
        raise Exception(f"{dataset} Not a valid dataset choice.")


def create_datasets(X, y, test_size=0.2, n_initial=100, seed=123):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # read training data


    # split the data into train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                        stratify=y)  # [:50000], X[50000:], y[:50000], y[50000:]



    if n_initial > 2:
        # easier to use train_test_split to create the initial dataset and the pool dataset, such that stratification remains true
        X_pool, X_initial, y_pool, y_initial = train_test_split(X_train, y_train, test_size=n_initial, random_state=seed,
                                                                stratify=y_train.ravel())

    else:
        # Separate indices of each class
        class_0_indices = np.where(y_train == 0)[0]
        class_1_indices = np.where(y_train == 1)[0]

        test_indices = np.concatenate([
            np.random.choice(class_0_indices, size=1, replace=False),
            np.random.choice(class_1_indices, size=1, replace=False)
        ])

        # Remaining indices for the training set
        train_indices = np.array([i for i in range(len(y_train)) if i not in test_indices])

        # Split the data
        X_pool, X_initial = X_train[train_indices], X_train[test_indices]
        y_pool, y_initial = y_train[train_indices], y_train[test_indices]


    data_dict = {"init": (X_initial, y_initial),
                 "pool": (X_pool, y_pool),
                 "test": (X_test, y_test)}

    return data_dict