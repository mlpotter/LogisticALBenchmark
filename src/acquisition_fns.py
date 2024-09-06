import numpy as np
from scipy.stats import entropy as entr


# UNIFORM SAMPLER
def random_partition(model,X_L,y_L,X_U,y_U,n_instances=1,**kwargs):
    """"
    random partition acquisition function
    """
    # random sampling acquisition function...
    N_U = X_U.shape[0]
    idx = np.random.choice(N_U, n_instances)
    return idx

def entropy(model,X_L,y_L,X_U,y_U,n_instances=1,**kwargs):
    """
    Entropy acquisition function
    """

    py = model.predict_proba(X_U)

    scores = entr(py,axis=-1)

    idx = np.argpartition(scores, -n_instances)[-n_instances:]

    return idx

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

    return idx

def max_error_reduction(model,X_L,y_L,X_U,y_U,n_instances=1,**kwargs):
    """
    Max Error reduction acquisition function
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

        y = np.argmin(py_given_L,axis=-1).reshape(1,1)

        model_L_plus.fit(np.vstack((X_L,x)),np.vstack((y_L,y)))

        scores[i] = entr(model_L_plus.predict_proba(X_U),axis=-1).sum()

    # take the minimum of the expected error
    scores = -scores

    idx = np.argpartition(scores, -n_instances)[-n_instances:]

    return idx

def max_model_change(model,X_L,y_L,X_U,y_U,n_instances=1,**kwargs):
    """
    Max Error reduction acquisition function
    """
    model_template = kwargs['model_template']

    N,F = X_U.shape


    pY_given_L = model.predict_proba(X_U)

    scores = (2*np.linalg.norm(X_U,axis=-1)) * (pY_given_L.prod(axis=-1))


    idx = np.argpartition(scores, -n_instances)[-n_instances:]

    return idx


def fisher_information_variance_reduction(model,X_L,y_L,X_U,y_U,n_instances=1,**kwargs):
    """
    Max Error reduction acquisition function
    """

    N_U,F = X_U.shape

    scores = np.zeros((N_U,))

    pY_given_L = model.predict_proba(X_U)

    # compute I_U
    I_U  = 0
    for i in range(N_U):
        # sigma_i * (1-sigma_i) * x_i x_i^T
        I_U = I_U + np.outer(X_U[i,:],X_U[i,:])*pY_given_L[i,:].prod(axis=-1)

    I_U = 1/N_U * I_U + kwargs['lam'] * np.eye(F)

    # write the computation of I_U in vectorized form
    # I_U = kwargs['lam'] * np.eye(F) + (X_U.T * pY_given_L.prod(axis=-1)).dot(X_U)

    # compute the score I_x(w)^-1 I_U(w) for each x in U
    for i in range(N_U):
        I_X = np.outer(X_U[i,:],X_U[i,:])*pY_given_L[i,:].prod(axis=-1) +  kwargs['lam'] * np.eye(F)
        ## I_x(w)^-1 I_U(w) future output variance once x has been labeled
        scores[i] = np.trace(np.linalg.inv(I_X)@I_U)

    # take the minimum of fivr
    scores = -scores

    idx = np.argpartition(scores, -n_instances)[-n_instances:]

    return idx