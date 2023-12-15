#!/usr/bin/env python

import logging

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("Example Kinships")

import numpy as np
from numpy import dot, array, zeros, setdiff1d
from numpy.linalg import norm
from numpy.random import shuffle
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from graph_package.src.rescal_als.rescal import rescal_als
from copy import deepcopy


def predict_rescal_als(T, config={}):
    A, R, _, _, _ = rescal_als(
        T,
        config["rank"],
        init="nvecs",
        conv=1e-5,
        lambda_A=config["reg"],
        lambda_R=config["reg"],
        maxIter=1000,
    )
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P


def normalize_predictions(P, e, k):
    for a in range(e):
        for b in range(e):
            nrm = norm(P[a, b, :k])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P


def innerfold(T, mask_idx, target_idx, e, k, sz, GROUND_TRUTH, config={}, test_syn=[]):
    Tc = [deepcopy(Ti) for Ti in T]
    mask_idx = np.unravel_index(mask_idx, (e, e, k))
    target_idx = np.unravel_index(target_idx, (e, e, k))

    # set values to be predicted to zero
    for i in range(len(mask_idx[0])):
        Tc[mask_idx[2][i]][mask_idx[0][i], mask_idx[1][i]] = 0
        # inverse triplets should also be set to 0 to prevent data leakage
        # Tc[mask_idx[2][i]][mask_idx[1][i], mask_idx[0][i]] = 0

    # predict unknown values
    P = predict_rescal_als(Tc, config=config)
    P = normalize_predictions(P, e, k)

    # compute area under precision recall curve
    if len(test_syn) > 0:
        true_targets = test_syn
    else:
        true_targets = GROUND_TRUTH[target_idx]

    prec, recall, _ = precision_recall_curve(true_targets, P[target_idx])
    auc_pr = auc(recall, prec)
    auc_roc = roc_auc_score(true_targets, P[target_idx])
    return auc_pr, auc_roc


if __name__ == "__main__":
    # load data
    mat = loadmat("graph_package/src/rescal_als/data/alyawarradata.mat")
    K = array(mat["Rs"], np.float32)
    e, k = K.shape[0], K.shape[2]
    SZ = e * e * k

    # copy ground truth before preprocessing
    GROUND_TRUTH = deepcopy(K)

    # construct array for rescal
    T = [lil_matrix(K[:, :, i]) for i in range(k)]

    _log.info(
        "Datasize: %d x %d x %d | No. of classes: %d" % (T[0].shape + (len(T),) + (k,))
    )

    # Do cross-validation
    FOLDS = 10
    IDX = list(range(SZ))
    shuffle(IDX)

    fsz = int(SZ / FOLDS)
    offset = 0
    AUC_PR_train = zeros(FOLDS)
    AUC_ROC_train = zeros(FOLDS)
    AUC_PR_test = zeros(FOLDS)
    AUC_ROC_test = zeros(FOLDS)
    for f in range(FOLDS):
        idx_test = IDX[offset : offset + fsz]
        idx_train = setdiff1d(IDX, idx_test)
        shuffle(idx_train)
        idx_train = idx_train[:fsz].tolist()
        _log.info("Train Fold %d" % f)
        AUC_PR_train[f], AUC_ROC_train[f] = innerfold(
            T, idx_train + idx_test, idx_train, e, k, SZ, GROUND_TRUTH
        )
        _log.info("Test Fold %d" % f)
        AUC_PR_test[f], AUC_ROC_test[f] = innerfold(
            T, idx_test, idx_test, e, k, SZ, GROUND_TRUTH
        )

        offset += fsz

    _log.info("AUC-PR Test Mean / Std: %f / %f" % (AUC_test.mean(), AUC_test.std()))
    _log.info("AUC-PR Train Mean / Std: %f / %f" % (AUC_train.mean(), AUC_train.std()))
