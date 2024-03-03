# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import issparse
from copy import deepcopy

from algorithm.BaseMissing import BaseMissing


class NMF(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask, seed, eta=0.2, beta=0.0000000000001, maxiter=100,
                 stopconv=1e-4):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.eta = eta
        self.beta = beta
        self.maxiter = maxiter
        self.stopconv = stopconv
        self.seed = seed
        self.solver = self._update_scipy_nnls

    def mainSNMF(self):
        # print("NMF 0")
        self.initVals()
        # print("NMF 1")
        X_comp_true, y_comp_true, X_comp_miss = self.ignore_down_stream()
        X = X_comp_miss.T
        # print("NMF 2")
        mask = (~np.isnan(X)).astype(int)
        X = self.norm_x(X)
        # print("NMF predict")
        # print("X",X.shape,"mask",mask.shape)
        NMF_res = self.predict(X, mask)
        # print("NMF 3")
        W, H = NMF_res.matrices
        return H.T, y_comp_true

    def predict(self, X, mask):
        m, n = X.shape

        if issparse(X):
            X = X.todense()
        else:
            X = X

        pdist = 1e10  

        np.random.seed(self.seed)
        W = np.random.rand(m, self.exp_cluster)
        H = np.random.rand(self.exp_cluster, n)

        dist = 0
        converged = False
        convgraph = np.zeros(self.maxiter)
        print("NMF MAXiter = ",self.maxiter)

        for i in range(self.maxiter):
            # print("iter = ",i)

            W, H = self.solver(W, H, X, mask)
            dist = self.frobenius(X, np.dot(W, H))
            convgraph[i] = dist

            if abs(pdist - dist) < self.stopconv:
                converged = True
                break

            pdist = dist

        W, H = self.normalize_factor_matrices(W, H)

        return NMFResult((W, H), convgraph, dist, converged)

    def _update_scipy_nnls(self, W, H, X, mask):
        Xaug = np.r_[X, np.zeros((1, H.shape[1]))]

        Waug = np.r_[W, np.sqrt(self.beta) * np.ones((1, H.shape[0]))]

        Htaug = np.r_[H.T, np.sqrt(self.eta) * np.eye(H.shape[0])]
        Xaugkm = np.r_[X.T, np.zeros(W.T.shape)]

        Wm = np.c_[mask, np.ones((X.shape[0], H.shape[0]))]
        Hm = np.r_[mask, np.ones((1, H.shape[1]))]
        rw, cw = np.where(Wm == 1)
        rh, ch = np.where(Hm == 1)

        for i in range(W.shape[0]):
            W[i, :] = nnls(Htaug[cw[rw == i], :], Xaugkm[cw[rw == i], i])[0]

        for j in range(H.shape[1]):
            H[:, j] = nnls(Waug[rh[ch == j], :], Xaug[rh[ch == j], j])[0]

        return W, H

    @staticmethod
    def frobenius(A, B):
        matrix_minus = A - B
        arr_minus = matrix_minus.flatten()
        arr_m = arr_minus[~np.isnan(arr_minus)]
        return np.linalg.norm(arr_m, 2)

    @staticmethod
    def normalize_factor_matrices(W, H):
        norms = np.linalg.norm(W, 2, axis=0)
        norm_gt_0 = norms > 0
        W[:, norm_gt_0] /= norms[norm_gt_0]
        H[norm_gt_0, :] = ((H[norm_gt_0, :].T) * norms[norm_gt_0]).T
        return W, H

    @staticmethod
    def norm_x(x):
        min_max = np.nanmax(x, axis=1) - np.nanmin(x, axis=1) + 1e-10
        min_max = min_max.reshape((-1, 1)).repeat(x.shape[1], axis=1)
        min_rp = np.nanmin(x, axis=1).reshape((-1, 1)).repeat(x.shape[1], axis=1)
        norm_x = (x - min_rp) / min_max
        return norm_x


class NMFResult:
    convgraph = None  
    matrices = None  
    objvalue = None  
    converged = None

    def __init__(self, matrices, convgraph=None, objvalue=None, converged=None):
        self.matrices = matrices
        self.convgraph = convgraph
        self.objvalue = objvalue
        self.converged = converged
