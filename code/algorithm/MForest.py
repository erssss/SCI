import math
from collections import Counter
from time import time

import numpy as np
from sklearn.cluster import KMeans

from algorithm.BaseMissing import BaseMissing
from util.Assist import Assist
from missingpy import KNNImputer

class MForest(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask, seed):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.Ncluster = Ncluster
        self.seed = seed

        self.Ndim = self.dbVals.shape[1]
        self.m_priors = np.ones(self.Ncluster) * (1 / Ncluster)
        self.m_means = np.zeros((self.Ncluster, self.Ndim))
        self.m_vars = np.ones((self.Ncluster, self.Ndim))
        self.m_minVars = np.zeros(self.Ndim)
        self.clusterCompRowIndexListMap = dict()
        self.trainSet = [[]]

    def mainGMM(self):
        self.initVals()
        self.trainSet = self.__initTrainSet()
        size = len(self.trainSet)
        self.train(self.trainSet, size)
        self.genCompIndexListForClusters()
        startTime = time()
        self.impute()
        self.algtime = time() - startTime
