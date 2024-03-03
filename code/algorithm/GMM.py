# -*- coding: utf-8 -*-

import math
from collections import Counter
from time import time

import numpy as np
from sklearn.cluster import KMeans

from algorithm.BaseMissing import BaseMissing
from util.Assist import Assist


class GMM(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask, K, Ncluster, seed):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.K = K
        self.Ncluster = Ncluster
        self.seed = seed
        self.m_maxIter = 100
        self.m_endError = 0.0001

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

    def genCompIndexListForClusters(self):
        self.clusterCompRowIndexListMap.clear()
        comSize = len(self.compRowIndexList)
        self.clusterCompRowIndexListMap = {ki: [] for ki in range(self.Ncluster)}
        for ci in range(comSize):
            rowIndex = self.compRowIndexList[ci]
            probabilities = np.array([self.getPartialProb(self.trainSet[ci], ki) for ki in range(self.Ncluster)])
            maxLocation = probabilities.argmax()
            if self.clusterCompRowIndexListMap.__contains__(maxLocation):
                clusterCompRowIndexList = self.clusterCompRowIndexListMap[maxLocation]
                clusterCompRowIndexList.append(rowIndex)
            else:
                self.clusterCompRowIndexListMap[maxLocation] = [rowIndex]

    def impute(self):
        missNum = len(self.cells)
        testSet = np.zeros((missNum, self.Ndim))
        attrNum = self.dbVals.shape[1]

        for mi in range(missNum):
            cell = self.cells[mi]
            rowIndex, misAttrIndex = cell.position

            for attrIndex in range(attrNum):
                if attrIndex == misAttrIndex:
                    if rowIndex != 0:
                        testSet[mi][attrIndex] = 0
                    else:
                        testSet[mi][attrIndex] = 0
                else:
                    testSet[mi][attrIndex] = float(self.dbVals[rowIndex][attrIndex])

            probabilities = np.array([self.getPartialProb(testSet[mi], k, misAttrIndex) for k in range(self.Ncluster)])
            maxLocation = probabilities.argmax()
            clusterCompRowIndexList = self.clusterCompRowIndexListMap[maxLocation]
            subDistances = np.zeros(len(self.dbVals))
            sIndexes = np.zeros(self.K, dtype=int)
            sDistances = np.zeros(self.K)
            self.calcDisDirFromList(rowIndex, subDistances, clusterCompRowIndexList)
            self.findCompleteKnn(subDistances, sIndexes, sDistances, clusterCompRowIndexList)
            knnIndexes = sIndexes
            modify = 0
            for kRowIndex in knnIndexes:
                modify += float(self.dbVals[kRowIndex][misAttrIndex])
            modify /= self.K
            cell.setModify(str(modify))

    def getPartialProb(self, x, k, misAttrIndex=None):
        if misAttrIndex is None:
            gamma_ik = self.gaussian(x, k) * self.m_priors[k] / self.getProbability(x)
            return gamma_ik
        else:
            probability = self.getProbability(x, misAttrIndex)
            if probability == 0:
                probability = 0.01
            gamma_ik = self.gaussian(x, k, misAttrIndex) * self.m_priors[k] / probability
            return gamma_ik

    def train(self, data, size):
        self.init(data, size, self.seed)
        loop = True
        iterNum = 0
        unchanged = 0
        lastL, currL = 0.0, 0.0
        while loop:
            next_priors = np.zeros(self.Ncluster)
            next_means = np.zeros((self.Ncluster, self.Ndim))
            next_vars = np.zeros((self.Ncluster, self.Ndim))

            lastL = currL
            currL = 0

            for i in range(size):
                x = data[i]
                p = self.getProbability(x)
                for k in range(self.Ncluster):
                    gamma_ik = self.gaussian(x, k) * self.m_priors[k] / p
                    next_priors[k] += gamma_ik
                    for d in range(self.Ndim):
                        next_means[k][d] += gamma_ik * x[d]
                        next_vars[k][d] += gamma_ik * x[d] * x[d]

                currL += math.log(p) if (p > 1E-20) else -20
            currL /= size

            for k in range(self.Ncluster):
                self.m_priors[k] = next_priors[k] / size
                if self.m_priors[k] > 0:
                    for d in range(self.Ndim):
                        self.m_means[k][d] = next_means[k][d] / next_priors[k]
                        self.m_vars[k][d] = next_vars[k][d] / next_priors[k] - self.m_means[k][d] ** 2
                        self.m_vars[k][d] = max(self.m_vars[k][d], self.m_minVars[d])

            iterNum += 1
            if abs(currL - lastL) < self.m_endError * abs(lastL):
                unchanged += 1
            if iterNum >= self.m_maxIter or unchanged >= 3:
                loop = False

    def getProbability(self, x, misAttrIndex=None):
        p = 0
        for k in range(self.Ncluster):
            p += self.m_priors[k] * self.gaussian(x, k, misAttrIndex)
        return p

    def gaussian(self, x, k, misAttrIndex=None):
        p = 1
        if misAttrIndex is None:
            for d in range(self.Ndim):
                p *= 1 / math.sqrt(2 * 3.14159 * self.m_vars[k][d])
                p *= math.exp(-0.5 * (x[d] - self.m_means[k][d]) * (x[d] - self.m_means[k][d]) / self.m_vars[k][d])
            return p
        else:
            for d in range(self.Ndim):
                if d == misAttrIndex:
                    continue
                p *= 1 / math.sqrt(2 * 3.14159 * self.m_vars[k][d])
                p *= math.exp(-0.5 * (x[d] - self.m_means[k][d]) * (x[d] - self.m_means[k][d]) / self.m_vars[k][d])
            return p

    def init(self, data, size, seed):
        MIN_VAR = 1e-10
        kmeans = KMeans(n_clusters=self.Ncluster, random_state=seed)
        kmeans = kmeans.fit(data)
        labels = kmeans.labels_
        cnt = Counter(labels)
        count = np.array([cnt[i] for i in range(self.Ncluster)])

        self.m_priors[:] = 0
        self.m_vars[:, :] = 0

        self.m_minVars = 0.01 * ((data ** 2).mean(0) - (data.mean(0)) ** 2)
        self.m_minVars = np.where(self.m_minVars > MIN_VAR, self.m_minVars, MIN_VAR)
        self.m_means = kmeans.cluster_centers_
        for k in range(self.Ncluster):
            self.m_priors[k] = 1.0 * count[k] / size
            if count[k] > 0:
                self.m_vars[k] = ((data - self.m_means[k]) ** 2).sum(0)
                for d in range(self.Ndim):
                    self.m_vars[k][d] = max(self.m_vars[k][d], self.m_minVars[d])
            else:
                for d in range(self.Ndim):
                    self.m_vars[k][d] = self.m_minVars[d]
                print("[WARNING] Gaussian " + str(k) + " of GMM is not used!")

    def __initTrainSet(self):
        return self.dbVals[self.compRowIndexList].astype(float)
