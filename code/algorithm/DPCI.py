# -*- coding: utf-8 -*-

from algorithm.BaseMissing import BaseMissing
from sklearn.neighbors import NearestNeighbors
from time import time
from sklearn.metrics import silhouette_samples
from sklearn import svm
import numpy as np


class DPCI(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.tmpDbValsNa = self.dbVals.astype(float)
        for cell in self.cells:
            self.tmpDbValsNa[cell.position] = np.nan
        self.modify_y = np.full(self.tmpDbValsNa.__len__(), 0)

    def setParams(self, k1, k2, alpha, beta,gamma):
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aKdensityPeak(self):
        compData = self.tmpDbValsNa[self.compRowIndexList]
        n_w = compData.shape[0]
        neigh = NearestNeighbors(n_neighbors=self.k1)
        neigh.fit(compData)
        distance, neighborsIndex = neigh.kneighbors()
        sigma = distance[:, -1]
        uk = np.mean(sigma)
        dc = uk + (sum((sigma - uk) ** 2) / (n_w - 1)) ** 0.5
        rho = np.sum(np.exp(-(distance ** 2) / (dc ** 2)), axis=1)
        delta = distance[:, 0]
        rhoMax = max(rho)
        for i in range(n_w):
            if rho[i] < rhoMax:
                delta[i] = min(self.calcDisNorm(compData[rho[i] < rho], compData[i]))
            else:
                delta[i] = max(self.calcDisNorm(compData, compData[i]))
        rhoDelta = rho * delta
        candidateCentersIndex = (rhoDelta >= sorted(rho * delta)[int(n_w * self.gamma)]) & (delta > (self.beta * dc))
        candidateCenters = compData[candidateCentersIndex]
        cluster = np.full(n_w, 0)
        centerDistance = np.full(n_w, np.inf)
        nCluster = sum(candidateCentersIndex)
        for cluster_i in range(nCluster):
            clDistance = self.calcDisNorm(candidateCenters[cluster_i], compData)
            cluster[clDistance < centerDistance] = cluster_i
            centerDistance[clDistance < centerDistance] = clDistance[clDistance < centerDistance]
        borderPointsIndexs = [[set() for _ in range(nCluster)] for _ in range(nCluster)]
        for k in range(n_w):
            for j in range(k, n_w):
                if (cluster[k] != cluster[j]) and ((np.sum((compData[k] - compData[j]) ** 2) ** 0.5) < dc):
                    borderPointsIndexs[cluster[k]][cluster[j]].update({k, j})
                    borderPointsIndexs[cluster[j]][cluster[k]].update({k, j})
        borderDensity = np.zeros((nCluster, nCluster))
        rhoClusterMean = np.zeros(nCluster)
        densityReachable = []
        for u in range(nCluster):
            rhoClusterMean[u] = np.mean(rho[cluster == u])
        for u in range(nCluster):
            for v in range(nCluster):
                if len(borderPointsIndexs[u][v]) > 0:
                    borderDensity[u, v] = np.mean(rho[list(borderPointsIndexs[u][v])])
                    if borderDensity[u, v] > min(rhoClusterMean[u], rhoClusterMean[v]):
                        densityReachable.append([u, v])

        densityReachableChain = self.findComponents(nCluster, densityReachable)
        for chain in densityReachableChain:
            chain = list(chain)
            chain.sort(reverse=True)
            for cluster_i in chain[:-1]:
                cluster[cluster == cluster_i] = chain[-1]
        return cluster, rho

    def main_DPCI(self):
        self.initVals()
        startTime = time()
        cluster, rho = self.aKdensityPeak()
        fwpdDistance = self.calFwpdDistance()
        self.impute(fwpdDistance)
        self.train_classifier(cluster, rho)
        self.algtime = time() - startTime

    def calFwpdDistance(self):
        nSamples = self.tmpDbValsNa.__len__()
        distanceObserved = np.full((nSamples, nSamples), -1.)
        for i in range(nSamples):
            distanceObserved[i] = np.nansum((self.tmpDbValsNa - self.tmpDbValsNa[i]) ** 2, axis=1) ** 0.5
        missingFeature = np.isnan(self.tmpDbValsNa)
        observedFeatureNum = np.sum(~missingFeature, axis=0)
        totalObservedFeatureNum = np.sum(observedFeatureNum)
        penaltyTerm = np.full((nSamples, nSamples), 0.)
        for i in range(nSamples):
            for j in range(nSamples):
                missingFeatureIJ = missingFeature[i] | missingFeature[j]
                if np.sum(missingFeatureIJ) > 0:
                    penaltyTerm[i, j] = np.sum(observedFeatureNum[missingFeatureIJ])
        penaltyTerm /= totalObservedFeatureNum
        maxDistanceObserved = np.max(distanceObserved)
        fwpdDistance = (1 - self.alpha) * (distanceObserved) / maxDistanceObserved + self.alpha * penaltyTerm
        return fwpdDistance

    def impute(self, fwpdDistance):
        neigh = NearestNeighbors(n_neighbors=self.k2, metric='precomputed')
        neigh.fit(fwpdDistance)
        distance, neighborsIndex = neigh.kneighbors()
        misNum = self.cells.__len__()
        for mi in range(misNum):
            cell = self.cells[mi]
            position = cell.position
            modify = np.nanmean(self.tmpDbValsNa[neighborsIndex[position[0]]][:, position[1]])
            if modify != modify:
                modify = 0
            self.tmpDbValsNa[position] = modify
            cell.setModify(str(modify))

    def train_classifier(self, cluster, rho):
        for i, ci in enumerate(list(set(cluster))):
            cluster[cluster == ci] = i

        compData = self.tmpDbValsNa[self.compRowIndexList]
        coreData = []
        coreLabel = []
        for ci in np.unique(cluster):
            dataCluster = compData[cluster == ci]
            rhoCluster = rho[cluster == ci]
            rhoMean = np.mean(rhoCluster)
            coreData.extend(dataCluster[rhoCluster > rhoMean].tolist())
            coreLabel.extend(cluster[cluster == ci][rhoCluster > rhoMean].tolist())
        if len(set(coreLabel)) >= 2:
            clf = svm.SVC(gamma="auto")
            clf.fit(coreData, coreLabel)
            incomH = clf.predict(self.tmpDbValsNa[self.misRowIndexList])
        else:
            incomH = coreLabel[0]
        self.modify_y[self.compRowIndexList] = cluster
        self.modify_y[self.misRowIndexList] = incomH

    def calcDisNorm(self, array1, array2, norm=2):
        return np.sum((array1 - array2) ** norm, axis=1) ** (1. / norm)

    def findComponents(self, n, edges):
        graph = [[] for _ in range(n)]
        seen = [False for _ in range(n)]
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        components = []
        for i in range(n):
            if not seen[i]:
                component = [i]
                seen[i] = True
                components.append(self.dfs(i, graph, seen, component))
        return components

    def dfs(self, i, graph, seen, component):
        for j in graph[i]:
            if not seen[j]:
                component.append(j)
                seen[j] = True
                self.dfs(j, graph, seen, component)
        return component
