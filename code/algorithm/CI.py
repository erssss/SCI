# -*- coding: utf-8 -*-

from algorithm.BaseMissing import BaseMissing
from time import time
from sklearn.cluster import KMeans
from entity.RegModel import RegModel
import numpy as np
import copy


class CI(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.tmpDbVals = self.dbVals.astype(float)
        self.mu = np.nanmean(self.tmpDbVals, axis=0)
        self.sigma = np.nanstd(self.tmpDbVals, axis=0)
        self.tmpDbVals = (self.tmpDbVals - self.mu) / self.sigma
        self.tmpDbValsNa = copy.deepcopy(self.tmpDbVals)
        for cell in self.cells:
            self.tmpDbValsNa[cell.position] = np.nan
        self.setCellMap()

    def setParams(self, K, maxIter, n_end, c_steps):
        self.initVals()
        self.K = K
        self.maxIter = maxIter
        self.n_end = n_end
        self.c_steps = c_steps
        self.tmpDbValsNa = self.tmpDbValsNa[self.misRowIndexList + self.compRowIndexList]

    def mainCI(self):
        for i in range(self.cells.__len__()):
            cell = self.cells[i]
            row_index = self.misRowIndexList.index(cell.position[0])
            self.tmpDbValsNa[row_index, cell.position[1]] = np.random.choice(self.tmpDbValsNa[self.misRowIndexList.__len__():, cell.position[1]])
        t0 = time()
        self.impute()
        self.algtime = time() - t0

    def impute(self):
        for iter in range(self.maxIter):
            kmeans = KMeans(self.K, max_iter=self.c_steps)
            kmeans = kmeans.fit(self.tmpDbValsNa)
            labels = kmeans.predict(self.tmpDbValsNa)
            omiga = min((iter + 1) / self.n_end, 1)
            for i in range(self.cells.__len__()):
                cell = self.cells[i]
                row_index = self.misRowIndexList.index(cell.position[0])
                U = self.tmpDbValsNa[labels == labels[row_index], cell.position[1]]
                if len(U) == 0:
                    U = self.tmpDbValsNa[:, cell.position[1]]
                modify = omiga * (np.random.choice(U))

                self.tmpDbValsNa[row_index, cell.position[1]] = modify
                if iter+1 == self.maxIter:
                    cell.modify = str((modify * self.sigma + self.mu)[cell.position[1]])
