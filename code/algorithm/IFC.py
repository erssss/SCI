# -*- coding: utf-8 -*-

from algorithm.BaseMissing import BaseMissing
from time import time
from sklearn.metrics import silhouette_samples
import numpy as np
from fcmeans import FCM


class IFC(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.model = None
        self.alpha = 1
        self.tmpDbVals = self.dbVals.astype(float)
        self.tmpDbValsNa = self.dbVals.astype(float)
        for cell in self.cells:
            self.tmpDbValsNa[cell.position] = np.nan
        self.setCellMap()

    def calcFuzzySilhouetteIndex(self, K):
        for i in range(self.tmpDbValsNa.shape[1]):
            temp_col = self.tmpDbValsNa[:, i]
            nan_num = temp_col[temp_col != temp_col].shape[0]
            if nan_num != 0:
                temp_col[np.isnan(temp_col)] = temp_col[temp_col == temp_col].mean()

        fcm = FCM(n_clusters=K)
        fcm.fit(self.tmpDbValsNa)
        fcm_labels = fcm.predict(self.tmpDbValsNa)
        fcm_soft_labels = fcm.soft_predict(self.tmpDbValsNa)
        fcm_soft_labels = np.sort(fcm_soft_labels, axis=1)
        largest = fcm_soft_labels[:, -1]
        second = fcm_soft_labels[:, -2]
        sik = silhouette_samples(self.tmpDbValsNa, fcm_labels)
        FSk = sum((largest - second) ** self.alpha * sik) / sum((largest - second) ** self.alpha)
        return FSk

    def setParams(self, min_k, max_k, maxIter, threshold):
        self.initVals()
        self.tmpDbValsNa = self.tmpDbValsNa[self.misRowIndexList + self.compRowIndexList]
        FSIndexMax = -1
        for k in range(min_k, max_k):
            FSIndex = self.calcFuzzySilhouetteIndex(k)
            if FSIndex > FSIndexMax:
                FSIndexMax = FSIndex
                self.K = k
        self.maxIter = maxIter
        self.threshold = threshold

    def mainIFC(self):
        t0 = time()
        self.impute()
        self.algtime = time() - t0

    def impute(self):
        for iter in range(self.maxIter):
            AVT = 0
            fcm = FCM(n_clusters=self.K)
            fcm.fit(self.tmpDbValsNa)
            centers = fcm.centers
            fcm_soft_labels = fcm.soft_predict(self.tmpDbValsNa[:self.misRowIndexList.__len__()])
            NV = fcm_soft_labels.dot(centers)
            for i in range(self.cells.__len__()):
                cell = self.cells[i]
                row_index = self.misRowIndexList.index(cell.position[0])
                cell.modify = str(NV[row_index, cell.position[1]])
                AVT += abs(float(cell.modify) - self.tmpDbValsNa[row_index, cell.position[1]]) / abs(self.tmpDbValsNa[row_index, cell.position[1]])
                self.tmpDbValsNa[row_index, cell.position[1]] = NV[row_index, cell.position[1]]

            if AVT / self.cells.__len__() <= self.threshold:
                break
