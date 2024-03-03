# -*- coding: utf-8 -*-

import math
import numpy as np
from copy import deepcopy
from util.Assist import Assist
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.cluster import AffinityPropagation

class BaseMissing:
    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        self.db = db
        self.label_idx = label_idx
        self.exp_cluster = exp_cluster
        self.cells = cells
        self.flags = mask
        self.tpList = []
        self.cellMap = dict()
        self.assist = Assist()
        self.EPSILON = 0.000000001
        self.rowIndexList = []
        self.misRowIndexList = []
        self.misRowAttrIndexList = []
        self.misAttrRowIndexList = []
        self.compRowIndexList = []

        self.delCompRowIndexList = []
        self.algtime = 0.0
        if self.label_idx == -1:
            self.dbVals = self.db.values
        else:
            self.dbVals = self.db.drop(self.label_idx, axis=1).values

        self.dbmax = np.nanmax(self.dbVals, axis=0)
        self.dbmin = np.nanmin(self.dbVals, axis=0)

    def setDelCompRowIndexList(self, delCompRowIndexList):
        self.delCompRowIndexList = delCompRowIndexList

    def setCellMap(self):
        self.cellMap = {self.cells[i].position: self.cells[i] for i in range(len(self.cells))}

    def initVals(self):
        self.rowIndexList = list(self.db.index)
        for i in range(self.flags.shape[0]):
            if self.flags[i].min() == 0:
                self.misRowIndexList.append(i)
                self.misRowAttrIndexList.append([j for j in range(self.flags.shape[1]) if self.flags[i][j] == 0])
        self.misAttrRowIndexList = [[j for j in range(self.flags.shape[0]) if self.flags[j][i] == 0] for i in
                                    range(self.flags.shape[1])]
        self.compRowIndexList = sorted(list(set(self.rowIndexList) - set(self.misRowIndexList) - set(self.delCompRowIndexList)))

    def calcNormDisBtwTwoArrays(self, vals1, vals2):
        dis = (np.array(vals1) - np.array(vals2)) / (self.dbmax - self.dbmin + self.EPSILON)
        dis = np.sum(dis ** 2) ** 0.5
        return dis

    def calcNormDisBtwTwoTp(self, rowIndex1, status1, rowIndex2):
        attrNum = self.dbVals.shape[1]
        vals1 = self.dbVals[rowIndex1]
        vals2 = self.dbVals[rowIndex2]
        sumUp, sumDown = 0, 0,
        for attri in range(attrNum):
            val1 = vals1[attri]
            val2 = vals2[attri]
            try:
                numVal1 = float(val1)
                numVal2 = float(val2)
                dis = self.assist.normNumDis(numVal1, numVal2, self.dbmax[attri], self.dbmin[attri])
                weight = status1[attri]
                sumDown += weight
                sumUp += weight * dis * dis 
            except:
                print(val1, val2)
        if sumDown == 0:
            dis = float('inf')
        elif sumUp == 0:
            dis = self.EPSILON
        else:
            dis = (sumUp / sumDown) ** 0.5
        return dis

    def calcDirDisBtwTwoTp(self, rowIndex1, status1, rowIndex2):
        attrNum = self.dbVals.shape[1]
        vals1 = self.dbVals[rowIndex1]
        vals2 = self.dbVals[rowIndex2]

        sumUp, sumDown = 0, 0
        for attri in range(attrNum):
            val1 = vals1[attri]
            val2 = vals2[attri]
            if isinstance(val1, str):
                if Assist().isNumber(val1) and Assist().isNumber(val2):
                    numVal1, numVal2 = float(val1), float(val2)
                    dis = Assist().numDis(numVal1, numVal2)
                else:
                    dis = Assist().strDis(val1, val2)
            else:
                dis = self.assist.numDis(val1, val2)
            weight = status1[attri]
            sumDown += weight
            sumUp += dis * dis * weight

        if sumDown == 0:
            dis = math.inf
        elif sumUp == 0:
            dis = self.EPSILON
        else:
            dis = math.sqrt(sumUp / sumDown)
        return dis

    def calcDirDisBtwTwoTpGivAttr(self, rowIndex1, status1, rowIndex2, features):
        """

        :param rowIndex1:
        :param status1: row1 mis flag list
        :param rowIndex2:
        :param features: selected feature list
        :return:
        """
        sumUp, sumDown = 0, 0
        vals1, vals2 = self.dbVals[rowIndex1], self.dbVals[rowIndex2]
        for fi in range(len(features)):
            attrIndex = features[fi]
            val1, val2 = vals1[attrIndex], vals2[attrIndex]
            if isinstance(val1, str):
                if self.assist.isNumber(val1) and self.assist.isNumber(val2):
                    numVal1, numVal2 = float(val1), float(val2)
                    dis = self.assist.numDis(numVal1, numVal2)
                else:
                    dis = self.assist.strDis(val1, val2)
            else:
                dis = self.assist.numDis(val1, val2)
            weight = status1[attrIndex]
            sumDown += weight
            sumUp += dis * dis * weight

        if sumDown == 0:
            dis = np.inf
        elif sumUp == 0:
            dis = self.EPSILON
        else:
            dis = math.sqrt(sumUp / sumDown)
        return dis

    def calcDisNorm(self, rowIndex, distances):
        status = self.flags[rowIndex]
        compSize = self.compRowIndexList.__len__()
        for ci in range(compSize):
            compRowIndex = self.compRowIndexList[ci]
            dis = self.calcNormDisBtwTwoTp(rowIndex, status, compRowIndex)
            distances[compRowIndex] = dis
        distances[rowIndex] = float('inf')

    def calcDisDirFromList(self, rowIndex, distances, givRowIndexList):
        status = self.flags[rowIndex]
        givSize = len(givRowIndexList)
        for ci in range(givSize):
            givRowIndex = givRowIndexList[ci]
            dis = self.calcDirDisBtwTwoTp(rowIndex, status, givRowIndex)
            distances[givRowIndex] = dis
        distances[rowIndex] = math.inf

    def calcDisDirGivFea(self, rowIndex, distances, features):
        """

        :param rowIndex: mis row index
        :param distances: distance matrix [mis rows * all rows]
        :param features: selected features list
        :return:
        """
        status = self.flags[rowIndex]
        for ci in range(len(self.compRowIndexList)):
            compRowIndex = self.compRowIndexList[ci]
            dis = self.calcDirDisBtwTwoTpGivAttr(rowIndex, status, compRowIndex, features)
            distances[compRowIndex] = dis
        distances[rowIndex] = np.inf

    def findCompKnn(self, distances, knnIndexes, knnDistances):
        """
        :param distances: []
        :param knnIndexes: []
        :param knnDistances: []
        :return:
        """
        if len(knnDistances) == 0:
            return
        length = len(knnIndexes)
        if length > len(self.compRowIndexList):
            print("length > len(self.compRowIndexList)")
            for i in range(len(self.compRowIndexList)):
                rowIndex = self.compRowIndexList[i]
                knnIndexes[i] = rowIndex
                knnDistances[i] = distances[rowIndex]
        else:
            print("length <= len(self.compRowIndexList)")
            for i in range(length):
                rowIndex = self.compRowIndexList[i]
                knnIndexes[i] = rowIndex
                knnDistances[i] = distances[rowIndex]
                # print("i = ",i,"knnIndexes = ",knnIndexes[i],"knnDistances = ",knnDistances[i])

            maxIndex = self.getMaxIndexfromK(knnDistances)
            maxVal = knnDistances[maxIndex]

            for i in range(length, len(self.compRowIndexList)):
                rowIndex = self.compRowIndexList[i]
                dis = distances[rowIndex]
                if dis < maxVal:
                    knnIndexes[maxIndex] = rowIndex
                    knnDistances[maxIndex] = dis

                    maxIndex = self.getMaxIndexfromK(knnDistances)
                    maxVal = knnDistances[maxIndex]

        kpList = []
        for i in range(length):
            kp = (knnDistances[i], knnIndexes[i])
            # print("i = ",i,"knnDistances = ",knnDistances[i],"knnIndexes = ",knnIndexes[i])
            kpList.append(kp)
        kpList = sorted(kpList, key=lambda x: x[0])

        for i in range(length):
            kp = kpList[i]
            knnIndexes[i] = kp[1]
            knnDistances[i] = kp[0]

    def findTopK(self, totalLikelihoods, topKIndexes, topKLikelihoods):
        if topKLikelihoods.__len__ == 0:
            return
        length = topKIndexes.__len__()
        if length > self.compRowIndexList.__len__():
            for i in range(self.compRowIndexList.__len__()):
                rowIndex = self.compRowIndexList[i]
                topKIndexes[i] = rowIndex
                topKLikelihoods[i] = totalLikelihoods[i]
        else:
            for i in range(length):
                rowIndex = self.compRowIndexList[i]
                topKIndexes[i] = rowIndex
                topKLikelihoods[i] = totalLikelihoods[i]

            minIndex = self.getMinIndexfromK(topKLikelihoods)
            minVal = topKLikelihoods[minIndex]

            for i in range(length, self.compRowIndexList.__len__()):
                rowIndex = self.compRowIndexList[i]
                dis = totalLikelihoods[i]
                if dis > minVal:
                    topKIndexes[minIndex] = rowIndex
                    topKLikelihoods[minIndex] = dis

                    minIndex = self.getMinIndexfromK(topKLikelihoods)
                    minVal = topKLikelihoods[minIndex]

        lpList = []
        for i in range(length):
            lp = (topKLikelihoods[i], topKIndexes[i])
            lpList.append(lp)
        lpList = sorted(lpList, key=lambda x: x[0])

        for i in range(length):
            lp = lpList[i]
            topKIndexes[i] = lp[1]
            topKLikelihoods[i] = lp[0]

    def findCompleteKnn(self, distances, knnIndexes, knnDistances, rowIndexList):
        if len(knnDistances) == 0:
            return None
        length = len(knnIndexes)
        if length > len(rowIndexList):
            for i in range(len(rowIndexList)):
                rowIndex = rowIndexList[i]
                knnIndexes[i] = rowIndex
                knnDistances[i] = distances[rowIndex]
        else:
            for i in range(length):
                rowIndex = rowIndexList[i]
                knnIndexes[i] = rowIndex;
                knnDistances[i] = distances[rowIndex]
            maxIndex = self.getMaxIndexfromK(knnDistances)
            maxVal = knnDistances[maxIndex]

            for i in range(len(rowIndexList)):
                rowIndex = rowIndexList[i]
                dis = distances[rowIndex]
                if dis < maxVal:
                    knnIndexes[maxIndex] = rowIndex
                    knnDistances[maxIndex] = dis

                    maxIndex = self.getMaxIndexfromK(knnDistances)
                    maxVal = knnDistances[maxIndex]

        kpList = []
        for i in range(length):
            kp = (knnDistances[i], knnIndexes[i])
            kpList.append(kp)
        kpList = sorted(kpList, key=lambda x: x[0])

        for i in range(length):
            kp = kpList[i]
            (knnDistances[i], knnIndexes[i]) = kp

    def getMaxIndexfromK(self, vals):
        vals = np.array(vals)
        return vals.argmax()

    def getMinIndexfromK(self, vals):
        vals = np.array(vals)
        return vals.argmin()

    def learnParamsOLS(self, xMatrix, yMatrix):
        attrXNum = xMatrix.shape[1]
        xMatrixT = xMatrix.T
        aMatrix = xMatrixT.dot(xMatrix)
        bMatrix = xMatrixT.dot(yMatrix)

        alpha = [[0] * attrXNum for _ in range(attrXNum)]
        for i in range(attrXNum):
            alpha[i][i] = self.EPSILON
        alphaMatrix = np.array(alpha)
        aMatrix = aMatrix + alphaMatrix
        middleMatrix = np.linalg.inv(aMatrix)
        beta = middleMatrix.dot(bMatrix)
        return beta

    def calcModelResidual(self, phis, attrY, xMatrix, yMatrix, size):
        sigma = 0
        x = xMatrix.tolist()
        y = yMatrix.tolist()
        for i in range(size):
            estimate = 0
            for j in range(phis.__len__()):
                estimate += phis[j] * x[i][j]
            residual = estimate - y[i][0]

            sigma += residual * residual

        sigma = sigma / size
        return sigma

    def getAttrXsFromMisList(self, misList):
        attrNum = self.dbVals.shape[1]
        attrXNum = attrNum - misList.__len__()
        attrXs = [0] * attrXNum
        curIndex = 0
        for attri in range(attrNum):
            if attri in misList:
                continue
            attrXs[curIndex] = attri
            curIndex += 1
        return attrXs

    def getAttrXsFromAttrY(self, attrY):
        attrNum = self.dbVals.shape[1]
        attrXNum = attrNum - 1

        attrXs = [0] * attrXNum
        curIndex = 0
        for attri in range(attrNum):
            if attrY == attri:
                continue
            attrXs[curIndex] == attri
            curIndex += 1

        return attrXs

    def getAlgtime(self):
        return self.algtime

    def setAlgtime(self, algtime):
        self.algtime = algtime

    def modify_down_stream(self, cells,name="tmp"):
        ds_idx = self.compRowIndexList + self.misRowIndexList
        if self.label_idx == -1:
            origin_x = deepcopy(self.dbVals)[ds_idx].astype(float)
            expection_n = self.exp_cluster
            origin_y = self.cls(origin_x, expection_n)
            modify_x = deepcopy(self.dbVals)
        else:
            origin_y = self.db.iloc[ds_idx, self.label_idx].values.astype(int)
            expection_n = self.exp_cluster
            modify_x = deepcopy(self.db.drop(self.label_idx, axis=1).values)
        for c in cells:
            modify_x[c.position] = c.modify

        modify_x = modify_x[ds_idx]
        modify_x = modify_x.astype(float)

        return origin_y, self.cls(np.nan_to_num(modify_x, nan=0), expection_n)

    def cls(self, x, n_cluster):
        
        kmeans = KMeans(n_clusters=n_cluster)
        y = kmeans.fit_predict(x)
        return y

    def ignore_down_stream(self):
        ds_idx = self.compRowIndexList + self.misRowIndexList
        test_pos = [cell.position for cell in self.cells]
        xp, yp = zip(*test_pos)
        dbarr = deepcopy(self.dbVals)
        dbarr = dbarr.astype(float)
        dbarr[(xp, yp)] = np.NAN
        X_comp_true = self.dbVals[ds_idx]
        X_comp_miss = dbarr[ds_idx]
        if self.label_idx == -1:
            y_comp_true = self.cls(X_comp_true, self.exp_cluster)
        else:
            y_true = self.db[self.label_idx].values
            y_comp_true = y_true[ds_idx]
        return X_comp_true, y_comp_true, X_comp_miss

    def calcNormDis(self, vals1, vals2):
        dis = np.sum((vals1 - vals2) ** 2) ** 0.5
        return dis