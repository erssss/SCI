# -*- coding: utf-8 -*-

from algorithm.BaseMissing import BaseMissing
from time import time
from entity.RegModel import RegModel
import numpy as np



class ERACER(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.setCellMap()

    def setParams(self, K, maxIter, threshold):
        self.K = K
        self.maxIter = maxIter
        self.threshold = threshold

    def mainEracer(self):
        print("0")
        self.initVals()
        print("1")
        self.tmpDbVals = self.dbVals.astype(float)
        print("2")
        # self.tmpDbVals = self.dbVals
        self.learnRegressionModel()
        print("3")
        t0 = time()
        self.impute()
        print("4")
        self.algtime = time() - t0

    def impute(self):
        missNum = self.cells.__len__()
        preVals = [0] * missNum
        curVals = [0] * missNum
        deltas = [0] * missNum

        meanVals = self.tmpDbVals.mean(axis=0)
        print("missNum = ",missNum)
        for ci in range(missNum):
            # print("ci = ",ci)
            cell = self.cells[ci]
            position = cell.position
            attrY = position[1]
            rowIndex = self.misRowIndexList.index(position[0])

            self.tmpDbVals[rowIndex][attrY] = meanVals[attrY]
            preVals[ci] = meanVals[attrY]
            cell.modify = str(meanVals[attrY])

        iterNum = 0
        isConverge = False
        print("start iter")
        while iterNum < self.maxIter and not isConverge:
            print("iterNum = ",iterNum)
            iterNum += 1
            subDistances = [0] * self.dbVals.shape[0]
            sIndexes = [0] * self.K
            sDistances = [0] * self.K
            for ci in range(missNum):
                cell = self.cells[ci]
                position = cell.position
                rowIndex = self.misRowIndexList.index(position[0])
                attrY = position[1]
                attrXs = self.modelMap[attrY].getAttrXs()
                phis = self.phiMap[attrY]
                self.calcDistanceInSetInDB(rowIndex, subDistances, attrXs, True, self.compRowIndexList, self.tmpDbVals)
                self.findCompleteKnnInSet(subDistances, sIndexes, sDistances, self.compRowIndexList)
                knnIndexed = sIndexes
                avgK = 0
                for kIndex in knnIndexed:
                    avgK += self.tmpDbVals[kIndex][attrY]
                avgK /= self.K

                estimate = self.getRegEstimation(rowIndex, attrXs, attrY, avgK, phis)
                curVals[ci] = estimate
                deltas[ci] = abs(estimate - preVals[ci])

            isConverge = True
            for ci in range(missNum):
                if deltas[ci] > self.threshold:
                    isConverge = False

        for ci in range(missNum):
            cell = self.cells[ci]
            position = cell.position
            rowIndex = self.misRowIndexList.index(position[0])
            attrY = position[1]

            modify = curVals[ci]
            self.tmpDbVals[rowIndex][attrY] = modify
            preVals[ci] = modify
            cell.modify = str(modify)

    def learnRegressionModel(self):
        self.modelMap = {}
        self.phiMap = {}
        models = self.getRegModels()
        print("models",len(models),models)

        size = self.compRowIndexList.__len__()
        # print("self.compRowIndexList.__len__()",size)
        print("mi TOTAL = ",models.__len__())
        print("ri TOTAL = ",size)
        for mi in range(models.__len__()):
            regModel = models[mi]
            tmpAttrXs = regModel.getAttrXs()
            attrXNum = tmpAttrXs.__len__()
            attrY = regModel.getAttrY()
            columnSize = attrXNum + 1 + 1
            phis = [0] * columnSize
            x = [[0] * columnSize for _ in range(size)]
            y = [[0] for _ in range(size)]

            subDistances = [0] * self.dbVals.shape[0]
            sIndexes = [0] * self.K
            sDistances = [0] * self.K
            for ri in range(size):
                # print("ri = ",ri)
                rowIndex = self.compRowIndexList[ri]
                self.calcDisDirGivFea(rowIndex, subDistances, tmpAttrXs)
                self.findCompKnn(subDistances, sIndexes, sDistances)

                knnIndexes = sIndexes
                avgK = 0
                for kIndex in knnIndexes:
                    avgK += float(self.dbVals[kIndex][attrY])
                avgK /= self.K
                for j in range(attrXNum):
                    attrX = tmpAttrXs[j]
                    x[ri][j + 1] = float(self.dbVals[rowIndex][attrX])
                x[ri][0] = 1
                y[ri][0] = float(self.dbVals[rowIndex][attrY])
                x[ri][columnSize - 1] = avgK

            isSingular = False
            lxMatrix = np.array(x)
            lyMatrix = np.array(y)

            try:
                phi = self.learnParamsOLS(lxMatrix, lyMatrix)
            except:
                isSingular = True

            if not isSingular:
                for i in range(columnSize):
                    phis[i] = phi[i][0]
            else:
                print("ERACER Singular Matrix!!!")

            self.phiMap[attrY] = phis

    def getRegModels(self):
        models = []
        attrNum = self.dbVals.shape[1]
        for i in range(attrNum):
            if len(self.misAttrRowIndexList[i]) > 0:
                attrXs = self.assist.getAttrXsFromAttrY(attrNum, i)
                regModel = RegModel(attrXs, i)
                models.append(regModel)
                self.modelMap[i] = regModel
        return models

    def calcDistanceInSetInDB(self, rowIndex, distances, usedAttrs, itselfSetMax, indexList, dbVals):
        vals = dbVals[rowIndex]
        sumDown = usedAttrs.__len__()
        for cIndex in indexList:
            sumUp = 0
            for attrIndex in usedAttrs:
                dis = vals[attrIndex] - dbVals[cIndex][attrIndex]
                sumUp += dis * dis

            if sumDown == 0:
                dis = float('inf')
            elif sumUp == 0:
                dis = self.EPSILON
            else:
                dis = (sumUp / sumDown) ** 0.5

            distances[cIndex] = dis
        if itselfSetMax:
            distances[rowIndex] = float('inf')

    def findCompleteKnnInSet(self, distances, knnIndexes, knnDistances, indexList):
        if knnDistances.__len__() == 0:
            return

        length = knnIndexes.__len__()
        if length > indexList.__len__():
            for i in range(indexList.__len__()):
                rowIndex = indexList[i]
                knnIndexes[i] = rowIndex
                knnDistances[i] = distances[rowIndex]
        else:
            for i in range(length):
                rowIndex = indexList[i]
                knnIndexes[i] = rowIndex
                knnDistances[i] = distances[rowIndex]
            maxIndex = self.getMaxIndexfromK(knnDistances)
            maxVal = knnDistances[maxIndex]
            for i in range(length, indexList.__len__()):
                rowIndex = indexList[i]
                dis = distances[rowIndex]
                if dis < maxVal:
                    knnIndexes[maxIndex] = rowIndex
                    knnDistances[maxIndex] = dis

                    maxIndex = self.getMinIndexfromK(knnDistances)
                    maxVal = knnDistances[maxIndex]

        kpList = []
        for i in range(length):
            kp = (knnDistances[i], knnIndexes[i])
            kpList.append(kp)
            kpList = sorted(kpList, key=lambda x: x[0])

        for i in range(length):
            kp = kpList[i]
            knnDistances = kp[0]
            knnIndexes[i] = kp[1]

    def getRegEstimation(self, rowIndex, tmpAttrXs, attrY, avgK, phis):
        estimate = 0
        attrXNum = tmpAttrXs.__len__()
        intercept = phis[0]

        for j in range(attrXNum):
            attrX = tmpAttrXs[j]
            val = self.tmpDbVals[rowIndex][attrX]
            estimate += val * phis[j + 1]

        estimate += intercept
        estimate += phis[attrXNum + 1] * avgK
        return estimate
