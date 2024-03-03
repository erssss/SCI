# -*- coding: utf-8 -*-

from algorithm.BaseMissing import BaseMissing
from time import time
from entity.Cluster import Cluster
import random
import math


class CMI(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.clusters = []
        self.centerList = []
        self.selList = []
        self.setCellMap()
        self.dbVals = self.dbVals.astype(float)

    def mainCMI(self):
        self.initVals()
        t0 = time()

        self.buildClusters()
        misRowNum = self.misRowIndexList.__len__()
        curchoice = [0] * misRowNum
        cnum = self.clusters.__len__()
        for cIndex in range(cnum):
            cluster = self.clusters[cIndex]
            centerRowIndex = self.centerList[cIndex]
            for rowIndex in cluster.getRowIndexList():
                if self.misRowIndexList.__contains__(rowIndex):
                    curchoice[self.getImIndex(self.misRowIndexList, rowIndex)] = centerRowIndex

        for cIndex in range(cnum):
            cluster = self.clusters[cIndex]
            loop = 3
            while loop > 0:
                for rowIndex in cluster.getRowIndexList():
                    if self.misRowIndexList.__contains__(rowIndex):
                        imindex = self.misRowIndexList.index(rowIndex)
                        kList = self.kNearest(self.misRowIndexList, curchoice, rowIndex, cluster)

                        if kList.__len__() == 0:
                            continue
                        elif kList.__len__() == 1:
                            kRowIndex = cluster.getRowIndexList()[kList[0]]
                            if self.misRowIndexList.__contains__(kRowIndex):
                                canRowIndex = curchoice[self.getImIndex(self.misRowIndexList, kRowIndex)]
                            else:
                                canRowIndex = kRowIndex
                        else:
                            tmpIndex = random.randint(0, kList.__len__()-1)
                            kIndex = kList[tmpIndex]
                            kRowIndex = cluster.getRowIndexList()[kIndex]
                            if self.misRowIndexList.__contains__(kRowIndex):
                                canRowIndex = curchoice[self.getImIndex(self.misRowIndexList, kRowIndex)]
                            else:
                                canRowIndex = kRowIndex
                        curchoice[imindex] = canRowIndex
                loop -= 1
        for ri in range(misRowNum):
            misRowIndex = self.misRowIndexList[ri]
            canRowIndex = curchoice[ri]
            misList = self.misRowAttrIndexList[ri]
            for mi in range(misList.__len__()):
                misAttrIndex = misList[mi]
                position = (misRowIndex, misAttrIndex)
                cell = self.cellMap[position]
                cell.modify = str(self.dbVals[canRowIndex][misAttrIndex])
        self.algtime = time() - t0

    def kNearest(self, tmpMisRowIndexList, curchoice, misRowIndex, cluster):
        kList = []
        simList = []
        sim = 0
        attrNum = self.dbVals.shape[1]
        vals1 = self.dbVals[misRowIndex]
        imIndex = self.getImIndex(tmpMisRowIndexList, misRowIndex)
        for rowIndex in cluster.getRowIndexList():
            sim = 0
            if misRowIndex == rowIndex:
                simList.append(-1.0)
                continue

            vals2 = self.dbVals[rowIndex]
            for attrIndex in range(attrNum):
                if self.selList.__contains__(attrIndex):
                    if tmpMisRowIndexList.__contains__(rowIndex):
                        if curchoice[imIndex] == curchoice[self.getImIndex(tmpMisRowIndexList, rowIndex)]:
                            sim += 1
                    elif self.dbVals[curchoice[imIndex]][attrIndex] == vals2[attrIndex]:
                        sim += 1
                elif math.isclose(vals1[attrIndex], vals2[attrIndex]):
                    sim += 1
            simList.append(sim)

        orderList = [0] * simList.__len__()
        sorted(orderList, reverse=True)
        cnum = cluster.getRowIndexList().__len__()
        kNear = 1 if cnum > 1 else cnum
        for ki in range(kNear):
            s = orderList[ki]
            if s > 0:
                for sindex in range(simList.__len__()):
                    if math.isclose(simList[sindex], s):
                        if sindex not in kList:
                            kList.append(sindex)
            else:
                break
        return kList

    def getImIndex(self, imList, tid):
        index = -1
        for i in range(imList.__len__()):
            if imList[i] == tid:
                return i
        return index

    def buildClusters(self):

        random.seed(self.seed)
        size = self.dbVals.shape[0]
        centerNum = int(self.compRowIndexList.__len__() * self.centerRatio)
        attrNum = self.dbVals.shape[1]
        rowIndex = 0
        cIndex = 0
        while cIndex < centerNum:
            rowIndex = random.randint(0, size-1)
            if (rowIndex in self.centerList) or (rowIndex in self.delCompRowIndexList):
                continue
            self.centerList.append(rowIndex)
            cluster = Cluster(cIndex)
            self.clusters.append(cluster)
            cIndex +=1
        sim = 0
        tempsim = 0
        centerIndex = -1
        loop = 5
        tau = 0.8
        preList = []
        itertime = 0
        while itertime < loop and not self.converge(self.centerList, preList, tau):
            preList.clear()
            preList.extend(self.centerList)
            for i in range(size):
                if self.delCompRowIndexList.__contains__(i):
                    continue
                vals1 = self.dbVals[i]
                sim = -1
                for cIndex in range(centerNum):
                    tempsim = 0
                    rowIndex = preList[cIndex]
                    if rowIndex == i:
                        centerIndex = cIndex
                        break

                    vals2 = self.dbVals[rowIndex]
                    for attrIndex in range(attrNum):
                        if self.selList.__contains__(attrIndex):
                            continue
                        if math.isclose(vals1[attrIndex], vals2[attrIndex]):
                            tempsim += 1
                    if tempsim > sim:
                        sim = tempsim
                        centerIndex = cIndex
                if centerIndex == -1:
                    continue
                self.clusters[centerIndex].addRowIndex(i)
            self.centerList.clear()
            for cIndex in range(centerNum):
                cluster = self.clusters[cIndex]
                tempIndex = self.getCenter(cluster)
                self.centerList.append(tempIndex)
            itertime += 1

    def getCenter(self, cluster):
        index = -1
        clusterRowIndexList = cluster.getRowIndexList()
        clusterRowNum = clusterRowIndexList.__len__()
        attrNum = self.dbVals.shape[1]
        simMatrix = [[0] * (clusterRowNum + 1) for _ in range(clusterRowNum)]
        rowIndexI = 0
        rowIndexJ = 0
        sim = 0
        for i in range(clusterRowNum):
            rowIndexI = clusterRowIndexList[i]
            vals1 = self.dbVals[rowIndexI]
            for j in range(i + 1, clusterRowNum):
                sim = 0
                rowIndexJ = clusterRowIndexList[j]
                vals2 = self.dbVals[rowIndexJ]
                for attrIndex in range(attrNum):
                    if self.selList.__contains__(attrIndex):
                        if (self.misRowIndexList.__contains__(rowIndexI)) or (self.misRowIndexList.__contains__(rowIndexJ)):
                            sim += 0
                        elif math.isclose(vals1[attrIndex], vals2[attrIndex]):
                            sim += 1
                    elif math.isclose(vals1[attrIndex], vals2[attrIndex]):
                        sim += 1
                simMatrix[i][j] = sim
                simMatrix[j][i] = sim
        sim = -1
        for i in range(clusterRowNum):
            for j in range(clusterRowNum):
                simMatrix[i][clusterRowNum] += simMatrix[i][j]
            if simMatrix[i][clusterRowNum] > sim:
                sim = simMatrix[i][clusterRowNum]
                index = i
        index = self.rowIndexList.index(clusterRowIndexList[index])
        return index

    def converge(self, centerList, preList, tau):
        isConverge = False
        centerSize = centerList.__len__()
        if (centerSize == 0) or (preList.__len__() == 0):
            return isConverge
        num = 0
        for i in range(centerSize):
            if math.isclose(centerList[i], preList[i]):
                num += 1
        if num / centerSize > tau:
            isConverge = True
        return isConverge

    def setCenterRatio(self, centerRatio):
        self.centerRatio = centerRatio

    def setSelList(self, selList):
        self.selList = selList

    def setCluster(self, clusters):
        self.clusters = clusters

    def setSeed(self, seed):
        self.seed = seed
