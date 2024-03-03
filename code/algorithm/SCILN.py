# -*- coding: utf-8 -*-
from algorithm.BaseMissing import BaseMissing
from time import time
from entity.TupleCandidate import TupleCandidate
from entity.Weight import Weight
import numpy as np
import copy
from sklearn.neighbors import KDTree

from collections import Iterable
from gurobipy import *


class SCILN(BaseMissing):

    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.setCellMap()
        self.__H = float('inf')
        self.misTpCanMap = {}
        self.stack = []
        self.WeightList = []
        self.WList = []
        self.clusterMembersList = []
        self.clusterCenterWeightList = []
        self.assignedRowIndexList = []
        self.distancePairs = [[[[]]]]
        self.WCopy = [[]]
        self.W = [[]]
        self.epsilon = None

    def mainSCI(self):
        print('initVals')
        self.initVals()

        print('genTpCans')
        self.genTpCans()

        print('lpSover')
        self.lpSover()

        print('round')
        self.myRound()

    def lpSover(self):

        print('set object')
        self.setObject()

        print('analyze solution')
        self.analyResult()

    def myRound(self):
        compSize = self.compRowIndexList.__len__()

        while (self.WList.__len__() > 0) and (self.clusterMembersList.__len__() < self.K):
            minElementIndex = self.WList.index(min(self.WList))
            weight = self.WeightList[minElementIndex]
            i = weight.getI()
            p = weight.getP()

            self.clusterCenterWeightList.append(weight)
            if i < compSize:
                rowIndex = self.compRowIndexList[i]
                clusterMemberList = []
                clusterMemberList.append(rowIndex)
                self.clusterMembersList.append(clusterMemberList)
                del (self.WList[minElementIndex])
                del (self.WeightList[minElementIndex])
                self.assignedRowIndexList.append(rowIndex)
                self.findExtNei(True, i, rowIndex, p)
            else:
                rowIndex = self.misRowIndexList[i - compSize]
                clusterMemberList = []
                clusterMemberList.append(rowIndex)
                self.clusterMembersList.append(clusterMemberList)
                self.assignedRowIndexList.append(rowIndex)
                tpCanLists = self.misTpCanMap[rowIndex]
                tpCandidate = tpCanLists[p]
                tpCanList = tpCandidate.getTpCanList()
                misAttrList = tpCandidate.getMisAttrList()
                misAttrNum = misAttrList.__len__()
                for mi in range(misAttrNum):
                    misAttrIndex = misAttrList[mi]
                    modify = tpCanList[mi]
                    position = (rowIndex, misAttrIndex)
                    cell = self.assist.getCellByPosition(self.cells, position)
                    cell.modify = modify

                tpCanSize = tpCanLists.__len__()
                for j in range(tpCanSize):
                    removeWeight = Weight(i, j)
                    removeElementIndex = self.WeightList.index(removeWeight) if removeWeight in self.WeightList else -1
                    if removeElementIndex != -1:
                        del (self.WList[removeElementIndex])
                        del (self.WeightList[removeElementIndex])
                self.findExtNei(False, i, rowIndex, p)

        if self.WList.__len__() > 0:
            self.handleRemain()

    def handleRemain(self):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        clusterNum = self.clusterCenterWeightList.__len__()

        for l in range(compSize):
            neiRowIndex = self.compRowIndexList[l]
            if self.assignedRowIndexList.__contains__(neiRowIndex):
                continue
            minDistance = float('inf')
            minCenterIndex = -1
            for ci in range(clusterNum):
                weight = self.clusterCenterWeightList[ci]
                i = weight.getI()
                p = weight.getP()
                if i < compSize:
                    distance = self.distancePairs[i][0][l][0]
                else:
                    distance = self.distancePairs[i][p][l][0]
                if distance < minDistance:
                    minDistance = distance
                    minCenterIndex = ci
            clusterMemberList = self.clusterMembersList[minCenterIndex]
            clusterMemberList.append(neiRowIndex)
            self.assignedRowIndexList.append(neiRowIndex)
        for l in range(compSize, totalSize):
            neiRowIndex = self.misRowIndexList[l - compSize]
            if self.assignedRowIndexList.__contains__(neiRowIndex):
                continue
            tpCanLists2 = self.misTpCanMap[neiRowIndex]
            tpCanSize2 = tpCanLists2.__len__()
            minDistance = float('inf')
            minCenterIndex = -1
            minCanIndex = -1
            for ci in range(clusterNum):
                weight = self.clusterCenterWeightList[ci]
                i = weight.getI()
                p = weight.getP()

                for q in range(tpCanSize2):
                    if self.flags[i][0][l][q] == True:
                        break
                    if i < compSize:
                        distance = self.distancePairs[i][0][l][q]
                    else:
                        distance = self.distancePairs[i][p][l][q]
                    if distance < minDistance:
                        minDistance = distance
                        minCanIndex = q
                        minCenterIndex = ci
            tpCandidate2 = tpCanLists2[minCanIndex]
            tpCanList2 = tpCandidate2.getTpCanList()
            misAttrList2 = tpCandidate2.getMisAttrList()
            misAttrNum2 = misAttrList2.__len__()
            for mi in range(misAttrNum2):
                misAttrIndex = misAttrList2[mi]
                modify = tpCanList2[mi]
                position = (neiRowIndex, misAttrIndex)
                cell = self.assist.getCellByPosition(self.cells, position)
                cell.modify = modify
            clusterMemberList = self.clusterMembersList[minCenterIndex]
            clusterMemberList.append(neiRowIndex)
            self.assignedRowIndexList.append(neiRowIndex)

    def findExtNei(self, isComp, i, rowIndex, tpCanIndex1):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        if isComp:
            Wip = self.WCopy[i][0]
            for l in range(compSize):
                neiRowIndex = self.compRowIndexList[l]
                if self.assignedRowIndexList.__contains__(neiRowIndex):
                    continue
                Wlq = self.WCopy[l][0]
                distance = self.distancePairs[i][0][l][0]
                threshold = (1 + self.epsilon) * Wip + (1 + self.epsilon) * Wlq
                if (distance <= threshold) or (distance < 0.000000000001):
                    self.flags[i][0][l][0] = True
                    clusterMemberList = self.clusterMembersList[self.clusterMembersList.__len__() - 1]
                    clusterMemberList.append(neiRowIndex)
                    weight = Weight(l, 0)
                    removeElementIndex = self.WeightList.index(weight) if weight in self.WeightList else -1
                    if removeElementIndex != -1:
                        del (self.WList[removeElementIndex])
                        del (self.WeightList[removeElementIndex])
                    self.assignedRowIndexList.append(neiRowIndex)

            for l in range(compSize, totalSize):
                neiRowIndex = self.misRowIndexList[l - compSize]
                if self.assignedRowIndexList.__contains__(neiRowIndex):
                    continue
                tpCanLists2 = self.misTpCanMap[neiRowIndex]
                tpCanSize2 = tpCanLists2.__len__()
                minDistance = float('inf')
                minCanIndex = -1
                for q in range(tpCanSize2):
                    if self.flags[i][0][l][q] == True:
                        break
                    Wlq = self.WCopy[l][q]
                    distance = self.distancePairs[i][0][l][q]
                    threshold = (1 + self.epsilon) * Wip + (1 + self.epsilon) * Wlq
                    if (distance <= threshold) or (distance < 0.000000000001):
                        if distance < minDistance:
                            minDistance = distance
                            minCanIndex = q

                if minCanIndex != -1:
                    tpCandidate2 = tpCanLists2[minCanIndex]
                    tpCanList2 = tpCandidate2.getTpCanList()
                    misAttrList2 = tpCandidate2.getMisAttrList()
                    misAttrNum2 = misAttrList2.__len__()
                    for mi in range(misAttrNum2):
                        misAttrIndex = misAttrList2[mi]
                        modify = tpCanList2[mi]
                        position = (neiRowIndex, misAttrIndex)
                        cell = self.assist.getCellByPosition(self.cells, position)
                        cell.modify = modify
                    clusterMemberList = self.clusterMembersList[self.clusterMembersList.__len__() - 1]
                    clusterMemberList.append(neiRowIndex)
                    for q in range(tpCanSize2):
                        self.flags[i][0][l][q] = True
                        weight = Weight(l, q)
                        removeElementIndex = self.WeightList.index(weight) if weight in self.WeightList else -1
                        if removeElementIndex != -1:
                            del (self.WList[removeElementIndex])
                            del (self.WeightList[removeElementIndex])
                    self.assignedRowIndexList.append(neiRowIndex)
        else:
            Wip = self.WCopy[i][tpCanIndex1]
            for l in range(compSize):
                if self.flags[i][tpCanIndex1][l][0] == True:
                    continue
                neiRowIndex = self.compRowIndexList[l]
                if self.assignedRowIndexList.__contains__(neiRowIndex):
                    continue
                Wlq = self.WCopy[l][0]
                distance = self.distancePairs[i][tpCanIndex1][l][0]
                threshold = (1 + self.epsilon) * Wip + (1 + self.epsilon) * Wlq
                if (distance <= threshold) or (distance < 0.000000000001):
                    self.flags[i][tpCanIndex1][l][0] = True
                    clusterMemberList = self.clusterMembersList[self.clusterMembersList.__len__() - 1]
                    clusterMemberList.append(neiRowIndex)
                    weight = Weight(l, 0)
                    removeElementIndex = self.WeightList.index(weight) if weight in self.WeightList else -1
                    if removeElementIndex != -1:
                        del (self.WList[removeElementIndex])
                        del (self.WeightList[removeElementIndex])
                    self.assignedRowIndexList.append(neiRowIndex)

            for l in range(compSize, totalSize):
                neiRowIndex = self.misRowIndexList[l - compSize]
                if self.assignedRowIndexList.__contains__(neiRowIndex):
                    continue
                tpCanLists2 = self.misTpCanMap[neiRowIndex]
                tpCanSize2 = tpCanLists2.__len__()
                minDistance = float('inf')
                minCanIndex = -1
                for q in range(tpCanSize2):
                    if self.flags[i][tpCanIndex1][l][q] == True:
                        break
                    Wlq = self.WCopy[l][q]
                    distance = self.distancePairs[i][tpCanIndex1][l][q]
                    threshold = (1 + self.epsilon) * Wip + (1 + self.epsilon) * Wlq
                    if (distance <= threshold) or (distance < 0.000000000001):
                        if distance < minDistance:
                            minDistance = distance
                            minCanIndex = q

                if minCanIndex != -1:
                    tpCandidate2 = tpCanLists2[minCanIndex]
                    tpCanList2 = tpCandidate2.getTpCanList()
                    misAttrList2 = tpCandidate2.getMisAttrList()
                    misAttrNum2 = misAttrList2.__len__()
                    for mi in range(misAttrNum2):
                        misAttrIndex = misAttrList2[mi]
                        modify = tpCanList2[mi]
                        position = (neiRowIndex, misAttrIndex)
                        cell = self.assist.getCellByPosition(self.cells, position)
                        cell.modify = modify
                    clusterMemberList = self.clusterMembersList[self.clusterMembersList.__len__() - 1]
                    clusterMemberList.append(neiRowIndex)
                    for q in range(tpCanSize2):
                        self.flags[i][tpCanIndex1][l][q] = True
                        weight = Weight(l, q)
                        removeElementIndex = self.WeightList.index(weight) if weight in self.WeightList else -1
                        if removeElementIndex != -1:
                            del (self.WList[removeElementIndex])
                            del (self.WeightList[removeElementIndex])

                    self.assignedRowIndexList.append(neiRowIndex)

    def analyResult(self):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        self.W = [[] for _ in range(totalSize)]
        self.WCopy = [[] for _ in range(totalSize)]

        for i in range(compSize):
            tmpVal = []
            self.W[i] = [0]
            self.WCopy[i] = [0]
            for l in range(compSize):
                distance = self.distancePairs[i][0][l][0]
                tmpVal.append(distance)

            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanLists2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanLists2.__len__()
                for q in range(tpCanSize2):
                    distance = self.distancePairs[i][0][l][q]
                    tmpVal.append(distance)
            self.W[i][0] = self.topK_distance(tmpVal, self.k_distance)
            self.WCopy[i][0] = self.W[i][0]
            weight = Weight(i, 0, self.W[i][0])
            self.WeightList.append(weight)
            self.WList.append(self.W[i][0])
        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanLists1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanLists1.__len__()
            self.W[i] = [0] * tpCanSize1
            self.WCopy[i] = [0] * tpCanSize1

            for p in range(tpCanSize1):
                tmpVal = []
                for l in range(compSize):
                    distance = self.distancePairs[i][p][l][0]
                    tmpVal.append(distance)

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanLists2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanLists2.__len__()
                    for q in range(tpCanSize2):
                        distance = self.distancePairs[i][p][l][q]
                        tmpVal.append(distance)
                self.W[i][p] = self.topK_distance(tmpVal, self.k_distance)
                self.WCopy[i][p] = self.W[i][p]
                weight = Weight(i, p, self.W[i][p])
                self.WeightList.append(weight)
                self.WList.append(self.W[i][p])

    def setObject(self):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        tpCanSize2_max = 1
        # print("self.misTpCanMap:",self.misTpCanMap)
        # print("len(self.misTpCanMap)",len(self.misTpCanMap))
        for i in self.misTpCanMap.keys():
            size = self.misTpCanMap[i].__len__()
            if size > tpCanSize2_max:
                tpCanSize2_max = size
                
        # print("compSize",compSize,"misSize",misSize,"totalSize",totalSize,"tpCanSize2_max",tpCanSize2_max)
        self.distancePairs = np.empty((totalSize,tpCanSize2_max, totalSize,tpCanSize2_max), dtype=np.float32)
        flags = np.empty((totalSize,tpCanSize2_max, totalSize,tpCanSize2_max), dtype=bool)
        dbmax,dbmin,EPSILON = self.dbmax,self.dbmin,self.EPSILON
        for i in range(compSize):
            if i % 100 ==0:
                print(i)
            compRowIndex1 = self.compRowIndexList[i]
            vals1 = self.dbVals[compRowIndex1]
            for l in range(compSize):
                compRowIndex2 = self.compRowIndexList[l]
                vals2 = self.dbVals[compRowIndex2]
                dis = (np.array(vals1) - np.array(vals2)) / (dbmax - dbmin + EPSILON)
                self.distancePairs[i][0][l][0] = np.sum(dis ** 2) ** 0.5
                flags[i][0][l][0] = False

            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanLists2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanLists2.__len__()
                vals2 = copy.deepcopy(self.dbVals[misRowIndex2])
                for q in range(tpCanSize2):
                    tpCandidate2 = tpCanLists2[q]
                    tpCanList2 = tpCandidate2.getTpCanList()
                    misAttrList2 = tpCandidate2.getMisAttrList()
                    for mi in range(misAttrList2.__len__()):
                        misAttrIndex = misAttrList2[mi]
                        canVal2 = tpCanList2[mi]
                        vals2[misAttrIndex] = canVal2
                    dis = (np.array(vals1) - np.array(vals2)) / (dbmax - dbmin + EPSILON)
                    self.distancePairs[i][0][l][q] = np.sum(dis ** 2) ** 0.5
                    flags[i][0][l][q] = False
                    
        for i in range(compSize, totalSize):
            if i % 1000 ==0:
                print(i)
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanLists1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanLists1.__len__()

            for p in range(tpCanSize1):
                tpCandidate1 = tpCanLists1[p]
                misAttrList1 = tpCandidate1.getMisAttrList()
                tpCanList1 = tpCandidate1.getTpCanList()
                vals1 = copy.deepcopy(self.dbVals[misRowIndex1])
                for attri in misAttrList1:
                    vals1[attri] = tpCanList1[misAttrList1.index(attri)]

                for l in range(compSize):
                    compRowIndex2 = self.compRowIndexList[l]
                    vals2 = self.dbVals[compRowIndex2]
                    dis = (np.array(vals1) - np.array(vals2)) / (dbmax - dbmin + EPSILON)
                    self.distancePairs[i][p][l][0] = np.sum(dis ** 2) ** 0.5
                    flags[i][p][l][0] = False

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanLists2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanLists2.__len__()
                    for q in range(tpCanSize2):
                        tpCandidate2 = tpCanLists2[q]
                        tpCanList2 = tpCandidate2.getTpCanList()
                        misAttrList2 = tpCandidate2.getMisAttrList()
                        vals2 = copy.deepcopy(self.dbVals[misRowIndex2])
                        for attrj in misAttrList2:
                            vals2[attrj] = tpCanList2[misAttrList2.index(attrj)]

                        dis = (np.array(vals1) - np.array(vals2)) / (dbmax - dbmin + EPSILON)
                        self.distancePairs[i][p][l][q] = np.sum(dis ** 2) ** 0.5
                        flags[i][p][l][q] = False

        self.flags = flags

    def genTpCans(self):
        misRowNum = self.misRowIndexList.__len__()
        subDistances = [[0] * self.dbVals.shape[0] for _ in range(misRowNum)]
        sIndexes = [[0] * self.K_Candidate for _ in range(misRowNum)]
        sDistances = [[0] * self.K_Candidate for _ in range(misRowNum)]
        for ri in range(misRowNum):
            rowIndex = self.misRowIndexList[ri]
            tpCanList = []
            self.misTpCanMap[rowIndex] = tpCanList
            misAttrIndexList = self.misRowAttrIndexList[ri]
            misAttrNum = misAttrIndexList.__len__()
            self.calcDisNorm(rowIndex, subDistances[ri])
            self.findCompKnn(subDistances[ri], sIndexes[ri], sDistances[ri])
            knnIndexes = sIndexes[ri]
            cellCanLists = []
            for mi in range(misAttrNum):
                misAttrIndex = misAttrIndexList[mi]
                canList = []
                for ki in range(self.K_Candidate):
                    kIndex = knnIndexes[ki]
                    canVal = self.dbVals[kIndex][misAttrIndex]
                    canList.append(canVal)
                cellCanLists.append(canList)

            iterateArray = [0] * self.K_Candidate
            for i in range(self.K_Candidate):
                iterateArray[i] = i

            self.myEnumerate(iterateArray, misAttrNum, 0, cellCanLists, misAttrIndexList, rowIndex)

    def myEnumerate(self, iterateArray, targ, cur, cellCanLists, misAttrIndexList, misRowIndex):
        if cur == targ:
            tpCan = []
            for mi in range(misAttrIndexList.__len__()):
                cellCanList = cellCanLists[mi]
                canIndex = self.stack[mi]
                cellCan = cellCanList[canIndex]
                tpCan.append(cellCan)

            tupleCandidate = TupleCandidate(misAttrIndexList, tpCan)
            tpCanLists = self.misTpCanMap[misRowIndex]
            if not tpCanLists.__contains__(tupleCandidate):
                tpCanLists.append(tupleCandidate)
            return
        for i in range(iterateArray.__len__()):
            self.stack.append(iterateArray[i])
            self.myEnumerate(iterateArray, targ, cur + 1, cellCanLists, misAttrIndexList, misRowIndex)
            self.stack.pop()

    def topK_distance(self, distance, k):
        distance = np.array(distance)
        tmp = distance[np.argpartition(distance, k)[:k]]
        return np.mean(tmp)

    def getK(self):
        return self.K

    def setK(self, k):
        self.K = k

    def setKDistance(self, k_distance):
        self.k_distance = k_distance

    def getK_Candidate(self):
        return self.K_Candidate

    def setK_Candidate(self, k_Candidate):
        self.K_Candidate = k_Candidate

    def getEpsilon(self):
        return self.epsilon

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
