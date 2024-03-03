# -*- coding: utf-8 -*-
from algorithm.BaseMissing import BaseMissing
from time import time
from entity.TupleCandidate import TupleCandidate
from entity.Weight import Weight
import numpy as np
import copy
from gurobipy import *


class SCILP(BaseMissing):

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
        self.zzzz=[]
        self.epsilon = None
        self.centerFlags = [[]]

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
        model = Model()
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        x = [[] for _ in range(totalSize)]
        y = [[] for _ in range(totalSize)]
        z = [[[[]]] for _ in range(totalSize)]

        for i in range(compSize):
            y[i] = [0]
            z[i] = [[[] for _ in range(totalSize)]]
            for l in range(compSize):
                z[i][0][l] = [0]
            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanList2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanList2.__len__()
                z[i][0][l] = [0] * tpCanSize2

        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanList1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanList1.__len__()
            x[i] = [0] * tpCanSize1
            y[i] = [0] * tpCanSize1
            z[i] = [[[] for _ in range(totalSize)] for _ in range(tpCanSize1)]
            for p in range(tpCanSize1):
                for l in range(compSize):
                    z[i][p][l] = [0]
                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanList2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanList2.__len__()
                    z[i][p][l] = [0] * tpCanSize2

        print('add variable')
        self.addVar(model, x, y, z)
        model.update()

        print('set object')
        self.setObject(model, z)

        print('add MaxOneConstraint')
        self.addMaxOneConstraint(model, x)

        print('add KClusterConstraint')
        self.addKClusterConstraint(model, y)

        print('add ClusterCenterConstraint')
        self.addClusterCenterConstraint(model, z)

        print('add ZYXSConstraint')
        self.addZYXConstraint(model, x, y, z)
        model.optimize()

        print('analyze solution')
        self.analyResult(z)
        

    def printResult(self, x, y, z):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        xVal = [[] for _ in range(totalSize)]
        yVal = [[] for _ in range(totalSize)]
        zVal = [[[[]]] for _ in range(totalSize)]

        for i in range(compSize):
            yVal[i] = [0]
            yVal[i][0] = y[i][0].X
            print("yVal[" + str(i) + "][0]:" + str(yVal[i][0]) + "," + "RowIndex:" + str(self.compRowIndexList[i]))
            zVal[i] = [[[] for _ in range(totalSize)]]
            for l in range(compSize):
                zVal[i][0][l] = [0]
                zVal[i][0][l][0] = z[i][0][l][0].X
                print("zVal[" + str(i) + "][0][" + str(l) + "][0]:" + str(zVal[i][0][l][0]))

            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanList2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanList2.__len__()
                zVal[i][0][l] = [0] * tpCanSize2
                for q in range(tpCanSize2):
                    zVal[i][0][l][q] = z[i][0][l][q].X
                    print("zVal[" + str(i) + "][0][" + str(l) + "][" + str(q) + "]:" + str(zVal[i][0][l][q]) + "\t")

        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanList1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanList1.__len__()
            xVal[i] = [0] * tpCanSize1
            yVal[i] = [0] * tpCanSize1

            for p in range(tpCanSize1):
                xVal[i][p] = x[i][p].X
                print("xVal[" + str(i) + "][" + str(p) + "]:" + str(xVal[i][p]) + "," + "RowIndex:" + str(misRowIndex1) + ","
                      + str(tpCanList1[p].getTpCanList()[0]) + "\t")

            for p in range(tpCanSize1):
                yVal[i][p] = y[i][p].X
                print("yVal[" + str(i) + "][" + str(p) + "]:" + str(yVal[i][p]) + "," + "RowIndex:" + str(misRowIndex1) + "\t")

            zVal[i] = [[[] for _ in range(totalSize)] for _ in range(tpCanSize1)]
            for p in range(tpCanSize1):
                for l in range(compSize):
                    zVal[i][p][l] = [0]
                    zVal[i][p][l][0] = z[i][p][l][0].X
                    print("zVal[" + str(i) + "][" + str(p) + "][" + str(l) + "][" + "0" + "]:" + str(zVal[i][p][l][0]) + "\t")

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanList2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanList2.__len__()
                    zVal[i][p][l] = [0] * tpCanSize2
                    for q in range(tpCanSize2):
                        zVal[i][p][l][q] = z[i][p][l][q].X
                        print("zVal[" + str(i) + "][" + str(p) + "][" + str(l) + "][" + str(q) + "]:" + str(zVal[i][p][l][q]) + "\t")

    def myRound(self):
        compSize = self.compRowIndexList.__len__()
        while (self.WList.__len__() > 0) and (self.clusterMembersList.__len__() < self.K):
            minElementIndex = self.WList.index(min(self.WList))
            weight = self.WeightList[minElementIndex]
            i = weight.getI()
            p = weight.getP()
            if (weight.getWVal() < 0.00000000000000000001) and (self.centerFlags[i][p] == False):
                del (self.WList[minElementIndex])
                del (self.WeightList[minElementIndex])
                continue

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
    def analyResult(self, z):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        distance = -1
        self.W = [[] for _ in range(totalSize)]
        self.WCopy = [[] for _ in range(totalSize)]
        tmpVal = 0
        zVal = 0
        self.centerFlags = [[] for _ in range(totalSize)]
        for i in range(compSize):
            self.W[i] = [0]
            self.WCopy[i] = [0]
            self.centerFlags[i] = [0]
            self.centerFlags[i][0] = True
            for l in range(compSize):
                distance = self.distancePairs[i][0][l][0]
                zVal = z[i][0][l][0].X
                tmpVal = zVal * distance
                self.W[i][0] += tmpVal
                self.WCopy[i][0] += tmpVal
            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanLists2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanLists2.__len__()
                for q in range(tpCanSize2):
                    distance = self.distancePairs[i][0][l][q]
                    zVal = z[i][0][l][q].X
                    tmpVal = zVal * distance
                    self.W[i][0] += tmpVal
                    self.WCopy[i][0] += tmpVal
            weight = Weight(i, 0, self.W[i][0])
            self.WeightList.append(weight)
            self.WList.append(self.W[i][0])
        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanLists1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanLists1.__len__()
            self.W[i] = [0] * tpCanSize1
            self.WCopy[i] = [0] * tpCanSize1
            self.centerFlags[i] = [0] * tpCanSize1
            for p in range(tpCanSize1):
                self.centerFlags[i][p] = False
                for l in range(compSize):
                    distance = self.distancePairs[i][p][l][0]
                    zVal = z[i][p][l][0].X
                    tmpVal = zVal * distance
                    self.W[i][p] += tmpVal
                    self.WCopy[i][p] += tmpVal
                    if zVal > 0:
                        self.centerFlags[i][p] = True
                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanLists2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanLists2.__len__()
                    for q in range(tpCanSize2):
                        distance = self.distancePairs[i][p][l][q]
                        zVal = z[i][p][l][q].X
                        tmpVal = zVal * distance
                        self.W[i][p] += tmpVal
                        self.WCopy[i][p] += tmpVal
                        if zVal > 0:
                            self.centerFlags[i][p] = True
                weight = Weight(i, p, self.W[i][p])
                self.WeightList.append(weight)
                self.WList.append(self.W[i][p])

    def addZYXConstraint(self, model, x, y, z):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        for i in range(compSize):
            for l in range(compSize):
                zyExpr = LinExpr()
                zyExpr.addTerms(1.0, z[i][0][l][0])
                model.addConstr(zyExpr, GRB.LESS_EQUAL, y[l][0], "ZYConstraint")
            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanList2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanList2.__len__()
                for q in range(tpCanSize2):
                    zyExpr = LinExpr()
                    zyExpr.addTerms(1.0, z[i][0][l][q])
                    model.addConstr(zyExpr, GRB.LESS_EQUAL, y[l][q], "ZYConstraint")
                    yxExpr = LinExpr()
                    yxExpr.addTerms(1.0, y[l][q])
                    model.addConstr(yxExpr, GRB.LESS_EQUAL, x[l][q], "YXConstraint")

        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanList1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanList1.__len__()
            for p in range(tpCanSize1):
                for l in range(compSize):
                    zyExpr = LinExpr()
                    zyExpr.addTerms(1.0, z[i][p][l][0])
                    model.addConstr(zyExpr, GRB.LESS_EQUAL, y[l][0], "ZYConstraint")
                    zx2Expr = LinExpr()
                    zx2Expr.addTerms(1.0, z[i][p][l][0])
                    model.addConstr(zx2Expr, GRB.LESS_EQUAL, x[i][p], "ZX2Constraint")

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanList2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanList2.__len__()
                    for q in range(tpCanSize2):
                        zyExpr = LinExpr()
                        zyExpr.addTerms(1.0, z[i][p][l][q])
                        model.addConstr(zyExpr, GRB.LESS_EQUAL, y[l][q], "ZYConstraint")
                        yxExpr = LinExpr()
                        yxExpr.addTerms(1.0, y[l][q])
                        model.addConstr(yxExpr, GRB.LESS_EQUAL, x[l][q], "YXConstraint")
                        zx2Expr = LinExpr()
                        zx2Expr.addTerms(1.0, z[i][p][l][q])
                        model.addConstr(zx2Expr, GRB.LESS_EQUAL, x[i][p], "ZX2Constraint")

    def addClusterCenterConstraint(self, model, z):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        for i in range(compSize):
            expr = LinExpr()
            for l in range(compSize):
                expr.addTerms(1.0, z[i][0][l][0])
            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanList2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanList2.__len__()
                for q in range(tpCanSize2):
                    expr.addTerms(1.0, z[i][0][l][q])
            model.addConstr(expr, GRB.EQUAL, 1.0, "ClusterCenterConstraint")

        for i in range(compSize, totalSize):
            expr = LinExpr()
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanList1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanList1.__len__()
            for p in range(tpCanSize1):
                for l in range(compSize):
                    expr.addTerms(1.0, z[i][p][l][0])

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanList2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanList2.__len__()
                    if misRowIndex1 == misRowIndex2:
                        for q in range(tpCanSize2):
                            if p != q:
                                exclusionExpr = LinExpr()
                                exclusionExpr.addTerms(1.0, z[i][p][l][q])
                                model.addConstr(exclusionExpr, GRB.EQUAL, 0, "ClusterExclusionConstraint")

                    for q in range(tpCanSize2):
                        expr.addTerms(1.0, z[i][p][l][q])
            model.addConstr(expr, GRB.EQUAL, 1.0, "ClusterCenterConstraint")

    def addKClusterConstraint(self, model, y):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        expr = LinExpr()
        for i in range(compSize):
            expr.addTerms(1.0, y[i][0])
        for i in range(compSize, totalSize):
            misRowIndex = self.misRowIndexList[i - compSize]
            tpCanList = self.misTpCanMap[misRowIndex]
            tpCanSize = tpCanList.__len__()
            for p in range(tpCanSize):
                expr.addTerms(1.0, y[i][p])
        model.addConstr(expr, GRB.LESS_EQUAL, self.K, "KClusterConstraint")

    def addMaxOneConstraint(self, model, x):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        for i in range(compSize, totalSize):
            misRowIndex = self.misRowIndexList[i - compSize]
            tpCanList = self.misTpCanMap[misRowIndex]
            tpCanSize = tpCanList.__len__()
            expr = LinExpr()
            for p in range(tpCanSize):
                expr.addTerms(1.0, x[i][p])
            model.addConstr(expr, GRB.EQUAL, 1.0, "MaxOneConstraint:" + str(i))

    def setObject(self, model, z):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        self.distancePairs = [[[[]]] for _ in range(totalSize)]
        flags = [[[[]]] for _ in range(totalSize)]
        expr = LinExpr()
        for i in range(compSize):
            self.distancePairs[i] = [[[] for _ in range(totalSize)]]
            flags[i] = [[[] for _ in range(totalSize)]]
            compRowIndex1 = self.compRowIndexList[i]
            vals1 = self.dbVals[compRowIndex1]
            for l in range(compSize):
                self.distancePairs[i][0][l] = [0]
                flags[i][0][l] = [0]
                compRowIndex2 = self.compRowIndexList[l]
                vals2 = self.dbVals[compRowIndex2]
                distance = self.calcNormDisBtwTwoArrays(vals1, vals2)
                self.distancePairs[i][0][l][0] = distance
                flags[i][0][l][0] = False
                expr.addTerms(distance, z[i][0][l][0])

            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanLists2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanLists2.__len__()
                self.distancePairs[i][0][l] = [0] * tpCanSize2
                flags[i][0][l] = [0] * tpCanSize2
                vals2 = copy.deepcopy(self.dbVals[misRowIndex2])
                for q in range(tpCanSize2):
                    tpCandidate2 = tpCanLists2[q]
                    tpCanList2 = tpCandidate2.getTpCanList()
                    misAttrList2 = tpCandidate2.getMisAttrList()
                    for mi in range(misAttrList2.__len__()):
                        misAttrIndex = misAttrList2[mi]
                        canVal2 = tpCanList2[mi]
                        vals2[misAttrIndex] = canVal2
                    distance = self.calcNormDisBtwTwoArrays(vals1, vals2)
                    self.distancePairs[i][0][l][q] = distance
                    flags[i][0][l][q] = False
                    expr.addTerms(distance, z[i][0][l][q])

        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanLists1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanLists1.__len__()
            self.distancePairs[i] = [[[] for _ in range(totalSize)] for _ in range(tpCanSize1)]
            flags[i] = [[[] for _ in range(totalSize)] for _ in range(tpCanSize1)]
            for p in range(tpCanSize1):
                tpCandidate1 = tpCanLists1[p]
                misAttrList1 = tpCandidate1.getMisAttrList()
                tpCanList1 = tpCandidate1.getTpCanList()
                vals1 = copy.deepcopy(self.dbVals[misRowIndex1])
                for attri in misAttrList1:
                    vals1[attri] = tpCanList1[misAttrList1.index(attri)]
                for l in range(compSize):
                    self.distancePairs[i][p][l] = [0]
                    flags[i][p][l] = [0]
                    compRowIndex2 = self.compRowIndexList[l]
                    vals2 = self.dbVals[compRowIndex2]
                    distance = self.calcNormDisBtwTwoArrays(vals1, vals2)
                    self.distancePairs[i][p][l][0] = distance
                    flags[i][p][l][0] = False
                    expr.addTerms(distance, z[i][p][l][0])

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanLists2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanLists2.__len__()
                    self.distancePairs[i][p][l] = [0] * tpCanSize2
                    flags[i][p][l] = [0] * tpCanSize2
                    for q in range(tpCanSize2):
                        tpCandidate2 = tpCanLists2[q]
                        tpCanList2 = tpCandidate2.getTpCanList()
                        misAttrList2 = tpCandidate2.getMisAttrList()
                        vals2 = copy.deepcopy(self.dbVals[misRowIndex2])
                        for attrj in misAttrList2:
                            vals2[attrj] = tpCanList2[misAttrList2.index(attrj)]
                        distance = self.calcNormDisBtwTwoArrays(vals1, vals2)
                        self.distancePairs[i][p][l][q] = distance
                        flags[i][p][l][q] = False
                        expr.addTerms(distance, z[i][p][l][q])
        self.flags = flags
        model.setObjective(expr, GRB.MINIMIZE)

    def addVar(self, model, x, y, z):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        for i in range(compSize):
            y[i][0] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "y" + "[" + str(i) + "," + "0" + "]")
            for l in range(compSize):
                z[i][0][l][0] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z" + "[" + str(i) + "," + "0" + "," + str(l) + "," + "0" + "]")
            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanList2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanList2.__len__()
                for q in range(tpCanSize2):
                    z[i][0][l][q] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "z" + "[" + str(i) + "," + "0" + "," + str(l) + "," + str(q) + "]")

        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanList1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanList1.__len__()
            for p in range(tpCanSize1):
                x[i][p] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "x" + "[" + str(i) + "," + str(p) + "]")
                y[i][p] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "y" + "[" + str(i) + "," + str(p) + "]")
                for l in range(compSize):
                    z[i][p][l][0] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS,
                                                 "z" + "[" + str(i) + "," + str(p) + "," + str(l) + "," + "0" + "]")

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanList2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanList2.__len__()
                    for q in range(tpCanSize2):
                        z[i][p][l][q] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS,
                                                     "z" + "[" + str(i) + "," + str(p) + "," + str(l) + "," + str(q) + "]")

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

    def getK(self):
        return self.K

    def setK(self, k):
        self.K = k

    def getK_Candidate(self):
        return self.K_Candidate

    def setK_Candidate(self, k_Candidate):
        self.K_Candidate = k_Candidate

    def getEpsilon(self):
        return self.epsilon

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
