# -*- coding: utf-8 -*-


from algorithm.BaseMissing import BaseMissing
from time import time
from entity.TupleCandidate import TupleCandidate
from entity.Weight import Weight
import numpy as np
import copy
from gurobipy import *


class SCIILP(BaseMissing):

    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.setCellMap()
        self.__H = float('inf')
        self.misTpCanMap = {}
        self.y=[]
        self.stack = []
        self.WeightList = []
        self.WList = []
        self.clusterMembersList = []
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
        self.analyResult(x, y, z)


    def printResult(self, x, y, z, filename="opdic_ilp_res.txt"):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        xVal = [[] for _ in range(totalSize)]
        yVal = [[] for _ in range(totalSize)]
        zVal = [[[[]]] for _ in range(totalSize)]

        with open(filename, 'w') as file:
            for i in range(compSize):
                yVal[i] = [0]
                yVal[i][0] = y[i][0].X
                file.write(f"yVal[{i}][0]:{yVal[i][0]}, RowIndex:{self.compRowIndexList[i]}\n")
                zVal[i] = [[[] for _ in range(totalSize)]]
                for l in range(compSize):
                    zVal[i][0][l] = [0]
                    zVal[i][0][l][0] = z[i][0][l][0].X
                    file.write(f"zVal[{i}][0][{l}][0]:{zVal[i][0][l][0]} - distancePairs[{i}][0][{l}][0]{self.distancePairs[i][0][l][0]}\n")

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanList2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanList2.__len__()
                    zVal[i][0][l] = [0] * tpCanSize2
                    for q in range(tpCanSize2):
                        zVal[i][0][l][q] = z[i][0][l][q].X
                        file.write(f"zVal[{i}][0][{l}][{q}]:{zVal[i][0][l][q]} - distancePairs[{i}][0][{l}][{q}]{self.distancePairs[i][0][l][q]}\t")

            for i in range(compSize, totalSize):
                misRowIndex1 = self.misRowIndexList[i - compSize]
                tpCanList1 = self.misTpCanMap[misRowIndex1]
                tpCanSize1 = tpCanList1.__len__()
                xVal[i] = [0] * tpCanSize1
                yVal[i] = [0] * tpCanSize1

                for p in range(tpCanSize1):
                    xVal[i][p] = x[i][p].X
                    file.write(f"xVal[{i}][{p}]:{xVal[i][p]}, RowIndex:{misRowIndex1}, {tpCanList1[p].getTpCanList()[0]}\t\n")

                for p in range(tpCanSize1):
                    yVal[i][p] = y[i][p].X
                    file.write(f"yVal[{i}][{p}]:{yVal[i][p]}, RowIndex:{misRowIndex1}\t\n")

                zVal[i] = [[[] for _ in range(totalSize)] for _ in range(tpCanSize1)]
                for p in range(tpCanSize1):
                    for l in range(compSize):
                        zVal[i][p][l] = [0]
                        zVal[i][p][l][0] = z[i][p][l][0].X
                        file.write(f"zVal[{i}][{p}][{l}][0]:{zVal[i][p][l][0]} - distancePairs[{i}][{p}][{l}][0]{self.distancePairs[i][p][l][0]}\t")

                    for l in range(compSize, totalSize):
                        misRowIndex2 = self.misRowIndexList[l - compSize]
                        tpCanList2 = self.misTpCanMap[misRowIndex2]
                        tpCanSize2 = tpCanList2.__len__()
                        zVal[i][p][l] = [0] * tpCanSize2
                        for q in range(tpCanSize2):
                            zVal[i][p][l][q] = z[i][p][l][q].X
                            file.write(f"zVal[{i}][{p}][{l}][{q}]:{zVal[i][p][l][q]} - distancePairs[{i}][{p}][{l}][{q}]{self.distancePairs[i][p][l][q]}\t")

    def analyResult(self, x, y, z):
        compSize = self.compRowIndexList.__len__()
        misSize = self.misRowIndexList.__len__()
        totalSize = compSize + misSize
        for i in range(compSize, totalSize):
            misRowIndex = self.misRowIndexList[i - compSize]
            tpCanLists = self.misTpCanMap[misRowIndex]
            tpCanSize = tpCanLists.__len__()
            maxImputationVal = -1
            maxImputationIndex = -1
            for p in range(tpCanSize):
                xVal = x[i][p].X
                if xVal > maxImputationVal:
                    maxImputationVal = xVal
                    maxImputationIndex = p

            tupleCandidate = tpCanLists[maxImputationIndex]
            misAttrIndexList = tupleCandidate.getMisAttrList()
            tpCanList = tupleCandidate.getTpCanList()
            for j in range(misAttrIndexList.__len__()):
                misAttrIndex = misAttrIndexList[j]
                canVal = tpCanList[j]
                position = (misRowIndex, misAttrIndex)
                cell = self.assist.getCellByPosition(self.cells, position)
                cell.modify = canVal

        clusterCenterMap = {}
        clusterCenterIndexList = []
        clusterIndex = -1
        for i in range(compSize):
            if clusterCenterIndexList.__len__() >= self.K:
                break
            rowIndex = self.compRowIndexList[i]
            yVal = y[i][0].X
            if yVal > 0.99:
                clusterIndex += 1
                clusterCenterIndexList.append(i)
                clusterCenterMap[rowIndex] = clusterIndex
                clusterMemberList = []
                self.clusterMembersList.append(clusterMemberList)

        for i in range(compSize, totalSize):
            if clusterCenterIndexList.__len__() >= self.K:
                break
            misRowIndex = self.misRowIndexList[i - compSize]
            tpCanLists = self.misTpCanMap[misRowIndex]
            tpCanSize = tpCanLists.__len__()

            for p in range(tpCanSize):
                yVal = y[i][p].X
                
                if yVal > 0.99:
                    clusterIndex += 1
                    clusterCenterIndexList.append(i)
                    clusterCenterMap[i] = clusterIndex
                    clusterMemberList = []
                    self.clusterMembersList.append(clusterMemberList)
                    break

        for i in range(compSize):
            rowIndex = self.compRowIndexList[i]
            maxVal = -1
            maxIndex = -1
            for l in range(compSize):
                zVal = z[i][0][l][0].X
                if (zVal > maxVal) and (clusterCenterIndexList.__contains__(l)):
                    maxVal = zVal
                    maxIndex = l

            for l in range(compSize, totalSize):
                misRowIndex = self.misRowIndexList[l - compSize]
                tpCanList = self.misTpCanMap[misRowIndex]
                tpCanSize = tpCanList.__len__()
                for q in range(tpCanSize):
                    zVal = z[i][0][l][q].X
                    if (zVal > maxVal) and (clusterCenterIndexList.__contains__(l)):
                        maxVal = zVal
                        maxIndex = l

            tmpClusterIndex = clusterCenterIndexList.index(maxIndex)
            clusterMemberList = self.clusterMembersList[tmpClusterIndex]
            clusterMemberList.append(rowIndex)

        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanList1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanList1.__len__()
            maxVal = -1
            maxIndex = -1

            for p in range(tpCanSize1):
                for l in range(compSize):
                    zVal = z[i][p][l][0].X
                    if (zVal > maxVal) and (clusterCenterIndexList.__contains__(l)):
                        maxVal = zVal
                        maxIndex = l

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanList2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanList2.__len__()
                    for q in range(tpCanSize2):
                        zVal = z[i][p][l][q].X
                        if (zVal > maxVal) and (clusterCenterIndexList.__contains__(l)):
                            maxVal = zVal
                            maxIndex = l
            tmpClusterIndex = clusterCenterIndexList.index(maxIndex)
            clusterMemberList = self.clusterMembersList[tmpClusterIndex]
            clusterMemberList.append(misRowIndex1)
        self.y=y

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
            y[i][0] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "y" + "[" + str(i) + "," + "0" + "]")
            for l in range(compSize):
                z[i][0][l][0] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "z" + "[" + str(i) + "," + "0" + "," + str(l) + "," + "0" + "]")
            for l in range(compSize, totalSize):
                misRowIndex2 = self.misRowIndexList[l - compSize]
                tpCanList2 = self.misTpCanMap[misRowIndex2]
                tpCanSize2 = tpCanList2.__len__()
                for q in range(tpCanSize2):
                    z[i][0][l][q] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "z" + "[" + str(i) + "," + "0" + "," + str(l) + "," + str(q) + "]")

        for i in range(compSize, totalSize):
            misRowIndex1 = self.misRowIndexList[i - compSize]
            tpCanList1 = self.misTpCanMap[misRowIndex1]
            tpCanSize1 = tpCanList1.__len__()
            for p in range(tpCanSize1):
                x[i][p] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "x" + "[" + str(i) + "," + str(p) + "]")
                y[i][p] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "y" + "[" + str(i) + "," + str(p) + "]")
                for l in range(compSize):
                    z[i][p][l][0] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY,
                                                 "z" + "[" + str(i) + "," + str(p) + "," + str(l) + "," + "0" + "]")

                for l in range(compSize, totalSize):
                    misRowIndex2 = self.misRowIndexList[l - compSize]
                    tpCanList2 = self.misTpCanMap[misRowIndex2]
                    tpCanSize2 = tpCanList2.__len__()
                    for q in range(tpCanSize2):
                        z[i][p][l][q] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY,
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
