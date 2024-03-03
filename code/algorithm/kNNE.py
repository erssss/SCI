# -*- coding: utf-8 -*-

from time import time

import numpy as np

from algorithm.BaseMissing import BaseMissing


class KNNE(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask, K, features, isNum=True):
        """
        :param db: dataframe
        :param cells:
        :param mask: mask array
        :param K: K neigh
        :param features: [[]], selected features index list
        :param isNum:
        """
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.K = K
        self.features = features
        self.isNum = isNum
        self.setCellMap()

    def mainKNNE(self):
        self.initVals()
        t0 = time()
        print("before knne impute")
        self.__impute()
        print("after knne impute")
        self.algtime = time() - t0

    def __impute(self):
        misRowNum = len(self.misRowIndexList)
        feaNum = len(self.features)
        subDistances = np.zeros((misRowNum, len(self.dbVals)))
        sIndexes = np.zeros((misRowNum, self.K), dtype=int)
        sDistances = np.zeros((misRowNum, self.K))
        print("misRowNum",misRowNum)
        for mi in range(misRowNum):
            # print("mi = ",mi)
            misRowIndex = self.misRowIndexList[mi]
            misAttrIndexList = self.misRowAttrIndexList[mi]
            for attri in range(len(misAttrIndexList)):
                misAttrIndex = misAttrIndexList[attri]
                position = (misRowIndex, misAttrIndex)
                cell = self.cellMap[position]

                if self.isNum:
                    self.modify = 0
                    for fvi in range(feaNum):
                        usingFeature = self.features[fvi]
                        self.calcDisDirGivFea(misRowIndex, subDistances[mi], usingFeature)
                        self.findCompKnn(subDistances[mi], sIndexes[mi], sDistances[mi])
                        knnIndexes = sIndexes[mi]
                        self.modify += self.__knnResultNum(knnIndexes, misAttrIndex)
                    self.modify /= feaNum
                    cell.setModify(str(self.modify))
                else:
                    modifyMap = dict()
                    for fvi in range(feaNum):
                        usingFeature = self.features[fvi]
                        self.calcDisDirGivFea(misRowIndex, subDistances[mi], usingFeature)
                        self.findCompKnn(subDistances[mi], sIndexes[mi], sDistances[mi])
                        knnIndexes = sIndexes[mi]
                        tmpModify = self.__knnResultStr(knnIndexes, misAttrIndex)
                        if tmpModify in modifyMap.keys():
                            modifyMap[tmpModify] = modifyMap[tmpModify] + 1
                        else:
                            modifyMap[tmpModify] = 1

                    modify = ""
                    maxTimes = -1
                    for k, v in modifyMap.items():
                        if v > maxTimes:
                            modify = k
                            maxTimes = v

                    cell.setModify(str(modify))

    def __knnResultNum(self, knnIndexes, misAttrIndex):
        modify = 0
        for i in range(self.K):
            rowIndex = knnIndexes[i]
            val = self.dbVals[rowIndex][misAttrIndex]
            numVal = float(val)
            modify += numVal
        modify /= self.K
        return modify

    def __knnResultStr(self, knnIndexes, misAttrIndex):
        modifyMap = dict()
        for i in range(self.K):
            rowIndex = knnIndexes[i]
            val = self.dbVals[rowIndex][misAttrIndex]
            if val in modifyMap.keys():
                modifyMap[val] = modifyMap[val] + 1
            else:
                modifyMap[val] = 1

        modify = ""
        maxTimes = -1
        for k, v in modifyMap.items():
            if v > maxTimes:
                modify = k
                maxTimes = v
        return modify
