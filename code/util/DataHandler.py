# -*- coding: utf-8 -*-

import random

import numpy as np

from entity.Cell import Cell


class DataHandler:
    def __init__(self):
        self.misRowIndexList = []

    def genMisCel(self, db, label_idx):
        if label_idx == -1:
            db = db.values
        else:
            db = db.drop(label_idx, axis=1).values
        size, attr_size = db.shape
        mask = np.ones((size, attr_size), dtype=np.int8)
        cells = []
        for r in range(size):
            for c in range(attr_size):
                if db[r, c] != db[r, c]:
                    pos = (r, c)
                    mask[r, c] = 0
                    cell = Cell(pos)
                    cells.append(cell)
        cells = sorted(cells, key=lambda x: (x.position[0], x.position[1]))
        return cells, mask


    def genMisCelMul(self, db, label_idx, misNum, size, seed, selList):
        if label_idx==-1:
            db = db.values
        else:
            db = db.drop(label_idx, axis=1).values

        size, attr_size = db.shape

        mask = np.ones((size, attr_size), dtype=np.int8)
        random.seed(seed)
        self.misRowIndexList = random.sample(range(0, size), misNum)
        cells = []
        for r in self.misRowIndexList:
            for c in selList:
                pos = (r, c)
                mask[r, c] = 0
                truth_value = db[pos]
                cell = Cell(pos)
                cell.setTruth(truth_value)
                cells.append(cell)
        cells = sorted(cells, key=lambda x: (x.position[0], x.position[1]))
        return cells, mask


    def getMisCelMul(self, db, label_idx, size):
        if label_idx==-1:
            db = db.values
        else:
            db = db.drop(label_idx, axis=1).values

        size, attr_size = db.shape

        mask = np.ones((size, attr_size), dtype=np.int8)
        cells = []
        for r in range(db.shape[0]):
            for c in range(db.shape[1]):
                pos = (r, c)
                if np.isnan(db[pos]):
                    mask[r, c] = 0
                    cell = Cell(pos)
                    cells.append(cell)
        cells = sorted(cells, key=lambda x: (x.position[0], x.position[1]))
        return cells, mask
    
    def genMissSelMulGivenMisRowList(self):
        pass

    def setmisRowIndexList(self,misRowIndexList):
        self.misRowIndexList = misRowIndexList

    def genMisCelMul_mode(self, df, label_idx, misNum, size, seed, selList, dependAttr,mode="MCAR"):
        if label_idx==-1:
            db = df.values
        else:
            db = df.drop(label_idx, axis=1).values

        size, attr_size = db.shape

        mask = np.ones((size, attr_size), dtype=np.int8)
        self.misRowIndexList = self.genMisRow(df, misNum, selList, seed, dependAttr,mode=mode)
        cells = []
        for r in self.misRowIndexList:
            for c in selList:
                pos = (r, c)
                mask[r, c] = 0
                truth_value = db[pos]
                cell = Cell(pos)
                cell.setTruth(truth_value)
                cells.append(cell)
        cells = sorted(cells, key=lambda x: (x.position[0], x.position[1]))
        return cells, mask

    def genMisRow(self, db, misNum, selList, seed, dependAttr, mode='MCAR'):
        size=db.shape[0]
        col=selList[0]
        random.seed(seed)
        if mode=="MCAR":
            misRowIndexList = random.sample(range(0, size), misNum)
        if mode=="MAR":
            sorted_list=list(db.sort_values(by=dependAttr).index)
            start_index=random.sample(range(0, size-misNum), 1)
            misRowIndexList=[sorted_list[start_index[0]+i] for i in range(misNum)]
        if mode=="MNAR":
            sorted_list=list(db.sort_values(by=col).index)
            start_index=random.sample(range(0, size-misNum), 1)
            misRowIndexList=[sorted_list[start_index[0]+i] for i in range(misNum)]
        return misRowIndexList

    def genDelCompRowIndexList(self, cleanRatio, size, seed, errorTupleNum):
        delComSize = int((size - errorTupleNum) * (1 - cleanRatio))
        random.seed(seed)
        delCompRowIndexList = random.sample(set(range(0, size)) - set(self.misRowIndexList), delComSize)
        return delCompRowIndexList

