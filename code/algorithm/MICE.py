# -*- coding: utf-8 -*-

from algorithm.BaseMissing import BaseMissing
from time import time
from entity.Cluster import Cluster
import random
import math
import numpy as np
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer


class MICE(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask):
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.tmpDbVals = self.dbVals.astype(float)
        self.tmpDbValsNa = self.dbVals.astype(float)
        for cell in self.cells:
            self.tmpDbValsNa[cell.position] = np.nan
        self.setCellMap()

    def main_mice(self):
        self.initVals()
        t0 = time()
        # imp_mean = IterativeImputer(random_state=0,n_nearest_features=2)
        imp_mean = IterativeImputer(random_state=42,n_nearest_features=2)
        self.tmpDbValsNa = imp_mean.fit_transform(self.tmpDbValsNa)
        for cell in self.cells:
            position = cell.position
            cell.modify = str(self.tmpDbValsNa[position])
        self.algtime = time() - t0
