# -*- coding: utf-8 -*-


import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class FileHandler:
    def __init__(self):
        self.PATH = ''
        self.data = pd.DataFrame()

    def readCompData(self, input, label_idx=None, header=None, index_col=None):
        data = np.array(pd.read_csv(os.path.join(self.PATH, input), header=header, index_col=index_col))
        scaler = MinMaxScaler()
        print("data",data)
        if label_idx != -1:
            print("label_idx != -1")
            label = data[:, label_idx]
            data = scaler.fit_transform(data)
            print(data)
            data[:, label_idx] = label
        else:
            data = scaler.fit_transform(data)
        self.data = pd.DataFrame(data)
        return self.data
    
    def findMis(self):
        return self.data[self.data.isnull().any(axis=1)].index.tolist()        
