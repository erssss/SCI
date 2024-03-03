# -*- coding: utf-8 -*-


class Cluster(object):
    def __init__(self, clusterIndex=0):
        self.__clusterIndex = None
        self.__rowIndexList = None
        self.setClusterIndex(clusterIndex)
        self.setRowIndexList([])

    def getClusterIndex(self):
        return self.__clusterIndex

    def setClusterIndex(self, clusterIndex):
        self.__clusterIndex = clusterIndex

    def getRowIndexList(self):
        return self.__rowIndexList

    def setRowIndexList(self, rowIndexList):
        self.__rowIndexList = rowIndexList

    def addRowIndex(self, rowIndex):
        if rowIndex not in self.__rowIndexList:
            self.__rowIndexList.append(rowIndex)
