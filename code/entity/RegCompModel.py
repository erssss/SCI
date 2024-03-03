# -*- coding: utf-8 -*-

class RegCompModel(object):
    def __init__(self, rowIndex, attrXs, attrY):
        self.setRowIndex(rowIndex)
        self.setAttrXs(attrXs)
        self.setAttrY(attrY)

    def setRowIndex(self, rowIndex):
        self.rowIndex = rowIndex

    def setAttrXs(self, attrXs):
        self.attrXs = attrXs

    def setAttrY(self, attrY):
        self.attrY = attrY

    def getRowIndex(self):
        return self.rowIndex

    def getAttrXs(self):
        return self.attrXs

    def getAttrY(self):
        return self.attrY

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        prime = 31
        result = 1

        for attrX in self.attrXs:
            result = prime * result + attrX
        result = prime * result + self.attrY
        result = prime * result + self.rowIndex
        return result
