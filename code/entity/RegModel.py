# -*- coding: utf-8 -*-


class RegModel(object):
    def __init__(self, attrXs, attrY):
        self.setAttrXs(attrXs)
        self.setAttrY(attrY)

    def setAttrXs(self, attrXs):
        self.attrXs = attrXs
        sorted(self.attrXs)

    def setAttrY(self, attrY):
        self.attrY = attrY

    def getAttrXs(self):
        return self.attrXs

    def getAttrY(self):
        return self.attrY
