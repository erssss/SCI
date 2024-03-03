# -*- coding: utf-8 -*-


class TupleCandidate(object):
    def __init__(self, misAttrList, tpCanList):
        self.misAttrList = misAttrList
        self.tpCanList = tpCanList
        
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def TupleCandidate(self, misAttrList, tpCanList):
        self.setMisAttrList(misAttrList)
        self.setTpCanList(tpCanList)

    def getMisAttrList(self):
        return self.misAttrList

    def setMisAttrList(self, misAttrList):
        self.misAttrList = misAttrList

    def getTpCanList(self):
        return self.tpCanList

    def setTpCanList(self, tpCanList):
        self.tpCanList = tpCanList
