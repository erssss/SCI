# -*- coding: utf-8 -*-



class Weight(object):
    def __init__(self, i, p, *args):
        self.__i = i
        self.__p = p
        if len(args) > 0:
            assert len(args) == 1
            self.__WVal = args[0]

    def __eq__(self, other):
        if other==None:
            return False
        if type(self) !=type(other):
            return False
        if self.__i != other.getI():
            return False
        if self.__p != other.getP():
            return False
        return True

    def getWVal(self):
        return self.__WVal

    def setWVal(self, wVal):
        self.__WVal = wVal

    def getI(self):
        return self.__i

    def setI(self, i):
        self.__i = i

    def getP(self):
        return self.__p

    def setWVal(self, p):
        self.__p = p
