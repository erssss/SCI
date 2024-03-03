# -*- coding: utf-8 -*-

class Cell:
    def __init__(self, position):
        self.position = position
        self.truth = ""
        self.modify = "0"

    def setTruth(self, truth):
        self.truth = truth

    def setModify(self, modify):
        self.modify = modify
