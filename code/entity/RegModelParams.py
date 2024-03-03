# -*- coding: utf-8 -*-

class RegModelParams(object):
    def __init__(self):
        self.betas = None
        self.sigma2 = None

    def getBetas(self):
        return self.betas

    def setBetas(self, betas):
        self.betas = betas

    def getSigma2(self):
        return self.sigma2

    def setSigma2(self, sigma2):
        self.sigma2 = sigma2
