from time import time

import numpy as np
import os
import pandas as pd
import warnings

from algorithm.ERACER import ERACER
from algorithm.GMM import GMM
from algorithm.kNNE import KNNE
from algorithm.CMI import CMI
from algorithm.MICE import MICE
from algorithm.BaseMissing import BaseMissing
from algorithm.SCIILP import SCIILP
from algorithm.SCILP import SCILP
from algorithm.SCILN import SCILN
from algorithm.IFC import IFC
from algorithm.CI import CI
from algorithm.DPCI import DPCI
from util.Assist import Assist
from algorithm.NMF import NMF
from algorithm.KPOD import KPOD
from util.DataHandler import DataHandler as dh
from util.FileHandler import FileHandler as fh

import logging

enable_debug = False
stop_check = False

logging.basicConfig(level=logging.INFO)
if enable_debug:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)

warnings.filterwarnings("ignore")


class CompTest:
    def __init__(self, filename, label_idx=-1, exp_cluster=None, header=None, index_col=None):
        self.name = filename.split('/')[-1].split('.')[0]
        self.REPEATLEN = 5
        self.RBEGIN = 998
        self.exp_cluster = exp_cluster
        self.label_idx = label_idx
        self.fh = fh()
        self.db = self.fh.readCompData(filename, label_idx=label_idx, header=header, index_col=index_col)
        self.dh = dh()
        self.delCompRowIndexList = self.fh.findMis()
        self.totalsize = self.db.shape[0]
        self.size = self.totalsize * 1

        self.ALGNUM = 14

        self.alg_flags = [True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        self.totalTime = np.zeros((1, self.ALGNUM))
        self.totalCost = np.zeros((1, self.ALGNUM))
        self.parasAlg = np.zeros((self.ALGNUM))
        self.purityAlg = np.zeros((self.ALGNUM))
        self.riAlg = np.zeros((self.ALGNUM))
        self.ariAlg = np.zeros((self.ALGNUM))
        self.fmeasureAlg = np.zeros((self.ALGNUM))
        self.recallAlg = np.zeros((self.ALGNUM))
        self.precAlg = np.zeros((self.ALGNUM))
        self.NMIAlg = np.zeros((self.ALGNUM))
        self.totalpurity = np.zeros((1, self.ALGNUM))
        self.totalfmeasure = np.zeros((1, self.ALGNUM))
        self.totalNMI = np.zeros((1, self.ALGNUM))
        self.totalri = np.zeros((1, self.ALGNUM))
        self.totalari = np.zeros((1, self.ALGNUM))
        self.totalRecall = np.zeros((1, self.ALGNUM))
        self.totalPrec = np.zeros((1, self.ALGNUM))
        self.features = [[0], [1]]

        self.K = 30
        self.N_Cluster = 50

        self.ERACER_K = 5
        self.ERACER_maxIter = 10
        self.ERACER_threshold = 0.01

        self.K_Candidate = 10
        self.L = 15
        self.C = 1

        self.IFC_min_k = 2
        self.IFC_max_k = 5
        self.IFC_maxIter = 20
        self.IFC_threshold = 0.001
        self.CI_K = 5
        self.CI_maxIter = 15
        self.CI_n_end = 10
        self.CI_c_steps = 10

        self.K_SCI = exp_cluster
        self.KDistance = 5
        self.K_Candidate_SCI = 1
        self.epsilon_SCI = 0.01
        
    def Dirty_exp(self):
        
        print("Dirty begin!")
        algIndex = 0
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                dirty = BaseMissing(self.db, self.label_idx, self.exp_cluster, cells, mask)
                dirty.setDelCompRowIndexList([])
                dirty.initVals()
                origin_y, modify_y = dirty.modify_down_stream(cells)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("dirty")
                if stop_check:
                    input("wait")
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]

            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("Dirty over!")

    def KNNE_exp(self):
        
        print("KNNE begin!")
        algIndex = 1
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                knne = KNNE(self.db, self.label_idx, self.exp_cluster, cells, mask, self.K, self.features)
                knne.setDelCompRowIndexList([])
                knne.mainKNNE()
                origin_y, modify_y = knne.modify_down_stream(cells)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("KNNE")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("KNNE over!")

    def GMM_exp(self):
        print("GMM begin!")
        algIndex = 2
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                gmm = GMM(self.db, self.label_idx, self.exp_cluster, cells, mask, self.K, self.N_Cluster, seed)
                gmm.setDelCompRowIndexList([])
                gmm.mainGMM()
                origin_y, modify_y = gmm.modify_down_stream(cells)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("GMM")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("GMM over!")

    def ERACER_exp(self):
        
        print("ERACER begin!")
        algIndex = 3
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                

                eracer = ERACER(self.db, self.label_idx, self.exp_cluster, cells, mask)
                eracer.setParams(self.ERACER_K, self.ERACER_maxIter, self.ERACER_threshold)
                eracer.setDelCompRowIndexList([])
                eracer.mainEracer()
                origin_y, modify_y = eracer.modify_down_stream(cells)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("ERACER")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("ERACER over!")

    def IFC_exp(self):
        print("IFC begin!")
        algIndex = 4
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                ifc = IFC(self.db, self.label_idx, self.exp_cluster, cells, mask)
                ifc.setDelCompRowIndexList([])
                ifc.setParams(self.IFC_min_k, self.IFC_max_k, self.IFC_maxIter, self.IFC_threshold)
                ifc.mainIFC()

                origin_y, modify_y = ifc.modify_down_stream(cells)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("IFC")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("IFC over!")

    def CI_exp(self):
        
        print("CI begin!")
        algIndex = 5
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                ci = CI(self.db, self.label_idx, self.exp_cluster, cells, mask)
                ci.setDelCompRowIndexList([])
                ci.setParams(self.CI_K, self.CI_maxIter, self.CI_n_end, self.CI_c_steps)
                ci.mainCI()

                origin_y, modify_y = ci.modify_down_stream(cells)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("CI")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime

        print("CI over!")

    def CMI_exp(self):
        
        print("CMI begin!")
        algIndex = 6
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                cmi = CMI(self.db, self.label_idx, self.exp_cluster, cells, mask)
                cmi.setCenterRatio(0.01)
                cmi.setSeed(seed)
                cmi.setDelCompRowIndexList([])
                cmi.mainCMI()
                
                origin_y, modify_y = cmi.modify_down_stream(cells)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("CMI")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime

        print("CMI over!")

    def MICE_exp(self):
        
        print("MICE begin!")
        algIndex = 7
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                mice = MICE(self.db, self.label_idx, self.exp_cluster, cells, mask)
                mice.setDelCompRowIndexList([])
                mice.main_mice()

                origin_y, modify_y = mice.modify_down_stream(cells)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("MICE")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime

        print("MICE over!")

    def DPCI_exp(self):
        
        print("DPCI begin!")
        algIndex = 8
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                dpci = DPCI(self.db, self.label_idx, self.exp_cluster, cells, mask)
                dpci.setDelCompRowIndexList([])
                dpci.setParams(k1=2, k2=1, alpha=0.1, beta=0.7, gamma=0.7)
                dpci.main_DPCI()

                origin_y, _ = dpci.modify_down_stream(cells)
                modify_y = dpci.modify_y[dpci.compRowIndexList + dpci.misRowIndexList]
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("DPCI")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("DPCI over!")

    def NMF_exp(self):
        
        algIndex = 9
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
               
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                
                snmf = NMF(self.db, self.label_idx, self.exp_cluster, cells, mask, self.N_Cluster, seed)
                snmf.setDelCompRowIndexList([])
                y, origin_y = snmf.mainSNMF()
                modify_y = np.argmax(y, axis=1)
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("NMF")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("NMF over!")

    def Kpod_exp(self):
        
        print("KPOD begin!")
        algIndex = 10
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)
                kpod = KPOD(self.db, self.label_idx, self.exp_cluster, cells, mask, seed)
                kpod.setDelCompRowIndexList([])
                modify_y, origin_y = kpod.mainkpod()
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("KPOD")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("KPOD over!")

    def SCI_LN(self):
        print("SCILN begin!")
        algIndex = 11
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)

                SCI = SCILN(self.db, self.label_idx, self.exp_cluster, cells, mask)
                SCI.setK(self.K_SCI)
                SCI.setK_Candidate(self.K_Candidate_SCI)
                SCI.setEpsilon(self.epsilon_SCI)
                SCI.setKDistance(self.KDistance)
                SCI.setDelCompRowIndexList([])
                SCI.mainSCI()

                cluster_dict = {}
                for i in range(len(SCI.clusterMembersList)):
                    for j in SCI.clusterMembersList[i]:
                        cluster_dict[j] = i
                modify_y = []
                for j in SCI.compRowIndexList + SCI.misRowIndexList:
                    modify_y.append(cluster_dict[j])
                origin_y, _ = SCI.modify_down_stream(cells)
                
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("SCILN")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("SCILN over")

    def SCI_LP(self):
        
        print("SCI begin!")
        algIndex = 12
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)

                SCI = SCILP(self.db, self.label_idx, self.exp_cluster, cells, mask)
                SCI.setK(self.K_SCI)
                SCI.setK_Candidate(self.K_Candidate_SCI)
                SCI.setEpsilon(self.epsilon_SCI)
                SCI.setDelCompRowIndexList([])
                SCI.mainSCI()
                cluster_dict = {}
                for i in range(len(SCI.clusterMembersList)):
                    for j in SCI.clusterMembersList[i]:
                        cluster_dict[j] = i
                modify_y = []
                for j in SCI.compRowIndexList + SCI.misRowIndexList:
                    modify_y.append(cluster_dict[j])
                origin_y, _ = SCI.modify_down_stream(cells)
                
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("SCI-ILP")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime
        print("SCI over")

    def SCI_ILP(self):
        
        print(" SCI_exact begin! ")
        algIndex = 13
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.getMisCelMul(self.db, self.label_idx, self.size)

                SCI = SCIILP(self.db, self.label_idx, self.exp_cluster, cells, mask)

                SCI.setK(self.K_SCI)
                SCI.setK_Candidate(self.K_Candidate_SCI)
                SCI.setEpsilon(self.epsilon_SCI)
                SCI.setDelCompRowIndexList([])
                SCI.mainSCI()

                self.totalTime[0][algIndex] += SCI.getAlgtime()

                cluster_dict = {}
                for i in range(len(SCI.clusterMembersList)):
                    for j in SCI.clusterMembersList[i]:
                        cluster_dict[j] = i
                modify_y = []
                for j in SCI.compRowIndexList + SCI.misRowIndexList:
                    modify_y.append(cluster_dict[j])
                origin_y, _ = SCI.modify_down_stream(cells)
                
                logging.debug("origin_y",origin_y)
                logging.debug("modify_y",modify_y)
                logging.debug("EXACT-SCI")
                if stop_check:
                    input("wait")
                
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.riAlg[algIndex] = Assist.RI(origin_y, modify_y)
                self.ariAlg[algIndex] = Assist.ARI(origin_y, modify_y)
                self.fmeasureAlg[algIndex],self.recallAlg[algIndex],self.precAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[0][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[0][algIndex] += self.NMIAlg[algIndex]
                self.totalri[0][algIndex] += self.riAlg[algIndex]
                self.totalari[0][algIndex] += self.ariAlg[algIndex]
                self.totalfmeasure[0][algIndex] += self.fmeasureAlg[algIndex]
                self.totalRecall[0][algIndex]+=self.recallAlg[algIndex]
                self.totalPrec[0][algIndex]+=self.precAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[0][algIndex] += algTime

        print("SCI_exact over")

    def alg_exp(self):
        self.Dirty_exp()
        self.KNNE_exp()
        self.GMM_exp()
        self.ERACER_exp()
        self.IFC_exp()
        self.CI_exp()
        self.CMI_exp()
        self.MICE_exp()
        self.DPCI_exp()
        self.NMF_exp()
        self.Kpod_exp()
        self.SCI_LN()
        self.SCI_LP()
        self.SCI_ILP()
        name1 = self.name + '_test'
        name2 = self.name
        ratio_arr = np.array([1]).reshape(-1, 1)
        columns = ["CompRatio", "Dirty", "kNNE", "GMM", "ERACER", "IFC", "CI", "CMI", "MICE", "DPCI", "NMF", "KPOD", "SCILN", "SCI",
                   "SCI_exact"]
        cost_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalCost / self.REPEATLEN), axis=1), columns=columns)
        time_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalTime / self.REPEATLEN), axis=1), columns=columns)
        purity_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalpurity / self.REPEATLEN), axis=1), columns=columns)
        nmi_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalNMI / self.REPEATLEN), axis=1), columns=columns)
        ri_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalri / self.REPEATLEN), axis=1), columns=columns)
        ari_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalari / self.REPEATLEN), axis=1), columns=columns)
        fmeasure_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalfmeasure / self.REPEATLEN), axis=1), columns=columns)
        recall_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalRecall / self.REPEATLEN), axis=1), columns=columns)
        prec_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalPrec / self.REPEATLEN), axis=1), columns=columns)
        if not os.path.exists(os.path.join("result/compare", name1)):
            os.makedirs(os.path.join("result/compare", name1))
        mode="a"
        params_str = "self.REPEATLEN = "+str(self.REPEATLEN)+"\tK_SCI: "+str(self.K_SCI)+"\tK_Candidate_SCI: "+str(self.K_Candidate_SCI)+"\tKDistance: "+str(self.KDistance)+"\tepsilon_SCI: "+str(self.epsilon_SCI)+"\texp_cluster: "+str(self.exp_cluster)+"\tK: "+str(self.K)+"\tN_Cluster: "+str(self.N_Cluster)+"\tERACER_K: "+str(self.ERACER_K)+"\tERACER_maxIter: "+str(self.ERACER_maxIter)+"\tERACER_threshold: "+str(self.ERACER_threshold)+"\tK_Candidate: "+str(self.K_Candidate)+"\tL: "+str(self.L)+"\tC: "+str(self.C)+"\tIFC_min_k: "+str(self.IFC_min_k)+"\tIFC_max_k: "+str(self.IFC_max_k)+"\tIFC_maxIter: "+str(self.IFC_maxIter)+"\tIFC_threshold: "+str(self.IFC_threshold)+"\tCI_K: "+str(self.CI_K)+"\tCI_maxIter: "+str(self.CI_maxIter)+"\tCI_n_end: "+str(self.CI_n_end)+"\tCI_c_steps: "+str(self.CI_c_steps)
            
        cost_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_cost" + ".tsv", sep='\t',
                       float_format="%.5f",
                       index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_cost" + ".tsv","a") as file:
            file.write(params_str+"\n")
            
        time_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_time" + ".tsv", sep='\t',
                       float_format="%.5f",
                       index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_time" + ".tsv","a") as file:
            file.write(params_str+"\n")
           
        purity_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_purity" + ".tsv", sep='\t',
                         float_format="%.5f",
                         index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_purity" + ".tsv","a") as file:
            file.write(params_str+"\n")
            
        nmi_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_nmi" + ".tsv", sep='\t',
                      float_format="%.5f",
                      index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_nmi" + ".tsv","a") as file:
            file.write(params_str+"\n")
            
        ri_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_ri" + ".tsv", sep='\t',
                      float_format="%.5f",
                      index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_ri" + ".tsv","a") as file:
            file.write(params_str+"\n")
            
        ari_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_ari" + ".tsv", sep='\t',
                      float_format="%.5f",
                      index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_ari" + ".tsv","a") as file:
            file.write(params_str+"\n")
            
        fmeasure_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_f1" + ".tsv", sep='\t',
                           float_format="%.5f",
                           index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_f1" + ".tsv","a") as file:
            file.write(params_str+"\n")
            
        recall_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_recall" + ".tsv", sep='\t',
                           float_format="%.5f",
                           index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_recall" + ".tsv","a") as file:
            file.write(params_str+"\n")
            
        prec_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_prec" + ".tsv", sep='\t',
                           float_format="%.5f",
                           index=False,mode=mode)
        with open("result/" + "compare/" + name1 + "/" + name2 + "_prec" + ".tsv","a") as file:
                file.write(params_str+"\n")

        print("all over!")


if __name__ == '__main__':
    ct = CompTest("../data/crx/crx.data", label_idx=15, exp_cluster=2, header=None, index_col=None)
    ct.alg_exp()
    
