# -*- coding: utf-8 -*-

import math
import re
import numpy as np
from sklearn import metrics


class Assist:
    def isNumber(self, string):
        return string

    def numDis(self, a, b):
        return abs(a - b)

    def normNumDis(self, a, b, maxVal, minVal):
        return abs(a - b) / (maxVal - minVal)

    def normStrDis(self, word1, word2):
        len1, len2 = len(word1), len(word2)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))

        gelStrDis = dp[len1][len2]
        norStrDis = (2 * gelStrDis) / (len1 + len2 + gelStrDis)
        return norStrDis

    def strDis(self, word1, word2):
        len1, len2 = len(word1), len(word2)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        strDis = dp[len1][len2]
        return strDis

    def calcRMS(self, cells):
        cost = 0.0
        l = len(cells)
        for cell in cells:
            tru_val = cell.truth
            modify = cell.modify
            if modify is None:
                delta = len(tru_val)
            else:
                delta = float(tru_val) - float(modify)
            cost += delta * delta
        cost /= l
        return math.sqrt(cost)

    def calcAcc(self, cells):
        pass

    def editDis(self, word1, word2):
        len1 = len(word1)
        len2 = len(word2)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        editDistance = dp[len1][len2]
        return editDistance

    def normEditDis(self, word1, word2):
        pass

    def getAttrXsFromAttrY(self, attrNum, attrY):
        misList = []
        misList.append(attrY)
        return self.getAttrXsFromMisList(attrNum, misList)

    def getAttrXsFromMisList(self, attrNum, misList):
        attrXNum = attrNum - len(misList)
        attrXs = [0] * attrXNum
        curIndex = 0
        for attri in range(attrNum):
            if attri in misList:
                continue
            attrXs[curIndex] = attri
            curIndex += 1
        return attrXs

    def getIndexOfMaxValFromArray(self, vals):
        pass

    def getCellByPosition(self, cells, position):
        cellSize = cells.__len__()
        for i in range(cellSize):
            cell = cells[i]
            tmpPosition = cell.position
            rowIndex1 = tmpPosition[0]
            attrIndex1 = tmpPosition[1]
            rowIndex2 = position[0]
            attrIndex2 = position[1]
            if rowIndex1 == rowIndex2 and attrIndex1 == attrIndex2:
                return cell
        return None

    @staticmethod
    def frobenius(A, B):
        return np.linalg.norm(A - B, 'fro')

    @staticmethod
    def purity(y_true, y_pred):
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    @staticmethod
    def NMI(y_true, y_pred):
        return metrics.cluster.normalized_mutual_info_score(y_true, y_pred)

    @staticmethod
    def ARI(y_true, y_pred):
        return metrics.cluster.adjusted_rand_score(y_true, y_pred)
    
    @staticmethod
    def RI(y_true, y_pred):
        return metrics.cluster.rand_score(y_true, y_pred)

    @staticmethod
    def f_measure(y_true, y_pred):
        print(y_pred)
        print(y_true)
        def invert_label(L):
            # print("in invert_label")
            ld = dict()
            for i, l in enumerate(L):
                if l not in ld.keys():
                    ld[l] = [i]
                else:
                    ld[l].append(i)
            # print("out invert_label")
            return list(ld.values())

        def pairset(D):
            if len(D) > 1:
                return [(D[i], D[j]) for i in range(len(D)) for j in range(i + 1, len(D))]
            else:
                return []

        def extend_pairset(P):
            # print("in extend_pairset")
            # print("P = ",P)
            # input("wait")
            res = []
            for p in P:
                # input("p:")
                print(p)
                res.extend(pairset(p))
            # print("out extend_pairset")
            return set(res)

        # f = metrics.f1_score(y_true, y_pred, average='micro')
        # recall = metrics.recall_score(y_true, y_pred, average='micro')
        # precision = metrics.precision_score(y_true, y_pred, average='micro')
        
        inv_t, inv_p = invert_label(y_true), invert_label(y_pred)
        pair_t, pair_p = extend_pairset(inv_t), extend_pairset(inv_p)
        a = len(pair_t & pair_p) # TP
        b = len(pair_t - pair_p) # FN
        c = len(pair_p - pair_t) # FP
        # print("a = ",a," b = ",b," c = ",c)
        f = 2 * a / (2 * a + b + c)
        recall = a/(a+b)
        precision = a/(a+c)
        return f,recall,precision