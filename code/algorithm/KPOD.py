# -*- coding: utf-8 -*-
"""
https://github.com/iiradia/kPOD
"""

import numpy as np
import pandas as pd
import sys
from scipy.spatial import distance
from algorithm.BaseMissing import BaseMissing


class KPOD(BaseMissing):
    def __init__(self, db, label_idx, exp_cluster, cells, mask, seed, max_iter=1, tol=0):
        """

        :param db:
        :param cells:
        :param mask:
        :param Ncluster:
        :param seed:
        """
        super().__init__(db, label_idx, exp_cluster, cells, mask)
        self.max_iter = max_iter
        self.seed = seed
        self.tol = tol

    def mainkpod(self):
        self.initVals()
        X_comp_true, y_comp_true, X_comp_miss = self.ignore_down_stream()
        clustered_data = self.predict(X_comp_miss)
        return clustered_data[0], y_comp_true

    def predict(self, data):
        data = np.array(data)
        N = data.shape[0]
        P = data.shape[1]
        K = self.exp_cluster
        num_iters = 0

        MISSING_DATA = data.copy()
        past_centroids = []
        cluster_centers = []
        cluster_assignment = []

        while num_iters < self.max_iter:
            if num_iters > 0:
                filled_data = self.__fill_data(MISSING_DATA, cluster_centers, cluster_assignment)
                filled_data = np.array(filled_data)
            else:
                data_frame = pd.DataFrame(data)
                filled_data = np.array(data_frame.fillna(np.nanmean(data)))
                cluster_centers = self.__initialize(filled_data, K)

            cluster_assignment = self.__cluster_assignment(filled_data, cluster_centers, N, K)
            cluster_centers = self.__move_centroids(filled_data, cluster_centers, cluster_assignment, N, K)
            centroids_complete = self.__check_convergence(cluster_centers, past_centroids, self.tol, num_iters)

            past_centroids = cluster_centers

            num_iters += 1

            if centroids_complete:
                break

        cluster_ret = {"ClusterAssignment": cluster_assignment, "ClusterCenters": cluster_centers}

        cluster_return = (cluster_assignment, cluster_centers)
        return cluster_return

    def __fill_data(self, MISSING_DATA, cluster_centers, cluster_assignment):
        filled_data = np.array(MISSING_DATA.copy())
        for i in range(len(filled_data)):
            obs_cluster = int(cluster_assignment[i])
            j = 0
            for val in filled_data[i]:
                if (np.isnan(val)):
                    filled_data[i][j] = cluster_centers[obs_cluster][j]
                j += 1
        return filled_data

    def __initialize(self, data, n_clusters):
        data = np.array(data)
        N = data.shape[0]

        np.random.seed(self.seed)
        centroids = []
        centroids.append(data[np.random.randint(N), :])

        for cluster in range(n_clusters - 1):
            distances = []

            for data_idx in range(N):
                point = data[data_idx, :]
                dist = sys.maxsize
                for centroid_idx in range(len(centroids)):
                    curr_distance = self.__euclidean_distance(point, centroids[centroid_idx])
                    dist = min(dist, curr_distance)
                distances.append(dist)

            distances = np.array(distances)

            center = data[np.argmax(distances), :]
            centroids.append(center)
            distances = []
        return centroids

    @staticmethod
    def __euclidean_distance(point_1, point_2):
        return np.sum((point_1 - point_2) ** 2)

    def __cluster_assignment(self, data, cluster_centers, N, K):
        cluster_assignment = np.zeros(N)
        dist = np.zeros(K)

        for num in range(0, N):
            for cluster in range(K):
                dist[cluster] = distance.euclidean(data[num], cluster_centers[cluster])
            cluster_assignment[num] = np.argmin(dist)
        return cluster_assignment

    def __move_centroids(self, data, cluster_centers, cluster_assignment, N, K):
        for num in range(1, K + 1):
            cluster_points = list()
            for i in range(0, N):
                if int(cluster_assignment[i]) == (num - 1):
                    cluster_points.append(data[i])
            cluster_points = np.array(cluster_points)
            cluster_centers[num - 1] = cluster_points.mean(axis=0)
        return cluster_centers

    def __check_convergence(self, cluster_centers, past_centroids, tol, num_iters):
        
        if num_iters == 0:
            return False

        centroids_complete = 0

        for i in range(len(cluster_centers)):
            if distance.euclidean(cluster_centers[i], past_centroids[i]) <= tol:
                centroids_complete += 1
        return centroids_complete
