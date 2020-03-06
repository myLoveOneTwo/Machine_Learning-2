# coding=utf-8
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class KMeans:
    def __init__(self, X, K, max_iters = 10):
        self.X = X
        self.K = K
        self.max_iters = max_iters
        #返回x的每行所属的中心点索引
    def findClosestCentroids(self, centroids):

        idx = np.zeros(len(self.X )).reshape( self.X .shape[0],-1)
        for i in range(len(self.X )):
            minDistance = float('inf');#初始无限大
            index = 0
            for k in range(len(centroids)):
                distance = np.sum(np.power(self.X [i]-centroids[k],2))
                if(distance<minDistance):
                    minDistance = distance
                    index = k
            idx[i]=index
        return idx
    #idx：X每行所属的中心点索引
    #K:聚类个数
    def computeCentroids(self , idx):
        k = set(np.ravel(idx).tolist()) #找到所有聚类中心索引
        k = list(k)
        centroids = np.ndarray((len(k),self.X .shape[1]))
        for i in range(len(k)):
            data = self.X [np.where(idx==k[i])[0]]
            centroids[i] = np.sum(data,axis=0)/len(data)
        return centroids


    def runkMeans(self):
        initial_centroids = self.InitCentroids()
        idx = self.findClosestCentroids(initial_centroids)
        centroids = self.computeCentroids(idx)
        for i in range(self.max_iters):
            idx = self.findClosestCentroids(centroids)
            centroids = self.computeCentroids(idx)
        return idx,centroids

    def InitCentroids(self):
        index = np.random.randint(0,len(self.X )-1,self.K)
        return self.X [index]

    def __repr__(self):
         return "KMeans(k=%d)" % self.K