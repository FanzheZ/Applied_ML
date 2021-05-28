# -*- coding: utf-8 -*-
"""
ECE5984 SP20 HW3A - clustering irises
Created on Thu Feb 20 17:41:33 2020
@author: crjones4
"""
from sklearn import neighbors
import pandas as pd
import numpy as np
import math

def distance(inA, inB, distSelector, mu, co):
    a = np.asarray(inA)
    b = np.asarray(inB)
    if(distSelector == 'Manhattan'): # Manhattan
        answer = sum(abs(a-b))
        return answer
    elif(distSelector == 'Mahalanobis'):# Mahalanobis
        diff_trans = np.transpose(a-b)
        cov_inv = np.linalg.inv(co)
        diff = a-b
        temp1 = np.dot(diff_trans, cov_inv)
        temp2 = np.dot(temp1, diff)
        answer = math.sqrt(temp2)
        return answer
    else:   #default is Euclidean
        answer = math.sqrt(np.dot(a-b, a-b))
        return answer
        # Eucl is the square root of the sum of the squares of the differences

pathName = "/Users/fanzhe/Documents/2020_Spring/Applied_ML/HW3/"
dataFrame = pd.read_excel(pathName + 'iris.xlsx', sheet_name='data')
trainX = dataFrame.drop(["species"], axis=1)
trainY = dataFrame.species
minDist = 100000
count = 0
mu = trainX.mean()  # for Mahalanobis
co = trainX.cov()   # for Mahalanobis

class1 = dataFrame.loc[dataFrame['species'] == 1].drop(["species"], axis=1)
class2 = dataFrame.loc[dataFrame['species'] == 2].drop(["species"], axis=1)
class3 = dataFrame.loc[dataFrame['species'] == 3].drop(["species"], axis=1)

mean = [class1.mean(), class2.mean(), class3.mean()]
cov = [class1.cov(), class2.cov(), class3.cov()]


ind = 0
for row in trainX.iterrows():   # NOTE! this is slow and only for use on small ADS
    samp = row[1]
    samp_class = trainY[ind]

    mu = mean[samp_class-1]
    co = cov[samp_class-1]

    #newX = [5.5, 3.0, 4.4, 1.4]
    #newX = [5.15, 3.25, 2.9, 0.9]
    newX = [6.7, 3.1, 5.15, 1.95]
    dist = distance(np.transpose(np.asarray(newX)), samp, "Mahalanobis", mu, co)
    if (dist < minDist): # find the smallest distance (most similar)
        minDist = dist
        minRow = row
        minTarget = trainY[count]
    count = count + 1

print("Min at:", minRow,"\n")
print("minDist: ",minDist," minTarget: ", minTarget)



