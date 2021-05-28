from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import preprocessing as preproc
import pandas as pd
import numpy as np
from queue import PriorityQueue

pathName = "/Users/fanzhe/Documents/2020_Spring/Applied_ML/HW5/"
ccpp_data= pd.read_excel(pathName + 'ccpp.xlsx', sheet_name='allBin')
x = ccpp_data.drop(["ID","TG"],axis = 1)
y = ccpp_data["TG"]

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.3,random_state = 2333)
testY = testY.tolist()
test_length = len(testY)

# Normalize
train_scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
train_scalerX.fit(trainX)
trainX = train_scalerX.transform(trainX)

test_scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
test_scalerX.fit(testX)
testX = test_scalerX.transform(testX)

activation_list = ['relu','identity','tanh']

AUROC_PQ = PriorityQueue()
MC_PQ = PriorityQueue()

from itertools import product
hidden_layer_number = 1

# go through the loop count of hidden layer = 1,2,3
while hidden_layer_number<=3:
    # create all possible tuples
    for hidden_layers_set in product(list(range(1, 21)), repeat=hidden_layer_number):
        print(hidden_layers_set)
        for current_activation in activation_list:
            setting = (hidden_layers_set, current_activation)
            #print(setting)
            # building clf
            clf = MLPClassifier(hidden_layer_sizes= hidden_layers_set, activation = current_activation, solver = 'adam',
                                    alpha = 0.001,learning_rate = 'invscaling',learning_rate_init = 0.1,
                                    early_stopping = True, validation_fraction = 0.25)
            clf.fit(trainX, trainY)

            # predict
            predicted = clf.predict(testX)

            # AUROC
            auroc = metrics.roc_auc_score(testY, predicted)
            AUROC_PQ.put((-auroc, setting))

            # misclassification rate
            missed = (testY!=predicted).sum()
            miss_rate = missed/test_length
            MC_PQ.put((miss_rate, setting))

    hidden_layer_number += 1

i = 1
while i<=10:
    if i==1:
        print("Ten best model architectures by AUROC:")
    auroc = -AUROC_PQ.get()[0]
    setting = AUROC_PQ.get()[1]
    print((auroc,setting))
    i+=1

print("\n")

i = 1
while i<=10:
    if i == 1:
        print("Ten best model architectures by misclassification rate:")
    print(MC_PQ.get())
    i+=1

