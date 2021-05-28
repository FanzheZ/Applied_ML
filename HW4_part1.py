from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing as preproc
import pandas as pd
import numpy as np
from statistics import mode

wifi_er_list = []
shuttle_er_list = []

def error_rate(wifi_expected, wifi_predicted, wifi_test_length, shuttle_expected, shuttle_predicted, shuttle_test_length, Classifier):
    wifi_conf = confusion_matrix(wifi_expected, wifi_predicted)
    shuttle_conf = confusion_matrix(shuttle_expected, shuttle_predicted)

    # wrong number = sum of elements in the confusion matrix except the elements at (0,0) (1,1) (2,2) (3,3)
    wifi_error = 0
    row = 0
    col = 0
    while row < 4:
        while col < 4:
            if row != col:
                wifi_error += wifi_conf[row][col]
            col += 1
        row += 1
        col = 0

    shuttle_error = 0
    row = 0
    col = 0
    while row < 7:
        while col < 7:
            if row != col:
                shuttle_error += shuttle_conf[row][col]
            col += 1
        row += 1
        col = 0

    # error rate
    wifi_error_rate = wifi_error / wifi_test_length
    wifi_er_list.append(wifi_error_rate)

    shuttle_error_rate = shuttle_error / shuttle_test_length
    shuttle_er_list.append(shuttle_error_rate)

    print("\n\n")
    print(Classifier, "on wifi: Number of mislabeled points out of a total", wifi_test_length, "points: ", wifi_error)
    print(Classifier, "wifi conf: \n", wifi_conf, "\n")

    print(Classifier, "on shuttle: Number of mislabeled points out of a total", shuttle_test_length, "points: ", shuttle_error)
    print(Classifier, "shuttle conf: \n", shuttle_conf)


pathName = "/Users/fanzhe/Documents/2020_Spring/Applied_ML/HW4/"
wifi_train = pd.read_excel(pathName + 'wifi.xlsx', sheet_name='train')
wifi_test = pd.read_excel(pathName + 'wifi.xlsx', sheet_name='test')

wifi_trainX = wifi_train.drop(["level"], axis=1)
wifi_trainY = wifi_train["level"]

wifi_testX = wifi_test.drop(["level"], axis=1)
wifi_testY = wifi_test["level"]
wifi_test_length = len(wifi_testY)

shuttle_train = pd.read_excel(pathName + 'shuttle.xlsx', sheet_name='train')
shuttle_test = pd.read_excel(pathName + 'shuttle.xlsx', sheet_name='test')

shuttle_trainX = shuttle_train.drop(["class"], axis=1)
shuttle_trainY = shuttle_train["class"]

shuttle_testX = shuttle_test.drop(["class"], axis=1)
shuttle_testY = shuttle_test["class"]
shuttle_test_length = len(shuttle_testY)

# Normalize
wifi_train_scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
wifi_train_scalerX.fit(wifi_trainX)
wifi_trainX = wifi_train_scalerX.transform(wifi_trainX)

wifi_test_scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
wifi_test_scalerX.fit(wifi_testX)
wifi_testX = wifi_test_scalerX.transform(wifi_testX)

shuttle_train_scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
shuttle_train_scalerX.fit(shuttle_trainX)
shuttle_trainX = shuttle_train_scalerX.transform(shuttle_trainX)

shuttle_test_scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
shuttle_test_scalerX.fit(shuttle_testX)
shuttle_testX = shuttle_test_scalerX.transform(shuttle_testX)

# KNN
wifi_knn = neighbors.KNeighborsClassifier(n_neighbors=3, p=2, weights='distance')
wifi_knn.fit(wifi_trainX, wifi_trainY)

shuttle_knn = neighbors.KNeighborsClassifier(n_neighbors=2, p=2, weights='distance')
shuttle_knn.fit(shuttle_trainX, shuttle_trainY)

wifi_knn_predicted = []
shuttle_knn_predicted = []


# Bayesian Classifier
wifi_bc_predicted = []
shuttle_bc_predicted = []

wifi_clf = GaussianNB()
wifi_bc = wifi_clf.fit(wifi_trainX, wifi_trainY)

shuttle_clf = BernoulliNB()
shuttle_bc = shuttle_clf.fit(shuttle_trainX, shuttle_trainY)


# Decision Tree
wifi_dt_predicted = []
shuttle_dt_predicted = []

wifi_dt_clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best")
wifi_dt_clf.max_depth = 4
wifi_dt = wifi_dt_clf.fit(wifi_trainX, wifi_trainY)

shuttle_dt_clf= tree.DecisionTreeClassifier(criterion="gini", splitter="best")
shuttle_dt_clf.max_depth = 1
shuttle_dt = shuttle_dt_clf.fit(shuttle_trainX, shuttle_trainY)


# Random Forest Classifier
wifi_rf_predicted = []
shuttle_rf_predicted = []
wifi_rf_clf = RandomForestClassifier(max_depth=5, random_state=0)
wifi_rf = wifi_rf_clf.fit(wifi_trainX, wifi_trainY)

shuttle_rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
shuttle_rf = shuttle_rf_clf.fit(shuttle_trainX, shuttle_trainY)

# Make Prediction
wifi_expected = []
shuttle_expected = []

wifi_voted = []
shuttle_voted = []

ind = 0
while ind < wifi_test_length:
    current_wifi_testX = wifi_testX[ind]
    actual_wifi_testY = wifi_testY[ind]
    wifi_expected.append(actual_wifi_testY)

    wifi_knn_predict = wifi_knn.predict(np.asarray(current_wifi_testX).reshape(1, -1))
    wifi_knn_predicted.append(wifi_knn_predict)

    wifi_bc_predict = wifi_bc.predict(np.asarray(current_wifi_testX).reshape(1, -1))
    wifi_bc_predicted.append(wifi_bc_predict)

    wifi_dt_predict = wifi_dt.predict(np.asarray(current_wifi_testX).reshape(1, -1))
    wifi_dt_predicted.append(wifi_dt_predict)

    wifi_rf_predict = wifi_rf.predict(np.asarray(current_wifi_testX).reshape(1, -1))
    wifi_rf_predicted.append(wifi_rf_predict)

    temp = [wifi_knn_predict[0], wifi_bc_predict[0], wifi_dt_predict[0]]
    voted = mode(temp)
    # print(voted)
    wifi_voted.append(voted)
    ind += 1

ind = 0
while ind < shuttle_test_length:
    current_shuttle_testX = shuttle_testX[ind]
    actual_shuttle_testY = shuttle_testY[ind]
    shuttle_expected.append(actual_shuttle_testY)

    shuttle_knn_predict = shuttle_knn.predict(np.asarray(current_shuttle_testX).reshape(1, -1))
    shuttle_knn_predicted.append(shuttle_knn_predict)

    shuttle_bc_predict = shuttle_bc.predict(np.asarray(current_shuttle_testX).reshape(1, -1))
    shuttle_bc_predicted.append(shuttle_bc_predict)

    shuttle_dt_predict = shuttle_dt.predict(np.asarray(current_shuttle_testX).reshape(1, -1))
    shuttle_dt_predicted.append(shuttle_dt_predict)
    #
    shuttle_rf_predict = shuttle_rf.predict(np.asarray(current_shuttle_testX).reshape(1, -1))
    shuttle_rf_predicted.append(shuttle_rf_predict)

    temp = [shuttle_knn_predict[0], shuttle_bc_predict[0], shuttle_dt_predict[0]]
    voted = mode(temp)
    shuttle_voted.append(voted)
    ind += 1


# calculate the confusion matrix, error rate and output them
error_rate(wifi_expected, wifi_knn_predicted, wifi_test_length, shuttle_expected, shuttle_knn_predicted, shuttle_test_length, "KNN")
error_rate(wifi_expected, wifi_bc_predicted, wifi_test_length, shuttle_expected, shuttle_bc_predicted, shuttle_test_length, "Bayesian Classifier")
error_rate(wifi_expected, wifi_dt_predicted, wifi_test_length, shuttle_expected, shuttle_dt_predicted, shuttle_test_length, "Decision Tree")
error_rate(wifi_expected, wifi_voted , wifi_test_length, shuttle_expected, shuttle_voted, shuttle_test_length, "Voting")
error_rate(wifi_expected, wifi_rf_predicted , wifi_test_length, shuttle_expected, shuttle_rf_predicted, shuttle_test_length, "Random Forest Classifier")

print("wifi error rate: knn bc dt voted rf", wifi_er_list)
print("shuttle error rate: knn bc dt voted rf", shuttle_er_list)