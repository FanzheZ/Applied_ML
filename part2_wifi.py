from sklearn import neighbors
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pathName = "/Users/fanzhe/Documents/2020_Spring/Applied_ML/HW3/"
dataFrame_train = pd.read_excel(pathName + 'wifi.xlsx', sheet_name='train')
dataFrame_test = pd.read_excel(pathName + 'wifi.xlsx', sheet_name='test')

trainX = dataFrame_train.drop(["level"], axis=1)
trainY = dataFrame_train.level

testX = dataFrame_test.drop(["level"], axis=1)
testY = dataFrame_test.level
testY_length = len(testY)

wm_er_list = []
we_er_list = []
uwm_er_list = []
uwe_er_list = []

k = 1
k_list = []
while k <= 25:
    k_list.append(k)

    # weighted
    weighted_clf_manhattan = neighbors.KNeighborsClassifier(n_neighbors=k, p=1,weights='distance')
    weighted_clf_manhattan.fit(trainX, trainY)

    weighted_clf_euclidean = neighbors.KNeighborsClassifier(n_neighbors=k, p=2, weights='distance')
    weighted_clf_euclidean.fit(trainX, trainY)

    # unweighted
    unweighted_clf_manhattan = neighbors.KNeighborsClassifier(n_neighbors=k, p=1, weights='uniform')
    unweighted_clf_manhattan.fit(trainX, trainY)

    unweighted_clf_euclidean = neighbors.KNeighborsClassifier(n_neighbors=k, p=2, weights='uniform')
    unweighted_clf_euclidean.fit(trainX, trainY)

    ind = 0
    expected = []
    wm_predicted = []
    we_predicted = []
    uwm_predicted = []
    uwe_predicted = []

    # making prediction an put into lists
    for row in testX.iterrows():
        current_testX = row[1]
        actual_testY = testY[ind]
        expected.append(actual_testY)

        w_m_Y = weighted_clf_manhattan.predict(np.asarray(current_testX).reshape(1,-1))
        wm_predicted.append(w_m_Y)

        w_e_Y = weighted_clf_euclidean.predict(np.asarray(current_testX).reshape(1,-1))
        we_predicted.append(w_e_Y)

        uw_m_Y = unweighted_clf_manhattan.predict(np.asarray(current_testX).reshape(1,-1))
        uwm_predicted.append(uw_m_Y)

        uw_e_Y = unweighted_clf_euclidean.predict(np.asarray(current_testX).reshape(1,-1))
        uwe_predicted.append(uw_e_Y)

        ind += 1


    # generating confusion matrix
    wm_conf = confusion_matrix(expected, wm_predicted)
    we_conf = confusion_matrix(expected, we_predicted)
    uwm_conf = confusion_matrix(expected, uwm_predicted)
    uwe_conf = confusion_matrix(expected, uwe_predicted)

    # wrong number = sum of elements in the confusion matrix except the elements at (0,0) (1,1) (2,2) (3,3)
    row = 0
    col = 0
    error_wm = 0
    error_we = 0
    error_uwm = 0
    error_uwe = 0

    while row < 4:
        while col < 4:
            if row != col:
                error_wm += wm_conf[row][col]
                error_we += we_conf[row][col]
                error_uwm += uwm_conf[row][col]
                error_uwe += uwe_conf[row][col]

            col += 1
        row += 1
        col = 0

    # error rate
    # print("error_wm: ", error_wm,"total: ", total_wm)
    wm_er = error_wm / testY_length
    we_er = error_we / testY_length
    uwm_er = error_uwm / testY_length
    uwe_er = error_uwe / testY_length

    wm_er_list.append(wm_er)
    we_er_list.append(we_er)
    uwm_er_list.append(uwm_er)
    uwe_er_list.append(uwe_er)

    # print confusion matrix for k=1,5,25
    if k == 1 or k == 5 or k == 25:
        print("k = ", k, "Confusion Matrix")
        print("Weighted Manhattan Distance: \n", wm_conf)
        print("Weighted Euclidean Distance: \n", we_conf)
        print("Unweighted Manhattan Distance: \n", uwm_conf)
        print("Unweighted Euclidean Distance: \n", uwe_conf, "\n\n")

    k += 1

# plot
f1 = plt.figure(1)
plt.plot(k_list, wm_er_list)
plt.xlabel('k')
plt.ylabel('error rate')
plt.title('Weighted Manhattan Distance')

f2 = plt.figure(2)
plt.plot(k_list, we_er_list)
plt.xlabel('k')
plt.ylabel('error rate')
plt.title('Weighted Euclidean Distance')

f3 = plt.figure(3)
plt.plot(k_list, uwm_er_list)
plt.xlabel('k')
plt.ylabel('error rate')
plt.title('Unweighted Manhattan Distance')

f4 = plt.figure(4)
plt.plot(k_list, uwe_er_list)
plt.xlabel('k')
plt.ylabel('error rate')
plt.title('Unweighted Euclidean Distance')

plt.show()