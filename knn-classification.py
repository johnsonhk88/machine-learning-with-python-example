import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

accuracies = []
for i in range(25):
    dataSetFile = "breast-cancer-wisconsin.data"

    #open csv file
    df =pd.read_csv(dataSetFile)


    #data preparation (cleaning)
    df.replace('?', -99999, inplace=True)

    #drop unused data (e.g. id)
    df.drop(['id'], 1, inplace=True)


    #define input x data for train
    X = np.array(df.drop(['class'], 1)) # only remove class label
    y = np.array(df['class']) # get label only

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    #inital KNN classifier
    clf = neighbors.KNeighborsClassifier(n_jobs=1)

    #train the model
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    # print("Accuary: ", accuracy)
    accuracies.append(accuracy)


    # exampleMeasures2 = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
    # exampleMeasures2 =exampleMeasures2.reshape(len(exampleMeasures2), -1)

    # predication2 = clf.predict(exampleMeasures2)

    # print("Predict2 ", predication2)


    # exampleMeasures = np.array([4,2,1,1,1,2,3,2,1])
    # print(len(exampleMeasures))
    # exampleMeasures =exampleMeasures.reshape(1, -1) # convert 1D array to 2D array
    # print(exampleMeasures)
    # predication1 = clf.predict(exampleMeasures)

    # print("Predict1 ", predication1)
    
print(sum(accuracies)/len(accuracies))

