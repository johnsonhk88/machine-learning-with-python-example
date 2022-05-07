
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import random
import pandas as pd



def kNN(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to value less than total voting group!")

    distance = [] # calulate list of distance
    for group in data:  #get data index 
        for features in data[group]:
            # euclideanDistances =sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2) # original 
            # euclideanDistances =np.sqrt(np.sum((np.array(features)-np.array(predict))**2)) # simplify version 1
            euclideanDistances =np.linalg.norm(np.array(features)- np.array(predict)) # simplify version 2 high level
            distance.append([euclideanDistances, group])
    votes =  [i[1] for i in sorted(distance)[:k]]  # sorted 
    voteResult = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k 
    # print("Votes: ", votes, "\n\rVote Result: ", Counter(votes).most_common(1), "\n\rConfidence Result: ", confidence)   
    
    return voteResult , confidence


# #test knn
# result = kNN(dataset, newFeatures, 3)
# print(result)
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset] # plot dataset
# plt.scatter(newFeatures[0], newFeatures[1], color=result)
# plt.show()

accuracies = []
for i in range(25):
    dataSetFile = "breast-cancer-wisconsin.data"

    #open csv file
    df =pd.read_csv(dataSetFile)
    #data preparation (clean)
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    # print(df.head())

    fullData = df.astype(float).values.tolist() # convert to list (Float type)
    # print(fullData[:5]) #
    random.shuffle(fullData)
    # print(20*'#')
    # print(fullData[:5])



    testSize = 0.4
    trainSet = {2:[],  4:[]}
    testSet = {2:[], 4:[]}
    trainData = fullData[:-int(testSize* len(fullData))]  # set Train Data
    testData =fullData[-int(testSize* len(fullData)):]  # set Test Data
    # print(trainData[:5])
    # print(20*'#')
    # print(testData[:5])

    for i in trainData:
        trainSet[i[-1]].append(i[:-1]) # get traindata last data for key , then append dataset

    for i in testData:
        testSet[i[-1]].append(i[:-1])

    # print(trainSet)
    # print(20*'#')
    # print(testSet)

    correct = 0
    total = 0

    for group in testSet: # get key
        for data in testSet[group]: # get each key data
            vote , confidence = kNN(trainSet, data, k=5)
            if group == vote: #check predict correct or not
                correct += 1 #correct counter +1
            # else :
            #     print(confidence) 
            total += 1

    # print("Accuarcy: ", correct/total *100 , "%")
    accuracies.append(correct/total)

print(sum(accuracies)/ len(accuracies) *100 )