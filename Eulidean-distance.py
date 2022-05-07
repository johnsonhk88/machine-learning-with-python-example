
from turtle import color
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
from sklearn.metrics import euclidean_distances
style.use('fivethirtyeight')

#test 1
plot1 = [1,3]
plot2 = [2, 5]

euclideanDistances = sqrt((plot1[0]- plot2[0])**2  + (plot1[1]- plot2[1])**2)

print(euclideanDistances)


#test2 
dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[6,5], [7,7], [8,6]]}

newFeatures = [5,7]


#plot dataset
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset] # plot dataset
# plt.scatter(newFeatures[0], newFeatures[1])
# plt.show()

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
    print("Votes: ", votes, "\n\rVote Result: ", Counter(votes).most_common(1))   
    
    return voteResult


#test knn
result = kNN(dataset, newFeatures, 3)
print(result)
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset] # plot dataset
plt.scatter(newFeatures[0], newFeatures[1], color=result)
plt.show()