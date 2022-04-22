from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


#define sample dataset 
xs = np.array([1,2,3, 4, 5, 6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# random generate correlation data 
def generateDataset(hm, variance, step=2, correlation=False):
    val =1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance , variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    


def bestFitSlopeAndIntercept(xs, ys):
    m= (((mean(xs) * mean(ys)) - mean(xs*ys)) /
        ((mean(xs)* mean(xs)) - mean(xs*xs)))
    b= mean(ys) - m * mean(xs) 
    return m , b

def squareError(ysOrigin, ysLine):
    return sum((ysOrigin- ysLine)**2) # calcuate error square

#calculate cofficient of det
def coefficientOfDetermination(ysOrigin, ysLine):
    yMeanLine = [mean(ysOrigin) for y in ysOrigin]
    squareErrorRegress = squareError(ysOrigin, ysLine) # square error for predict regression line
    squareErrorYMean = squareError(ysOrigin, yMeanLine) # square error for mean of y
    return 1 - (squareErrorRegress / squareErrorYMean)
    
xs, ys = generateDataset(40, 20, 2, correlation='pos')

    
    

#train model 
m , b = bestFitSlopeAndIntercept(xs, ys)




print(m, b) 

# draw line input xs 
regressionLine = [(m*x)+b for x in xs] 
print(regressionLine) 

#predication 
predictX = 8
predictY = (m * predictX) + b

rSquared = coefficientOfDetermination(ys, regressionLine)
print("R Squared : ", rSquared)



plt.scatter(xs, ys) # draw point
plt.scatter(predictX, predictY, color='g') 
plt.plot(xs, regressionLine) # draw line
plt.show()
