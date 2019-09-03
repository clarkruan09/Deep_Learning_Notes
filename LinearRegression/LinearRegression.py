import numpy as np
import matplotlib.pyplot as plt

fileName = "ex0.txt"

# Get number of features
numFeatures = len(open("ex0.txt").readline().split('\t')) - 1
train_X = []
train_y = []

# Read Data
fr = open(fileName)
for line in fr.readlines():
	lineArr = []
	curLine = line.strip().split('\t')
	for i in range(numFeatures):
		lineArr.append(float(curLine[i]))

	train_X.append(lineArr)
	train_y.append(float(curLine[-1]))

# Visualize Data
XX = []
for i in range(len(train_X)):
	XX.append(train_X[i][1])

plt.plot(XX, train_y, 'ro')

# Regression
xMat = np.mat(train_X)
yMat = np.mat(train_y).T
xTx = xMat.T * xMat
w = xTx.I * (xMat.T * yMat)

# Show result
prediction = xMat * w
plt.plot(xMat[:, 1], prediction)
plt.show()