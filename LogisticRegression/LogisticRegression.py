from numpy import *
import logRegres

dataMat = []; labelMat = []
fr = open('testSet.txt')
for line in fr.readlines():
    lineArr = line.strip().split()
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    labelMat.append(int(lineArr[2]))

weights = logRegres.stocGradAscent0(array(dataMat), labelMat)
logRegres.plotBestFit(weights)

