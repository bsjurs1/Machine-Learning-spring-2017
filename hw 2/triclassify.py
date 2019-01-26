import sys
import numpy as np

trainingDataFilePath = sys.argv[1]
testDataFilePath = sys.argv[2]

trainingData = open(trainingDataFilePath,'r')
testData = open(testDataFilePath,'r')

#Variables used in training
i = 0
dim = 0
aCount = 0
bCount = 0
cCount = 0

cumulativeAVecValue = np.zeros(1)
cumulativeBVecValue = np.zeros(1)
cumulativeCVecValue = np.zeros(1)

for line in trainingData:
	nrs = [float(s) for s in line.split(' ')]
	if i==0:
		dim = int(nrs[0])
		aCount = nrs[1]
		bCount = nrs[2]
		cCount = nrs[3]
		cumulativeAVecValue = np.zeros(dim)
		cumulativeBVecValue = np.zeros(dim)
		cumulativeCVecValue = np.zeros(dim)
	else:
		featureVec = np.array(nrs)
		if i < (aCount+1):
			cumulativeAVecValue = np.add(cumulativeAVecValue, featureVec)
		elif i >= (aCount+1) and i < (aCount+bCount+1):
			cumulativeBVecValue = np.add(cumulativeBVecValue, featureVec)
		elif i >= (aCount+bCount+1):
			cumulativeCVecValue = np.add(cumulativeCVecValue, featureVec)
	i+=1

aCentroid = np.divide(cumulativeAVecValue, aCount)
bCentroid = np.divide(cumulativeBVecValue, bCount)
cCentroid = np.divide(cumulativeCVecValue, cCount)

#Construct A/B discriminant
p = aCentroid
n = bCentroid
abw = p - n
ab = np.dot((p-n),(p+n))/2

#Construct B/C discriminant
p = bCentroid
n = cCentroid
bcw = p - n
bc = np.dot((p-n),(p+n))/2

#Construct A/C discriminant
p = aCentroid
n = cCentroid
acw = p - n
ac = np.dot((p-n),(p+n))/2

#Evaluate the dot product of x and w: If x*w>t, then+ Otherwise –


#Variables used in training
i = 0
testDim = 0
testALen = 0
testBLen = 0
testCLen = 0

contingencyMatrix = np.zeros((3,3))

abContingencyMatrix = np.zeros((2,2))
acContingencyMatrix = np.zeros((2,2))
bcContingencyMatrix = np.zeros((2,2))

for line in testData:
	nrs = [float(s) for s in line.split(' ')]
	if i==0:
		dim = nrs[0]
		testALen = nrs[1]
		testBLen = nrs[2]
		testCLen = nrs[3]
	else:
		featureVec = np.array(nrs)
		evalABValue = np.dot(featureVec, abw)
		evalBCValue = np.dot(featureVec, bcw)
		evalACValue = np.dot(featureVec, acw)

		if evalABValue>=ab and evalACValue>=ac:
			#print("Predict A")
			if i < testALen+1:
				contingencyMatrix[0][0] += 1
				#print("It is A")
				abContingencyMatrix[0][0] += 1
				acContingencyMatrix[0][0] += 1
			elif i >= testALen+1 and i < testALen+testBLen+1: 
				contingencyMatrix[0][1] += 1
				#print("It is B")
				abContingencyMatrix[0][1] += 1
			elif i >= testALen+testBLen+1:
				contingencyMatrix[0][2] += 1
				#print("It is C")
				acContingencyMatrix[0][1] += 1

		elif evalABValue < ab and evalBCValue >= bc:
			#print("Predict B")
			if i < testALen+1:
				contingencyMatrix[1][0] += 1
				#print("It is A")
				abContingencyMatrix[1][0] += 1
			elif i >= testALen+1 and i < testALen+testBLen+1: 
				contingencyMatrix[1][1] += 1
				#print("It is B")
				abContingencyMatrix[1][1] += 1
				bcContingencyMatrix[0][0] += 1
			elif i >= testALen+testBLen+1:
				contingencyMatrix[1][2] += 1
				#print("It is C")
				bcContingencyMatrix[0][1] += 1

		else:
			#print("Predict C")
			if i < testALen+1:
				contingencyMatrix[2][0] += 1
				#print("It is A")
				acContingencyMatrix[1][0] += 1
			elif i >= testALen+1 and i < testALen+testBLen+1: 
				contingencyMatrix[2][1] += 1
				#print("It is B")
				bcContingencyMatrix[1][0] += 1
			elif i >= testALen+testBLen+1:
				contingencyMatrix[2][2] += 1
				#print("It is C")
				acContingencyMatrix[1][1] += 1
				bcContingencyMatrix[1][1] += 1
	i+=1

#Assign true positives
abTP = abContingencyMatrix[0][0]
bcTP = bcContingencyMatrix[0][0]
acTP = acContingencyMatrix[0][0]

#Assign false positives
abFP = abContingencyMatrix[0][1]
bcFP = bcContingencyMatrix[0][1]
acFP = acContingencyMatrix[0][1]

#Assign false negatives
abFN = abContingencyMatrix[1][0]
bcFN = bcContingencyMatrix[1][0]
acFN = acContingencyMatrix[1][0]

#Assign true negatives
abTN = abContingencyMatrix[1][1]
bcTN = bcContingencyMatrix[1][1]
acTN = acContingencyMatrix[1][1]

#Calculate positive and negative values
abP = abTP + abFN
abN = abFP + abTN
acP = acTP + acFN
acN = acFP + acTN
bcP = bcTP + bcFN
bcN = bcFP + bcTN


cumulativeP = abP+acP+bcP
cumulativeN = abN+acN+bcN
cumulativeTP = abTP+acTP+bcTP
cumulativeFP = abFP+acFP+bcFP
cumulativeFN = abFN + acFN + bcFN
cumulativeTN = abTN + acTN + bcTN
cumulativeTP = abTP + acTP + bcTP
cumulativeEstimatedP = np.sum(acContingencyMatrix[0,:]) + np.sum(abContingencyMatrix[0,:]) + np.sum(bcContingencyMatrix[0,:])

#calculate avg true positive rate
avgTPR = cumulativeTP/cumulativeP

#calculate avg false positive rate
avgFPR = cumulativeFP/cumulativeN

# error rate = FP+FN/P+N
avgErrorRate = (cumulativeFP+cumulativeFN)/(cumulativeP+cumulativeN)

# accuracy = (TP+TN)/(P+N)
avgAccuracy = (cumulativeTP + cumulativeTN)/(cumulativeP+cumulativeN)

#precision TP/(Estimated P)
avgPrecision = cumulativeTP/cumulativeEstimatedP

print("True positive rate = ", avgTPR)
print("False positive rate = ", avgFPR)
print("Error rate = ", avgErrorRate)
print("Accuracy = ", avgAccuracy)
print("Precision = ", avgPrecision)


