from sklearn import tree 
import sys

def initDataFromFile(file, data):
	i = 0
	for line in file:
		if i ==0:
			i+=1
			continue
		nrs = [int(s) for s in line.split(' ')]
		data.append(nrs)

trainingPath = sys.argv[1]
testPath = sys.argv[2]
testFile = open(testPath,'r')
trainingFile = open(trainingPath,'r')

trainingInstances = []
initDataFromFile(trainingFile, trainingInstances)

testInstances = []
initDataFromFile(testFile, testInstances)

trainingInstanceLabels = []

goodMovieIndex = len(trainingInstances[0])-1
for trainingInstance in trainingInstances:
	goodMovie = trainingInstance[goodMovieIndex]
	trainingInstanceLabels.append(goodMovie)
	trainingInstance.pop(goodMovieIndex)

testInstanceLabels = []
goodMovieIndex = len(testInstances[0])-1
for testInstance in testInstances:
	goodMovie = testInstance[goodMovieIndex]
	testInstanceLabels.append(goodMovie)
	testInstance.pop(goodMovieIndex)

decisionTree = tree.DecisionTreeClassifier()
decisionTree.criterion = "entropy"
decisionTree = decisionTree.fit(trainingInstances, trainingInstanceLabels)
predictions = decisionTree.predict(testInstances)

correctPredictions = []

TP = 0
FP = 0
TN = 0
FN = 0
P = sum(testInstanceLabels)
N = len(testInstanceLabels) - P

for i in range(len(predictions)):
	#true prediction
	if predictions[i] == testInstanceLabels[i]:
		correctPredictions.append(True)
		#true positive
		if predictions[i] == 1:
			TP+=1
		#true negative
		elif predictions[i] == 0:
			TN += 1

	#false prediction
	else:
		correctPredictions.append(False)
		#false positive
		if predictions[i] == 1:
			FP+=1
		#false negative
		elif predictions[i] == 0:
			FN += 1

errorRate = float(FP+FN)/float(P+N)

print("Number of movies in test set: " + str(len(testInstances)))
print("True positives: " + str(TP))
print("True negatives: " + str(TN))
print("False positives: " + str(FP))
print("False negatives: " + str(FN))
print("Error rate: " + str(errorRate))