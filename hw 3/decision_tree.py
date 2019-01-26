import sys
import numpy as np
from copy import *
from math import *
import collections

class FeatureNode:
	def __init__(self, label, splitValues, parentNode):
		self.parentNode = parentNode
		self.label = label
		self.splitValues = splitValues
		self.split = {}
		for splitValue in splitValues:
			self.split[splitValue] = []

	def description(self):
		if self.parentNode != None:
			print("The parent node has feature name: " + str(self.parentNode.featureName))
		else:
			print("There is no parent node")
		print("Feature name is: " + self.featureName)
		print("The split values are: " + str(self.splitValues))
		print("The split is populated with: " + str(self.split))
		for (key, value) in self.split.iteritems():
			print("The nr of " + str(key) + " are: " + str(len(value)))

class ConceptNode:
	def __init__(self, parentNode, label, instances):
		self.parentNode = parentNode
		self.label = label
		self.instances = []


class T:
	def __init__(self,label, splitValues, parentNode, instances, features):
		instances = []
		splitValues = splitValues
		self.parentNode = parentNode
		self.label = label
		self.splitValues = []
		self.features = features
		self.instances = instances
		self.split = {}
		if splitValues != None:
			for splitValue in splitValues:
				self.split[splitValue] = []

	def description(self):
		if self.parentNode != None:
			print("The parent node has label: " + str(self.parentNode.label))
		else:
			print("There is no parent node")
		print("Label name is: " + self.label)
		print("The split values are: " + str(self.splitValues))
		print("The split is populated with: " + str(self.split))
		print("The instances here are: " + str(self.instances))
		# for (key, value) in self.split.iteritems():
		# 	print(key, value)

	def homogeneous(self, D):
		D = deepcopy(D)
		#print("homogeneous")
		h = 0
		for row in D:
			if row[len(row)-1] ==1:
				h+=1

		h = float(h)/float(len(D))
		return h >= 0.8 or h<=0.2

	def Label(self, D, F):
		D = deepcopy(D)
		instance = D[0]
		label = ""

		h = 0
		for row in D:
			if row[len(row)-1] ==1:
				h+=1

		if h>= 0.8:
			label = F[len(F)-1][0]
		else:
			label = "Not " + F[len(F)-1][0]

		t = T(label, None, self, D, F)
		t.parentNode.split[label] = t

		return t

	def bestSplit(self, D,F):
		D = deepcopy(D)
		#print("bestSplit")
		iMin = 1 
		fBest = None
		D_sub = {}

		#init D_sub
		for feature in F:
			if feature == F[len(F)-1]:
				continue
			D_sub[feature[0]] = {}
			for value in feature[1]:
				D_sub[feature[0]][value] = {"Positive":0, "Negative":0}

		#Count positives and negatives in the data by
		#Iterate over the instances in the data
		for i in range(len(D)):
			instance = D[i]
			#iterate over the features in F
			for j in range(len(F)-1):
				feature = F[j]
				for value in feature[1]:
					if value == instance[j+1] and instance[len(instance)-1]==1:
						D_sub[feature[0]][value]["Positive"] += 1
						break
					elif value == instance[j+1] and instance[len(instance)-1]==0:
						D_sub[feature[0]][value]["Negative"] += 1
						break


		#Find the best split feature
		for feature in D_sub:
			impTot = 0
			for value in D_sub[feature]:
				imp = None
				p = D_sub[feature][value]["Positive"]
				n = D_sub[feature][value]["Negative"]
				if p+n == 0:
					continue
				pDot = float(p)/float(p+n)
				if pDot == 0 or pDot == 1:
					imp = 0
				else:
					imp = -pDot*log(pDot,2)-(1-pDot)*log((1-pDot),2)
				impTot += (float(p+n)/float(len(D)))*imp

			if impTot < iMin:
				iMin = impTot
				fBest = feature

		return fBest

	def splitData(self, D ,F,feature):
		D = deepcopy(D)
		#print("splitData")
		index = 0
		for i in range(len(feature)):
			if feature == F[i][0]:
				index = i
				break

		valueCount = len(F[index][1])
		
		D_sub = []
		for i in range(valueCount):
			splitValue = F[index][1][i]
			D_sub.append((splitValue,[]))

		for instance in D:
			for j in range(0,valueCount):
				value = F[index][1][j]
				if instance[index+1] == value:
					D_sub[j][1].append(instance)
					break

		return D_sub

	def growTree(self, D,F):
		D = deepcopy(D)
		#print("growTree")
		if self.homogeneous(D):
			#print("Label: " + str(D))
			return self.Label(D,F)

		feature = self.bestSplit(D,F)
		self.label = feature
		splitData = self.splitData(D,F,feature)
		index = 0
		for row in F:
			if row[0] == feature:
				break
			index+=1
		F.pop(index)
		for row in splitData:
			self.splitValues.append(row[0])
		for splitValue, extension in splitData:
			if extension:
				#print("EXTENSION" + str(extension))
				self.growTree(extension, F)
			else:
				#print("NOT EXTENSION" + str(extension))
				self.Label(D,F)

		return self

def initDataFromFile(file, data):
	i = 0
	for line in file:
		if i ==0:
			i+=1
			continue
		nrs = [int(s) for s in line.split(' ')]
		data.append(nrs)

#def growTree(data, featureTree):


filePath = sys.argv[1]
file = open(filePath,'r')

data = []
initDataFromFile(file, data)


for elem in data:
	nr = elem[0]
	budget = elem[1]
	genre = elem[2]
	famActors = elem[3]
	director = elem[4]
	goodMovie = elem[5]

# gillsNode = FeatureNode("Gills", [True, False], None)
# lengthNode = FeatureNode(("Length"),[3,4,5], gillsNode)
# teethNode = FeatureNode("Teeth", ["Many", "Few"], lengthNode)

# gillsNode.description()

# D_sub = {}
# F = {"Gills": [True, False], "Length":[3,4,5], "Teeth":["Many", "Few"]}
# for key in F:
# 	D_sub[key] = {}
# 	for value in F[key]:
# 		D_sub[key][value] = [0]*2

# print(D_sub)

F = [("Length", [3,4,5]), ("Gills",[0,1]), ("Beak",[0,1]), ("Teeth",["Many","Few"]), ("Dolphin",[0,1])]
p1 = [1,3,0,1,"Many",1]
p2 = [2,4,0,1,"Many",1]
p3 = [3,3,0,1,"Few",1]
p4 = [4,5,0,1,"Many",1]
p5 = [5,5,0,1,"Few",1]
n1 = [6,5,1,1,"Many",0]
n2 = [7,4,1,1,"Many",0]
n3 = [8,5,1,0,"Many",0]
n4 = [9,4,1,0,"Many",0]
n5 = [10,4,0,1,"Few",0]
data = [p1,p2,p3,p4,p5,n1,n2,n3,n4,n5]

t = T("",None, None, data, F)
t.growTree(data, F)
t.description()
for key in t.split:
	print(t.split[key].description())





