import sys
import numpy as np
from copy import *
from math import *

#CONSTANTS 
NUMBER = 'Number'

LENGTH = 'Length'
GILLS = 'Gills'
BEAK = 'Beak'
TEETH = 'Teeth'
DOLPHIN = 'Dolphin'

POSITIVE = 'Positive'
NEGATIVE = 'Negative'

BUDGET = 'Budget'
GENRE = 'Genre'
FAMOUSACTORS = 'FamousActors'
DIRECTOR = 'Director'
GOODMOVIE = 'GoodMovie'

class T:
	def __init__(self, parent=None, label='', children={}):
		self.parent = parent
		self.label = label
		self.children = children

def bestSplit(D,F):
	print("bestSplit")
	#print(D)
	D = deepcopy(D)
	iMin = 1 
	fBest = None
	D_sub = {}

	#init D_sub
	for key in F:
		if key == DOLPHIN:
			continue
		D_sub[key] = {}
		for value in F[key]:
			D_sub[key][F[key][value]] = {POSITIVE:0, NEGATIVE:0}

	#Count positives and negatives in the data by
	#Iterate over the instances in the data
	for instance in D:
		for key in F:
			if key == DOLPHIN:
				continue
			for value in F[key]:
				value = F[key][value]
				if value == instance[key] and instance[DOLPHIN] == True:
					D_sub[key][F[key][value]][POSITIVE] += 1
					break
				elif value == instance[key] and instance[DOLPHIN] == False:
					D_sub[key][F[key][value]][NEGATIVE] += 1
					break

	for key in D_sub:
		impTot = 0
		for value in D_sub[key]:
			imp = None
			p = D_sub[key][value][POSITIVE]
			n = D_sub[key][value][NEGATIVE]
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
			fBest = key

	return fBest

def splitData(D,F,feature):
	print("splitData")
	D = deepcopy(D)
	#print(D)

	D_sub = {}
	for value in F[feature]:
		value = F[feature][value]
		D_sub[value] = []

	for instance in D:
		for value in F[feature]:
			if instance[feature] == value:
				D_sub[value].append(instance)
				break

	return D_sub

def homogeneous(D):
	print("homogeneous")
	D = deepcopy(D)
	#print(D)
	
	dolphinCount = 0

	for instance in D:
		if instance[DOLPHIN] == True:
			dolphinCount += 1

	homogeneousFactor = float(dolphinCount)/float(len(D))

	print("Is Homogenous: " + str(homogeneousFactor >= 0.8 or homogeneousFactor <= 0.2))

	return homogeneousFactor >= 0.8 or homogeneousFactor <= 0.2

def Label(D, F):
	print("Label")
	#print(D)
	D = deepcopy(D)

	label = ""

	dolphinCount = 0

	for instance in D:
		if instance[DOLPHIN] == True:
			dolphinCount += 1

	homogeneousFactor = float(dolphinCount)/float(len(D))

	if homogeneousFactor >= 0.8:
		label = DOLPHIN
	else: #homogeneousFactor <= 0.2:
		label = "Not " + DOLPHIN

	return label

def growTree(root,D,F):
	print("growTree")
	#print(D)
	D = deepcopy(D)
	F = deepcopy(F)

	if homogeneous(D):
		return Label(D,F)

	feature = bestSplit(D,F)
	root.label = feature
	split = splitData(D,F,feature)
	F.pop(feature)


	for key in split:
		t = T()
		if split[key]:
			print("EXTENSION")
			t = growTree(root, split[key], F)
		else:
			print("NOT EXTENSION")
			t = Label(root, D, F)

		if t.label in root.children.keys():
			root.children[t.label].append(t)
		else:
			root.children[t.label] = []
			root.children[t.label].append(t)

	return root

featurePath = '/Users/bjartesjursen/Desktop/Homework 3/HW3data/Key.txt'
trainingPath = '/Users/bjartesjursen/Desktop/Homework 3/HW3data/training.txt'
featureData = open(featurePath,'r')
trainingData = open(trainingPath,'r')

# featureValues = {BUDGET:{0:'Low', 1:'Medium', 2:'High'}, GENRE:{0:'Documentary',1:'Drama',2:'Comedy'}, FAMOUSACTORS:{0:False, 1:True}, DIRECTOR:{0:'Unknown',1:'Great'},GOODMOVIE:{0:False, 1:True}}
# features = [BUDGET,GENRE,FAMOUSACTORS,DIRECTOR,GOODMOVIE]

# i = 0
# for row in trainingData:
# 	if i == 0:
# 		i += 1
# 		continue

# 	row = [int(s) for s in row.split(' ')]
# 	nr = row[0]
# 	budget = row[1]
# 	genre = row[2]
# 	famActors = row[3]
# 	director = row[4]
# 	goodMovie = row[5]
# 	instance = [nr,featureValues[BUDGET][budget],featureValues[GENRE][genre],featureValues[FAMOUSACTORS][famActors],featureValues[DIRECTOR][director],featureValues[GOODMOVIE][goodMovie]]
# 	instances.append(instance)

instances = []


featureValues = {LENGTH:{3:3, 4:4, 5:5}, GILLS:{0:False,1:True}, BEAK:{0:False, 1:True}, TEETH:{'Many':'Many','Few':'Few'},DOLPHIN:{0:False, 1:True}}
features = [LENGTH,GILLS,BEAK,TEETH,DOLPHIN]

p1 = [1,3,0,1,'Many',1]
p2 = [2,4,0,1,'Many',1]
p3 = [3,3,0,1,'Few',1]
p4 = [4,5,0,1,'Many',1]
p5 = [5,5,0,1,'Few',1]
n1 = [6,5,1,1,'Many',0]
n2 = [7,4,1,1,'Many',0]
n3 = [8,5,1,0,'Many',0]
n4 = [9,4,1,0,'Many',0]
n5 = [10,4,0,1,'Few',0]

trainingData = [p1,p2,p3,p4,p5,n1,n2,n3,n4,n5]

i = 0
for row in trainingData:
	nr = row[0]
	length = row[1]
	gills = row[2]
	beak = row[3]
	teeth = row[4]
	dolphin = row[5]
	instance = {NUMBER:nr, LENGTH:featureValues[LENGTH][length],GILLS:featureValues[GILLS][gills], BEAK:featureValues[BEAK][beak], TEETH:featureValues[TEETH][teeth], DOLPHIN:featureValues[DOLPHIN][dolphin]}
	instances.append(instance)

t = T()

growTree(t, instances, featureValues)

print t.children




