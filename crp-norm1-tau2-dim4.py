# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import myCrpFunctions
import os

#Make Folder
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def state_phase(v, start, dim, tau):

	return [v[start + i*tau] for i in range(0, dim, 1)]



def predict_diagonal(trainSet, testSet, dim=5, tau=2, epsilon=0.7, lambd=3, percent=0.6, distNorm = 1,
	titleOfGraph = 'Pretty Girl', figureName = 'GrilLikeYou', pathSaveFigure = None):

	vectors_train_1 = []
	for i in range(len(trainSet)-(dim-1)*tau):
		vectors_train_1.append(state_phase(trainSet, i, dim, tau)) 

	#tách statephases
	vectors_test_1 = []
	for i in range(len(testSet)-(dim-1)*tau):
		vectors_test_1.append(state_phase(testSet, i, dim, tau))

	#ép kiểu về array
	vectors_train_1 = np.array(vectors_train_1)
	vectors_test_1 = np.array(vectors_test_1)

	r_dist = cdist(vectors_train_1, vectors_test_1, 'minkowski', p=distNorm)

	
	# print('vectors_train.shape: ', vectors_train_1.shape) #in ra shape
	# print('vectors_test.shape: ', vectors_test_1.shape)
	# print('r_dist.shape: ', r_dist.shape)
	# print('r_dist min', r_dist.min())
	# print('epsilon: ', epsilon)
	# print('r_dist max', r_dist.max())
	# print('lambd: ', lambd)
	
	r1 = np.array((r_dist < epsilon)-2)


	len_r1 = int(np.size(r1[0]))
	high_r1 = int(np.size(r1)/len_r1)

	diagonals_have_predict = []
	for index_diagonal in range(-(high_r1 - lambd + 1), len_r1 - lambd + 2, 1):
		offset = index_diagonal
		#---offset = x - y
		y = -offset if (index_diagonal < 0) else 0

		while (y < high_r1 and y+offset < len_r1):
			if (r1[y][y+offset] == -1):
				start = y
				while ( y+1<high_r1 and y+1+offset < len_r1 and r1[y+1][y+1+offset] == -1):
					y+=1
				if (y - start + 1 >= lambd):
					predicts = np.full(y+1, -2)

					for index in range(start, y+1, 1):
						predicts[index] = index + offset
					diagonals_have_predict.append(predicts)
			y+=1

	indexSampleOrigin = diagonals_have_predict

	for i_row in range(len(indexSampleOrigin)):
		for i_in_1_State in range(len(indexSampleOrigin[i_row])):
			if (indexSampleOrigin[i_row][i_in_1_State] >= 0):
				#Chọn số index state cuối cùng
				if ((i_in_1_State == len(indexSampleOrigin[i_row]) - 1) or (indexSampleOrigin[i_row][i_in_1_State+1] < 0)):
					for j in range(((int(dim*percent))-1)*tau):
						indexSampleOrigin[i_row] = np.insert(indexSampleOrigin[i_row], i_in_1_State+j+1, indexSampleOrigin[i_row][i_in_1_State] + j + 1)

	# Lấy giá trị từ index của sample trong testSet
	valueOfSample = []

	for i in range(len(indexSampleOrigin)):
		arr = []
		for j in range(len(indexSampleOrigin[i])):
			if (indexSampleOrigin[i][j] < 0): 
				arr.append(None)
			else:
				arr.append(testSet[indexSampleOrigin[i][j]])

		valueOfSample.append(arr)

	f_line = plt.figure(figureName, figsize = (12.8, 7.2), dpi = 100)

	print("_________num predict: ", len(valueOfSample))

	if (len(valueOfSample) > 0):
		# f_line.set_size_inches(1080, 720)
		for i in range(0, len(valueOfSample)):
			plt.plot( valueOfSample[i], ':', label=i)
		plt.plot(trainSet, 'r', label='train')
		plt.legend(loc=0)
		plt.xlabel('index')
		plt.ylabel('feature')

		titleOfGraph = titleOfGraph + " - lamb_" + str(lambd) + ' - minkowski_' + str(distNorm) + " - numPredict_" + str(len(valueOfSample))
		plt.title(titleOfGraph)

		if (pathSaveFigure != None):
			plt.savefig(pathSaveFigure, dpi = 200)
		# plt.ylim(ymin = min(trainSet) - 0.09)
		# plt.show()

	return f_line


###############################-----________MAIN________-----###############################

if (__name__ == "__main__"):

	indexOfCol = 5
	print("indexOfCol: ", indexOfCol)

	dataTrain = myCrpFunctions.readCSVFile("data/1.csv", indexOfCol)
	dataTest = myCrpFunctions.readCSVFile("data/2.csv", indexOfCol)
	dataTest += (myCrpFunctions.readCSVFile("data/3.csv", indexOfCol))

	if (min(dataTrain) > min(dataTest)) :
		minOfNorm = min(dataTest) 
	else: 
		minOfNorm = min(dataTrain) 

	if max(dataTrain) < max(dataTest) : 
		maxOfNorm = max(dataTest) 
	else: 
		maxOfNorm = max(dataTrain) 
	
	trainSet = myCrpFunctions.ConvertSetNumber(dataTrain, minOfSet = minOfNorm, maxOfSet = maxOfNorm)
	testSet = myCrpFunctions.ConvertSetNumber(dataTest, minOfSet = minOfNorm, maxOfSet = maxOfNorm)

	print("len(trainSet): ", len(trainSet))
	print("len(testSet): ", len(testSet))

	numSample = 45
	myLambd=3
	myDim=4
	distNorm = 1
	tau = 2
	formatSave = ".png"

	pathFolder = "output03012019/" + "minkowski_" + str(distNorm) + " - tau = " + "/"

	for markEpsilon in range(85, 1, -5):
		epsilon = float(markEpsilon/10000);
	# epsilon = 0.00345;
		subFolderName = "col_" + str(indexOfCol) + " - dim_" + str(myDim) + " - numSamp_" + str(numSample) + " - epsilon_" + str(epsilon)

		print("\n------------------------------------------------", subFolderName,"------------------------------------------------\n")
		pathNewFolder = pathFolder + subFolderName + "/"
		createFolder(pathNewFolder)

		for start in range(2180, len(trainSet), numSample - myLambd):
			finish = start+numSample
			title = "index_" + str(start) + " - num_" + str(numSample) + " - epsil_" + str(epsilon)

			pathSave = pathNewFolder + title + formatSave
			print("---------------------------", title, "---------------------------\n")

			f2 = predict_diagonal(trainSet[start:finish], testSet ,
			 						dim=myDim, tau=tau, epsilon=epsilon, lambd=myLambd, percent=1, distNorm=distNorm, titleOfGraph = title,  figureName = title, pathSaveFigure = pathSave)

			# plt.show()
	print("\n------------------------------------------Xong!------------------------------------------")
		
		