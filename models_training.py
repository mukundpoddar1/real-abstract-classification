import cv2
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

'''Enter required features as single string. No spaces'''

def load_csv(file_name, choose_cols):
	dataset = pd.read_csv (file_name, header=None)
	dataset = dataset.values
	rows = len(dataset)
	cols = len(dataset[0])
	feature_data = []
	label_data = []
	colStart = 0
	colEnd = cols-1
	for r in range (0, rows):
		for c in range (colStart, colEnd):
			dataset[r][c] = float(dataset[r][c])
		feature_data.append (dataset[r][choose_cols])
		label_data.append (int(dataset[r][cols-1]))
	'''print (dataset[0])
	print (cols)
	for r in range(0, rows):
		for c in range(cols-1):
			dataset[r][c] = float(dataset[r][c])
		dataset[r][cols-1] = int(dataset[r][cols-1])
		feature_data.append(dataset[r][:cols])
		label_data.append(dataset[r][-1])'''
	#print (feature_data[0], label_data[0])
	return feature_data, label_data

def make_training_and_testing(features, labels):
	c = list(zip(features, labels))
	np.random.shuffle(c)
	features, labels = zip(*c)
	features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.33, random_state=42)
	return features_train, features_test, labels_train, labels_test

def ml_logistic_regression(features_train, features_test, labels_train, labels_test):
	model = LogisticRegression()
	#features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.33, random_state=42)
	scores = model_selection.cross_val_score(model, features_train, labels_train, cv=5)
	print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	model.fit(features_train, labels_train)
	predictions = model.predict(features_test)
	print (classification_report(labels_test, predictions))
	print (confusion_matrix(labels_test, predictions))
	print (accuracy_score(labels_test, predictions))

def ml_k_nearest_neighbours(features_train, features_test, labels_train, labels_test):
	model = KNeighborsClassifier()
	scores = model_selection.cross_val_score(model, features_train, labels_train, cv=5)
	print("KNN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	model.fit(features_train, labels_train)
	predictions = model.predict(features_test)
	print (classification_report(labels_test, predictions))
	print (confusion_matrix(labels_test, predictions))
	print (accuracy_score(labels_test, predictions))

def ml_support_vector_machine(features_train, features_test, labels_train, labels_test):
	model = SVC(kernel='sigmoid');
	scores = model_selection.cross_val_score(model, features_train, labels_train, cv=5)
	print("SVM Gaussian: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	model.fit(features_train, labels_train)
	predictions = model.predict(features_test)
	print (classification_report(labels_test, predictions))
	print (confusion_matrix(labels_test, predictions))
	print (accuracy_score(labels_test, predictions))

def ml_decision_tree(features_train, features_test, labels_train, labels_test):
	model = DecisionTreeClassifier()
	scores = model_selection.cross_val_score(model, features_train, labels_train, cv=5)
	print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	model.fit(features_train, labels_train)
	predictions = model.predict(features_test)
	print (classification_report(labels_test, predictions))
	print (confusion_matrix(labels_test, predictions))
	print (accuracy_score(labels_test, predictions))

file_name = './output_features_hh.csv' #input ('Enter the path of features.csv: ')

print('1: Texture 2: Hue Histogram 3: Contrast 4: Coherence 5: Hue Count 6: Sobel Otsu')
print('Choose subset:')
in_string=input()
choose_cols=[]
for num in in_string:
	num=int(num)
	if num==1:
		for item in range(6):
			choose_cols.append(item)
	elif num==2:
		choose_cols.append(6)
	elif num==3:
		choose_cols.append(7)
	elif num==4:
		choose_cols.append(8)
	elif num==5:
		choose_cols.append(9)
	elif num==6:
		for item in range(10,35):
			choose_cols.append(item)
print(choose_cols)

features,labels = load_csv(file_name, choose_cols)
features_train, features_test, labels_train, labels_test = make_training_and_testing(features, labels)
ml_logistic_regression(features_train, features_test, labels_train, labels_test)
ml_k_nearest_neighbours(features_train, features_test, labels_train, labels_test)
#ml_support_vector_machine(features_train, features_test, labels_train, labels_test)
ml_decision_tree(features_train, features_test, labels_train, labels_test)