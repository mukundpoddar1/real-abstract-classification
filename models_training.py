import cv2
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def load_csv(file_name):
	reader = csv.reader(open(file_name, "rt", encoding="utf8"))
	dataset = list(reader)
	feature_data = []
	label_data = []
	rows = len(dataset)
	cols = len(dataset[0])
	for r in range(rows):
		for c in range(cols-1):
			dataset[r][c] = float(dataset[r][c])
		dataset[r][cols-1] = int(dataset[r][cols-1])
		feature_data.append(dataset[r][33:])
		label_data.append(dataset[r][-1])
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
	model = SVC();
	scores = model_selection.cross_val_score(model, features_train, labels_train, cv=5)
	print("SVM Linear: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	model.fit(features_train, labels_train)
	predictions = model.predict(features_test)
	print (classification_report(labels_test, predictions))
	print (confusion_matrix(labels_test, predictions))
	print (accuracy_score(labels_test, predictions))
	model = SVC(kernel='rbf');
	scores = model_selection.cross_val_score(model, features_train, labels_train, cv=5)
	print("SVM Gaussian: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	model.fit(features_train, labels_train)
	predictions = model.predict(features_test)
	print (classification_report(labels_test, predictions))
	print (confusion_matrix(labels_test, predictions))
	print (accuracy_score(labels_test, predictions))

seed = 7
file_name = input ('Enter the path of features.csv: ')
features,labels = load_csv(file_name)
features_train, features_test, labels_train, labels_test = make_training_and_testing(features, labels)
ml_logistic_regression(features_train, features_test, labels_train, labels_test)
ml_k_nearest_neighbours(features_train, features_test, labels_train, labels_test)
ml_support_vector_machine(features_train, features_test, labels_train, labels_test)