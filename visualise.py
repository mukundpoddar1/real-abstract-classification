import csv
import numpy as np
import pandas as pd
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
		feature_data.append(dataset[r][:-1])
		label_data.append(dataset[r][-1])
	return feature_data, label_data


file_name = "./output_features_less_edge.csv"
features,labels = load_csv(file_name)
features1 = features[:164]
features0 = features[164:]
df1 = pd.DataFrame(features1)
df0 = pd.DataFrame(features0)

'''
comb = list(zip(features,labels))
flat_comb=[]
for row in comb:
	flat_comb.append(np.array(row).flatten())
df1=[]
df2=[]
for row in comb:
	if row[-1] == 1:
		df1.append(row)
	else:
		df2.append(row)
df1=pd.DataFrame(df1)
df2=pd.DataFrame(df2)
print (df1.describe(), df2.describe())
'''