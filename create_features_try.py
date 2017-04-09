import cv2
import numpy as np
import csv
#import matplotlib
from  matplotlib import pyplot as plt
from collections import Counter
#from matplotlib import rcsetup

def find_neighbours(input_img, row, column):
	neighbour_indices=[]
	if row>0 and column>0:
		neighbour_indices.append(input_img[row - 1][column - 1])
	else:
		neighbour_indices.append(0)
	if row>0:
		neighbour_indices.append(input_img[row - 1][column])
	else:
		neighbour_indices.append(0)
	if row>0 and column<299:
		neighbour_indices.append(input_img[row - 1][column + 1])
	else:
		neighbour_indices.append(0)
	if column<299:
		neighbour_indices.append(input_img[row][column + 1])
	else:
		neighbour_indices.append(0)
	if row < 199 and column < 399:
		neighbour_indices.append(input_img[row + 1][column + 1])
	else:
		neighbour_indices.append(0)
	if row<199:
		neighbour_indices.append(input_img[row + 1][column])
	else:
		neighbour_indices.append(0)
	if row<199 and column>0:
		neighbour_indices.append(input_img[row + 1][column - 1])
	else:
		neighbour_indices.append(0)
	if column>0:
		neighbour_indices.append(input_img[row][column - 1])
	else:
		neighbour_indices.append(0)
	return neighbour_indices

def calculate_lbp_mask (input_img):
	lbp_mask=[[0 for x in range(input_img[0].size)]for y in range(len(input_img))]
	for row in range(200):
		for column in range(0, input_img[row].size):
			neighbour_indices=find_neighbours(input_img, row, column)
			result=0
			sum1= sum0 =0
			num1= num0=0
			multiplier=1
			for i in range(8):
				if input_img[row][column]<neighbour_indices[i]:
					result+=neighbour_indices[i]*multiplier
					sum1+=neighbour_indices[i]
					num1+=1
				else:
					sum0+=neighbour_indices[i]
					num0+=1
				multiplier*=2
			lbp_mask[row][column]=result
	#print (lbp_mask[0])
	return lbp_mask

def normalize (arr):
	arr = np.array(arr)
	arr = arr/arr.max()
	arr = np.around (arr, decimals=2)
	return arr.tolist()

def reduce_hues(hues, bins):
	rows, cols=hues.shape
	step=180/bins
	for i in range(rows):
		for j in range(cols):
			hues[i][j]=int(hues[i][j]/step)*step
	return hues

def find_connected_components(hues):
	rows, cols = hues.shape
	connected_components=[[0 for x in range(cols)] for y in range(rows)]
	connected_components=np.array(connected_components)
	l=1
	connected_components[0][0]=1
	EQ={}
	for c in range(1, cols):
		if hues[0][c]==hues[0][c-1]:
			connected_components[0][c]=connected_components[0][c-1]	
		else:
			l=l+1
			connected_components[0][c]=l
	for r in range(1,rows):
		if hues[r][0]==hues[r-1][0]:
			connected_components[r][0]=connected_components[r-1][0]
		else:
			l=l+1
			connected_components[0][c]=l
		for c in range(1,cols):
			if (hues[r][c]==hues[r][c-1] and hues[r][c]!=hues[r-1][c]):
				connected_components[r][c]=connected_components[r][c-1]
			if (hues[r][c]==hues[r-1][c] and hues[r][c]!=hues[r][c-1]):
				connected_components[r][c]=connected_components[r-1][c]
			if (hues[r][c]!=hues[r][c-1] and hues[r][c]!=hues[r-1][c]):
				l=l+1
				connected_components[r][c]=l
			if (hues[r][c]==hues[r][c-1] and hues[r][c]==hues[r-1][c] and connected_components[r][c-1]==connected_components[r-1][c]):
				connected_components[r][c]=connected_components[r][c-1]
			if (hues[r][c]==hues[r][c-1] and hues[r][c]==hues[r-1][c] and connected_components[r][c-1]!=connected_components[r-1][c]):
				connected_components[r][c]=min([connected_components[r-1][c], connected_components[r][c-1]])
				EQ[min([hues[r-1][c], hues[r][c-1]])]=min([hues[r-1][c], hues[r][c-1]])
	for key in reversed(sorted(EQ.keys())):
		print(key)
		np.place(connected_components, connected_components==key, EQ[key])
	return connected_components

def connected_components_table(hues, stats):
	component_table={}
	for label in stats:
		hue=hues[label[1]][label[0]]
		if hue not in component_table.keys():
			if label[4]>2400:
				component_table[hue]=[label[4], 0]
			else:
				component_table[hue]=[0, label[4]]
		else:
			if label[4]>2400:
				component_table[hue]=[component_table[hue][0]+label[4], 0]
			else:
				component_table[hue]=[0, component_table[hue][0]+label[4]]
	return component_table

def convert_edge_features (img, index):
	img = cv2.Canny (img, 100, 200)
	#cv2.imshow ('Edges', img)
	#cv2.waitKey (0)
	row_inter = []
	col_inter = []
	rows, cols = img.shape
	for curr_row in range (100, rows, 100):
		row_intersection_counter = 0
		for col_num in range (0, cols):
			if img.item (curr_row, col_num) != 0:
				row_intersection_counter += 1
		row_inter.append (row_intersection_counter)
	row_inter = normalize (row_inter)
	for item in row_inter:
		features[index].append(item)
	for curr_col in range (100, cols, 100):
		col_intersection_counter = 0
		for row_num in range (0, rows):
			if img.item (row_num, curr_col) != 0:
				col_intersection_counter += 1
		col_inter.append (col_intersection_counter)
	col_inter = normalize(col_inter)
	for item in col_inter:
		features[index].append(item)
	return;

def convert_colour_features (img, index):
	curr_img_hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
	colour_hist = cv2.calcHist (curr_img_hsv, [0], None, [15], [0,179]).flatten().tolist()
	#cv2.imshow ("for HSV", img)
	#cv2.waitKey(0)
	colour_hist = normalize (colour_hist)
	for item in colour_hist:
		features[index].append (item)
	return

def convert_colour_coherence_features(img, index):
	curr_img_hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
	curr_img_hsv = np.array(curr_img_hsv)
	hues = np.zeros((curr_img_hsv.shape[0],curr_img_hsv.shape[1]), dtype=curr_img_hsv.dtype)
	hues[:,:] = curr_img_hsv[:, :, 0]
	hues=reduce_hues(hues, 15)
	connectivity=8
	ret, connected_components, stats, c=cv2.connectedComponentsWithStats(hues)
	component_table=connected_components_table(hues, stats)
	features[index].append(component_table)

def convert_texture_features (img, index):
	scaled_img = cv2.resize(img, (200, 300), interpolation=cv2.INTER_AREA)
	lbp_mask = calculate_lbp_mask(scaled_img)
	lbp_mask = np.array(lbp_mask)
	hist_input = lbp_mask.flatten()
	lbp_histogram, bin_edges = np.histogram(hist_input, 8, density=True)
	for val in lbp_histogram:
		features[index].append(val)

def convert_colour_number_features(img, index):
	curr_img_hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
	hue_dict = {}
	rows=400
	cols=600
	for r in range(rows):
		for c in range(cols):
			hue = curr_img_hsv[r][c][0]
			sat = curr_img_hsv[r][c][1]/255
			val = curr_img_hsv[r][c][2]/255
			if sat>=0.2 and val>=0.15 and val<=0.95 and hue not in hue_dict:
				hue_dict[hue] = 1
	features[index].append(len(hue_dict))

img_bgr = []
img_gray = []
features = []
no_images = 1

folder_name = input ('Enter path to main folder: ')
for index in range (no_images):
	file_name = folder_name+'/real'+str(index)+'.jpg'
	img_bgr.append (cv2.imread (file_name, 1))
	img_gray.append (cv2.imread (file_name, 0))
	features.append([])
	print(index)
	#convert_edge_features (img_gray[index], index)    
	#convert_colour_features (img_bgr[index], index)
	#convert_texture_features(img_gray[index], index)
	#convert_colour_number_features(img_bgr[index], index)
	convert_colour_coherence_features(img_bgr[index], index)
#print (features)
#print (matplotlib.matplotlib_fname())
out_file_path = '/home/shriya/ML_project/real-abstract-classification/output_features_connected.csv'

with open(out_file_path, "a") as out_file:
    writer = csv.writer(out_file)
    writer.writerows(features)
#for index in range(no_images):
#img_hist = cv2.calcHist ([cv2.Canny(img_gray[0], 100, 200)], [0], None, [16], [0,256])
#print (img_hist)
#plt.plot (img_hist)
#plt.xlim ([0,16])
#plt.show()
