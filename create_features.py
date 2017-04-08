import cv2
import numpy as np
import csv
from  matplotlib import pyplot as plt

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
	if row < 199 and column < 299:
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
	lbp_mask=[[0 for x in range(input_img[0].size)] for y in range(len(input_img))]
	for row in range(input_img.shape[0]):
		for column in range(0, input_img.shape[1]):
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
	if arr.max() != 0:
		arr = arr/arr.max()
	arr = np.around (arr, decimals=2)
	return arr.tolist()

def convert_edge_features (img, index):
	img = cv2.Canny (img, 100, 200)
	#cv2.imshow ('Edges', img)
	#cv2.waitKey (0)
	row_inter = []
	col_inter = []
	rows, cols = img.shape
	for curr_row in range (50, rows, 50):
		row_intersection_counter = 0
		for col_num in range (0, cols):
			if img.item (curr_row, col_num) != 0:
				row_intersection_counter += 1
		row_inter.append (row_intersection_counter)
	#row_inter = normalize (row_inter)
	for item in row_inter:
		features[index].append(item)
	for curr_col in range (50, cols, 50):
		col_intersection_counter = 0
		for row_num in range (0, rows):
			if img.item (row_num, curr_col) != 0:
				col_intersection_counter += 1
		col_inter.append (col_intersection_counter)
	#col_inter = normalize(col_inter)
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
	return;

def convert_texture_features (img, index):
	scaled_img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
	lbp_mask = calculate_lbp_mask(scaled_img)
	lbp_mask = np.array(lbp_mask)
	hist_input = lbp_mask.flatten()
	lbp_histogram, bin_edges = np.histogram(hist_input, 8)
	lbp_histogram = normalize(lbp_histogram)
	for val in lbp_histogram:
		features[index].append(val)

img_bgr = []
img_gray = []
features = []
num_images_abs = 164
num_images_real = 194
abs_folder_name = input ('Enter path to abstract folder: ')
real_folder_name = input ('Enter path to real folder: ')
for index in range (num_images_abs):
	file_name = abs_folder_name+'/abstract'+str(index)+'.jpg'
	img_bgr.append (cv2.imread (file_name, 1))
	img_gray.append (cv2.imread (file_name, 0))
	features.append([])
	print(index)
	convert_edge_features (img_gray[index], index)    
	convert_colour_features (img_bgr[index], index)
	convert_texture_features(img_gray[index], index)
	features[index].append(1)
for index in range (num_images_real):
	file_name = real_folder_name+'/real'+str(index)+'.jpg'
	img_bgr.append (cv2.imread (file_name, 1))
	img_gray.append (cv2.imread (file_name, 0))
	features.append([])
	print(index)
	convert_edge_features (img_gray[index+num_images_abs], index+num_images_abs)
	convert_colour_features (img_bgr[index+num_images_abs], index+num_images_abs)
	convert_texture_features(img_gray[index+num_images_abs], index+num_images_abs)
	features[index+num_images_abs].append(0)
#print (features)
#print (matplotlib.matplotlib_fname())
out_file_path = './output_features_less_edge.csv'

with open(out_file_path, "a") as out_file:
    writer = csv.writer(out_file)
    writer.writerows(features)
#for index in range(no_images):
#img_hist = cv2.calcHist ([cv2.Canny(img_gray[0], 100, 200)], [0], None, [16], [0,256])
#print (img_hist)
#plt.plot (img_hist)
#plt.xlim ([0,16])
#plt.show()