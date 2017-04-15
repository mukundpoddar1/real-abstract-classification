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

def reduce_hues(img, bins):
	curr_img_hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
	curr_img_hsv = np.array(curr_img_hsv)
	hues = np.zeros((curr_img_hsv.shape[0],curr_img_hsv.shape[1]), dtype=curr_img_hsv.dtype)
	hues[:,:] = curr_img_hsv[:, :, 0]
	rows, cols=hues.shape
	step=180/(bins*2)
	for i in range(rows):
		for j in range(cols):
			div=int(hues[i][j]/step)
			if div%2!=0:
				div=(div+1)%30
			hues[i][j]=div*step*2
	return hues

def connected_components_table(hues, stats):
	component_table={0:[0,0], 12:[0,0], 24:[0,0], 36:[0,0], 48:[0,0], 60:[0,0], 72:[0,0], 84:[0,0], 96:[0,0], 108:[0,0], 120:[0,0], 132:[0,0], 144:[0,0], 156:[0,0], 168:[0,0]}
	for label in stats:
		if label[0]>=600 or label[1]>=400:
			continue
		hue=hues[label[cv2.CC_STAT_TOP]][label[cv2.CC_STAT_LEFT]]
		if label[4]>2400:
			component_table[hue]=[component_table[hue][0]+label[4], 0]
		else:
			component_table[hue]=[0, component_table[hue][0]+label[4]]
	return component_table


def normalize (arr):
	arr = np.array(arr)
	if arr.max() != 0:
		arr = arr/arr.max()
	arr = np.around (arr, decimals=2)
	return arr.tolist()

def L1(v1,v2):
      if len(v1)!=len(v2):
        print ('error')
      return sum([abs(v1[i]-v2[i]) for i in range(len(v1))])

def hog (img, index):
	winSize = (512, 384)
	blockSize = (128, 128)
	cellSize = (64, 64)
	blockStride = (64, 64)
	nbins = 9
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	#compute(img[, winStride[, padding[, locations]]]) -> descriptors
	winStride = (8,8)
	padding = (8,8)
	locations = ((10,20),)
	hist = hog.compute(img)
	hist = np.around (hist, decimals=4)
	for item in hist:
		features[index].append (item[0])

def sobel_otsu (img, index):
	sobelx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=1)
	sobelx = np.absolute(sobelx)
	sobelx = np.uint8(sobelx)
	sobelx = cv2.GaussianBlur(sobelx,(5,5),0)
	ret3,sobelx = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	sobely = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=1)
	sobely = np.absolute(sobely)
	sobely = np.uint8(sobely)
	sobely = cv2.GaussianBlur(sobely, (5,5),0)
	ret3,sobely = cv2.threshold(sobely,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	edge_density = []
	for tilex in range (0, 400, 100):
		for tiley in range (0, 600, 100):
			edge_density.append(0)
			for x in range(100):
				for y in range (100):
					edge_density[int(tilex/25) + int(tiley/100)] += (sobelx.item(tilex+x, tiley+y) or sobely.item(tilex+x, tiley+y))/255
	edge_density = normalize(edge_density)
	std_dev = np.std (edge_density)
	for item in edge_density:
		features[index].append(item)
	features[index].append(std_dev)


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
	'''cv2.imshow ("Sobel", sobel)
	if cv2.waitKey(0) == 27:
		quit()'''

def colour_coherence(img, index):
	connectivity=8
	ret, connected_components, stats, c=cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
	component_count=0
	for stat in stats:
		component_count=component_count+1
	features[index].append(component_count)


def hue_count(img, index):
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


def hue_histogram (img, index):
	img=np.array(img)
	colour_tile_hists=[[[] for x in range(6)] for y in range(4)]
	for tilex in range (0, 400, 100):
		for tiley in range (0, 600, 100):
			tile = img[tilex: tilex+100, tiley: tiley+100]
			'''cv2.imshow ("Sobel", curr_tile_hsv)
			if cv2.waitKey(0) == 27:
				quit()'''
			colour_hist, _ = np.histogram(tile.flatten(), bins = 15)
			colour_tile_hists[int(tilex/100)][int(tiley/100)]=np.array(colour_hist, np.float32)
			#colour_hist = cv2.calcHist (curr_tile_hsv, [0], None, [18], [0,180]).flatten().tolist()
	for row in range(4):
		for col in range(6):
			neighbours=[]
			dist=0
			if (row>0):
				neighbours.append(colour_tile_hists[row-1][col])
			if (col>0):
				neighbours.append(colour_tile_hists[row][col-1])
			if (row<3):
				neighbours.append(colour_tile_hists[row+1][col])
			if (col<5):
				neighbours.append(colour_tile_hists[row][col+1])
			for neighbour in neighbours:
				dist=dist + cv2.compareHist(colour_tile_hists[row][col], neighbour, method=cv2.HISTCMP_CORREL)
			features[index].append(dist)
	
	#cv2.imshow ("for HSV", img)
	#cv2.waitKey(0)

def lbp_texture (img, index):
	scaled_img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
	lbp_mask = calculate_lbp_mask(scaled_img)
	lbp_mask = np.array(lbp_mask)
	hist_input = lbp_mask.flatten()
	lbp_histogram, bin_edges = np.histogram(hist_input, 8)
	lbp_histogram_norm = normalize(lbp_histogram)
	lbp_histogram_norm=np.array(lbp_histogram_norm, np.float32)
	lbp_histogram_dists=[]
	for row in range(0, 200, 100):
		for col in range(0, 300, 100):
			lbp_histogram_tile, bin_edges=np.histogram(lbp_mask[row:row+100, col:col+100].flatten(), 8)
			lbp_histogram_tile = np.array(normalize(lbp_histogram_tile), np.float32)
			lbp_histogram_dists.append(cv2.compareHist(lbp_histogram_tile, lbp_histogram_norm, method=cv2.HISTCMP_CHISQR_ALT))
	for val in lbp_histogram_dists:
		features[index].append(val)

def contrast (img, index):
	img_stddev = []
	for tilex in range (0, 400, 100):
		for tiley in range (0, 600, 100):
			_, tile_stddev = cv2.meanStdDev(img[tilex:tilex+100, tiley:tiley+100])
			img_stddev.append(tile_stddev[0][0])
	for item in img_stddev:
		features[index].append (item)


img_bgr = []
img_gray = []
img_hues = []
features = []
num_images_abs = 164
num_images_real = 194
abs_folder_name = input ('Enter path to abstract folder: ')
real_folder_name = input ('Enter path to real folder: ')
for index in range (num_images_abs):
	file_name = abs_folder_name+'/abstract'+str(index)+'.jpg'
	img_bgr.append (cv2.imread (file_name, 1))
	img_gray.append (cv2.imread (file_name, 0))
	img_hues.append(reduce_hues(img_bgr[index], 15))
	features.append([])
	print(index)
	lbp_texture(img_gray[index], index) #6
	#hue_histogram (img_hues[index], index) #24
	contrast (img_gray[index], index) #24
	colour_coherence (img_hues[index], index) #1
	hue_count (img_bgr[index], index) #1
	#convert_edge_features (img_gray[index], index)
	#hog(img_gray[index], index)
	sobel_otsu(img_gray[index], index) #25
	features[index].append(1)
for index in range (num_images_real):
	file_name = real_folder_name+'/real'+str(index)+'.jpg'
	img_bgr.append (cv2.imread (file_name, 1))
	img_gray.append (cv2.imread (file_name, 0))
	img_hues.append(reduce_hues(img_bgr[index+num_images_abs], 15))
	features.append([])
	print(index)
	lbp_texture(img_gray[index+num_images_abs], index+num_images_abs)
	#hue_histogram (img_hues[index+num_images_abs], index+num_images_abs)
	contrast (img_gray[index+num_images_abs], index+num_images_abs)
	colour_coherence (img_hues[index+num_images_abs], index+num_images_abs)
	hue_count (img_bgr[index+num_images_abs], index+num_images_abs)
	#convert_edge_features (img_gray[index+num_images_abs], index+num_images_abs)
	#hog (img_gray[index+num_images_abs], index+num_images_abs)
	sobel_otsu(img_gray[index+num_images_abs], index+num_images_abs)
	features[index+num_images_abs].append(0)
#print (features)
#print (matplotlib.matplotlib_fname())
out_file_path = './new_output_featuresnohh.csv'

with open(out_file_path, "a") as out_file:
    writer = csv.writer(out_file)
    writer.writerows(features)
#for index in range(no_images):
#img_hist = cv2.calcHist ([cv2.Canny(img_gray[0], 100, 200)], [0], None, [16], [0,256])
#print (img_hist)
#plt.plot (img_hist)
#plt.xlim ([0,16])
#plt.show()
