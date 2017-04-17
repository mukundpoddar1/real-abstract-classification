import cv2
import numpy as np

def display (img):
	cv2.imshow("Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result

abs_folder_name = '/home/shriya/ML_project/dataset_new/Abstract_250'
file_name = abs_folder_name+'/abstract0.jpg'
img = cv2.imread (file_name, 1)
rows, cols, _ = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine (img, M, (rows, cols))
#M = np.float32([[1,0,100],[0,1,50]])
dst2 = rotateImage (img, 90) #cv2.warpAffine(dst,M, dst.shape[0:2])
resize_img = cv2.resize(img, (600,400), interpolation=cv2.INTER_AREA)