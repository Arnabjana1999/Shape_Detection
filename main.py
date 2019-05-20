import numpy as np
import sys 
import cv2
import os

from utils import *
from Neural_Network import *
from image_proc import *

if __name__ == "__main__":

	boundary = 50                     # dark pixel, indication shape. ideal = 0                 
	fuzz_range = 15                   # ideal is 0, covers a little tilt of a line
	iter_lower_limit = 90
	iter_upper_limit = 100
	
	train_image_dir = sys.argv[1]     #training set directory name
	test_image_dir = sys.argv[2]      #testing set directory name

	print(train_image_dir)
	print(test_image_dir)

	train_image_names = os.listdir(train_image_dir)      #list of training image_names 
	test_image_names = os.listdir(test_image_dir)        #list of testing image_names

	print(train_image_names)
	print(test_image_names)

	train_images = process_namelist(train_image_dir, train_image_names)     #list of training images (2D arrays)
	train_output = train_output(train_image_names)                          #list of shapes corresponding to training images (2D array)
	test_images = process_namelist(test_image_dir, test_image_names)        #list of testing images (2D arrays) 

	#print(train_images[0])
	#r=get_rowsize(train_images[0])
	#c=get_columnsize(train_images[0])
	#ctr=0
	#for i in range (0,r):
	#	for j in range (0, c):
	#		if(train_images[0][i][j]<30):
	#			ctr+=1

	#print(ctr)
	#print(r*c)


	#print(train_images)
	#print(train_images[0].shape[0])
	#print(train_images[0].shape[1])
	train_input = get_feature_list(train_images, boundary, fuzz_range)      #list of feature-set corresponding to train images
	#print(train_input)
	test_input = get_feature_list(test_images, boundary, fuzz_range)        #list of feature-set corresponding to test images
	print("train features")
	print(train_input)
	print()

	neural_network = NeuralNetwork(train_input, train_output, 3)            #initializing a Neural-Network

	for iter in range (iter_lower_limit,iter_upper_limit+1):
		neural_network.train(iter)
		new_w1 = neural_network.weights1
		new_w2 = neural_network.weights2

		#print("weight1")
		#print(new_w1)
		#print()
		#print("weight2")
		#print(new_w2)
		#print()
		#print("test_features")
		#print(test_input)

		predicted_factors = predict_output(test_input, new_w1, new_w2)
		prediction = predict_geometry(predicted_factors)
		#print("prediction")
		#for debug in predicted_factors:
		#	print(debug)
		#print(iter)
		print(prediction)                                                 #list of predicted shapes corresponding to each test-image
		#print()


