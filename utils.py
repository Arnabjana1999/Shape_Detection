import cv2
import numpy as np
from image_proc import *

def sigmoid(x):
	#print("x= ")
	#print(x)
	return (1 / (1 + np.exp(-x)))

def sigmoid_derivative(x):
	return sigmoid(x)*(1-sigmoid(x))

def train_output(name_list):
	outlist=np.array([])
	#print(name_list)
	for element in name_list:
		if (element[0]=='s'):
			outlist = np.append(outlist,[1,0,0])
		elif (element[0]=='c'):
			outlist = np.append(outlist,[0,1,0])
		elif (element[0]=='t'):
			outlist = np.append(outlist,[0,0,1])
	#print(outlist)
	outputlist = outlist.reshape(len(name_list),3)
	#print(outputlist)
	return outputlist

def process_namelist(dir,list):
	outlist=[]
	#print("image_list")
	#print(list)
	for element in list :
		outlist.append(read_image(dir+"/"+element))
	#print(outlist)
	return outlist


def read_image(filename):
	#print("filename")
	#print(filename)
	#print(type(filename))
	image = cv2.imread(filename,0)
	#print(image)
	return image


def get_feature_list(image_list, boundary, fuzz_range):
	feature_list=np.array([])
	for element in image_list:
		feature_list = np.append(feature_list, np.array([feature1(element, boundary, fuzz_range), feature2(element, boundary, fuzz_range)]))

	return feature_list.reshape(len(image_list),2) 




def predict_output(image_list, weights1, weights2):
	output=[]
	for image in image_list:
		layer1 = sigmoid(np.dot(image, weights1))
		output.append(sigmoid(np.dot(layer1, weights2)))

	return output 

def predict_geometry(output_list):
	output=[]
	for element in output_list:
		#print(element)
		#print(element[1])
		#print(element[2])
		#print(element[0]-element[2])
		if (element[0]-element[1]>=0 and element[0]-element[2]>=0):
			output.append("Square")
		elif (element[1]-element[0]>=0 and element[1]-element[2]>=0):
			output.append("Circle")
		elif (element[2]-element[0]>=0 and element[2]-element[1]>=0):
			output.append("Triangle")

	return output