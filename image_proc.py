
def get_rowsize(image):
	return image.shape[0]

def get_columnsize(image):
	return image.shape[1]

def dist_from_left(image, boundary, fuzz_range):
	row_size = get_rowsize(image)
	column_size = get_columnsize(image)
	flag = False
	closest_column_index = -1

	for column_index in range (0,column_size):
		for row_index in range (0,row_size):
			pixel = image[row_index][column_index]
			if (pixel < boundary):
				closest_column_index = column_index
				flag = True
				break

		if (flag):
			break

	return closest_column_index + fuzz_range

def dist_from_bottom(image, boundary, fuzz_range):
	row_size = get_rowsize(image)
	column_size = get_columnsize(image)
	flag = False
	closest_row_index = -1

	for row_index in range (0,row_size):
		for column_index in range (0,column_size):
			pixel = image[row_size-row_index-1][column_index]
			if (pixel < boundary):
				closest_row_index = row_size - row_index - 1
				flag = True
				break

		if (flag):
			break

	return closest_row_index - fuzz_range

def feature1(image, boundary, fuzz_range):
	row_size = get_rowsize(image)
	column_size = get_columnsize(image)
	row_lower_limit = dist_from_bottom(image, boundary, fuzz_range)

	ctr=0

	for row_index in range (row_lower_limit,row_size):
		for column_index in range (0,column_size):
			pixel = image[row_index][column_index]
			if (pixel < boundary):
				ctr+=1

	#print(ctr)
	#print(row_size)
	#print(column_size)
	val = ctr*100/(row_size*column_size)
	return val


def feature2(image, boundary, fuzz_range):
	row_size = get_rowsize(image)
	column_size = get_columnsize(image)
	column_upper_limit = dist_from_left(image, boundary, fuzz_range)

	ctr=0

	for row_index in range (0,row_size):
		for column_index in range (0,column_upper_limit):
			pixel = image[row_index][column_index]
			if (pixel < boundary):
				ctr+=1

	#print(ctr)
	#print()
	val = ctr*100/(row_size*column_size)
	return val