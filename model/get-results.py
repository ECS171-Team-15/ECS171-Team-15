from sklearn.model_selection import train_test_split
from keras.models import load_model
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow

import common

def eval_model(model_filename, test_x, test_y):
	model = load_model(model_filename)
	results = model.evaluate(test_x, test_y)
	print(f"{model_filename} results")
	print(results)

if __name__ == '__main__':
	_, test_x, _, test_y = common.load_and_split_data()

	# Scale test data to [0, 1] range
	scaler = MinMaxScaler()
	test_x = scaler.fit_transform(test_x)

	# Reshape test data to tensor of 2D images
	test_x = tensorflow.reshape(test_x, (-1, 302, 425, 1))

	files = os.listdir()
	for filename in files:
		if filename.endswith('.h5'):
			eval_model(filename, test_x, test_y)

