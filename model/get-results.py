from sklearn.model_selection import train_test_split
from keras.models import load_model
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from sklearn.metrics import classification_report
import numpy as np

import common

def eval_model(model_filename, test_x, test_y):
	model = load_model(model_filename)
	predicted_y = model.predict(test_x)
	print(predicted_y)
	print(test_y)
	# predicted_y = np.asarray(predicted_y)
	# print(type(predicted_y), type(test_y))
	report = classification_report(test_y, predicted_y)

	print(f"{model_filename} results:")
	print(report)

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

