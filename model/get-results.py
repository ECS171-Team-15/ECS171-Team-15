import os
import tensorflow
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

import common


def eval_model(model_filename, test_x, test_y):
    model = load_model(model_filename)
    predicted_y = model.predict(test_x)
    predicted_y = [round(y[0], 0) for y in predicted_y]
    # print(predicted_y)
    # print(test_y)
    report = classification_report(test_y, predicted_y)

    print(f"CLASSIFICATION REPORT: {model_filename} results:\n")
    print(report)
    cm = confusion_matrix(test_y, predicted_y)
    print(f"\nCONFUSION MATRIX: {model_filename} results:\n")

    # Display confusion matrix
    print("\t\tNON-COVID\tCOVID")
    for row in range(len(cm)):
        if row == 0:
            print("NON-COVID", end="\t")
        else:
            print("COVID", end="\t\t")
        for col in cm[row]:
            print(col, end="\t\t")
        print("\n")


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
