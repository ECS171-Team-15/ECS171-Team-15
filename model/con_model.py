import tensorflow
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def build_conv_model(train_x, train_y, test_x, test_y, input_dim):
    print("Building model...", end=" ")
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l1_l2'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l1_l2'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=50, validation_data=(test_x, test_y), verbose=1)
    print("Done.")
    return model

def eval_model(test_x, test_y, model):
    test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
    print(f"Loss is: {test_loss}\tAccuracy is: {test_acc}")
    return

def parse_test_train():
    print("Loading data...", end=" ")
    df = pd.read_csv('../processed_data/original.csv')
    print("Done.")
    feature_data = df.drop(columns='class')
    label_data = df.iloc[:, df.shape[1]-1]
    print("Splitting data...", end=" ")
    train_x, test_x, train_y, test_y = train_test_split(feature_data, label_data, test_size=0.2, random_state=6)
    print("Done.")
    return (train_x, train_y, test_x, test_y)

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = parse_test_train()
    scaler = MinMaxScaler()
    print("Normalizing data...", end=" ")
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    print("Done.")
    train_x = tensorflow.reshape(train_x, (-1, 302, 425, 1))
    test_x = tensorflow.reshape(test_x, (-1, 302, 425, 1))
    model = build_conv_model(train_x, train_y, test_x, test_y, (302, 425, 1))
    print("Save model...", end=" ")
    model.save("./model.h5")
    print("Done.")
    eval_model(test_x, test_y, model)
