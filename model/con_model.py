import tensorflow
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def build_conv_model(train_x, train_y, test_x, test_y, input_dim):
    # Image augmentation
    # Only applied to our training data
    model = models.Sequential([
        layers.InputLayer(input_shape=input_dim),
        layers.experimental.preprocessing.RandomFlip(),
        # 36 degree rotation range in both directions
        layers.experimental.preprocessing.RandomRotation(0.1),
        # Zoom range for both width & height: -10% to 10%
        # We want to fill the edges with black if we zoom out
        layers.experimental.preprocessing.RandomZoom(0.1, fill_mode="constant"),
        # Stretch image vertically by [-10%, 10%]
        layers.experimental.preprocessing.RandomHeight(0.1),
        # Do the same horizontally
        layers.experimental.preprocessing.RandomWidth(0.1)
    ])

    # Convolution layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l1_l2'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l1_l2'))
    model.add(layers.Flatten())
    
    # Scramble some of the results to avoid overfitting
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    model.summary()

    model.fit(train_x, train_y, epochs=50, validation_data=(test_x, test_y), verbose=1)
    return model

def eval_model(test_x, test_y, model):
    test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
    print(f"Loss is: {test_loss}\tAccuracy is: {test_acc}")
    return

def parse_test_train():
    # Load data
    df = pd.read_csv('../processed_data/original.csv')
    feature_data = df.drop(columns='class')
    label_data = df.iloc[:, df.shape[1]-1]

    # Split data
    train_x, test_x, train_y, test_y = train_test_split(feature_data, label_data, test_size=0.2, random_state=6)
    return (train_x, train_y, test_x, test_y)

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = parse_test_train()
    
    # Scale data
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)

    # Train model
    train_x = tensorflow.reshape(train_x, (-1, 302, 425, 1))
    test_x = tensorflow.reshape(test_x, (-1, 302, 425, 1))
    model = build_conv_model(train_x, train_y, test_x, test_y, (302, 425, 1))
    model.save("./model.h5")

    # Report results
    eval_model(test_x, test_y, model)
