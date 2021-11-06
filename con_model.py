import tensorflow
import os

# import pandas as pd
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def build_conv_model(input_dim):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # from_logits = False because our output is from 0 to 1
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

    return model

# def parse_test_train():
#     print("Loading data...", end=" ")
#     df = pd.read_hdf('../data.h5', 'df')
#     print("Done.")
#     feature_data = df.drop(df.shape[1]-1, axis=1)
#     label_data = df.iloc[:, df.shape[1]-1]
#     print("Splitting data...", end=" ")
#     train_x, test_x, train_y, test_y = train_test_split(feature_data, label_data, test_size=0.2, random_state=6)
#     print("Done.")
#     return (train_x, train_y, test_x, test_y)

# Get the average width and height of all images in the dataset
def get_mean_dimensions(dirs) -> list:
    height = []
    width = []
    for dir in dirs:
        files = os.listdir(dir)
        for file in files:
            im = Image.open(f"{dir}/{file}")
            w, h = im.size
            width.append(w)
            height.append(h)

	# Return averages
    return (sum(width)//len(width), sum(height)//len(height))

if __name__ == "__main__":
    #gpus = tensorflow.config.list_physical_devices('GPU')
    #if gpus:
    #    try:
    #        tensorflow.config.experimental.set_memory_growth(gpus[0], True)
    #    except RuntimeError as e:
    #        print(e)
    # train_x, train_y, test_x, test_y = parse_test_train()
    # train_x = tensorflow.reshape(train_x, (-1, train_x.shape[1], 1, 1))
    # test_x = tensorflow.reshape(test_x, (-1, test_x.shape[1], 1, 1))
    dataset = 'original'

    # Read images from directories as a stream
    # Convert them to grayscale
    training_gen = ImageDataGenerator(rescale=1./255)
    validation_gen = ImageDataGenerator(rescale=1./255)
    testing_gen = ImageDataGenerator(rescale=1./255)

    # Rescale images
    dirs = [
        "data/{dataset}/training/CT_COVID",
        "data/{dataset}/training/CT_NonCOVID",
        "data/{dataset}/testing/CT_COVID",
        "data/{dataset}/testing/CT_NonCOVID",
        "data/{dataset}/validation/CT_COVID",
        "data/{dataset}/validation/CT_NonCOVID",
            ]
    mean_dims = get_mean_dimensions(dirs)

    training_gen = training_gen.flow_from_directory(f"data/{dataset}/training",
                                                    target_size=mean_dims,
                                                    class_mode='binary',
                                                    batch_size=1)
    validation_gen = validation_gen.flow_from_directory(f"data/{dataset}/validation",
                                                    target_size=mean_dims,
                                                    class_mode='binary',
                                                    batch_size=1)
    testing_gen = testing_gen.flow_from_directory(f"data/{dataset}/testing",
                                                    target_size=mean_dims,
                                                    class_mode='binary',
                                                    batch_size=1)

    model = build_conv_model()
    model.summary()

    model.fit_generator(training_gen, epochs=10, validation_data=validation_gen, verbose=1)
    model.save("./model.h5")

    test_loss, test_acc = model.evaluate_generator(testing_gen)
    print(f"Loss is: {test_loss}\tAccuracy is: {test_acc}")
