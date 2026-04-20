import numpy as np
import cv2
import argparse
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
from myutil import probas_to_classes

# Load model
model = load_model('drawing_classification.h5')

label = {0: "Circle", 1: "Square", 2: "Triangle"}


def predict_one(file_name):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.reshape(img, [1, 28, 28, 3])

    prediction = model.predict(img)
    classes = np.argmax(prediction, axis=1)[0]

    category = label[classes]
    print(f"\n{file_name} is predicted as: {category}")


def predict_dataset(input_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        input_dir,
        target_size=(28, 28),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=1
    )

    nb_samples = len(test_generator.filenames)

    predict = model.predict(test_generator, steps=nb_samples)

    return predict, test_generator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--testdata', action='store_true')
    parser.add_argument('--validationdata', action='store_true')
    parser.add_argument('--image')

    args = parser.parse_args()

    if args.testdata:
        print("Running on test dataset...")
        predict, test_generator = predict_dataset("shapes/test")

    elif args.validationdata:
        print("Running on validation dataset...")
        predict, test_generator = predict_dataset("shapes/validation")

    elif args.image:
        predict_one(args.image)

    else:
        print("Please provide --image or dataset flag")


if __name__ == '__main__':
    main()