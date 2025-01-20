import numpy as np

from digit_classifier import DigitClassifier
from enums.model_type import ModelType

if __name__ == "__main__":
    image = np.random.rand(28, 28, 1)

    cnn_classifier = DigitClassifier(ModelType.CNN)
    print(f"CNN Prediction: {cnn_classifier.predict(image)}")

    rf_classifier = DigitClassifier(ModelType.RANDOM_FOREST)
    print(f"Random Forest Prediction: {rf_classifier.predict(image.flatten())}")

    rand_classifier = DigitClassifier(ModelType.RANDOM)
    print(f"Random Prediction: {rand_classifier.predict(image[9:19, 9:19, 0])}")
