import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras import layers, models, Input

from interfaces.digit_classification import DigitClassificationInterface

class CNNModel(DigitClassificationInterface):
    """A Convolutional Neural Network (CNN) model for MNIST digit classification."""
    
    def __init__(self):
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """
        Build and compile the CNN model.

        Returns:
            tf.keras.Model: A compiled CNN model.
        """
        model = models.Sequential()

        model.add(Input(shape=(28, 28, 1)))

        model.add(layers.Conv2D(32, (3, 3), activation='relu'))

        model.add(layers.Flatten())

        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        return model

    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict the digit from the input image.

        Args:
            input_data (np.ndarray): A 28x28x1 tensor representing the input image.

        Returns:
            int: The predicted digit (0-9).

        Raises:
            ValueError: If the input tensor does not have the shape (28, 28, 1).
        """
        if input_data.shape != (28, 28, 1):
            raise ValueError("Input tensor must have shape (28, 28, 1)")
        
        input_data = np.expand_dims(input_data, axis=0)

        predictions = self.model.predict(input_data, verbose=0)
        
        return int(np.argmax(predictions, axis=1)[0])
