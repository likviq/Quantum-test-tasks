import numpy as np
from sklearn.ensemble import RandomForestClassifier

from interfaces.digit_classification import DigitClassificationInterface

class RandomForestModel(DigitClassificationInterface):
    """A Random Forest model for MNIST digit classification."""

    def __init__(self):
        self.model = self._initialize_model()

    def _initialize_model(self):
        """
        Build and train the Random Forest model with fake data to init weights.

        Returns:
            sklearn.ensemble.RandomForestClassifier: A trained Random Forest model.
        """
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=3407,
        )

        fake_data = np.random.rand(10, 784)
        fake_labels = np.random.randint(0, 10, size=10)
        model.fit(fake_data, fake_labels)

        return model

    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict the digit from the input array.

        Args:
            input_data (np.ndarray): A 1D array of length 784 representing the input image.

        Returns:
            int: The predicted digit (0-9).

        Raises:
            ValueError: If the input array does not have the shape (784,).
        """
        if input_data.shape != (784,):
            raise ValueError("Input array must have shape (784,)")
        
        input_data_2d = input_data.reshape(1, -1)

        prediction = self.model.predict(input_data_2d)[0]
        
        return prediction
