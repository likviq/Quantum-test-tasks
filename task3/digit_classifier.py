import numpy as np

from models.cnn_model import CNNModel
from models.random_forest_model import RandomForestModel
from models.random_model import RandomModel
from enums.model_type import ModelType

class DigitClassifier:
    """A classifier that uses a specified model for MNIST digit prediction."""

    def __init__(self, algorithm: ModelType):
        """
        Initialize the DigitClassifier with the specified algorithm.

        Args:
            algorithm (ModelType): The type of model to use. Must be one of the values from `ModelType`.
        """
        self.algorithm = algorithm
        self.model = self._get_model()

    def _get_model(self):
        """
        Retrieve the model based on the specified algorithm.

        Returns:
            DigitClassificationInterface: An instance of the selected model.

        Raises:
            ValueError: If the algorithm is not recognized.
        """
        if self.algorithm == ModelType.CNN:
            return CNNModel()
        elif self.algorithm == ModelType.RANDOM_FOREST:
            return RandomForestModel()
        elif self.algorithm == ModelType.RANDOM:
            return RandomModel()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
    def train(self):
        """Raise an error if training is attempted."""
        raise NotImplementedError("Training is not implemented.")

    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict the digit from the input data using the selected model.

        Args:
            input_data (np.ndarray): The input data for prediction.

        Returns:
            int: The predicted digit (0-9).
        """
        return self.model.predict(input_data)
