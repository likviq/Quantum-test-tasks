import numpy as np

from interfaces.digit_classification import DigitClassificationInterface

class RandomModel(DigitClassificationInterface):
    """A random model for MNIST digit classification."""

    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict a random digit from the input array.

        Args:
            input_data (np.ndarray): A 10x10 array representing the center crop of the input image.

        Returns:
            int: A random digit between 0 and 9.

        Raises:
            ValueError: If the input array does not have the shape (10, 10).
        """
        if input_data.shape != (10, 10):
            raise ValueError("Input array must have shape (10, 10)")
        
        return np.random.randint(0, 10)
