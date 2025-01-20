from abc import ABC, abstractmethod
import numpy as np

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict the digit from the input data.
        :param input_data: Input data for the model.
        :return: Predicted digit (int).
        """
        pass
