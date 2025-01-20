from enum import Enum

class ModelType(Enum):
    """
    Enum for selecting model type.
    """
    CNN = "cnn"
    RANDOM_FOREST = "rf"
    RANDOM = "rand"
