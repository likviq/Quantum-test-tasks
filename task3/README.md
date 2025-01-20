# MNIST Digit Classifier (Task 3)

This project implements a `DigitClassifier` class that uses different models (CNN, Random Forest, Random) to classify images from the MNIST database. The project is organized using Object-Oriented Programming (OOP) principles and includes an interface to support the addition of new models.

## Project Description
The goal of this project is to create a `DigitClassifier` class that can use different models to classify MNIST images. The project supports three models:

- **CNN (Convolutional Neural Network)**: Works with a 28x28x1 tensor.
- **Random Forest**: Works with a 1D array of length 784 (28x28 pixels).
- **Random Model**: Returns a random value based on a 10x10 center crop of the image.

## File Structure

```bash
task3/
│
├── models/
│   ├── cnn_model.py
│   ├── random_forest_model.py
│   └── random_model.py
│
├── interfaces/
│   └── digit_classification.py
│
├── digit_classifier.py
├── test_digit_classifier.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/task3.git
cd task3
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Testing

The ``test_digit_classifier.py`` file contains examples of using the DigitClassifier class with different models.

```bash
python test_digit_classifier.py
```

## Adding New Models

To add a new model:

1. Create a new class in the models/ folder.

2. Implement the DigitClassificationInterface (the predict method).

3. Add a new value to ModelType in the digit_classifier.py file.
