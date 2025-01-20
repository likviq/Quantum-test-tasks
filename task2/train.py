import argparse

from src.data_preprocessor import DataPreprocessor
from src.model_trainer import H2OModelTrainer
from config import DEFAULTS

def train_model(args):
    preprocessor = DataPreprocessor(
        data_path=args.data_path,
        target=args.target,
        features=args.features
    )
    processed_data = preprocessor.preprocess()

    model_trainer = H2OModelTrainer(
        data=processed_data,
        target=args.target,
        features=args.features,
        cv_column=args.cv_column
    )
    model_trainer.train_model(args.model_output_path)

    performance = model_trainer.evaluate_model()
    print(f"Model performance: {performance}")

    model_trainer.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H2O Model Training Script")

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the training data CSV file.",
        default=DEFAULTS["data_path"],
    )
    parser.add_argument(
        "--target",
        type=str,
        help="The target column in the dataset.",
        default=DEFAULTS["target"],
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs='+',
        help="List of feature columns.",
        default=DEFAULTS["features"],
    )
    parser.add_argument(
        "--cv_column",
        type=str,
        help="The cross-validation column.",
        default=DEFAULTS["cv_column"],
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        help="Path to save the trained model.",
        default=DEFAULTS["model_output_path"],
    )

    args = parser.parse_args()

    train_model(args)
