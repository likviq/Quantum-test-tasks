import argparse

from src.inference import H2OModelInference
from config import DEFAULTS

def inference(args):
    h2o_inference = H2OModelInference(args.model_path)

    predictions_df = h2o_inference.predict_dataframe(
        args.input_csv,
        args.output_csv,
    )
    print(f"Predictions completed. Number of records: {len(predictions_df)}")

    h2o_inference.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H2O Model Inference Script")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="Path to the saved H2O model.",
        default=DEFAULTS["model_path"],
    )
    parser.add_argument(
        "--input_csv", 
        type=str, 
        help="Path to the input CSV file with data.",
        default=DEFAULTS["input_csv"],
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        help="Path to save the prediction results as CSV.",
        default=DEFAULTS["output_csv"],
    )

    args = parser.parse_args()

    inference(args)
