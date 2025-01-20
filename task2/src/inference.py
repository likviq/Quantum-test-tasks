import pandas as pd

import h2o
from h2o.frame import H2OFrame

class H2OModelInference:
    """
    A class for loading a trained H2O model and performing inference on data.
    """

    def __init__(self, model_path):
        """
        Initializes the inference class and loads the model.

        Args:
            model_path (str): Path to the saved H2O model.
        """
        h2o.init(verbose=False)
        h2o.no_progress()
        self.model = h2o.load_model(model_path)

    def predict_dataframe(self, input_df_path, output_df_path):
        """
        Makes predictions on a dataset loaded from a CSV file and saves the results to another CSV file.

        Args:
            input_df_path (str): Path to the input CSV file.
            output_df_path (str): Path to save the prediction results as a CSV file.

        Returns:
            pandas.DataFrame: A DataFrame containing the predictions.
        """
        df = pd.read_csv(input_df_path)
        h2o_frame = H2OFrame(df)

        df_predictions = self.model.predict(h2o_frame).as_data_frame()

        df['y_pred'] = df_predictions['predict']
        df.to_csv(output_df_path)

        return df_predictions

    def predict_single(self, input_data):
        """
        Makes a prediction for a single data record.

        Args:
            input_data (dict): A dictionary of feature values for prediction.

        Returns:
            float: The predicted value for the single record.
        """
        df = pd.DataFrame([input_data])
        predictions = self.predict_dataframe(df)
        return predictions.iloc[0, 0]

    def shutdown(self):
        """
        Shuts down the H2O cluster session.
        """
        h2o.cluster().shutdown()
