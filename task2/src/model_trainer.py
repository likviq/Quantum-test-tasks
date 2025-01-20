import h2o
from h2o.automl import H2OAutoML

class H2OModelTrainer:
    """
    A class to train and evaluate a model using H2O AutoML.
    """

    def __init__(self, data, target, features, cv_column):
        """
        Initializes the H2OModelTrainer with the given data, target, features, and cross-validation column.

        Args:
            data (pandas.DataFrame): The dataset to train the model on.
            target (str): The name of the target column.
            features (list): A list of feature column names.
            cv_column (str): The name of the column for cross-validation folds.
        """
        self.data = data
        self.target = target
        self.features = features
        self.cv_column = cv_column
        self.seed = 3407
        
        h2o.init()
        
        self.h2o_data = h2o.H2OFrame(self.data)
        
        self.best_params = None
        self.final_model = None

    def get_fold(self, fold_idx):
        """
        Splits the data into training and validation sets based on the given fold index.

        Args:
            fold_idx (int): The index of the fold to use for validation.

        Returns:
            tuple: A tuple of H2OFrame objects (training data, validation data).
        """
        train_data = self.data[self.data[self.cv_column] != fold_idx]
        valid_data = self.data[self.data[self.cv_column] == fold_idx]
        return h2o.H2OFrame(train_data), h2o.H2OFrame(valid_data)

    def train_model(self, save_path=None):
        """
        Trains the final model using H2O AutoML on the entire dataset.

        Args:
            save_path (str, optional): The path to save the trained model. If None, the model will not be saved.
        """
        aml = H2OAutoML(
            seed=self.seed,
            max_runtime_secs_per_model=300,
        )

        aml.train(
            x=self.features,
            y=self.target,
            training_frame=self.h2o_data,
            fold_column=self.cv_column
        )
        
        self.final_model = aml.leader

        if save_path is None:
            return

        self.save_model()
        
    def evaluate_model(self):
        """
        Evaluates the performance of the final trained model on the entire dataset.

        Returns:
            H2OAutoML: The performance metrics of the model.

        Raises:
            ValueError: If the final model has not been trained.
        """
        if not self.final_model:
            raise ValueError("The final model has not been trained yet!")
        
        performance = self.final_model.model_performance()
        return performance
    
    def save_model(self):
        """
        Saves the trained model to the specified path.

        Returns:
            str: The path where the model was saved.
        """
        model_path = h2o.save_model(
            model=self.final_model, 
            path="task2/data/models", 
            force=True
        )
        return model_path

    def shutdown(self):
        """
        Shuts down the H2O instance.
        """
        h2o.cluster().shutdown()
