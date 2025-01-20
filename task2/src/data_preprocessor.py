import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer

class DataPreprocessor:
    """
    A class for preprocessing data by handling missing values, outliers, feature normalization, and cross-validation indices.
    """

    def __init__(self, data_path, target, features, n_splits=5):
        """
        Initializes the DataPreprocessor with data, target variable, features, and number of cross-validation splits.

        Args:
            data_path (str): Path to the input data CSV file.
            target (str): Name of the target column.
            features (list): List of feature column names.
            n_splits (int): Number of splits for cross-validation (default is 5).
        """
        self.data_path = data_path
        self.target = target
        self.features = features
        self.n_splits = n_splits
        self.data = None

    def load_data(self):
        """
        Loads the data from the CSV file into a Pandas DataFrame.
        """
        self.data = pd.read_csv(self.data_path)

    def normalize_features(self):
        """
        Normalizes the features in the dataset using the Normalizer.
        """
        scale = Normalizer()
        feature_names_to_normalize = []
        for feature_name in self.features:
            if len(self.data[feature_name].unique()) < len(self.data[feature_name]) or \
               self.data[feature_name].dtype == int or feature_name == self.target:
                continue
            feature_names_to_normalize.append(feature_name)

        self.data[feature_names_to_normalize] = scale.fit_transform(
            self.data[feature_names_to_normalize]
        )

    @staticmethod
    def find_outliers(column, IQR_threshold=1.5):
        """
        Identifies the outliers in a column based on the Interquartile Range (IQR) method.

        Args:
            column (pandas.Series): The column in which to find outliers.
            IQR_threshold (float): The multiplier for the IQR to determine outlier bounds (default is 1.5).

        Returns:
            int: The count of outliers in the column.
        """
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR_threshold * IQR
        upper_bound = Q3 + IQR_threshold * IQR
        return column[(column < lower_bound) | (column > upper_bound)].count()

    def remove_outliers(self, IQR_threshold=1.5):
        """
        Removes outliers from the dataset based on the IQR method.

        Args:
            IQR_threshold (float): The multiplier for the IQR to determine outlier bounds (default is 1.5).
        """
        outliers_count = self.data[self.features].apply(self.find_outliers)
        total_outliers = outliers_count.sum()

        if total_outliers > 0:
            for feature in self.features:
                Q1 = self.data[feature].quantile(0.25)
                Q3 = self.data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - IQR_threshold * IQR
                upper_bound = Q3 + IQR_threshold * IQR
                self.data = self.data[(self.data[feature] >= lower_bound) & (self.data[feature] <= upper_bound)]

    def add_cv_indices(self):
        """
        Adds cross-validation indices to the dataset based on the target variable.

        This method assigns a fold index for each sample in the dataset using StratifiedKFold.
        """
        bins = np.linspace(self.data[self.target].min(), self.data[self.target].max(), self.n_splits + 1)
    
        stratified_target = np.digitize(self.data[self.target], bins)
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        cv_indices = np.zeros(len(self.data), dtype=int)
        for fold_idx, (_, valid_idx) in enumerate(skf.split(self.data, stratified_target)):
            cv_indices[valid_idx] = fold_idx
        
        self.data['fold_id'] = cv_indices

    def preprocess(self):
        """
        Performs all preprocessing steps including data loading, outlier removal, feature normalization, and cross-validation index assignment.

        Returns:
            pandas.DataFrame: The preprocessed dataset with cross-validation indices added.
        """
        self.load_data()
        self.remove_outliers()
        self.normalize_features()
        self.add_cv_indices()
        return self.data
