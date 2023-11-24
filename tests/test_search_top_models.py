import sys
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.search_top_models import fit_and_return_top_models


class TestFitAndReturnTopModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method will run once before all tests
        cls.X_train, cls.y_train = cls.create_dummy_dataset()

    @staticmethod
    def create_dummy_dataset():
        X, y = make_classification(n_samples=30, n_features=50, random_state=123)
        return pd.DataFrame(X), pd.Series(y)

    @staticmethod
    def create_search_object(search_type="random"):
        ridge = Ridge()
        if search_type == "random":
            search = RandomizedSearchCV(ridge, {"alpha": [1, 2, 3, 4, 5]}, n_iter=3)
        elif search_type == "grid":
            search = GridSearchCV(ridge, {"alpha": [1, 2, 3, 4, 5]})
        return search

    def test_return_type(self):
        """
        Test that the return type is a DataFrame
        """
        search = self.create_search_object()
        result = fit_and_return_top_models(search, 3, self.X_train, self.y_train)
        self.assertIsInstance(result, pd.DataFrame, "Result should be a DataFrame")

    def test_correct_columns(self):
        """
        Test that the returned DataFrame has the correct columns
        """
        search = self.create_search_object()
        columns_expected = [
            "mean_test_score",
            "mean_fit_time",
            "mean_train_score",
            "param_alpha",
        ]
        result = fit_and_return_top_models(
            search, 3, self.X_train, self.y_train, ["param_alpha"], True
        ).T
        self.assertEqual("rank_test_score", result.index.name)
        for column in columns_expected:
            self.assertIn(column, result.columns, f"Missing expected column: {column}")

    def test_train_score_false(self):
        """
        Test that the returned DataFrame has the correct columns
        """
        search = self.create_search_object()
        columns_expected = [
            "mean_test_score",
            "mean_fit_time",
            "param_alpha",
        ]
        result = fit_and_return_top_models(
            search, 3, self.X_train, self.y_train, ["param_alpha"], False
        ).T
        self.assertEqual("rank_test_score", result.index.name)
        self.assertNotIn("mean_train_score", result.columns)

    def test_grid_search(self):
        """
        Test GridSearchCV
        """
        search = self.create_search_object()
        columns_expected = [
            "mean_test_score",
            "mean_fit_time",
            "mean_train_score",
            "param_alpha",
        ]
        result = fit_and_return_top_models(
            search, 3, self.X_train, self.y_train, ["param_alpha"], True
        ).T
        self.assertEqual("rank_test_score", result.index.name)
        for column in columns_expected:
            self.assertIn(column, result.columns, f"Missing expected column: {column}")


if __name__ == "__main__":
    unittest.main()
