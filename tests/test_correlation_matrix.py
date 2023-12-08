import pandas as pd
import sys
import os
import numpy as np
import re


import pandas as pd
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.helper.correlation_matrix import pearson_correlation_matrix


class TestPearsonCorrelationMatrix(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 6, 7, 8, 9],
            "C": [1, 1, 1, 1, 1],  # Constant column
            "D": [2, 2, 2, 2, 2],  # Constant column
            "E": [1, 2, 1, 2, None],  # Variable column with a null value
        }
        self.sample_df = pd.DataFrame(data)

    def test_pearson_correlation_matrix_basic(self):
        print("Running unit tests for correlation_matrix.py")

        # Test basic functionality without specifying colormap
        pearson_matrix = pearson_correlation_matrix(self.sample_df)

        # Check if the result is a Styler object
        self.assertIsInstance(pearson_matrix, pd.io.formats.style.Styler)

        # Check if the expected constant columns are in the excluded list
        expected_columns = ["A", "B"]
        print(pearson_matrix.columns.tolist())
        self.assertEqual(expected_columns, pearson_matrix.columns.tolist())

    def test_pearson_correlation_matrix_empty_df(self):
        empty_df = pd.DataFrame()
        pearson_matrix = pearson_correlation_matrix(empty_df)
        # Verify that the function handles an empty DataFrame
        self.assertIsInstance(pearson_matrix, pd.io.formats.style.Styler)
        self.assertEqual([], pearson_matrix.columns.tolist())

    def test_pearson_correlation_matrix_all_constant(self):
        constant_df = pd.DataFrame(
            {
                "A": [1, 1, 1, 1, 1],
                "B": [2, 2, 2, 2, 2],
            }
        )
        pearson_matrix = pearson_correlation_matrix(constant_df)
        # Verify that constant columns are excluded
        self.assertIsInstance(pearson_matrix, pd.io.formats.style.Styler)
        self.assertEqual([], pearson_matrix.columns.tolist())

    def test_pearson_correlation_matrix_single_unique_value(self):
        unique_value_df = pd.DataFrame(
            {
                "A": [1, 1, 1, 1, 1],
                "B": [1, 2, 1, 2, 1],
            }
        )
        pearson_matrix = pearson_correlation_matrix(unique_value_df)
        # Verify that columns with a single unique value are excluded
        self.assertIsInstance(pearson_matrix, pd.io.formats.style.Styler)
        self.assertEqual(["B"], pearson_matrix.columns.tolist())

    def test_pearson_correlation_matrix_custom_colormap(self):
        custom_colormap = "viridis"
        pearson_matrix = pearson_correlation_matrix(
            self.sample_df, colormap=custom_colormap
        )

        # Verify that the custom colormap is applied correctly
        self.assertIsInstance(pearson_matrix, pd.io.formats.style.Styler)

        # Extract the style information
        style = pearson_matrix.export()

        # Extract the cmap parameter from the apply function
        cmap_paras = style["apply"][0][2]["cmap"]

        # Assert that the extracted colormap parameter matches the expected custom_colormap
        self.assertEqual(custom_colormap, cmap_paras)


if __name__ == "__main__":
    unittest.main()
