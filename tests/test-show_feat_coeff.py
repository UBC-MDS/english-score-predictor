# Adapted from https://github.com/ttimbers/demo-tests-ds-analysis-python/blob/main/tests/test-count_classes.py

import pandas as pd
import numpy as np
import pytest
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import unittest

# Import the show_feat_coeff function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.show_feat_coeff import show_feat_coeff

class TestShowFeatCoeff(unittest.TestCase):

    def setUp(self):
        # Create simple data for testing
        pipe = Pipeline([('stdsclr', StandardScaler()), ('ridge', Ridge())])
        y = np.array([3, 6, 4, 4, 1, 6]).T
        self.model_name = 'ridge'
        self.X = pd.DataFrame(
            data = np.array(
                [[  1,    5,    4,   3,   0,    9],
                [0.1,  0.3, -0.4,   0, 0.8, -0.9],
                [100,  101,  230, 333, 305,  440]]
            ).T,
            columns = ['feat1', 'feat2', 'feat1']
        )
        # Fit pipeline so that we can access coefficents
        self.pipe_obj = pipe.fit(self.X, y)

    def test_show_feat_coeff_returns_dataframe(self):
        # Test for correct return type
        result = show_feat_coeff(self.pipe_obj, self.model_name, self.X)
        self.assertIsInstance(result, pd.DataFrame, "show_feat_coeff should return a pandas DataFrame")

    def test_show_feat_coeff_number_of_rows(self):
        # Test for correct number of rows in returned Data Frame as in input X
        result = show_feat_coeff(self.pipe_obj, self.model_name, self.X)
        self.assertEqual(result.shape[0], 
                         self.X[0], 
                         "result should have same number of rows as input data frame"
        )

    def test_show_feat_coeff_output(self):
        # Test for correct values in the data frame
        result = show_feat_coeff(self.pipe_obj, self.model_name, self.X)
        output_df = pd.DataFrame(
            data = np.array(
                [1.511837, 0.169247, -0.433128]).T,
            index = ['feat1', 'feat2', 'feat1'],
            columns=["Coefficients"]
            )
        self.assertEqual(result, output_df)

    def test_show_feat_coeff_model_name_in_pipeline(self):
        # Test for model name present in pipeline named steps
        self.assertIn(self.model_name, 
                      self.pipe_obj.named_steps,
                      "model name must be present in pipeline named steps")
    
    def test_show_feat_coeff_column_name(self):
        # Test if the column name is 'Coefficients'
        result_df = show_feat_coeff(self.pipe_obj, self.model_name, self.X)
        self.assertEqual(result_df.columns.tolist(), ['Coefficients'])

    def test_show_feat_coeff_index_match(self):
        # Test if the index is the same as the input dataframe columns
        result_df = show_feat_coeff(self.pipe_obj, self.model_name, self.X)
        self.assertEqual(result_df.index.tolist(), self.X.columns.tolist())

    def test_value_type(self):
        # Test if the coefficent values are floats
        result_df = show_feat_coeff(self.pipe_obj, self.model_name, self.X)
        self.assertEqual(result_df['Coefficients'].dtype, float)


if __name__ == "__main__":
    pytest.main()