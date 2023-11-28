# Adapted from https://github.com/ttimbers/demo-tests-ds-analysis-python/blob/main/tests/test-count_classes.py

import pandas as pd
import numpy as np
import sys
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.linear_model import Ridge

import unittest

# Import the show_feat_coeff function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.show_feat_coeff import show_feat_coeff

class TestShowFeatCoeff(unittest.TestCase):

    def setUp(self):
        # Create simple data for testing
        self.y = np.array([3, 6, 4, 4, 1, 6]).T
        self.X = pd.DataFrame(
            data = np.array(
                [[True, True, False, True, False, False],
                [0.1,  0.3, -0.4,   0, 0.8, -0.9],
                [100,  101,  230, 333, 305,  440]]
            ).T,
            columns = ['feat1', 'feat2', 'feat3']
        )
        self.preprocessor = make_column_transformer(
            (StandardScaler(), ['feat2', 'feat3']),
            (OneHotEncoder(), ['feat1']),
        )
        pipe = make_pipeline(self.preprocessor, Ridge())
        self.model_name = 'ridge'
        # Fit pipeline so that we can access coefficents
        self.pipe_obj = pipe.fit(self.X, self.y)

    def test_show_feat_coeff_returns_dataframe(self):
        # Test for correct return type
        result = show_feat_coeff(self.pipe_obj, self.model_name, self.preprocessor)
        self.assertIsInstance(result, pd.DataFrame, "show_feat_coeff should return a pandas DataFrame")

    def test_show_feat_coeff_output(self):
        # Test for correct values in the data frame
        result = show_feat_coeff(self.pipe_obj, self.model_name, self.preprocessor)
        output_df = pd.DataFrame(
            data = np.array(
                [0.424879, -0.151969, -0.424879, -1.151919]).T,
            index = ['feat1_1.0', 
                     'feat3', 
                     'feat1_0.0',
                     'feat2'],
            columns=["Coefficients"]
            )
        self.assertTrue(all(result == output_df))

    def test_show_feat_coeff_model_name_in_pipeline(self):
        # Test for model name present in pipeline named steps
        self.assertIn(self.model_name, 
                      self.pipe_obj.named_steps,
                      "model name must be present in pipeline named steps")
    
    def test_show_feat_coeff_preprocessor_in_pipeline(self):
        # Test for preprocessor present in pipeline steps
        check_condition = any(isinstance(step[1], ColumnTransformer) for step in self.pipe_obj.steps)
        self.assertTrue(check_condition, "preprocessor must be present in pipeline steps")
    
    def test_show_feat_coeff_column_name(self):
        # Test if the column name is 'Coefficients'
        result_df = show_feat_coeff(self.pipe_obj, self.model_name, self.preprocessor)
        self.assertEqual(result_df.columns.tolist(), ['Coefficients'])

    def test_value_type(self):
        # Test if the coefficent values are floats
        result_df = show_feat_coeff(self.pipe_obj, self.model_name, self.preprocessor)
        self.assertEqual(result_df['Coefficients'].dtype, float)


if __name__ == "__main__":
    unittest.main()