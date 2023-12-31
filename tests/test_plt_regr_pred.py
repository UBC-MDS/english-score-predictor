# Adapted from https://github.com/ttimbers/demo-tests-ds-analysis-python/blob/main/tests/test-count_classes.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import unittest

# Import the plt_regr_pred function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.helper.plt_regr_pred import plt_regr_pred


class TestPltRegrPred(unittest.TestCase):
    def setUp(self):
        # Create simple data for testing
        pipe = Pipeline([("stdsclr", StandardScaler()), ("ridge", Ridge())])
        self.y = np.array([3, 6, 4, 4, 1, 6]).T
        self.X = pd.DataFrame(
            data=np.array(
                [
                    [1, 5, 4, 3, 0, 9],
                    [0.1, 0.3, -0.4, 0, 0.8, -0.9],
                    [100, 101, 230, 333, 305, 440],
                ]
            ).T,
            columns=["feat1", "feat2", "feat1"],
        )
        # Fit pipeline so that we can predict
        self.pipe_obj = pipe.fit(self.X, self.y)

    def test_plt_regr_pred_returns_axis(self):
        print("Running unit tests for plt_regr_pred.py")

        # Test for correct return type
        _, ax = plt_regr_pred(self.X, self.y, self.pipe_obj)
        self.assertIsInstance(
            ax, plt.Axes, "plt_regr_pred should return an Axes object"
        )

def test_plt_regr_labels(self):
        # Test if scatter plot has correct labels
        _, ax = plt_regr_pred(self.X, self.y, self.pipe_obj)
        self.assertEqual(ax.get_xlabel(), "Actual Target")
        self.assertEqual(ax.get_ylabel(), "Predicted Target")
        self.assertEqual(ax.get_title(), "2D Histogram of Actual vs. Predicted Target Values")


if __name__ == "__main__":
    unittest.main()
