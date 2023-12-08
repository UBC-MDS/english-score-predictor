import pandas as pd
import numpy as np
import sys
import os
import unittest
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.helper.plot_histogram_with_exclusions import plot_histogram_with_exclusions


class TestPlotHistogramWithExclusions(unittest.TestCase):
    def setUp(self):
        # Test data setup
        self.df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [5, 4, 3, 2, 1],
                "C": [2, 3, 4, 5, 6],
                "D": ["a", "b", "c", "d", "e"],  # Non-numeric column
            }
        )

    def test_dataframe_validation(self):
        print("Running unit tests for plot_histogram_with_exclusions.py...")

        # Test that a TypeError is raised when a non-dataframe is passed
        with self.assertRaises(TypeError):
            plot_histogram_with_exclusions("not a dataframe")

    def test_excluded_columns(self):
        # Test that excluded columns are not in the output
        excluded_column = "B"
        axes = plot_histogram_with_exclusions(
            self.df, columns_to_exclude=[excluded_column]
        )
        included_columns = [
            ax.get_title().strip("Histogram of ") for ax in axes.flatten()
        ]
        self.assertNotIn(excluded_column, included_columns)

    def test_numeric_columns_only(self):
        # Test that only numeric columns are considered for histograms
        axes = plot_histogram_with_exclusions(self.df)
        included_columns = [
            ax.get_title().strip("Histogram of ") for ax in axes.flatten()
        ]
        self.assertNotIn(
            "D", included_columns
        )  # 'D' is non-numeric and should not be included

    def test_return_type(self):
        # Test that the return type is correct (matplotlib AxesSubplot or ndarray of them)
        result = plot_histogram_with_exclusions(self.df)
        self.assertTrue(isinstance(result, (plt.Axes, np.ndarray)))

    def test_fig_size(self):
        # Test that the figure size is correctly applied
        fig_width, fig_height = 10, 5
        plot_histogram_with_exclusions(self.df, fig_size=(fig_width, fig_height))
        figures = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figures:
            self.assertEqual(fig.get_size_inches()[0], fig_width)
            self.assertEqual(fig.get_size_inches()[1], fig_height)

    def tearDown(self):
        # Cleanup (close figures, etc.)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
