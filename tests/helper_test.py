import unittest
import helper
import pandas as pd
import numpy as np
import altair as alt
from helper import plot_histogram_with_exclusions


class TestHelperMethods(unittest.TestCase):
    # Example test function
    def test_sum(self):
        self.assertEqual(helper.sum(1, 2), 3)
        self.assertEqual(helper.sum(1, -1), 0)
        self.assertEqual(helper.sum(1, 0), 1)
        self.assertEqual(helper.sum(1, 1), 2)


class TestPlotHistogramWithExclusions(unittest.TestCase):

    def setUp(self):
        # Basic numeric DataFrame
        self.df_numeric = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        # Mixed data types
        self.df_mixed = pd.DataFrame({'Numeric': [1, 2, 3], 'Text': ['x', 'y', 'z'], 'Boolean': [True, False, True]})

        # DataFrame with missing values
        self.df_missing = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, np.nan, 6]})

        # Empty DataFrame
        self.df_empty = pd.DataFrame()

        # Large DataFrame
        self.df_large = pd.DataFrame(np.random.rand(1000, 4), columns=['A', 'B', 'C', 'D'])

    def test_numeric_columns_only(self):
        result = plot_histogram_with_exclusions(self.df_numeric)
        self.assertIsInstance(result, alt.vegalite.v4.api.Chart)

    def test_mixed_column_types(self):
        result = plot_histogram_with_exclusions(self.df_mixed)
        self.assertIsInstance(result, alt.vegalite.v4.api.Chart)

    def test_with_missing_values(self):
        result = plot_histogram_with_exclusions(self.df_missing)
        self.assertIsInstance(result, alt.vegalite.v4.api.Chart)

    def test_empty_dataframe(self):
        result = plot_histogram_with_exclusions(self.df_empty)
        self.assertIsInstance(result, alt.vegalite.v4.api.Chart)

    def test_large_dataframe(self):
        result = plot_histogram_with_exclusions(self.df_large)
        self.assertIsInstance(result, alt.vegalite.v4.api.Chart)

    def test_columns_exclusion_existing(self):
        result = plot_histogram_with_exclusions(self.df_numeric, ['A'])
        self.assertNotIn('A', result.to_dict()['datasets'].keys())

    def test_columns_exclusion_non_existing(self):
        result = plot_histogram_with_exclusions(self.df_numeric, ['NonExistingColumn'])
        self.assertIsInstance(result, alt.vegalite.v4.api.Chart)

    def test_all_columns_excluded(self):
        result = plot_histogram_with_exclusions(self.df_numeric, ['A', 'B'])
        self.assertIsInstance(result, alt.vegalite.v4.api.Chart)

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            plot_histogram_with_exclusions("not a dataframe")


if __name__ == "__main__":
    unittest.main()
