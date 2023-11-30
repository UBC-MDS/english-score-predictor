import sys
import pandas as pd
import numpy as np
import click
import matplotlib.pyplot as plt
import altair as alt
import dataframe_image as dfi
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer,
    
)
import pickle
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

from correlation_matrix import pearson_correlation_matrix
from plot_histogram_with_exclusions import plot_histogram_with_exclusions

@click.command()
@click.option('--training-data', type=str, help="Path to training data ex: data/raw/")
@click.option('--pickle-file', type=str, help="Path to where preprocessor should be written ex: results/models/preprocessor" )

def map_to_other(df, categories_list):
    return (
        df["education"].apply(
            lambda x: x if x in categories_list else "Others")
    ).to_frame()

def main(training_data, pickle_file):
    train_df = pd.read_csv(
        training_data, sep=",", on_bad_lines="skip", low_memory=False
    )
     # Get value counts for the 'education' column
    education_counts = train_df["education"].value_counts()

    # Filter to include only counts greater than 1
    education_counts_greater_than_one = education_counts[education_counts > 100]

    categories_list = education_counts_greater_than_one.index.tolist()
    with open(pickle_file, 'rb') as file:
        preprocessor = pickle.load(file)