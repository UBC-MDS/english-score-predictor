import sys
import os
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

sys.path.append('src')

from helper.correlation_matrix import pearson_correlation_matrix
from helper.plot_histogram_with_exclusions import plot_histogram_with_exclusions


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Will print verbose messages.")
@click.option("--training-data", type=str, 
              default="data/raw/train_data.csv",
              help="Path to training data ex: data/raw/")
@click.option(
    "--plot-to",
    type=str,
    default="results/figures/",
    help="Path to where all plots should be written ex: results/figures/",
)
@click.option(
    "--pickle-to",
    type=str,
    default="results/models/preprocessor/",
    help="Path to where preprocessor should be written ex: results/models/preprocessor",
)
@click.option(
    "--tables-to",
    type=str,
    default="results/tables/",
    help="Path to where tables should be written ex: results/tables/",
)
def main(verbose,training_data, plot_to, pickle_to, tables_to):
    """
    Main function for data preprocessing and visualization.

    Args:
        training_data (str): Path to the training data CSV file.
        plot_to (str): Path to save the generated plots.
        pickle_to (str): Path to save the preprocessor object.
        tables_to (str): Path to save the tables.

    Returns:
        None
    """
    # Read the training data
    train_df = pd.read_csv(
        training_data, sep=",", on_bad_lines="skip", low_memory=False
    )
    if verbose:
        click.echo("Loaded Training Dataset...")
    

    # Columns to be dropped
    drop_feats = [
        "id", "date", "time", "Unnamed: 0", "tests", "elogit", "dyslexia",
        "dictionary", "already_participated", "natlangs", "primelangs",
        "Can_region", "Ir_region", "US_region", "UK_region", "UK_constituency",
        "gender", "type", "currcountry", "countries",
    ]

    # Additional columns related to questions to be dropped
    question_columns_to_drop = [col for col in train_df.columns if col.startswith("q")]
    drop_feats += question_columns_to_drop

    if verbose:
        click.echo("Creating Numeric Feature Plots...")

    # Create the plots for numeric figures plotting
    plot_1 = plot_histogram_with_exclusions(train_df, columns_to_exclude=drop_feats)

    # Specify the grid layout
    rows = 3
    cols = plot_1.size // rows


    # Create subplots
    for i, plot in enumerate(np.ravel(plot_1)):
        plt.subplot(rows, cols, i + 1)
        #plt.axis("off")  # Turn off axis for each subplot

    # Save the numeric plots as a single figure
    # Add a title to your plot
    plt.suptitle('Distribution of Numeric Features')
    plt.savefig(plot_to + "feat-numeric-figs.png")

    # Categorical features for plotting
    categorical_education = ["education"]
    categorical_feats = ["Eng_little", "speaker_cat"]

    # Education Level Plot
    education_level_plot = (
        alt.Chart(train_df)
        .mark_bar()
        .encode(
            x=alt.X("education:N", title="Education Level"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["count()"],
        )
        .properties(title="Distribution of Education Levels")
    )

    # Save the education level plot
    education_level_plot.save(plot_to + "/education-level-fig.png", format="png")

    if verbose:
        click.echo("Creating Categorical Feature PLots...")

    # Categorical Feature Plot
    categorical_plot = (
        alt.Chart(train_df, title="Distribution of the Categorical features")
        .mark_bar()
        .encode(
            x=alt.X(alt.repeat()),
            y="count()",
        )
        .properties(height=200, width=800)
        .repeat(categorical_feats, columns=1)
    )

    # Save the categorical plots
    plt.suptitle('Distribution of Categorical Features')
    categorical_plot.save(plot_to + "/feat-categoric-figs.png", format="png")
    

    # Columns for the correlation matrix
    normal_cols = list(
        set(train_df.columns.tolist())
        - set(
            [col for col in train_df.columns if col.startswith("q")]
            + ["Unnamed: 0", "date", "time", "id"]
        )
    )
    if verbose:
        click.echo("Creating Correlation Matrix...")

    # Generate and save the Pearson correlation matrix
    cm = pearson_correlation_matrix(train_df[normal_cols], "viridis")
    dfi.export(cm, plot_to + "feat-correlation-matrix.png")

    cm.data.to_csv(tables_to+"correlation-matrix.csv")

    # Final additional feature definitions
    numeric_feats = ["age", "Eng_start", "Eng_country_yrs", "Lived_Eng_per"]

    binary_feats = ["psychiatric"]
    target = ["correct"]

    binary_withNA = [
        "house_Eng", "nat_Eng", "prime_Eng",
    ]

    # Pairwise Correlation Plot 
    # Generate pairwise correlation plots for numeric features
    fig, axs = plt.subplots(len(numeric_feats), len(numeric_feats), figsize=(15, 15))

    for i in range(len(numeric_feats)):
        for j in range(len(numeric_feats)):
            if i == j:
                axs[i, j].hist(train_df[numeric_feats[i]].dropna(), bins=30, color='skyblue', edgecolor='black')
            else:
                axs[i, j].hexbin(train_df[numeric_feats[j]], train_df[numeric_feats[i]], gridsize=30, cmap='Blues')

            if i < len(numeric_feats) - 1:
                axs[i, j].xaxis.set_visible(False)
            if j > 0:
                axs[i, j].yaxis.set_visible(False)

            if j == 0:
                axs[i, j].set_ylabel(numeric_feats[i])
            if i == len(numeric_feats) - 1:
                axs[i, j].set_xlabel(numeric_feats[j])

    plt.tight_layout()
    plt.suptitle('Pairwise Correlation Plots of Numeric Features', y=1.02)
    plt.savefig(plot_to + "/pairwise-correlation-plots.png", format="png")
    plt.show()

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="little"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    binary_NA_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        OneHotEncoder(handle_unknown="ignore", drop="if_binary", dtype=int),
    )

    # Get value counts for the 'education' column
    education_counts = train_df["education"].value_counts()

    # Filter to include only counts greater than 1
    education_counts_greater_than_one = education_counts[education_counts > 100]

    categories_list = education_counts_greater_than_one.index.tolist()

    # Defines the order for Education to be used in the OrdinalEncoder
    education_order = [
        "Graduate Degree", "Some Graduate School", "Undergraduate Degree (3-5 years higher ed)",
        "Some Undergrad (higher ed)", "High School Degree (12-13 years)",
        "Haven't Finished High School (less than 13 years ed)",
        "Didn't Finish High School (less than 13 years ed)", "Others",
    ]

    # Create a transformer using FunctionTransformer
    categorical_education_tranformer = make_pipeline(
        FunctionTransformer(map_to_other, feature_names_out="one-to-one"),
        OrdinalEncoder(categories=[education_order]),
    )

    if verbose:
        click.echo("Creating preprocessor...")

    preprocessor = make_column_transformer(
        ("drop", drop_feats),
        (numeric_transformer, numeric_feats),
        (categorical_transformer, categorical_feats),
        ("passthrough", binary_feats),
        # ("passthrough", target),# for now it is pass through but later most likely log transformation will be applied hence added like this
        (binary_NA_transformer, binary_withNA),
        (categorical_education_tranformer, categorical_education),
    )

    # Save the preprocessor object using pickle
    with open(pickle_to + "preprocessor.pkl", "wb") as file:
         pickle.dump(preprocessor, file)
    if verbose:
        click.echo("Finished Pickling the preprocessor...")



# Define the custom function to map values to 'Other'
def map_to_other(df, categories_list):
    return (
        df["education"].apply(lambda x: x if x in categories_list else "Others")
    ).to_frame()


if __name__ == "__main__":
    main()
    if verbose:
        click.echo("Done!")
