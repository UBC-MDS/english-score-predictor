import click
import pandas as pd
import pickle
import sys
import dataframe_image as dfi
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
sys.path.append("src")
from helper.show_feat_coeff import show_feat_coeff

@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Will print verbose messages.")
@click.option(
    "--train",
    default="data/raw/train_data.csv",
    help="Path to the training data.",
)
@click.option(
    "--test",
    default="data/raw/test_data.csv",
    help="Path to the test data.",
)
@click.option(
    "--plot_to",
    type=str,
    default="results/figures/",
    help="Path to where all plots should be written ex: results/figures/",
)
@click.option(
    "--tables_to",
    type=str,
    default="results/tables/",
    help="Path to where tables should be written ex: results/tables/",
)
@click.option(
    "--preprocessor_path",
    default="results/models/preprocessor/preprocessor.pkl",
    help="Path to the preprocessor object (default: results/models/preprocessor/preprocessor.pkl)",
)
@click.option(
        "--best_model_path",
    default="results/models/ridge_best_model.pkl",
    help="Path to the preprocessor object (default: results/models/ridge_best_model.pkl)",
)

def main(verbose, train, test, plot_to, tables_to, preprocessor_path, best_model_path):
    '''
        Main function for discussion and results of final model. 
        Includes reporting test score, showing feature coefficients, and plotting actual vs. predicted values
        Args:
            verbose (str): flag to print verbose messages.
            train (str): Path to the train data CSV file.
            test (str): Path to the test data CSV file.
            plot_to (str): Path to save the generated plots.
            tables_to (str): Path to save the tables.
            preprocessor_path (str): Path to preprocessor object.
            pipeline_path (str): Path to pipeline object.
    Returns:
        None
    '''
    if verbose:
        click.echo("Getting the train and test data...")
    # Get train data
    train_df = pd.read_csv(train, sep=",", on_bad_lines="skip", low_memory=False)
    X_train = train_df.drop(columns="correct")
    y_train = train_df["correct"]
    # Get test data
    test_df = pd.read_csv(test, sep=",", on_bad_lines="skip", low_memory=False)
    X_test = test_df.drop(columns="correct")
    y_test = test_df["correct"]

    if verbose:
        click.echo("Getting and fitting the preprocessor...")
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    preprocessor.fit(X_train, y_train)

    if verbose:
        click.echo("Getting and fitting the pipeline...")
    with open(best_model_path, "rb") as f:
        best_model = pickle.load(f)
    best_model.fit(X_train, y_train)

    if verbose:
        click.echo("Getting Best Test Score...")
    # Get the test score of the best model
    score = pd.DataFrame([best_model.score(X_test, y_test)], columns=["r2"])
    score.to_csv(tables_to + "test-score.csv")

    if verbose:
        click.echo("Running the feature coefficient section...")
    # Get the feature coefficient values
    sfc = show_feat_coeff(best_model, "ridge", preprocessor)
    dfi.export(sfc, plot_to + "feat-coefs.png")
        
    if verbose:
        click.echo("Plotting the actual vs predicted...")
    # Plot actual vs predicted
    ped = PredictionErrorDisplay.from_estimator(
        best_model, X_test, y_test, kind="actual_vs_predicted", subsample=None
    ).plot()
    figure = ped.figure_
    figure.savefig(plot_to + "act-vs-pred.png")

    if verbose:
        click.echo("Done!")

def map_to_other(df):
    """helper function to map categories to others, used in the preprocessor"""
    categories_list = [
        "Graduate Degree",
        "Some Graduate School",
        "Undergraduate Degree (3-5 years higher ed)",
        "Some Undergrad (higher ed)",
        "High School Degree (12-13 years)",
        "Haven't Finished High School (less than 13 years ed)",
        "Didn't Finish High School (less than 13 years ed)",
        "Others",
    ]

    return (
        df["education"].apply(lambda x: x if x in categories_list else "Others")
    ).to_frame()

if __name__ == "__main__":
    main()